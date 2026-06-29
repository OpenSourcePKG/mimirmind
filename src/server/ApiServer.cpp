#include "server/ApiServer.hpp"

#include "model/ChatTemplate.hpp"
#include "model/LlmConfig.hpp"
#include "model/Tokenizer.hpp"
#include "runtime/InferenceEngine.hpp"
#include "runtime/Log.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <ctime>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace mimirmind::server {

using nlohmann::json;

namespace {

std::string makeRequestId() {
    static std::mt19937_64 rng{std::random_device{}()};
    static constexpr char kAlpha[] =
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    std::string out = "chatcmpl-";
    out.reserve(out.size() + 24);
    std::uniform_int_distribution<std::size_t> d{0, sizeof(kAlpha) - 2};
    for (int i = 0; i < 24; ++i) {
        out.push_back(kAlpha[d(rng)]);
    }
    return out;
}

std::int64_t unixNow() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

void sendJson(httplib::Response& res, int status, const json& body) {
    res.status = status;
    res.set_content(body.dump(), "application/json");
}

void sendError(httplib::Response& res, int status,
               std::string_view type, std::string_view message) {
    json body = {
        {"error", {
            {"message", std::string{message}},
            {"type",    std::string{type}},
            {"code",    nullptr},
        }},
    };
    sendJson(res, status, body);
}

struct ChatRequest {
    std::vector<model::ChatMessage> messages;
    std::size_t                     maxTokens{0};   // 0 => use server default
    float                           temperature{0.0F};
    bool                            hasTemperature{false};
    float                           topP{1.0F};
    std::size_t                     topK{0};
    std::uint64_t                   seed{0};
    std::vector<std::string>        stopStrings;
    bool                            stream{false};
    std::string                     model;
};

[[nodiscard]] ChatRequest parseChatRequest(const json& body) {
    ChatRequest req;

    if (!body.is_object()) {
        throw std::runtime_error("request body must be a JSON object");
    }

    if (body.contains("model") && body["model"].is_string()) {
        req.model = body["model"].get<std::string>();
    }

    if (!body.contains("messages") || !body["messages"].is_array()) {
        throw std::runtime_error("messages: missing or not an array");
    }
    for (const auto& m : body["messages"]) {
        if (!m.is_object() || !m.contains("role") || !m.contains("content")) {
            throw std::runtime_error(
                "messages[]: each entry needs role + content");
        }
        const auto roleStr = m["role"].get<std::string>();
        model::ChatRole role;
        if (!model::parseChatRole(roleStr, role)) {
            throw std::runtime_error(
                "messages[].role: unsupported value '" + roleStr + "'");
        }
        std::string content;
        if (m["content"].is_string()) {
            content = m["content"].get<std::string>();
        } else if (m["content"].is_null()) {
            content = "";
        } else {
            // OpenAI also accepts content arrays (multimodal). Not supported.
            throw std::runtime_error(
                "messages[].content: only plain strings are supported");
        }
        req.messages.push_back({role, std::move(content)});
    }

    auto readSize = [&](const char* key, std::size_t& dst) {
        if (body.contains(key) && body[key].is_number_integer()) {
            const auto v = body[key].get<std::int64_t>();
            if (v > 0) {
                dst = static_cast<std::size_t>(v);
            }
        }
    };
    auto readFloat = [&](const char* key, float& dst, bool& has) {
        if (body.contains(key) && body[key].is_number()) {
            dst = body[key].get<float>();
            has = true;
        }
    };

    // OpenAI: max_completion_tokens (current) overrides max_tokens (legacy).
    readSize("max_tokens", req.maxTokens);
    readSize("max_completion_tokens", req.maxTokens);
    readSize("top_k", req.topK);

    bool hasTopP = false;
    readFloat("temperature", req.temperature, req.hasTemperature);
    readFloat("top_p", req.topP, hasTopP);
    (void)hasTopP;

    if (body.contains("seed") && body["seed"].is_number_integer()) {
        req.seed = body["seed"].get<std::uint64_t>();
    }

    if (body.contains("stream") && body["stream"].is_boolean()) {
        req.stream = body["stream"].get<bool>();
    }

    if (body.contains("stop")) {
        const auto& s = body["stop"];
        if (s.is_string()) {
            req.stopStrings.push_back(s.get<std::string>());
        } else if (s.is_array()) {
            for (const auto& e : s) {
                if (e.is_string()) {
                    req.stopStrings.push_back(e.get<std::string>());
                }
            }
        }
    }

    return req;
}

/// Extend the engine's stopIds with the first token of each user-supplied
/// stop string. Multi-token stops are flagged with a warning — robust
/// substring matching belongs in a later iteration.
void extendStopIds(const model::Tokenizer&         tok,
                   const std::vector<std::string>& strings,
                   std::vector<std::int32_t>&      stopIds) {
    for (const auto& s : strings) {
        if (s.empty()) {
            continue;
        }
        const auto ids = tok.encode(s, /*addBos=*/false);
        if (ids.empty()) {
            continue;
        }
        if (ids.size() > 1) {
            MM_LOG_WARN("server",
                        "stop string '{}' encodes to {} tokens; only the "
                        "first ({}) is used — substring matching is TODO",
                        s, ids.size(), ids[0]);
        }
        if (std::find(stopIds.begin(), stopIds.end(), ids[0]) == stopIds.end()) {
            stopIds.push_back(ids[0]);
        }
    }
}

} // namespace

struct ApiServer::Impl {
    runtime::InferenceEngine&  engine;
    ServerConfig               cfg;
    model::ChatTemplate::Style chatStyle;
    httplib::Server            server;
    std::mutex                 engineMutex;
    std::atomic_bool           started{false};

    Impl(runtime::InferenceEngine& e, ServerConfig c)
        : engine{e}, cfg{std::move(c)},
          chatStyle{model::ChatTemplate::detectFromArch(
              engine.config().architecture)} {
        installRoutes();
    }

    void installRoutes() {
        server.Get("/health", [this](const httplib::Request& req,
                                     httplib::Response&       res) {
            handleHealth(req, res);
        });
        server.Get("/v1/models", [this](const httplib::Request& req,
                                        httplib::Response&       res) {
            handleModels(req, res);
        });
        server.Post("/v1/chat/completions",
                    [this](const httplib::Request& req,
                           httplib::Response&       res) {
                        handleChatCompletions(req, res);
                    });

        server.set_exception_handler(
            [](const httplib::Request&,
               httplib::Response& res,
               const std::exception_ptr& ep) {
                std::string msg = "internal error";
                try {
                    if (ep) std::rethrow_exception(ep);
                } catch (const std::exception& e) {
                    msg = e.what();
                } catch (...) {
                    msg = "non-std exception in handler";
                }
                MM_LOG_ERROR("server", "handler exception: {}", msg);
                sendError(res, 500, "server_error", msg);
            });

        server.set_logger([](const httplib::Request& req,
                             const httplib::Response& res) {
            MM_LOG_INFO("server", "{} {} -> {}",
                        req.method, req.path, res.status);
        });
    }

    void handleHealth(const httplib::Request&, httplib::Response& res) {
        sendJson(res, 200, {
            {"status", "ok"},
            {"model",  cfg.modelId},
        });
    }

    void handleModels(const httplib::Request&, httplib::Response& res) {
        json entry = {
            {"id",       cfg.modelId},
            {"object",   "model"},
            {"created",  0},
            {"owned_by", "mimirmind"},
        };
        sendJson(res, 200, {
            {"object", "list"},
            {"data",   json::array({entry})},
        });
    }

    void handleChatCompletions(const httplib::Request& req,
                               httplib::Response&       res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (const std::exception& e) {
            sendError(res, 400, "invalid_request_error",
                      std::string{"invalid JSON: "} + e.what());
            return;
        }

        ChatRequest cr;
        try {
            cr = parseChatRequest(body);
        } catch (const std::exception& e) {
            sendError(res, 400, "invalid_request_error", e.what());
            return;
        }

        if (cr.stream) {
            sendError(res, 501, "server_error",
                      "stream=true is not implemented yet (M7e)");
            return;
        }
        if (cr.messages.empty()) {
            sendError(res, 400, "invalid_request_error",
                      "messages must not be empty");
            return;
        }

        const auto& tok = engine.tokenizer();
        const auto promptIds = model::ChatTemplate::encode(
            chatStyle, tok, cr.messages, /*addGenerationPrompt=*/true);

        std::vector<std::int32_t> stopIds =
            model::ChatTemplate::stopIds(chatStyle, tok);
        extendStopIds(tok, cr.stopStrings, stopIds);

        runtime::GenerateParams params;
        params.maxNewTokens = cr.maxTokens > 0 ? cr.maxTokens : cfg.defaultMaxNew;
        params.stopIds      = stopIds;
        if (cr.hasTemperature) {
            params.sampling.temperature = cr.temperature;
        }
        params.sampling.topP = cr.topP;
        params.sampling.topK = cr.topK;
        params.sampling.seed = cr.seed;

        runtime::GenerateStats stats;
        std::vector<std::int32_t> generated;

        {
            // Engine is single-instance, mutable scratch + sampler. Serialise.
            std::lock_guard<std::mutex> lk{engineMutex};
            try {
                generated = engine.generate(promptIds, params, {}, &stats);
            } catch (const std::exception& e) {
                MM_LOG_ERROR("server", "generate failed: {}", e.what());
                sendError(res, 500, "server_error",
                          std::string{"generate: "} + e.what());
                return;
            }
        }

        // Strip a trailing stop token from the rendered text so clients
        // don't see "<|im_end|>" tacked onto the answer.
        std::vector<std::int32_t> visible = generated;
        bool hitStop = false;
        if (!visible.empty()) {
            const auto last = visible.back();
            if (last == tok.eosId() ||
                std::find(stopIds.begin(), stopIds.end(), last) != stopIds.end()) {
                visible.pop_back();
                hitStop = true;
            }
        }
        // hitEos in stats means we broke out of the decode loop early
        // because the *previously* sampled token was a stop. The token
        // itself is still in `generated` (added before the check). Mirror
        // that behaviour for the simple last-token strip above.
        hitStop = hitStop || stats.hitEos;

        const std::string text = tok.decode(visible, /*skipSpecial=*/true);

        const std::string respId = makeRequestId();
        const std::int64_t now   = unixNow();
        const std::string finish = hitStop ? "stop" : "length";

        const std::string echoModel = cr.model.empty() ? cfg.modelId : cr.model;

        json response = {
            {"id",      respId},
            {"object",  "chat.completion"},
            {"created", now},
            {"model",   echoModel},
            {"choices", json::array({
                json{
                    {"index", 0},
                    {"message", {
                        {"role",    "assistant"},
                        {"content", text},
                    }},
                    {"finish_reason", finish},
                },
            })},
            {"usage", {
                {"prompt_tokens",     promptIds.size()},
                {"completion_tokens", visible.size()},
                {"total_tokens",      promptIds.size() + visible.size()},
            }},
        };

        MM_LOG_INFO("server",
                    "chat.completion id={} model={} prompt={} new={} "
                    "prefill={:.0f}ms decode={:.0f}ms finish={}",
                    respId, echoModel,
                    promptIds.size(), visible.size(),
                    stats.prefillMs, stats.decodeMs, finish);

        sendJson(res, 200, response);
    }

    void run() {
        MM_LOG_INFO("server", "binding {}:{} (model={})",
                    cfg.host, cfg.port, cfg.modelId);
        started.store(true);
        if (!server.listen(cfg.host, cfg.port)) {
            started.store(false);
            throw std::runtime_error("ApiServer: failed to bind " +
                                     cfg.host + ":" + std::to_string(cfg.port));
        }
    }

    void stop() {
        if (started.exchange(false)) {
            server.stop();
        }
    }
};

ApiServer::ApiServer(runtime::InferenceEngine& engine, ServerConfig cfg)
    : _impl{std::make_unique<Impl>(engine, std::move(cfg))} {}

ApiServer::~ApiServer() = default;

void ApiServer::run()  { _impl->run(); }
void ApiServer::stop() { _impl->stop(); }

} // namespace mimirmind::server