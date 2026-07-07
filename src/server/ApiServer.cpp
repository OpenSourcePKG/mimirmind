#include "server/ApiServer.hpp"

#include "model/ChatTemplate.hpp"
#include "model/LlmConfig.hpp"
#include "model/ResponseCleaner.hpp"
#include "model/Tokenizer.hpp"
#include "runtime/FanController.hpp"
#include "runtime/GpuClockGovernor.hpp"
#include "runtime/InferenceEngine.hpp"
#include "runtime/Log.hpp"
#include "runtime/PerfRegressionDetector.hpp"
#include "runtime/PowerMonitor.hpp"
#include "runtime/SpeculativeDecoder.hpp"
#include "runtime/ThermalGuard.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <mutex>
#include <optional>
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

    // M7f — repetition-control penalties. `has*` flags distinguish
    // "client sent 0" from "client sent nothing"; the latter picks up
    // the server-side default while the former stays at exactly 0.
    float                           frequencyPenalty{0.0F};
    bool                            hasFrequencyPenalty{false};
    float                           presencePenalty{0.0F};
    bool                            hasPresencePenalty{false};
    float                           repetitionPenalty{1.0F};
    bool                            hasRepetitionPenalty{false};
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

    // M7f — penalties. `frequency_penalty` + `presence_penalty` match
    // the OpenAI schema; `repetition_penalty` is a mimirmind extension
    // matching llama.cpp's convention (multiplicative, range ~1.0-1.3).
    readFloat("frequency_penalty",  req.frequencyPenalty,  req.hasFrequencyPenalty);
    readFloat("presence_penalty",   req.presencePenalty,   req.hasPresencePenalty);
    readFloat("repetition_penalty", req.repetitionPenalty, req.hasRepetitionPenalty);

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

// ---- M-PT: server-side length discipline ----------------------------------
//
// Prompt trimming and max_new clamping so a request that doesn't fit the
// KV cache budget gracefully degrades to a smaller-but-served response,
// instead of the pre-M-PT behaviour where InferenceEngine::ensureCapacity
// throws and the client sees a 500. See the M-PT Synaipse note for the
// design rationale.

struct TrimReport {
    std::size_t droppedMessages{0};
    std::size_t originalPromptTokens{0};
    std::size_t effectivePromptTokens{0};
    std::size_t maxNewClampedFrom{0};   // 0 = no clamp
    std::size_t maxNewClampedTo{0};
    bool        contextExtrapolated{false};
    std::size_t contextExtrapolatedBy{0};

    [[nodiscard]] bool fired() const noexcept {
        return droppedMessages > 0
            || maxNewClampedFrom > 0
            || contextExtrapolated;
    }
};

// Slack — matches InferenceEngine::ensureCapacity's `+ 4`. If those two
// ever drift, ensureCapacity will start throwing again with a request the
// trim helper thought was safe.
constexpr std::size_t kMPtCapSlack = 4;
constexpr std::size_t kMPtTrimIterLimit = 20;

/// Iteratively drop the oldest non-system, non-last-user message until
/// `promptIds.size() + maxNewTokens + slack <= maxContextTokens`. Then
/// clamp `maxNewTokens` if the prompt alone still leaves too little
/// budget. Emits an error message (via `errorMessage` out-param) and
/// returns false only in the extreme case that the prompt itself is
/// larger than the context budget minus slack.
[[nodiscard]] bool applyPromptTrim(std::vector<model::ChatMessage>& msgs,
                                    std::vector<std::int32_t>&       promptIds,
                                    std::size_t&                     maxNewTokens,
                                    std::size_t                      maxContextTokens,
                                    std::size_t                      modelContextLength,
                                    const model::Tokenizer&          tok,
                                    model::ChatTemplate::Style       chatStyle,
                                    TrimReport&                      report,
                                    std::string&                     errorMessage) {
    report.originalPromptTokens = promptIds.size();

    // 1. Message-drop-trim loop.
    for (std::size_t iter = 0; iter < kMPtTrimIterLimit; ++iter) {
        if (promptIds.size() + maxNewTokens + kMPtCapSlack <= maxContextTokens) {
            break;
        }
        // Find the LAST user message — must be preserved. Walk backwards.
        std::size_t lastUserIdx = static_cast<std::size_t>(-1);
        for (std::size_t i = msgs.size(); i-- > 0; ) {
            if (msgs[i].role == model::ChatRole::User) {
                lastUserIdx = i;
                break;
            }
        }
        // Pick the earliest droppable index: not system, not last-user.
        std::size_t dropIdx = static_cast<std::size_t>(-1);
        for (std::size_t i = 0; i < msgs.size(); ++i) {
            if (msgs[i].role == model::ChatRole::System) continue;
            if (i == lastUserIdx) continue;
            dropIdx = i;
            break;
        }
        if (dropIdx == static_cast<std::size_t>(-1)) {
            // Only system + last-user left — cannot drop more.
            break;
        }
        msgs.erase(msgs.begin() + static_cast<std::ptrdiff_t>(dropIdx));
        ++report.droppedMessages;
        promptIds = model::ChatTemplate::encode(chatStyle, tok, msgs,
                                                /*addGenerationPrompt=*/true);
    }

    // 2. Post-trim: clamp maxNew against remaining budget if still too big.
    const std::size_t Tp = promptIds.size();
    if (Tp + maxNewTokens + kMPtCapSlack > maxContextTokens) {
        if (Tp + kMPtCapSlack >= maxContextTokens) {
            // Prompt alone exhausts the budget even without any completion.
            // Extreme edge — reject with the clearest possible message.
            errorMessage =
                "prompt too long: " + std::to_string(Tp) +
                " tokens after trimming " + std::to_string(report.droppedMessages) +
                " message(s) + slack " + std::to_string(kMPtCapSlack) +
                " does not fit context budget " + std::to_string(maxContextTokens) +
                " — raise MIMIRMIND_MAX_CONTEXT_TOKENS or shorten the last "
                "user message";
            return false;
        }
        const std::size_t newMax = maxContextTokens - Tp - kMPtCapSlack;
        report.maxNewClampedFrom = maxNewTokens;
        report.maxNewClampedTo   = newMax;
        maxNewTokens             = newMax;
    }

    report.effectivePromptTokens = Tp;

    // 3. Model-native context-length warning (RoPE extrapolation zone).
    //    Zero means "not populated by GGUF" — skip.
    if (modelContextLength > 0 && Tp > modelContextLength) {
        report.contextExtrapolated   = true;
        report.contextExtrapolatedBy = Tp - modelContextLength;
    }
    return true;
}

/// Apply the M-PT report to an httplib response as `x-mimirmind-*`
/// headers. Only headers for fields that actually fired are set —
/// clients that never trigger the trim path see zero overhead.
void attachTrimHeaders(httplib::Response& res, const TrimReport& r) {
    if (r.droppedMessages > 0) {
        res.set_header("x-mimirmind-dropped-messages",
                       std::to_string(r.droppedMessages));
    }
    if (r.maxNewClampedFrom > 0) {
        res.set_header("x-mimirmind-max-new-clamped",
                       std::to_string(r.maxNewClampedFrom) + "->" +
                       std::to_string(r.maxNewClampedTo));
    }
    if (r.contextExtrapolated) {
        res.set_header("x-mimirmind-context-extrapolated-by",
                       std::to_string(r.contextExtrapolatedBy));
    }
}

/// Attach the M-PT report to a chat-completion `usage` JSON block.
/// Fields prefixed `mimirmind_` are additive extensions over OpenAI
/// — vanilla clients ignore unknown keys.
void attachTrimUsage(json& usage, const TrimReport& r) {
    if (r.droppedMessages > 0) {
        usage["mimirmind_dropped_messages"]        = r.droppedMessages;
        usage["mimirmind_original_prompt_tokens"]  = r.originalPromptTokens;
    }
    if (r.maxNewClampedFrom > 0) {
        usage["mimirmind_max_new_clamped_from"] = r.maxNewClampedFrom;
        usage["mimirmind_max_new_clamped_to"]   = r.maxNewClampedTo;
    }
    if (r.contextExtrapolated) {
        usage["mimirmind_context_extrapolated_by"] = r.contextExtrapolatedBy;
    }
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

// ---- SSE / streaming chunk builders ---------------------------------------

json streamChunkSkeleton(const std::string& id,
                         std::int64_t       created,
                         const std::string& model) {
    return json{
        {"id",      id},
        {"object",  "chat.completion.chunk"},
        {"created", created},
        {"model",   model},
    };
}

json buildRoleChunk(const std::string& id, std::int64_t created,
                    const std::string& model) {
    json out = streamChunkSkeleton(id, created, model);
    out["choices"] = json::array({
        json{
            {"index",         0},
            {"delta",         json{{"role", "assistant"}}},
            {"finish_reason", nullptr},
        },
    });
    return out;
}

json buildContentChunk(const std::string& id, std::int64_t created,
                       const std::string& model, std::string_view text) {
    json out = streamChunkSkeleton(id, created, model);
    out["choices"] = json::array({
        json{
            {"index",         0},
            {"delta",         json{{"content", std::string{text}}}},
            {"finish_reason", nullptr},
        },
    });
    return out;
}

json buildFinishChunk(const std::string& id, std::int64_t created,
                      const std::string& model, std::string_view finishReason) {
    json out = streamChunkSkeleton(id, created, model);
    out["choices"] = json::array({
        json{
            {"index",         0},
            {"delta",         json::object()},
            {"finish_reason", std::string{finishReason}},
        },
    });
    return out;
}

/// Format `payload` as one SSE event and push it onto `sink`. Returns
/// false if the sink refused the write — caller should treat that as
/// "client disconnected".
[[nodiscard]] bool writeSseEvent(httplib::DataSink& sink, const json& payload) {
    const std::string line = "data: " + payload.dump() + "\n\n";
    return sink.write(line.data(), line.size());
}

/// Named SSE event: `event: <name>\ndata: <json>\n\n`. OpenAI-style
/// stream consumers ignore named events (they only demux `data:` lines
/// that don't have a preceding `event:` field), so this is a safe
/// place to publish mimirmind-specific side-channel signals like the
/// `prefill_done` progress hint that browsers can pick up via
/// `EventSource.addEventListener('prefill_done', ...)`.
[[nodiscard]] bool writeSseNamedEvent(httplib::DataSink&     sink,
                                      std::string_view       name,
                                      const json&            payload) {
    std::string line;
    line.reserve(name.size() + payload.dump().size() + 16);
    line.append("event: ").append(name).append("\n");
    line.append("data: ").append(payload.dump()).append("\n\n");
    return sink.write(line.data(), line.size());
}

[[nodiscard]] bool writeSseDone(httplib::DataSink& sink) {
    static constexpr std::string_view kDone = "data: [DONE]\n\n";
    return sink.write(kDone.data(), kDone.size());
}

/// Index of the first byte of an incomplete UTF-8 codepoint at the end
/// of `s`, or `s.size()` if the buffer ends on a codepoint boundary.
/// Used to split a token-stream into "safe-to-emit" prefix + "buffer
/// for next token" suffix — GPT-2 BPE can put a multi-byte UTF-8 char
/// across two tokens, and SSE consumers choke on partial codepoints.
[[nodiscard]] std::size_t utf8IncompleteTailStart(const std::string& s) {
    if (s.empty()) return 0;

    // Walk back over continuation bytes (10xxxxxx). Max 3 trailing
    // continuations are legal in UTF-8 (4-byte codepoint).
    std::size_t i        = s.size();
    std::size_t contSeen = 0;
    while (i > 0 && (static_cast<unsigned char>(s[i - 1]) & 0xC0) == 0x80
                 && contSeen < 3) {
        --i;
        ++contSeen;
    }
    if (i == 0) {
        // Pathological: buffer is all continuation bytes. Emit and let
        // the receiver deal with it — we can't keep buffering forever.
        return s.size();
    }

    const auto lead = static_cast<unsigned char>(s[i - 1]);
    std::size_t expectedCont;
    if      ((lead & 0x80) == 0)    expectedCont = 0;  // ASCII
    else if ((lead & 0xE0) == 0xC0) expectedCont = 1;
    else if ((lead & 0xF0) == 0xE0) expectedCont = 2;
    else if ((lead & 0xF8) == 0xF0) expectedCont = 3;
    else                            return s.size();   // malformed lead

    if (contSeen >= expectedCont) {
        return s.size();        // last codepoint is complete
    }
    return i - 1;               // last codepoint starts here and is partial
}

} // namespace

struct ApiServer::Impl {
    runtime::InferenceEngine&                        engine;
    runtime::InferenceEngine*                        draftEngine{nullptr};
    // M9.11.4 — spec-dec orchestrator. Owned; only constructed when a
    // draft engine is present. When null, all generate calls go straight
    // to `engine.generate()`. When set, callers route through
    // `speculativeDecoder->generate()` which falls through to
    // target-only for sampled requests and the env kill-switch.
    std::unique_ptr<runtime::SpeculativeDecoder>     speculativeDecoder;
    ServerConfig                          cfg;
    model::ChatTemplate::Style            chatStyle;
    httplib::Server                       server;
    std::mutex                            engineMutex;
    std::atomic_bool                      started{false};

    // Baseline RAPL snapshot taken once after the engine reports the
    // model is loaded — represents "engine idle, server warmed up".
    // Total-since-start energy is computed against this; before it's
    // captured, totals report 0.
    std::mutex                            powerStateMutex;
    runtime::PowerMonitor::Snapshot       powerBaseline{};
    runtime::PowerMonitor::Snapshot       powerLastStatus{};
    std::chrono::steady_clock::time_point baselineWallStart{};
    bool                                  baselineCaptured{false};

    // In-flight request snapshot for /v1/system/status.current_request.
    // A UI (e.g. Pegenaut) can poll this to render "prefill 24/42, decode
    // 15/4096" instead of showing a spinner for the ~15 minutes a long
    // prompt with a large max_tokens can take. Guarded by
    // `currentRequestMutex`; only one chat request runs at a time thanks
    // to `engineMutex`, so writers never contend, only the status poller
    // reads while a writer may be updating.
    struct CurrentRequest {
        std::string request_id;
        std::chrono::steady_clock::time_point started_at;
        std::size_t prompt_tokens{0};
        std::size_t max_new_tokens{0};
        std::size_t prefill_blocks_done{0};
        std::size_t prefill_blocks_total{0};
        double      prefill_elapsed_ms{0.0};
        bool        prefill_done{false};
        std::size_t decode_tokens_emitted{0};
        bool        streaming{false};
    };
    mutable std::mutex                    currentRequestMutex;
    std::optional<CurrentRequest>         currentRequest;

    Impl(runtime::InferenceEngine& e, ServerConfig c,
         runtime::InferenceEngine* draft)
        : engine{e}, draftEngine{draft}, cfg{std::move(c)},
          chatStyle{model::ChatTemplate::detectFromArch(
              engine.config().architecture)} {
        // M9.11.4 — stand up the spec-dec orchestrator now that both
        // engines are ready. Reads MIMIRMIND_SPEC_DEC_N from env for
        // draft-N tuning; default 4 matches the roadmap prediction.
        if (draftEngine != nullptr) {
            runtime::SpeculativeDecoder::Config sdcfg{};
            if (const char* envN = std::getenv("MIMIRMIND_SPEC_DEC_N")) {
                if (envN[0] != '\0') {
                    const auto n = std::strtoull(envN, nullptr, 10);
                    if (n > 0 && n < 64) {
                        sdcfg.draftN = static_cast<std::size_t>(n);
                    } else {
                        MM_LOG_WARN("server",
                                    "MIMIRMIND_SPEC_DEC_N={} out of range "
                                    "[1..63], keeping default {}",
                                    envN, sdcfg.draftN);
                    }
                }
            }
            speculativeDecoder =
                std::make_unique<runtime::SpeculativeDecoder>(
                    engine, *draftEngine, sdcfg);
        }
        installRoutes();
        // Engine is constructed model-loaded already (loadModel is called
        // before ApiServer wraps it in main.cpp). Capture the baseline
        // here so it represents "right after model load, before any
        // requests". If the monitor is unavailable the snapshot will be
        // empty and totals stay at 0.
        if (auto* mon = engine.powerMonitor(); mon != nullptr && mon->available()) {
            std::lock_guard lk{powerStateMutex};
            powerBaseline      = mon->snapshot();
            powerLastStatus    = powerBaseline;
            baselineWallStart  = std::chrono::steady_clock::now();
            baselineCaptured   = true;
        }
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
        server.Get("/v1/system/status",
                   [this](const httplib::Request& req,
                          httplib::Response&       res) {
                       handleSystemStatus(req, res);
                   });
        server.Get("/v1/system/info",
                   [this](const httplib::Request& req,
                          httplib::Response&       res) {
                       handleSystemInfo(req, res);
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

    /// Static-only companion to /v1/system/status. Everything reported
    /// here is fixed for the lifetime of the process — model config,
    /// KV-cache size, hardware descriptor, thermal-profile limits,
    /// governor envelope, autotune bench results, perf-regression
    /// constants. Clients are expected to call this ONCE at startup and
    /// then poll /v1/system/status for the dynamic values (temp, cap,
    /// watts, throttle state, current/baseline p50, last alert).
    void handleSystemInfo(const httplib::Request&, httplib::Response& res) {
        const auto& modelCfg = engine.config();
        const auto& tok      = engine.tokenizer();
        const auto& devInfo  = engine.ctx().info();
        const auto& usmLim   = engine.allocator().limits();

        // Model architecture + dims
        json model = {
            {"id",                   cfg.modelId},
            {"arch",                 modelCfg.architecture},
            {"block_count",          modelCfg.blockCount},
            {"context_length",       modelCfg.contextLength},
            {"embedding_length",     modelCfg.embeddingLength},
            {"feed_forward_length",  modelCfg.feedForwardLength},
            {"head_count",           modelCfg.headCount},
            {"head_count_kv",        modelCfg.headCountKv},
            {"key_length",           modelCfg.keyLength},
            {"value_length",         modelCfg.valueLength},
            {"rms_norm_eps",         modelCfg.rmsNormEps},
            {"rope_freq_base",       modelCfg.ropeFreqBase},
        };
        if (modelCfg.slidingWindow > 0) {
            model["sliding_window"]     = modelCfg.slidingWindow;
            model["rope_freq_base_swa"] = modelCfg.ropeFreqBaseSwa;
            model["key_length_swa"]     = modelCfg.keyLengthSwa;
            model["value_length_swa"]   = modelCfg.valueLengthSwa;
            std::size_t swa = 0;
            for (bool b : modelCfg.slidingWindowPattern) {
                if (b) ++swa;
            }
            model["swa_layer_count"]  = swa;
            model["full_layer_count"] =
                modelCfg.slidingWindowPattern.size() - swa;
        }
        if (modelCfg.expertCount > 0) {
            model["expert_count"]      = modelCfg.expertCount;
            model["expert_used_count"] = modelCfg.expertUsedCount;
        }

        // Tokenizer
        json tokenizer = {
            {"model",      std::string{tok.modelType()}},
            {"vocab_size", tok.vocabSize()},
            {"bos_id",     tok.bosId()},
            {"eos_id",     tok.eosId()},
            {"unk_id",     tok.unknownId()},
            {"pad_id",     tok.padId()},
        };

        // KV cache — hard limit the engine will admit. Prompt +
        // max_new_tokens + a small slack (4 tokens) must fit.
        json kvCache = {
            {"max_context_tokens", engine.maxContextTokens()},
            {"layer_count",        modelCfg.blockCount},
            {"dtype",              (engine.kvDtype() ==
                                        runtime::KvDtype::FP16
                                    ? "fp16" : "f32")},
            {"element_bytes",      runtime::kvElementBytes(engine.kvDtype())},
        };

        // Level-Zero device descriptor
        json hardware = {
            {"device_name",             devInfo.name},
            {"device_uuid",             devInfo.uuid},
            {"vendor_id",               devInfo.vendorId},
            {"device_id",               devInfo.deviceId},
            {"num_compute_units",       devInfo.numComputeUnits},
            {"core_clock_rate_mhz",     devInfo.coreClockRate},
            {"total_local_mem_bytes",   devInfo.totalLocalMem},
            {"usm_per_alloc_max_bytes", usmLim.perAllocMaxBytes},
        };

        // GPU clock envelope — the static parts of /system/status.gpu_clock
        // (current_cap_mhz is dynamic and stays in /status).
        json gpuClockEnvelope;
        if (auto* gov = engine.gpuClockGovernor();
            gov != nullptr && gov->available()) {
            gpuClockEnvelope = {
                {"card_path",     std::string{gov->cardPath()}},
                {"rp0_mhz",       gov->rp0Mhz()},
                {"rpn_mhz",       gov->rpnMhz()},
                {"target_temp_c", gov->targetTempC()},
            };
            // M9.11.a — bench-repeatability pin. When present every
            // perf number in the same session was taken under this cap,
            // so the ledger can attribute it correctly.
            if (gov->pinned()) {
                gpuClockEnvelope["pin"] = {
                    {"intent",  std::string{gov->pinIntent()}},
                    {"raw_env", std::string{gov->pinRawEnv()}},
                    {"cap_mhz", gov->pinnedMhz()},
                };
            }
        } else {
            gpuClockEnvelope = nullptr;
        }

        // Thermal profile — the static limits (readings.package_temp_c and
        // throttle state stay in /status).
        json thermalProfile;
        if (auto* guard = engine.thermalGuard(); guard != nullptr) {
            const auto& p = guard->profile();
            thermalProfile = {
                {"name",                    p.name},
                {"description",             p.description},
                {"package_temp_hard_c",     p.package_temp_hard_c.has_value()
                                              ? json(*p.package_temp_hard_c)
                                              : json(nullptr)},
                {"package_temp_soft_c",     p.package_temp_soft_c.has_value()
                                              ? json(*p.package_temp_soft_c)
                                              : json(nullptr)},
                {"package_throttle_max_ms", p.package_throttle_max_ms},
            };
        } else {
            thermalProfile = nullptr;
        }

        // Perf-regression tuning constants. Alert thresholds, windows,
        // warmup — everything that is compile-time constexpr and can't
        // change without a rebuild.
        json perfRegressionConfig = {
            {"threshold_ratio",
             runtime::PerfRegressionDetector::kAlertThreshold},
            {"baseline_window_days",
             runtime::PerfRegressionDetector::kBaselineDays},
            {"warmup_tokens",
             runtime::PerfRegressionDetector::kWarmupTokens},
            {"rolling_window",
             runtime::PerfRegressionDetector::kRollingWindow},
            {"min_run_samples",
             runtime::PerfRegressionDetector::kMinRunSamples},
            {"min_baseline_n",
             runtime::PerfRegressionDetector::kMinBaselineN},
        };

        // Build / process identity. internal_version bumps on every
        // container restart; a client that caches /system/info should
        // key its cache on this value.
        json build = json::object();
        if (auto* det = engine.perfRegressionDetector()) {
            build["internal_version"] = det->internalVersion();
        }

        // Fan controller envelope — chip identity + baseline values
        // captured at startup. Absent (null) when no writable pwm/pwm_enable
        // pair was found. Live pwm/rpm belong in /system/status, not here.
        json fanEnvelope;
        if (auto* fc = engine.fanController();
            fc != nullptr && fc->available()) {
            fanEnvelope = {
                {"chip_name",         std::string{fc->chipName()}},
                {"chip_path",         std::string{fc->chipPath()}},
                {"pwm_path",          std::string{fc->pwmPath()}},
                {"pwm_enable_path",   std::string{fc->pwmEnablePath()}},
                {"fan_input_path",    std::string{fc->fanInputPath()}},
                {"original_pwm",      fc->originalPwm()},
                {"original_enable",   fc->originalEnableMode()},
                {"boost_pwm",         fc->boostPwm()},
                {"min_safe_pwm",      fc->minSafePwm()},
            };
        } else {
            fanEnvelope = nullptr;
        }

        // M9.11.1 + M9.11.4 — Speculative-decoding readiness. `status`
        // reflects whether the draft engine loaded AND the M9.11.4
        // orchestrator was wired up. `draft_n` is the per-round draft
        // token budget; `mode` is currently always "greedy" — the
        // sampled path falls through to target-only until M9.11.4b.
        json speculativeDecoding;
        if (draftEngine != nullptr && speculativeDecoder != nullptr) {
            const auto& draftCfg = draftEngine->config();
            speculativeDecoding = {
                {"status",                 "ready"},
                {"mode",                   "greedy"},
                {"draft_n",                speculativeDecoder->config().draftN},
                {"draft_model_arch",       draftCfg.architecture},
                {"draft_block_count",      draftCfg.blockCount},
                {"draft_embedding_length", draftCfg.embeddingLength},
            };
        } else {
            speculativeDecoding = {
                {"status", "disabled"},
            };
        }

        json body = {
            {"model",                  model},
            {"tokenizer",              tokenizer},
            {"kv_cache",               kvCache},
            {"hardware",               hardware},
            {"gpu_clock_envelope",     gpuClockEnvelope},
            {"fan_envelope",           fanEnvelope},
            {"thermal_profile",        thermalProfile},
            {"perf_regression_config", perfRegressionConfig},
            {"kernels",                buildKernelsBlock()},
            {"speculative_decoding",   speculativeDecoding},
            {"build",                  build},
        };
        sendJson(res, 200, body);
    }

    void handleSystemStatus(const httplib::Request&, httplib::Response& res) {
        auto* guard = engine.thermalGuard();
        json body{
            {"profile_active", guard != nullptr},
        };
        if (guard == nullptr) {
            body["warning"] =
                "no thermal profile configured — engine is unprotected. "
                "Set --thermal-profile or MIMIRMIND_THERMAL_PROFILE.";
            sendJson(res, 200, body);
            return;
        }

        const auto& p        = guard->profile();
        const auto  decision = guard->decide();
        const auto  reading  = guard->lastReading();

        json profileJson{
            {"name",        p.name},
            {"description", p.description},
        };
        if (p.hasPackageLimits()) {
            profileJson["package_temp_soft_c"]    = *p.package_temp_soft_c;
            profileJson["package_temp_hard_c"]    = *p.package_temp_hard_c;
            profileJson["package_throttle_max_ms"] = p.package_throttle_max_ms;
        }

        json readingsJson = json::object();
        if (reading.package_temp_c.has_value()) {
            readingsJson["package_temp_c"] = *reading.package_temp_c;
        }
        if (reading.ram_total_mib.has_value()) {
            readingsJson["ram_total_mib"] = *reading.ram_total_mib;
        }
        if (reading.ram_available_mib.has_value()) {
            readingsJson["ram_available_mib"] = *reading.ram_available_mib;
        }

        const char* stateStr =
            decision.state == runtime::ThermalDecision::State::Critical   ? "critical"
            : decision.state == runtime::ThermalDecision::State::Throttling ? "throttling"
                                                                            : "ok";

        body["profile"]   = std::move(profileJson);
        body["readings"]  = std::move(readingsJson);
        body["throttle"]  = json{
            {"state",                stateStr},
            {"current_pause_ms",     static_cast<int>(decision.pause.count())},
            {"next_request_allowed", decision.admit_new_request},
            {"reason",               decision.reason.empty()
                                       ? json{}
                                       : json{decision.reason}},
        };
        body["power"]           = buildPowerBlock();
        body["gpu_clock"]       = buildGpuClockBlock();
        body["fan"]             = buildFanBlock();
        body["kernels"]         = buildKernelsBlock();
        body["perf_regression"] = buildPerfRegressionBlock();
        body["current_request"] = buildCurrentRequestBlock();
        sendJson(res, 200, body);
    }

    // --- In-flight request tracking for status polling ------------------
    //
    // Called at chat-handler entry / exit so the status endpoint can
    // report progress instead of leaving a UI stuck on a spinner for the
    // 15 minutes a large prompt with max_tokens=4096 can take. The four
    // updaters are cheap (a mutex + a few writes) and safe to call from
    // per-block / per-token callbacks.

    void beginCurrentRequest(std::string   id,
                             std::size_t   promptTokens,
                             std::size_t   maxNew,
                             bool          streaming) {
        std::lock_guard<std::mutex> lk{currentRequestMutex};
        CurrentRequest r;
        r.request_id     = std::move(id);
        r.started_at     = std::chrono::steady_clock::now();
        r.prompt_tokens  = promptTokens;
        r.max_new_tokens = maxNew;
        r.streaming      = streaming;
        currentRequest   = std::move(r);
    }

    void endCurrentRequest() noexcept {
        std::lock_guard<std::mutex> lk{currentRequestMutex};
        currentRequest.reset();
    }

    void updateCurrentPrefillProgress(
        const runtime::InferenceEngine::PrefillProgress& p) {
        std::lock_guard<std::mutex> lk{currentRequestMutex};
        if (!currentRequest.has_value()) return;
        currentRequest->prefill_blocks_done  = p.blocksDone;
        currentRequest->prefill_blocks_total = p.blocksTotal;
        currentRequest->prefill_elapsed_ms   = p.elapsedMs;
    }

    void markCurrentPrefillDone() {
        std::lock_guard<std::mutex> lk{currentRequestMutex};
        if (!currentRequest.has_value()) return;
        currentRequest->prefill_done = true;
    }

    void incrementCurrentDecodeTokens() {
        std::lock_guard<std::mutex> lk{currentRequestMutex};
        if (!currentRequest.has_value()) return;
        ++currentRequest->decode_tokens_emitted;
    }

    /// RAII wrapper — clears the snapshot on scope exit even if generate()
    /// throws. Blocking and streaming chat handlers both use this so no
    /// exception path leaks a stale "active" entry into the status poll.
    struct CurrentRequestGuard {
        Impl* self;
        explicit CurrentRequestGuard(Impl* s) noexcept : self{s} {}
        ~CurrentRequestGuard() noexcept { if (self) self->endCurrentRequest(); }
        CurrentRequestGuard(const CurrentRequestGuard&) = delete;
        CurrentRequestGuard& operator=(const CurrentRequestGuard&) = delete;
    };

    json buildCurrentRequestBlock() const {
        std::lock_guard<std::mutex> lk{currentRequestMutex};
        if (!currentRequest.has_value()) {
            return json{{"active", false}};
        }
        const auto& r = *currentRequest;
        const double elapsedMs =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - r.started_at).count();
        return json{
            {"active",          true},
            {"request_id",      r.request_id},
            {"streaming",       r.streaming},
            {"elapsed_ms",      elapsedMs},
            {"prompt_tokens",   r.prompt_tokens},
            {"prefill", {
                {"blocks_done",  r.prefill_blocks_done},
                {"blocks_total", r.prefill_blocks_total},
                {"elapsed_ms",   r.prefill_elapsed_ms},
                {"done",         r.prefill_done},
            }},
            {"decode", {
                {"tokens_emitted", r.decode_tokens_emitted},
                {"max_new_tokens", r.max_new_tokens},
            }},
        };
    }

    /// Compose the "perf_regression" sub-object of /v1/system/status.
    /// Reports the current-run p50 decode time, the rolling baseline
    /// p50 (median over runs in the last kBaselineDays), how many prior
    /// runs contributed, and the most recent alert if one has fired.
    /// Absent (= "available": false with reason) when no detector is
    /// installed — happens in smoke/parity mode or when
    /// MIMIRMIND_REGRESSION_ALERT=off was passed to serve.
    json buildPerfRegressionBlock() {
        auto* det = engine.perfRegressionDetector();
        if (det == nullptr) {
            return json{
                {"available", false},
                {"reason",    "no perf-regression detector installed"},
            };
        }
        json body{
            {"available",              true},
            {"internal_version",       det->internalVersion()},
            {"threshold_ratio",        runtime::PerfRegressionDetector::kAlertThreshold},
            {"baseline_window_days",   runtime::PerfRegressionDetector::kBaselineDays},
            {"warmup_tokens",          runtime::PerfRegressionDetector::kWarmupTokens},
            {"baseline_sample_count",  det->baselineSampleCount()},
        };
        const double curP50 = det->currentP50Ms();
        const double basP50 = det->baselineP50Ms();
        // json{X} is an initializer-list ctor and produces [X] (array of one);
        // json(X) is the function-style ctor and produces the scalar X.
        // Same distinction for the "not available" case — json(nullptr) is
        // null, json{} is {}. Consumers want a scalar-or-null shape here.
        if (curP50 > 0.0) {
            body["current_p50_ms"] = curP50;
        } else {
            body["current_p50_ms"] = nullptr;
        }
        if (basP50 > 0.0) {
            body["baseline_p50_ms"] = basP50;
        } else {
            body["baseline_p50_ms"] = nullptr;
        }
        if (auto alert = det->lastAlert()) {
            body["last_alert"] = json{
                {"current_p50_ms",   alert->current_p50_ms},
                {"baseline_p50_ms",  alert->baseline_p50_ms},
                {"delta_ratio",      alert->delta_ratio},
                {"internal_version", alert->internal_version},
                {"detected_unix",    alert->detected_unix},
            };
        } else {
            body["last_alert"] = json{};
        }
        return body;
    }

    /// Compose the "gpu_clock" sub-object of /v1/system/status.
    /// Reports the current iGPU max-freq cap, the hardware envelope,
    /// the target temperature, and which sysfs card the governor
    /// landed on. Absent (= "available": false with reason) when the
    /// LXC/Docker mount config does not let us write the freq file.
    json buildGpuClockBlock() {
        auto* gov = engine.gpuClockGovernor();
        if (gov == nullptr) {
            return json{
                {"available", false},
                {"reason",    "no GPU clock governor installed (profile "
                              "has no gpu_target_temp_c)"},
            };
        }
        if (!gov->available()) {
            return json{
                {"available", false},
                {"reason",    std::string{gov->unavailableReason()}},
            };
        }
        json body{
            {"available",       true},
            {"card_path",       std::string{gov->cardPath()}},
            {"rp0_mhz",         gov->rp0Mhz()},
            {"rpn_mhz",         gov->rpnMhz()},
            {"current_cap_mhz", gov->currentCapMhz()},
            {"target_temp_c",   gov->targetTempC()},
        };
        if (gov->pinned()) {
            body["pin"] = {
                {"intent",  std::string{gov->pinIntent()}},
                {"raw_env", std::string{gov->pinRawEnv()}},
                {"cap_mhz", gov->pinnedMhz()},
            };
        }
        return body;
    }

    /// Compose the "fan" sub-object of /v1/system/status.
    /// Live pwm value / RPM / mode (auto vs manual) — the parts of the
    /// FanController state that change per request. Static chip identity
    /// stays in /system/info.fan_envelope.
    json buildFanBlock() {
        auto* fc = engine.fanController();
        if (fc == nullptr) {
            return json{
                {"available", false},
                {"reason",    "no FanController installed "
                              "(MIMIRMIND_FAN_BOOST=off or probe failed)"},
            };
        }
        if (!fc->available()) {
            return json{
                {"available", false},
                {"reason",    std::string{fc->unavailableReason()}},
            };
        }
        return json{
            {"available",   true},
            {"chip_name",   std::string{fc->chipName()}},
            {"current_pwm", fc->currentPwm()},
            {"current_rpm", fc->currentFanRpm()},
            {"mode",        fc->currentEnableMode()},  // 1=manual, 2..=auto
            {"boost_active", fc->boostActive()},
        };
    }

    /// Compose the "kernels" sub-object of /v1/system/status.
    ///
    /// Reports the autotune dispatch decision per QuantType, the
    /// selfTest state, and the FusedQkvWeights counts — enough to
    /// diagnose an M5i regression via `curl /v1/system/status` alone,
    /// without pulling docker logs.
    json buildKernelsBlock() {
        json body = json::object();

        // Per-type matmul autotune.
        json matmulByType = json::object();
        for (const auto& r : engine.gpuMatmul().autotuneReport()) {
            // M8.J — expose per-M timings + threshold. gemm_min_m is
            // null when GEMM never wins (matvec-loop applies at every M).
            json vecMsAtM = json::object();
            json gemmMsAtM = json::object();
            for (std::size_t i = 0; i < r.mBuckets.size(); ++i) {
                const std::string key = std::to_string(r.mBuckets[i]);
                vecMsAtM[key]  = r.vecMsAtM[i];
                gemmMsAtM[key] = r.gemmMsAtM[i];
            }
            json gemmMinMJson = nullptr;
            if (r.gemmMinM != 0 &&
                r.gemmMinM != std::numeric_limits<std::size_t>::max())
            {
                gemmMinMJson = r.gemmMinM;
            }
            // M8.K.1 — expose v2 GEMM bench alongside v1 so we can
            // eyeball the crossover from /v1/system/status.
            json gemmV2MsAtM = json::object();
            for (std::size_t i = 0; i < r.mBuckets.size(); ++i) {
                gemmV2MsAtM[std::to_string(r.mBuckets[i])] =
                    r.gemmV2MsAtM[i];
            }
            matmulByType[r.name] = json{
                {"gemm_available",    r.gemmAvailable},
                {"gemm_picked",       r.gemmPicked},
                {"gemm_min_m",        gemmMinMJson},
                {"vec_ms",            r.vecMs},
                {"gemm_ms",           r.gemmMs},
                {"vec_ms_at_m",       std::move(vecMsAtM)},
                {"gemm_ms_at_m",      std::move(gemmMsAtM)},
                {"dp4a_available",    r.dp4aAvailable},
                {"dp4a_picked",       r.dp4aPicked},
                {"dp4a_ms",           r.dp4aMs},
                {"gemm_v2_available", r.gemmV2Available},
                {"gemm_v2_picked",    r.gemmV2Picked},
                {"gemm_v2_ms_at_m",   std::move(gemmV2MsAtM)},
                {"source",            r.source},
            };
        }
        body["matmul"] = std::move(matmulByType);

        // Fused QKV block counts + USM footprint.
        json fusedJson;
        if (const auto* fq = engine.fusedQkv()) {
            fusedJson = json{
                {"disabled_by_env", fq->disabledByEnv()},
                {"blocks_fused",    fq->fusedCount()},
                {"blocks_skipped",  fq->skippedCount()},
                {"usm_mib",         static_cast<unsigned long long>(
                                        (fq->totalUsmBytes()
                                         + (1ULL << 20) - 1) >> 20)},
            };
        } else {
            fusedJson = json{{"available", false}};
        }
        body["fused_qkv"] = std::move(fusedJson);

        // GpuOps self-test result.
        body["selftest"] = std::string{engine.gpuOps().selfTestStatus()};

        return body;
    }

    /// Compose the "power" sub-object of /v1/system/status.
    ///
    /// Three views on RAPL energy counters:
    ///   - watts_now      Average Watts since the last poll of this endpoint
    ///                    (rolling window — whatever cadence the operator
    ///                    polls at sets the smoothing interval).
    ///   - total_joules   Energy consumed since the baseline snapshot
    ///                    (taken once just after model load).
    ///   - uptime_s       Seconds since that same baseline.
    ///
    /// All are reported per RAPL domain (package / core / uncore / etc.)
    /// so dashboards can pick what they need.
    json buildPowerBlock() {
        auto* mon = engine.powerMonitor();
        if (mon == nullptr) {
            return json{
                {"available", false},
                {"reason",    "no power monitor installed"},
            };
        }
        if (!mon->available()) {
            return json{
                {"available", false},
                {"reason",    std::string{mon->unavailableReason()}},
            };
        }

        const auto now = mon->snapshot();
        std::vector<double>                  totalJoules;
        std::vector<double>                  wattsNow;
        std::chrono::steady_clock::time_point baselineAt;
        bool                                  haveBaseline = false;
        {
            std::lock_guard lk{powerStateMutex};
            if (baselineCaptured) {
                totalJoules   = mon->energyBetween(powerBaseline, now);
                wattsNow      = mon->averageWattsBetween(powerLastStatus, now);
                baselineAt    = baselineWallStart;
                powerLastStatus = now;
                haveBaseline  = true;
            }
        }

        json domains = json::array();
        const auto names = mon->domainNames();
        for (std::size_t i = 0; i < names.size(); ++i) {
            json d{{"name", names[i]}};
            if (haveBaseline && i < wattsNow.size()) {
                d["watts_now"] = wattsNow[i];
            }
            if (haveBaseline && i < totalJoules.size()) {
                d["total_joules"] = totalJoules[i];
            }
            domains.push_back(std::move(d));
        }

        json out{
            {"available", true},
            {"domains",   std::move(domains)},
        };
        if (haveBaseline) {
            const auto uptime_s = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - baselineAt).count();
            out["uptime_s"] = uptime_s;
        }
        return out;
    }

    /// Common pre-generation work shared by streaming and non-streaming.
    /// Mutates `params` and `stopIds` in place. Populates `report` with
    /// any M-PT length-discipline actions taken. Returns false (and writes
    /// the error response) if the request is invalid or trim cannot
    /// recover it.
    [[nodiscard]] bool prepareChatRequest(const ChatRequest&             cr,
                                          httplib::Response&             res,
                                          std::vector<std::int32_t>&     promptIds,
                                          std::vector<std::int32_t>&     stopIds,
                                          runtime::GenerateParams&       params,
                                          TrimReport&                    report) {
        if (cr.messages.empty()) {
            sendError(res, 400, "invalid_request_error",
                      "messages must not be empty");
            return false;
        }

        const auto& tok = engine.tokenizer();
        // M-PT: work on a mutable copy so the trim loop can drop entries
        // without touching the parsed request. Also lets us report the
        // original prompt-token count in the response.
        std::vector<model::ChatMessage> msgs = cr.messages;
        promptIds = model::ChatTemplate::encode(
            chatStyle, tok, msgs, /*addGenerationPrompt=*/true);

        stopIds = model::ChatTemplate::stopIds(chatStyle, tok);
        extendStopIds(tok, cr.stopStrings, stopIds);

        params.maxNewTokens = cr.maxTokens > 0 ? cr.maxTokens : cfg.defaultMaxNew;
        params.stopIds      = stopIds;

        // M-PT — server-side length discipline. Trim messages + clamp
        // max_new so the prompt fits _maxContextTokens without tripping
        // InferenceEngine::ensureCapacity's throw. See the M-PT Synaipse
        // note for the design rationale (why message-drop and not client-
        // side, why the last user message stays intact, why system stays).
        {
            std::string trimErr;
            if (!applyPromptTrim(msgs, promptIds, params.maxNewTokens,
                                 engine.maxContextTokens(),
                                 engine.config().contextLength,
                                 tok, chatStyle, report, trimErr)) {
                sendError(res, 400, "invalid_request_error", trimErr);
                return false;
            }
            if (report.fired()) {
                MM_LOG_INFO("server",
                    "prompt-trim: dropped={} orig_tokens={} eff_tokens={} "
                    "max_new={}→{} extrapolated_by={}",
                    report.droppedMessages, report.originalPromptTokens,
                    report.effectivePromptTokens,
                    report.maxNewClampedFrom, report.maxNewClampedTo,
                    report.contextExtrapolatedBy);
                if (report.contextExtrapolated) {
                    MM_LOG_WARN("server",
                        "prompt Tp={} exceeds model-native context {} by {} "
                        "tokens — RoPE extrapolation zone",
                        report.effectivePromptTokens,
                        engine.config().contextLength,
                        report.contextExtrapolatedBy);
                }
            }
            attachTrimHeaders(res, report);
        }
        if (cr.hasTemperature) {
            params.sampling.temperature = cr.temperature;
        }
        params.sampling.topP = cr.topP;
        params.sampling.topK = cr.topK;
        params.sampling.seed = cr.seed;

        // M7f — repetition-control penalties.
        //
        // Server-side defaults are opinionated: we saw a 26B-A4B-it
        // Q6_K model fall into a 1300-token repetition loop when no
        // client-side penalty was set, because SamplingParams alone
        // has no history-based mechanism to break out. Applying a mild
        // frequency + repetition penalty by default protects vanilla
        // OpenAI clients (that don't send any penalty) from the same
        // failure mode. Clients that explicitly set a penalty value
        // (including 0) override the default.
        constexpr float        kDefaultFrequencyPenalty  = 0.5F;
        constexpr float        kDefaultRepetitionPenalty = 1.10F;
        constexpr std::uint32_t kDefaultPenaltyWindow    = 64U;

        params.sampling.frequencyPenalty =
            cr.hasFrequencyPenalty  ? cr.frequencyPenalty  : kDefaultFrequencyPenalty;
        params.sampling.presencePenalty =
            cr.hasPresencePenalty   ? cr.presencePenalty   : 0.0F;
        params.sampling.repetitionPenalty =
            cr.hasRepetitionPenalty ? cr.repetitionPenalty : kDefaultRepetitionPenalty;
        params.sampling.penaltyWindow = kDefaultPenaltyWindow;
        return true;
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

        // Thermal admission BEFORE we commit to a stream: a 503 must
        // ship as a plain JSON response, not as half a chunked SSE
        // body. The engine repeats this check internally for callers
        // that bypass the server, so the worst-case cost of doing it
        // twice is two sysfs reads.
        if (auto* guard = engine.thermalGuard(); guard != nullptr) {
            try {
                guard->checkAdmission();
            } catch (const runtime::ThermalLimitExceeded& e) {
                MM_LOG_INFO("server",
                            "thermal refusal: {}", e.what());
                res.set_header("Retry-After", "10");
                sendError(res, 503, "service_unavailable", e.what());
                return;
            }
        }

        if (cr.stream) {
            handleChatCompletionsStream(cr, res);
        } else {
            handleChatCompletionsBlocking(cr, res);
        }
    }

    void handleChatCompletionsBlocking(const ChatRequest& cr,
                                       httplib::Response& res) {
        std::vector<std::int32_t> promptIds;
        std::vector<std::int32_t> stopIds;
        runtime::GenerateParams   params;
        TrimReport                trimReport;
        if (!prepareChatRequest(cr, res, promptIds, stopIds, params,
                                trimReport)) {
            return;
        }

        const auto& tok = engine.tokenizer();

        runtime::GenerateStats stats;
        std::vector<std::int32_t> generated;

        // Reserve the response id up-front so the /v1/system/status
        // snapshot can carry it while the request is still running.
        const std::string respId = makeRequestId();
        beginCurrentRequest(respId, promptIds.size(),
                            params.maxNewTokens, /*streaming=*/false);
        CurrentRequestGuard requestGuard{this};

        auto onPrefillProgress =
            [this](const runtime::InferenceEngine::PrefillProgress& p)
                -> bool {
                updateCurrentPrefillProgress(p);
                return true;
            };
        auto onPrefillDone =
            [this](const runtime::InferenceEngine::PrefillDone&) {
                markCurrentPrefillDone();
            };
        auto onToken = [this](std::int32_t) -> bool {
            incrementCurrentDecodeTokens();
            return true;
        };

        {
            // Engine is single-instance, mutable scratch + sampler. Serialise.
            std::lock_guard<std::mutex> lk{engineMutex};
            try {
                // M9.11.4 — route through the spec-dec orchestrator when a
                // draft is loaded. It falls through to engine.generate() bit-
                // identically for sampled / kill-switched requests.
                if (speculativeDecoder != nullptr) {
                    generated = speculativeDecoder->generate(
                        promptIds, params, onToken, &stats,
                        onPrefillDone, onPrefillProgress);
                } else {
                    generated = engine.generate(promptIds, params,
                                                onToken, &stats,
                                                onPrefillDone,
                                                onPrefillProgress);
                }
            } catch (const runtime::ThermalLimitExceeded& e) {
                MM_LOG_INFO("server",
                            "thermal refusal at engine entry: {}", e.what());
                res.set_header("Retry-After", "10");
                sendError(res, 503, "service_unavailable", e.what());
                return;
            } catch (const std::exception& e) {
                MM_LOG_ERROR("server", "generate failed: {}", e.what());
                sendError(res, 500, "server_error",
                          std::string{"generate: "} + e.what());
                return;
            }
        }

        // Strip trailing stop tokens from the rendered text so clients
        // don't see "<|im_end|>" tacked onto the answer. Loop because at
        // low temperatures the model could in theory sample several stop
        // tokens in a row before the engine's stop-check breaks the loop.
        std::vector<std::int32_t> visible = generated;
        bool hitStop = stats.hitStop;
        auto isStop = [&](std::int32_t id) {
            if (id == tok.eosId()) return true;
            return std::find(stopIds.begin(), stopIds.end(), id) != stopIds.end();
        };
        while (!visible.empty() && isStop(visible.back())) {
            visible.pop_back();
            hitStop = true;
        }

        const std::string rawText  = tok.decode(visible, /*skipSpecial=*/true);
        const std::string text     = cfg.preserveThinking
            ? rawText
            : model::ChatTemplate::cleanResponse(chatStyle, rawText);

        const std::int64_t now   = unixNow();
        const std::string finish = hitStop ? "stop" : "length";

        const std::string echoModel = cr.model.empty() ? cfg.modelId : cr.model;

        json usage = {
            {"prompt_tokens",     promptIds.size()},
            {"completion_tokens", visible.size()},
            {"total_tokens",      promptIds.size() + visible.size()},
        };
        // Extension over the OpenAI shape: per-request energy delta from
        // the RAPL package counter. Quietly omitted when no power
        // monitor was active for this call.
        if (stats.packageJoules > 0.0) {
            usage["package_joules"] = stats.packageJoules;
        }
        // M-PT — length-discipline metadata. Only present when trim /
        // clamp / extrapolation-warn actually fired.
        attachTrimUsage(usage, trimReport);

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
            {"usage", std::move(usage)},
        };

        // Spec-dec accept-rate is the headline diagnostic for M9.11.4 —
        // it tells operators whether the draft is earning its keep.
        // Suffix stays empty when spec-dec was disabled or fell through.
        std::string specSuffix;
        if (stats.specDecRounds > 0 && stats.specDecDrafted > 0) {
            const double acceptRate = static_cast<double>(stats.specDecAccepted)
                                    / static_cast<double>(stats.specDecDrafted);
            specSuffix = " spec_rounds=" + std::to_string(stats.specDecRounds)
                       + " spec_acc=" + std::to_string(stats.specDecAccepted)
                       + "/" + std::to_string(stats.specDecDrafted)
                       + " spec_rate=" + std::to_string(acceptRate);
        }
        MM_LOG_INFO("server",
                    "chat.completion id={} model={} prompt={} cached={} new={} "
                    "prefill={:.0f}ms decode={:.0f}ms energy={:.1f}J finish={}{}",
                    respId, echoModel,
                    promptIds.size(), stats.cachedTokens, visible.size(),
                    stats.prefillMs, stats.decodeMs, stats.packageJoules,
                    finish, specSuffix);

        sendJson(res, 200, response);
    }

    void handleChatCompletionsStream(const ChatRequest& cr,
                                     httplib::Response& res) {
        std::vector<std::int32_t> promptIds;
        std::vector<std::int32_t> stopIds;
        runtime::GenerateParams   params;
        // M-PT — headers are attached inside prepareChatRequest before the
        // stream body starts, so a streaming client sees the same
        // x-mimirmind-* signals as the blocking path. We don't push the
        // report into a usage chunk here because mimirmind's SSE format
        // doesn't emit a terminal usage chunk (see prefill_done named
        // event for the token-count signal instead).
        TrimReport                trimReport;
        if (!prepareChatRequest(cr, res, promptIds, stopIds, params,
                                trimReport)) {
            return;
        }

        const std::string respId    = makeRequestId();
        const std::int64_t created  = unixNow();
        const std::string echoModel = cr.model.empty() ? cfg.modelId : cr.model;

        // Disable proxy buffering / browser caching so SSE chunks reach the
        // client as they are produced.
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");
        res.set_header("X-Accel-Buffering", "no");

        // The provider lambda owns the heavy work — taking the engine
        // mutex, running generate, writing chunks. The HTTP response
        // headers (200 + content-type) go out as soon as cpp-httplib
        // starts invoking the provider, so the client gets immediate
        // feedback that the request was accepted.
        //
        // Shared state lives on the heap so the provider lambda can be
        // re-entered safely if cpp-httplib calls it more than once.
        struct StreamState {
            std::vector<std::int32_t>     promptIds;
            std::vector<std::int32_t>     stopIds;
            runtime::GenerateParams       params;
            std::string                   respId;
            std::int64_t                  created{};
            std::string                   echoModel;
            // Buffers a trailing incomplete UTF-8 codepoint between
            // tokens so SSE deltas always carry valid UTF-8.
            std::string                   utf8Pending;
            // Per-stream filter that swallows the Gemma 4
            // <|channel>thought<channel|> wrapper at the token level,
            // matching the behaviour ChatTemplate::cleanResponse applies
            // in the non-streaming path. No-op for other chat styles.
            model::ResponseCleaner        cleaner;
            bool                          done{false};

            StreamState(model::ChatTemplate::Style style,
                        const model::Tokenizer&    tok,
                        bool                       preserveThinking)
                : cleaner{preserveThinking
                    ? model::ResponseCleaner{
                          model::ChatTemplate::Style::QwenChatML, -1, -1}
                    : model::ResponseCleaner::forStyle(style, tok)} {}
        };
        auto state = std::make_shared<StreamState>(chatStyle, engine.tokenizer(),
                                                   cfg.preserveThinking);
        state->promptIds = std::move(promptIds);
        state->stopIds   = std::move(stopIds);
        state->params    = std::move(params);
        state->respId    = respId;
        state->created   = created;
        state->echoModel = echoModel;

        res.set_chunked_content_provider(
            "text/event-stream",
            [this, state](std::size_t /*offset*/,
                          httplib::DataSink& sink) -> bool {
                if (state->done) {
                    return false;
                }
                state->done = true;

                const auto& tok = engine.tokenizer();

                // 1. Initial role chunk so clients see {role:"assistant"}.
                if (!writeSseEvent(
                        sink,
                        buildRoleChunk(state->respId, state->created,
                                       state->echoModel))) {
                    MM_LOG_INFO("server",
                                "stream {}: client closed before role chunk",
                                state->respId);
                    return false;
                }

                // 2. Generate under the engine mutex; emit each non-stop
                //    token as a delta-content chunk. The onToken returning
                //    false causes generate() to abort cleanly.
                std::size_t              emittedTokens = 0;
                runtime::GenerateStats   stats;
                bool                     clientGone    = false;

                auto isStop = [&](std::int32_t id) {
                    if (id == tok.eosId()) return true;
                    for (auto s : state->stopIds) {
                        if (id == s) return true;
                    }
                    return false;
                };

                auto onToken = [&](std::int32_t id) -> bool {
                    if (isStop(id)) {
                        // Do not surface the stop token's text. The engine
                        // checks isStop on the next iteration and exits.
                        return true;
                    }
                    // Snapshot the per-token progress even for stripped
                    // tokens so /v1/system/status reflects real decode work.
                    incrementCurrentDecodeTokens();
                    std::string txt = tok.decode(
                        std::span<const std::int32_t>{&id, 1},
                        /*skipSpecial=*/true);
                    if (txt.empty()) {
                        return true;
                    }

                    // Drop structural markup (Gemma 4 channel wrapper).
                    if (!state->cleaner.feed(id, txt)) {
                        return true;
                    }

                    state->utf8Pending.append(txt);
                    const std::size_t cut =
                        utf8IncompleteTailStart(state->utf8Pending);
                    if (cut == 0) {
                        return true;     // entire buffer is partial; keep waiting
                    }

                    std::string emit = state->utf8Pending.substr(0, cut);
                    state->utf8Pending.erase(0, cut);

                    if (!writeSseEvent(
                            sink,
                            buildContentChunk(state->respId, state->created,
                                              state->echoModel, emit))) {
                        clientGone = true;
                        return false;   // abort generate()
                    }
                    ++emittedTokens;
                    return true;
                };

                // Named SSE event fired between prefill and the first
                // decode token so a streaming client can flip its UX
                // from "reading your prompt" to "answering". The
                // OpenAI stream demuxer ignores named events, browsers
                // pick it up via EventSource.addEventListener.
                auto onPrefillDone =
                    [&](const runtime::InferenceEngine::PrefillDone& p) {
                        markCurrentPrefillDone();
                        const json payload = {
                            {"prompt_tokens",    p.promptTokens},
                            {"prefilled_tokens", p.prefilledTokens},
                            {"prefill_ms",       p.prefillMs},
                            {"response_id",      state->respId},
                        };
                        if (!writeSseNamedEvent(sink, "prefill_done", payload)) {
                            clientGone = true;
                        }
                    };

                // Per-block prefill progress, rate-limited so a fast
                // prefill (~10 ms per block on Gemma 4 26B) doesn't
                // fire 34 SSE events in half a second. First and last
                // blocks always emit; in between we throttle to one
                // event per ~200 ms so a browser progress bar updates
                // smoothly on long prompts without spamming short ones.
                double lastProgressMs = -1.0;
                constexpr double kProgressMinIntervalMs = 200.0;
                auto onPrefillProgress =
                    [&](const runtime::InferenceEngine::PrefillProgress& p)
                        -> bool {
                        // Snapshot for /v1/system/status polling — kept
                        // outside the SSE-rate-limit branch so the status
                        // block sees every completed transformer layer.
                        updateCurrentPrefillProgress(p);
                        // M7g — poll for a client-side disconnect on every
                        // block. If clientGone flipped between blocks (via
                        // the throttle branch below OR via a broken write
                        // from onToken during a previous prefill call —
                        // rare but possible in edge flows), tell the engine
                        // to stop at the next barrier.
                        if (clientGone) {
                            return false;
                        }
                        const bool isFirst = (p.blocksDone == 1);
                        const bool isLast  = (p.blocksDone == p.blocksTotal);
                        const bool dueByTime =
                            (p.elapsedMs - lastProgressMs) >=
                            kProgressMinIntervalMs;
                        if (!isFirst && !isLast && !dueByTime) {
                            return true;  // no SSE this tick, keep going
                        }
                        lastProgressMs = p.elapsedMs;
                        const json payload = {
                            {"blocks_done",  p.blocksDone},
                            {"blocks_total", p.blocksTotal},
                            {"elapsed_ms",   p.elapsedMs},
                            {"response_id",  state->respId},
                        };
                        if (!writeSseNamedEvent(sink, "prefill_progress",
                                                payload)) {
                            clientGone = true;
                            return false;  // abort prefill at next barrier
                        }
                        return true;
                    };

                std::vector<std::int32_t> generated;
                std::string               errorMessage;
                // Register this request with the status snapshot so
                // /v1/system/status can report prefill+decode progress
                // while it runs. The RAII guard releases on scope exit —
                // covers early error exits and generate() throws.
                beginCurrentRequest(state->respId,
                                    state->promptIds.size(),
                                    state->params.maxNewTokens,
                                    /*streaming=*/true);
                CurrentRequestGuard requestGuard{this};
                {
                    std::lock_guard<std::mutex> lk{engineMutex};
                    try {
                        if (speculativeDecoder != nullptr) {
                            generated = speculativeDecoder->generate(
                                state->promptIds, state->params,
                                onToken, &stats,
                                onPrefillDone, onPrefillProgress);
                        } else {
                            generated = engine.generate(state->promptIds,
                                                        state->params,
                                                        onToken,
                                                        &stats,
                                                        onPrefillDone,
                                                        onPrefillProgress);
                        }
                    } catch (const std::exception& e) {
                        errorMessage = e.what();
                    }
                }

                if (clientGone) {
                    // Distinguish prefill vs decode abort in the log so
                    // the operator can spot Pegenaut cancels happening
                    // during the (potentially long) prefill phase — those
                    // are the highest-value ones to see, they show the
                    // user gave up while waiting for the first token.
                    const bool cancelledInPrefill = (emittedTokens == 0);
                    MM_LOG_INFO("server",
                                "stream {}: aborted mid-{} "
                                "(emitted={}, prefill={:.0f}ms)",
                                state->respId,
                                cancelledInPrefill ? "prefill" : "decode",
                                emittedTokens, stats.prefillMs);
                    sink.done();
                    return false;
                }

                if (!errorMessage.empty()) {
                    // Surface the error as an SSE event so the client sees
                    // a structured failure rather than just a half-stream.
                    json errChunk = streamChunkSkeleton(state->respId,
                                                        state->created,
                                                        state->echoModel);
                    errChunk["error"] = json{
                        {"message", errorMessage},
                        {"type",    "server_error"},
                    };
                    (void)writeSseEvent(sink, errChunk);
                    MM_LOG_ERROR("server",
                                 "stream {}: generate failed: {}",
                                 state->respId, errorMessage);
                    sink.done();
                    return false;
                }

                // Flush any UTF-8 buffer leftover (partial codepoint at
                // the end — rare but possible when the model stops mid-
                // grapheme on length cutoff). Receivers may show U+FFFD.
                if (!state->utf8Pending.empty()) {
                    (void)writeSseEvent(
                        sink,
                        buildContentChunk(state->respId, state->created,
                                          state->echoModel,
                                          state->utf8Pending));
                    state->utf8Pending.clear();
                }

                // 3. Finish chunk + DONE sentinel. finish_reason is "stop"
                //    if a stop token broke the decode loop, "length" if we
                //    exhausted maxNewTokens.
                const bool hitStop = stats.hitStop
                    || (!generated.empty() && isStop(generated.back()));
                const std::string finish = hitStop ? "stop" : "length";

                (void)writeSseEvent(
                    sink,
                    buildFinishChunk(state->respId, state->created,
                                     state->echoModel, finish));
                (void)writeSseDone(sink);

                std::string streamSpecSuffix;
                if (stats.specDecRounds > 0 && stats.specDecDrafted > 0) {
                    const double acceptRate =
                        static_cast<double>(stats.specDecAccepted)
                      / static_cast<double>(stats.specDecDrafted);
                    streamSpecSuffix =
                          " spec_rounds=" + std::to_string(stats.specDecRounds)
                        + " spec_acc="   + std::to_string(stats.specDecAccepted)
                        + "/"            + std::to_string(stats.specDecDrafted)
                        + " spec_rate="  + std::to_string(acceptRate);
                }
                MM_LOG_INFO("server",
                            "stream {} model={} prompt={} cached={} emitted={} "
                            "prefill={:.0f}ms decode={:.0f}ms energy={:.1f}J "
                            "finish={}{}",
                            state->respId, state->echoModel,
                            state->promptIds.size(), stats.cachedTokens,
                            emittedTokens,
                            stats.prefillMs, stats.decodeMs,
                            stats.packageJoules, finish, streamSpecSuffix);

                sink.done();
                return false;
            });
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

ApiServer::ApiServer(runtime::InferenceEngine& engine,
                     ServerConfig              cfg,
                     runtime::InferenceEngine* draftEngine)
    : _impl{std::make_unique<Impl>(engine, std::move(cfg), draftEngine)} {}

ApiServer::~ApiServer() = default;

void ApiServer::run()  { _impl->run(); }
void ApiServer::stop() { _impl->stop(); }

} // namespace mimirmind::server