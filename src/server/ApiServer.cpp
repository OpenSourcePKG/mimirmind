#include "server/ApiServer.hpp"

#include "server/ApiHelpers.hpp"
#include "server/ChatCompletionHandler.hpp"
#include "server/RequestDispatcher.hpp"
#include "server/RequestTracker.hpp"
#include "server/SystemStatusBuilder.hpp"

#include "model/ChatTemplate.hpp"
#include "runtime/InferenceEngine.hpp"
#include "core/log/Log.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <exception>
#include <stdexcept>
#include <string>
#include <utility>

namespace mimirmind::server {

using nlohmann::json;

struct ApiServer::Impl {
    RequestDispatcher                     dispatcher;
    runtime::InferenceEngine&             engine;    // == dispatcher.defaultEngine()
    ServerConfig                          cfg;
    model::ChatTemplate::Style            chatStyle;
    httplib::Server                       server;
    std::atomic_bool                      started{false};

    // In-flight request snapshot for /v1/system/status.current_request.
    RequestTracker                        requestTracker;

    // Populates /v1/system/info (static) and /v1/system/status (dynamic).
    // Owns the RAPL power baseline snapshot captured at construction.
    SystemStatusBuilder                   statusBuilder;

    // POST /v1/chat/completions dispatcher + streaming/blocking handlers.
    ChatCompletionHandler                 chatHandler;

    Impl(std::vector<LoadedEngine> in, ServerConfig c,
         runtime::InferenceEngine* draft)
        : dispatcher{std::move(in), c.modelId, draft,
                     c.speculativeTargetId, c.speculative},
          engine{dispatcher.defaultEngine()},
          cfg{std::move(c)},
          chatStyle{model::ChatTemplate::detectFromArch(
              engine.config().architecture)},
          statusBuilder{engine, dispatcher, requestTracker, cfg.modelId},
          chatHandler{dispatcher, requestTracker, chatStyle, cfg} {
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
        server.Get("/v1/system/status",
                   [this](const httplib::Request&, httplib::Response& res) {
                       sendJson(res, 200, statusBuilder.buildStatus());
                   });
        server.Get("/v1/system/info",
                   [this](const httplib::Request&, httplib::Response& res) {
                       sendJson(res, 200, statusBuilder.buildInfo());
                   });
        server.Post("/v1/chat/completions",
                    [this](const httplib::Request& req,
                           httplib::Response&       res) {
                        // M9.8b observability follow-up: log request
                        // acceptance BEFORE handing off to the
                        // ChatCompletionHandler. The httplib access-log
                        // (set_logger below) only fires when the handler
                        // returns — for streaming responses that means
                        // the log is silent from request-accept until the
                        // client either receives all chunks or
                        // disconnects. A start-log makes the request
                        // visible in stdout the moment it arrives, so
                        // hung/disconnected requests can be correlated
                        // against a clear "accepted" line.
                        MM_LOG_INFO(
                            "server",
                            "POST /v1/chat/completions accepted "
                            "from {} (content-length={} B)",
                            req.remote_addr, req.body.size());
                        chatHandler.handle(req, res);
                    });

        server.set_exception_handler(
            [](const httplib::Request& req,
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
                MM_LOG_ERROR("server",
                             "handler exception on {} {} from {}: {}",
                             req.method, req.path, req.remote_addr, msg);
                sendError(res, 500, "server_error", msg);
            });

        server.set_logger([](const httplib::Request& req,
                             const httplib::Response& res) {
            // Access-log with remote IP so operators can correlate
            // request-start (chat-completions POST accept-log above)
            // with request-end. Fires when the handler returns; for
            // streaming responses that is after the last SSE chunk or
            // client disconnect.
            MM_LOG_INFO("server", "{} {} -> {} (from {})",
                        req.method, req.path, res.status, req.remote_addr);
        });
    }

    void handleHealth(const httplib::Request&, httplib::Response& res) {
        sendJson(res, 200, {
            {"status", "ok"},
            {"model",  cfg.modelId},
        });
    }

    void handleModels(const httplib::Request&, httplib::Response& res) {
        json data = json::array();
        for (const auto& m : dispatcher.listModels()) {
            data.push_back(json{
                {"id",       m.id},
                {"title",    m.title.empty() ? m.id : m.title},
                {"object",   "model"},
                {"created",  0},
                {"owned_by", "mimirmind"},
            });
        }
        sendJson(res, 200, {
            {"object", "list"},
            {"data",   std::move(data)},
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

ApiServer::ApiServer(std::vector<LoadedEngine>   engines,
                     ServerConfig                cfg,
                     runtime::InferenceEngine*   draftEngine)
    : _impl{std::make_unique<Impl>(std::move(engines),
                                   std::move(cfg), draftEngine)} {}

ApiServer::~ApiServer() = default;

void ApiServer::run()  { _impl->run(); }
void ApiServer::stop() { _impl->stop(); }

} // namespace mimirmind::server