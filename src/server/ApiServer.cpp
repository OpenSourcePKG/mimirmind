// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "server/ApiServer.hpp"

#include "server/ApiHelpers.hpp"
#include "server/ChatCompletionHandler.hpp"
#include "server/RerankHandler.hpp"
#include "server/RequestDispatcher.hpp"
#include "server/RequestTracker.hpp"
#include "server/SystemStatusBuilder.hpp"
#include "server/TenantMetrics.hpp"

#include "model/ChatTemplate.hpp"
#include "runtime/InferenceEngine.hpp"
#include "runtime/serving/ContinuousBatcher.hpp"
#include "core/log/Log.hpp"
#include "core/security/ScopedTenant.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
#include <openssl/pem.h>
#include <openssl/x509.h>
#endif

#include <atomic>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace mimirmind::server {

using nlohmann::json;

namespace {

/// Extract the raw token from an `Authorization: Bearer <token>` header.
/// Returns an empty view when the scheme is absent or malformed.
std::string_view bearerToken(std::string_view header) noexcept {
    constexpr std::string_view kPrefix = "Bearer ";
    if (header.size() <= kPrefix.size()) return {};
    if (header.substr(0, kPrefix.size()) != kPrefix) return {};
    std::string_view tok = header.substr(kPrefix.size());
    // Tolerate leading spaces after the scheme (some clients emit two).
    while (!tok.empty() && tok.front() == ' ') tok.remove_prefix(1);
    return tok;
}

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT

// Owns the OpenSSL cert/key objects for the SSLServer's lifetime. httplib's
// in-memory SSLServer ctor installs these into its SSL_CTX; keeping our own
// references alive here is robust regardless of upstream ref-count
// semantics, and frees them deterministically on server teardown.
struct TlsMaterial {
    X509*     cert{nullptr};
    EVP_PKEY* key{nullptr};

    ~TlsMaterial() {
        if (cert) X509_free(cert);
        if (key)  EVP_PKEY_free(key);
    }
    TlsMaterial() = default;
    TlsMaterial(const TlsMaterial&)            = delete;
    TlsMaterial& operator=(const TlsMaterial&) = delete;
};

// Parse in-memory PEM strings into an X509 + EVP_PKEY. Returns false and
// leaves `out` empty on any parse failure.
bool parsePemMaterial(const std::string& certPem, const std::string& keyPem,
                      TlsMaterial& out) {
    BIO* cbio = BIO_new_mem_buf(certPem.data(),
                                static_cast<int>(certPem.size()));
    BIO* kbio = BIO_new_mem_buf(keyPem.data(),
                                static_cast<int>(keyPem.size()));
    bool ok = false;
    if (cbio && kbio) {
        out.cert = PEM_read_bio_X509(cbio, nullptr, nullptr, nullptr);
        out.key  = PEM_read_bio_PrivateKey(kbio, nullptr, nullptr, nullptr);
        ok = out.cert != nullptr && out.key != nullptr;
    }
    if (cbio) BIO_free(cbio);
    if (kbio) BIO_free(kbio);
    return ok;
}

#endif // CPPHTTPLIB_OPENSSL_SUPPORT

} // namespace

struct ApiServer::Impl {
    RequestDispatcher                     dispatcher;
    runtime::InferenceEngine&             engine;    // == dispatcher.defaultEngine()
    ServerConfig                          cfg;
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    TlsMaterial                           tls;       // kept alive for `server`
#endif
    std::unique_ptr<httplib::Server>      server;
    std::atomic_bool                      started{false};

    // In-flight request snapshot for /v1/system/status.current_request.
    RequestTracker                        requestTracker;

    // Per-tenant (per-API-key) usage accounting for /v1/admin/tenants +
    // /metrics. Loads persisted totals + starts its flush thread on ctor;
    // declared before chatHandler because the handler binds it by reference.
    TenantMetrics                         tenantMetrics;

    // Populates /v1/system/info (static) and /v1/system/status (dynamic).
    // Owns the RAPL power baseline snapshot captured at construction.
    SystemStatusBuilder                   statusBuilder;

    // POST /v1/chat/completions dispatcher + streaming/blocking handlers.
    ChatCompletionHandler                 chatHandler;

    // POST /v1/rerank — cross-encoder reranker(s). Empty when no model
    // with task=rerank is configured.
    RerankHandler                         rerankHandler;

    Impl(std::vector<LoadedEngine> in, ServerConfig c,
         runtime::Drafter*         drafter,
         std::vector<LoadedReranker> rerankers)
        : dispatcher{std::move(in), c.modelId, drafter,
                     c.speculativeTargetId, c.speculative},
          engine{dispatcher.defaultEngine()},
          cfg{std::move(c)},
          tenantMetrics{cfg.tenantMetricsPath},
          statusBuilder{engine, dispatcher, requestTracker, cfg.modelId},
          chatHandler{dispatcher, requestTracker, tenantMetrics, cfg},
          rerankHandler{std::move(rerankers), cfg} {
        makeServer();
        installRoutes();
    }

    // Build the concrete listener: an OpenSSL SSLServer when TLS is enabled
    // (from the in-memory PEM material ServeMode provisioned), otherwise the
    // plain HTTP Server. SSLServer derives from Server, so every route /
    // pool / listen call below is identical through the base pointer.
    void makeServer() {
        if (cfg.tls.enabled) {
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
            if (cfg.tls.certPem.empty() || cfg.tls.keyPem.empty()) {
                throw std::runtime_error(
                    "ApiServer: TLS enabled but no certificate material "
                    "provided");
            }
            if (!parsePemMaterial(cfg.tls.certPem, cfg.tls.keyPem, tls)) {
                throw std::runtime_error(
                    "ApiServer: failed to parse TLS certificate/key PEM");
            }
            auto ssl = std::make_unique<httplib::SSLServer>(tls.cert, tls.key);
            if (!ssl->is_valid()) {
                throw std::runtime_error(
                    "ApiServer: SSLServer rejected the certificate/key "
                    "(mismatched pair or unsupported key?)");
            }
            server = std::move(ssl);
            MM_LOG_INFO("server", "TLS: in-process HTTPS enabled ({}cert)",
                        cfg.tls.selfSigned ? "self-signed " : "");
#else
            throw std::runtime_error(
                "ApiServer: TLS requested but binary built without OpenSSL "
                "(CPPHTTPLIB_OPENSSL_SUPPORT)");
#endif
        } else {
            server = std::make_unique<httplib::Server>();
            MM_LOG_WARN("server",
                        "TLS: DISABLED — HTTP is served in cleartext");
        }
    }

    // Bearer-token gate. Installed as httplib's pre-routing handler so it
    // runs before any route match and can short-circuit with a 401. When
    // auth is enabled EVERY path is gated with no exceptions — including
    // `/health` — so an internet scanner cannot read or POST to anything
    // without a valid key. Health checks must therefore send the key too
    // (see the Docker note) or probe TLS reachability instead.
    void installAuthHook() {
        server->set_pre_routing_handler(
            [this](const httplib::Request& req, httplib::Response& res) {
                using HR = httplib::Server::HandlerResponse;
                if (!cfg.auth.enabled) return HR::Unhandled;

                // get_header_value returns by value — keep it alive so the
                // string_view carved out of it does not dangle.
                const std::string authHeader =
                    req.get_header_value("Authorization");
                const std::string_view tok = bearerToken(authHeader);
                const core::security::ApiKey* hit = cfg.auth.store.match(tok);
                if (hit == nullptr) {
                    core::security::ScopedTenant::set({}, /*isAdmin=*/false);
                    res.set_header("WWW-Authenticate",
                                   "Bearer realm=\"mimirmind\"");
                    res.set_header("Cache-Control", "no-store");
                    sendError(res, 401, "unauthorized",
                              "invalid or missing API key");
                    return HR::Handled;
                }
                // Carry the tenant label + admin flag for the rest of this
                // request on this pool thread. Per-tenant serving admission
                // reads the label; the operator-only routes read isAdmin.
                core::security::ScopedTenant::set(hit->tenantId, hit->isAdmin);
                return HR::Unhandled;
            });
    }

    void installRoutes() {
        installAuthHook();   // pre-routing: runs before every route match
        server->Get("/health", [this](const httplib::Request& req,
                                     httplib::Response&       res) {
            handleHealth(req, res);
        });
        server->Get("/v1/models", [this](const httplib::Request& req,
                                        httplib::Response&       res) {
            handleModels(req, res);
        });
        server->Get("/v1/system/status",
                   [this](const httplib::Request&, httplib::Response& res) {
                       sendJson(res, 200, statusBuilder.buildStatus());
                   });
        server->Get("/v1/system/info",
                   [this](const httplib::Request&, httplib::Response& res) {
                       sendJson(res, 200, statusBuilder.buildInfo());
                   });

        // Operator-only per-tenant usage accounting. Auth already gated the
        // request to a valid key (pre-routing hook); these two additionally
        // require an ADMIN key so a normal tenant cannot read every tenant's
        // usage. When auth is off (dev), there is no admin distinction — the
        // routes are open like everything else.
        server->Get("/v1/admin/tenants",
                   [this](const httplib::Request&, httplib::Response& res) {
                       if (!requireAdmin(res)) return;
                       sendJson(res, 200,
                                tenantMetrics.snapshotJson(cfg.batcher));
                   });
        server->Get("/metrics",
                   [this](const httplib::Request&, httplib::Response& res) {
                       if (!requireAdmin(res)) return;
                       // OpenMetrics/Prometheus text exposition format.
                       res.set_content(
                           tenantMetrics.snapshotOpenMetrics(cfg.batcher),
                           "text/plain; version=0.0.4; charset=utf-8");
                       res.status = 200;
                   });
        server->Post("/v1/chat/completions",
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

        server->Post("/v1/rerank",
                    [this](const httplib::Request& req,
                           httplib::Response&       res) {
                        MM_LOG_INFO(
                            "server",
                            "POST /v1/rerank accepted from {} "
                            "(content-length={} B)",
                            req.remote_addr, req.body.size());
                        rerankHandler.handle(req, res);
                    });

        server->set_exception_handler(
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

        server->set_logger([](const httplib::Request& req,
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

    // Gate the operator-only routes on an admin-scoped key. Returns true to
    // proceed. When auth is disabled there is no privilege distinction, so
    // the routes are as open as the rest of the API. A valid-but-non-admin
    // key gets a 403 (distinct from the 401 an invalid key already got).
    bool requireAdmin(httplib::Response& res) {
        if (cfg.auth.enabled && !core::security::ScopedTenant::activeIsAdmin()) {
            sendError(res, 403, "forbidden",
                      "admin API key required for this route");
            return false;
        }
        return true;
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
        for (const auto& m : rerankHandler.listModels()) {
            data.push_back(json{
                {"id",       m.id},
                {"title",    m.title},
                {"object",   "model"},
                {"created",  0},
                {"owned_by", "mimirmind"},
                {"task",     "rerank"},
            });
        }
        sendJson(res, 200, {
            {"object", "list"},
            {"data",   std::move(data)},
        });
    }

    void run() {
        // Serving-class load management: size the HTTP worker pool to the
        // batcher's admission bound (+ headroom for health/monitoring routes)
        // so the batcher — which knows the GPU's real serving capacity — is the
        // authoritative admission controller and its clean 503 actually fires.
        // With the default pool (~hw_concurrency) smaller than maxInflight, the
        // undersized pool becomes the silent bottleneck: excess requests queue
        // in httplib's unbounded task queue for tens of seconds instead of
        // getting a fast 503. Pool threads waiting on the batcher are just
        // blocked on a condvar (no CPU), so a larger pool is cheap. The accept
        // queue is bounded too, so an extreme flood sheds (connection close)
        // rather than growing httplib's queue without limit. Non-serving
        // (single-session) keeps httplib's default pool.
        if (cfg.batcher != nullptr) {
            constexpr std::size_t kHeadroom = 8;   // health/models/system routes
            const std::size_t poolSize =
                cfg.batcher->maxInflight() + kHeadroom;
            const std::size_t queueMax = cfg.batcher->maxInflight() * 4;
            server->new_task_queue = [poolSize, queueMax] {
                return new httplib::ThreadPool(poolSize, queueMax);
            };
            MM_LOG_INFO("server",
                        "serving-class HTTP pool = {} workers (maxInflight {} + "
                        "headroom {}), accept-queue bound {} — batcher governs "
                        "admission with clean 503s",
                        poolSize, cfg.batcher->maxInflight(), kHeadroom, queueMax);
        }
        MM_LOG_INFO("server", "binding {}:{} (model={})",
                    cfg.host, cfg.port, cfg.modelId);
        started.store(true);
        if (!server->listen(cfg.host, cfg.port)) {
            started.store(false);
            throw std::runtime_error("ApiServer: failed to bind " +
                                     cfg.host + ":" + std::to_string(cfg.port));
        }
    }

    void stop() {
        if (started.exchange(false)) {
            server->stop();
            // Persist the final metrics window promptly on shutdown (the
            // dtor also flushes, but stop() may run well before teardown).
            tenantMetrics.flush();
        }
    }
};

ApiServer::ApiServer(std::vector<LoadedEngine>     engines,
                     ServerConfig                  cfg,
                     runtime::Drafter*             drafter,
                     std::vector<LoadedReranker>   rerankers)
    : _impl{std::make_unique<Impl>(std::move(engines),
                                   std::move(cfg), drafter,
                                   std::move(rerankers))} {}

ApiServer::~ApiServer() = default;

void ApiServer::run()  { _impl->run(); }
void ApiServer::stop() { _impl->stop(); }

} // namespace mimirmind::server