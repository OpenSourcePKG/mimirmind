// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/security/ApiKeyStore.hpp"
#include "runtime/spec/SpeculativeDecoder.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mimirmind::runtime {
class Drafter;
class InferenceEngine;
}
namespace mimirmind::runtime::serving {
class ContinuousBatcher;
}
namespace mimirmind::runtime::encoder {
class RerankEngine;
class EmbedEngine;
}

namespace mimirmind::server {

/**
 * One loaded model exposed by the server. Each entry gets its own dispatch
 * mutex inside ApiServer, so requests to different models can run in
 * parallel (the engines are independent InferenceEngine instances with
 * separate USM, KV cache, and sampler state).
 *
 * `engine` is non-owning — `main()` keeps the InferenceEngine alive for
 * the process lifetime.
 */
struct LoadedEngine {
    /// OpenAI-facing name. Matched against request.model. Must be unique.
    std::string                 id;
    /// Human-readable name for UI dropdowns. Falls back to `id` in
    /// /v1/models when empty.
    std::string                 title;
    runtime::InferenceEngine*   engine{nullptr};
};

/**
 * One loaded cross-encoder reranker exposed under /v1/rerank. `engine` is
 * non-owning — ServeMode keeps the RerankEngine (and its compute stack)
 * alive for the process lifetime. Each entry gets its own serialisation
 * mutex inside RerankHandler.
 */
struct LoadedReranker {
    std::string                     id;
    std::string                     title;
    runtime::encoder::RerankEngine* engine{nullptr};
};

/**
 * One loaded bi-encoder embedding model exposed under /v1/embeddings. `engine`
 * is non-owning — ServeMode keeps the EmbedEngine (and its compute stack) alive
 * for the process lifetime. Each entry gets its own serialisation mutex inside
 * EmbeddingsHandler.
 */
struct LoadedEmbedder {
    std::string                    id;
    std::string                    title;
    runtime::encoder::EmbedEngine* engine{nullptr};
};

/// In-process TLS termination settings. When `enabled`, the listener is a
/// cpp-httplib `SSLServer` built from an in-memory PEM cert/key pair; the
/// certificate is provisioned by ServeMode (real files, or an auto self-
/// signed pair) so the server itself never touches disk here. OpenSSL types
/// are deliberately kept out of this header — the PEM strings cross the
/// boundary instead.
struct TlsConfig {
    /// Secure-by-default: TLS on unless explicitly disabled in config.
    bool        enabled{true};
    /// PEM-encoded certificate (leaf, optionally chain). Empty when
    /// `enabled` is false.
    std::string certPem{};
    /// PEM-encoded private key matching `certPem`.
    std::string keyPem{};
    /// True when the loaded cert was auto-generated (self-signed) rather
    /// than operator-provided — drives the startup WARN banner.
    bool        selfSigned{false};
};

/// Bearer-token API-key auth settings. When `enabled`, a pre-routing hook
/// requires `Authorization: Bearer <key>` on protected routes and answers
/// 401 otherwise. Keys are provisioned by ServeMode (explicit config, a
/// keyfile, or a single auto-generated key).
struct AuthConfig {
    bool                        enabled{false};
    core::security::ApiKeyStore store{};
};

struct ServerConfig {
    /// Bind address. "0.0.0.0" for all interfaces; "127.0.0.1" for loopback.
    std::string   host{"0.0.0.0"};

    /// In-process TLS (OpenSSL via httplib SSLServer).
    TlsConfig     tls{};

    /// Bearer-token API-key auth.
    AuthConfig    auth{};

    /// Persistence file for per-tenant usage metrics (per-API-key request /
    /// token accounting shown on the admin routes). Empty => metrics kept
    /// in-memory only (or disabled — ServeMode leaves it empty when
    /// `server.metrics.enabled` is false).
    std::string   tenantMetricsPath{};

    /// TCP port for the listener.
    std::uint16_t port{8080};

    /// Fallback model id when a chat request omits `model` or sends an
    /// empty string. Must match one of the LoadedEngine.id values passed
    /// to the ApiServer constructor, or an empty string to force the
    /// caller to always specify the model.
    std::string   modelId{};

    /// Max new tokens when the request does not specify `max_tokens` /
    /// `max_completion_tokens`.
    std::size_t   defaultMaxNew{512};

    /// Preserve the architecture's internal thinking-channel markup in
    /// the assistant response (currently only Gemma 4's
    /// `<|channel>thought<channel|>` wrapper). Off by default so
    /// OpenAI-compatible clients see clean text. Turn on when the
    /// client round-trips the assistant turn back into a follow-up
    /// prompt and you want the M9.1 prefix cache to match across
    /// turns — the cache is keyed on raw decoded tokens, and stripping
    /// the wrapper from the response makes the round-trip diverge
    /// from the cached tokens at the first assistant turn.
    bool          preserveThinking{false};

    /// Speculative-decoding settings from `speculative` in config.json.
    /// Applied inside ApiServer::Impl when a draft engine is present.
    runtime::SpeculativeDecoder::Config speculative{};

    /// Model id (matching one of the LoadedEngine.id values) that the
    /// speculative-decoding orchestrator should treat as its target.
    /// When empty, or when it matches the default engine's id, spec-dec
    /// wires up on the default engine. When it names a non-default
    /// engine, spec-dec stays off with a warning (target-on-extra is
    /// not wired yet).
    std::string speculativeTargetId{};

    /// M-Cuda.Batch D2e.2 — continuous-batching worker for the DEFAULT
    /// engine (serving-class). Non-owning; the caller keeps it alive for
    /// the server's lifetime. When set, chat requests that resolve to the
    /// default engine are serviced through the batcher (multi-tenant
    /// continuous batching) instead of the serialised single-session
    /// generate() path. nullptr => classic per-engine-mutex generate().
    runtime::serving::ContinuousBatcher* batcher{nullptr};
};

/**
 * OpenAI-compatible HTTP front-end for the inference engine.
 *
 * Endpoints (M7d):
 *   - `GET  /health`               -> liveness probe
 *   - `GET  /v1/models`            -> single model entry
 *   - `POST /v1/chat/completions`  -> non-streaming (streaming = 501 here,
 *                                     M7e implements it)
 *
 * The engine is not thread-safe (mutable sampler + scratch); requests are
 * serialised through an internal mutex. cpp-httplib runs handlers on
 * worker threads, so the server stays responsive for /health while a
 * generation is in flight.
 *
 * PIMPL so cpp-httplib + nlohmann/json stay out of the public header.
 */
class ApiServer {
public:
    /// `engines` must be non-empty. Each entry maps to an engine that has
    /// already finished loadModel(). `cfg.modelId`, if set, must match one
    /// of the engine ids and picks the default engine for requests that
    /// omit `model`.
    ///
    /// `drafter` is optional — when non-null, and `cfg.speculative`
    /// resolves `speculative.target` against the default engine's id,
    /// the ApiServer wires up the M9.11.4 spec-dec orchestrator for
    /// that target. Other loaded engines dispatch through the plain
    /// generate() path.
    ApiServer(std::vector<LoadedEngine>     engines,
              ServerConfig                  cfg,
              runtime::Drafter*             drafter = nullptr,
              std::vector<LoadedReranker>   rerankers = {},
              std::vector<LoadedEmbedder>   embedders = {});
    ~ApiServer();

    ApiServer(const ApiServer&)            = delete;
    ApiServer& operator=(const ApiServer&) = delete;
    ApiServer(ApiServer&&)                 = delete;
    ApiServer& operator=(ApiServer&&)      = delete;

    /// Bind + listen. Blocks until stop() is called or listening fails.
    void run();

    /// Stop the listener. Safe to call from a signal handler.
    void stop();

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace mimirmind::server