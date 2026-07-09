#pragma once

#include "runtime/SpeculativeDecoder.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mimirmind::runtime {
class InferenceEngine;
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

struct ServerConfig {
    /// Bind address. "0.0.0.0" for all interfaces; "127.0.0.1" for loopback.
    std::string   host{"0.0.0.0"};

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
    /// `draftEngine` is optional — when non-null, and `cfg.speculative`
    /// resolves against one of the loaded engines' id via
    /// `speculative.target`, the ApiServer wires up the M9.11.4 spec-dec
    /// orchestrator for that specific target engine only. Other loaded
    /// engines dispatch through the plain generate() path.
    ApiServer(std::vector<LoadedEngine>   engines,
              ServerConfig                cfg,
              runtime::InferenceEngine*   draftEngine = nullptr);
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