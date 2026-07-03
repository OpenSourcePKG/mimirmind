#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace mimirmind::runtime {
class InferenceEngine;
}

namespace mimirmind::server {

struct ServerConfig {
    /// Bind address. "0.0.0.0" for all interfaces; "127.0.0.1" for loopback.
    std::string   host{"0.0.0.0"};

    /// TCP port for the listener.
    std::uint16_t port{8080};

    /// Identifier reported by `GET /v1/models` and echoed in chat-completion
    /// responses. Empty => derived from the model filename's stem.
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
    /// `draftEngine` is optional — when non-null the server exposes
    /// speculative-decoding metadata via /v1/system/info. The actual
    /// speculation loop lands in M9.11.2+, until then the draft engine
    /// is held but not called during chat/completions.
    ApiServer(runtime::InferenceEngine& engine,
              ServerConfig              cfg,
              runtime::InferenceEngine* draftEngine = nullptr);
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