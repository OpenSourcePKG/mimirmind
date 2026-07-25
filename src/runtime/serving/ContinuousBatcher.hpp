// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace mimirmind::runtime {
class InferenceEngine;
}

namespace mimirmind::runtime::serving {

/**
 * A single in-flight serving request. Shared between the submitting HTTP
 * thread and the batcher worker: the worker appends generated token ids
 * under `mtx` and signals `cv`; the HTTP thread waits on `cv` to block for
 * completion or to stream tokens as they arrive.
 *
 * M-Cuda.Batch D2e.2.
 */
struct ServingRequest {
    std::mutex               mtx;
    std::condition_variable  cv;
    std::vector<std::int32_t> tokens;    // generated ids, appended by worker
    bool                     done{false};
    bool                     cancelled{false};
    std::string              error;      // non-empty => request failed

    /// Block until the request completes; returns the full token stream.
    /// On failure the stream may be partial and `error` is set.
    std::vector<std::int32_t> waitAll();

    /// Streaming: wait until token index `next` is available or the request
    /// finishes. Returns true and writes `out` if a token at `next` exists;
    /// returns false when the request is done and no token `next` remains.
    bool waitToken(std::size_t next, std::int32_t& out);
};

/**
 * Continuous-batching serving loop — Sub-Step D2e.2 of M-Cuda.Batch.
 *
 * Owns a single worker thread that drives `InferenceEngine::stepServing`
 * over a fixed set of `maxBatch` physical slots. HTTP threads `submit()`
 * prompts asynchronously; the worker admits waiting prompts into free slots
 * (lowest index first, so the active set stays a tight prefix), steps every
 * active slot ONE token per iteration — at the slot's OWN position, so a
 * freshly admitted prompt prefills at position 0 while others decode at
 * position 500 in the SAME batched forward — and frees a slot the moment its
 * request hits EOS / a stop id / its token budget.
 *
 * v1 uses token-by-token prefill (one prompt token per slot per iteration):
 * correct and genuinely continuous. Chunked multi-token prefill is a later
 * optimization (needs a var-length attention kernel).
 *
 * CUDA + qwen35moe only (the engine's stepServing throws otherwise).
 */
class ContinuousBatcher {
public:
    /// `engine` must outlive the batcher. `maxBatch` physical slots,
    /// `maxContext` tokens (prompt + generated) per slot. `eosId` ends a
    /// request when generated (pass negative to disable the implicit EOS
    /// stop). The serving state is built eagerly here.
    ContinuousBatcher(InferenceEngine& engine, std::size_t maxBatch,
                      std::size_t maxContext, std::int32_t eosId);
    ~ContinuousBatcher();

    ContinuousBatcher(const ContinuousBatcher&)            = delete;
    ContinuousBatcher& operator=(const ContinuousBatcher&) = delete;
    ContinuousBatcher(ContinuousBatcher&&)                 = delete;
    ContinuousBatcher& operator=(ContinuousBatcher&&)      = delete;

    /// Enqueue a request. `prompt` must be non-empty; `maxNew` caps the
    /// generated tokens; `stopIds` are extra single-token stops (checked in
    /// addition to the ctor eosId). Returns immediately with a handle the
    /// caller waits/streams on. Thread-safe.
    std::shared_ptr<ServingRequest> submit(std::vector<std::int32_t> prompt,
                                           std::size_t               maxNew,
                                           std::vector<std::int32_t> stopIds);

    [[nodiscard]] std::size_t maxBatch()   const noexcept { return _maxBatch; }
    [[nodiscard]] std::size_t maxContext() const noexcept { return _maxContext; }

private:
    struct Pending {
        std::shared_ptr<ServingRequest> req;
        std::vector<std::int32_t>       prompt;
        std::size_t                     maxNew{0};
        std::vector<std::int32_t>       stopIds;
    };

    struct Slot {
        bool                            occupied{false};
        std::shared_ptr<ServingRequest> req;
        std::vector<std::int32_t>       prompt;
        std::size_t                     promptLen{0};
        std::size_t                     pos{0};       // next input index / position
        std::int32_t                    lastTok{0};
        std::size_t                     maxNew{0};
        std::size_t                     produced{0};  // generated tokens so far
        std::vector<std::int32_t>       stopIds;
    };

    void workerLoop();
    [[nodiscard]] bool isStop(std::int32_t tok, const Slot& s) const;

    InferenceEngine& _engine;
    std::size_t      _maxBatch;
    std::size_t      _maxContext;
    std::int32_t     _eosId;

    std::vector<Slot>                     _slots;
    std::deque<Pending>                   _waiting;
    std::mutex                            _mtx;
    std::condition_variable               _cv;      // wakes worker on submit/stop
    std::thread                           _worker;
    bool                                  _running{false};
};

} // namespace mimirmind::runtime::serving
