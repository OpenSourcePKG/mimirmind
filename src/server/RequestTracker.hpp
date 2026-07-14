// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstddef>
#include <mutex>
#include <optional>
#include <string>

namespace mimirmind::server {

/// In-flight chat-request snapshot for /v1/system/status.current_request.
///
/// A UI (e.g. Pegenaut) can poll this to render "prefill 24/42, decode
/// 15/4096" instead of showing a spinner for the ~15 minutes a long
/// prompt with a large max_tokens can take. Guarded by an internal
/// mutex; only one chat request runs at a time thanks to the engine
/// dispatch mutex, so writers never contend — the status poller reads
/// while a writer may be updating.
class RequestTracker {
public:
    RequestTracker() = default;

    RequestTracker(const RequestTracker&)            = delete;
    RequestTracker& operator=(const RequestTracker&) = delete;
    RequestTracker(RequestTracker&&)                 = delete;
    RequestTracker& operator=(RequestTracker&&)      = delete;

    void begin(std::string id, std::size_t promptTokens,
               std::size_t maxNew, bool streaming);
    void end() noexcept;

    void updatePrefillProgress(std::size_t blocksDone,
                                std::size_t blocksTotal,
                                double      elapsedMs);
    void markPrefillDone();
    void incrementDecodeTokens();

    /// JSON block for /v1/system/status.current_request. Returns
    /// `{"active": false}` when idle.
    [[nodiscard]] nlohmann::json buildStatusBlock() const;

    /// RAII helper — clears the snapshot on scope exit even if the
    /// generate() call throws.
    class Guard {
    public:
        explicit Guard(RequestTracker* t) noexcept : _t{t} {}
        ~Guard() noexcept { if (_t) _t->end(); }
        Guard(const Guard&)            = delete;
        Guard& operator=(const Guard&) = delete;
        Guard(Guard&&)                 = delete;
        Guard& operator=(Guard&&)      = delete;
    private:
        RequestTracker* _t;
    };

private:
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

    mutable std::mutex               _mutex;
    std::optional<CurrentRequest>    _current;
};

} // namespace mimirmind::server