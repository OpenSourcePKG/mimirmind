// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstddef>
#include <mutex>
#include <string>
#include <unordered_map>

namespace mimirmind::server {

/// In-flight chat-request snapshots for /v1/system/status.current_request.
///
/// A UI (e.g. Pegenaut) can poll this to render "prefill 24/42, decode
/// 15/4096" instead of showing a spinner for the ~15 minutes a long
/// prompt with a large max_tokens can take.
///
/// Keyed by request id: the RequestDispatcher hands each model its own
/// mutex, so requests to *different* models run concurrently. A single
/// slot would let one request clobber another's snapshot (and the first
/// Guard to destruct would wipe a still-running request). Every mutating
/// call therefore names the request it touches; `end()` erases only that
/// id. Guarded by an internal mutex — writers on parallel engine threads
/// and the status poller may touch the map at the same time.
class RequestTracker {
public:
    RequestTracker() = default;

    RequestTracker(const RequestTracker&)            = delete;
    RequestTracker& operator=(const RequestTracker&) = delete;
    RequestTracker(RequestTracker&&)                 = delete;
    RequestTracker& operator=(RequestTracker&&)      = delete;

    void begin(std::string id, std::size_t promptTokens,
               std::size_t maxNew, bool streaming);
    void end(const std::string& id) noexcept;

    void updatePrefillProgress(const std::string& id,
                                std::size_t blocksDone,
                                std::size_t blocksTotal,
                                double      elapsedMs);
    void markPrefillDone(const std::string& id);
    void incrementDecodeTokens(const std::string& id);

    /// JSON block for /v1/system/status.current_request. Returns
    /// `{"active": false}` when idle. When at least one request is in
    /// flight it reports `{"active": true, "count": N, "requests": [...]}`;
    /// for single-tenant back-compat the fields of the oldest active
    /// request are also mirrored at the top level.
    [[nodiscard]] nlohmann::json buildStatusBlock() const;

    /// RAII helper — erases the request's snapshot on scope exit even if
    /// the generate() call throws. Holds the id so it only clears its own
    /// entry, never a concurrently-running request's.
    class Guard {
    public:
        Guard(RequestTracker* t, std::string id) noexcept
            : _t{t}, _id{std::move(id)} {}
        ~Guard() noexcept { if (_t) _t->end(_id); }
        Guard(const Guard&)            = delete;
        Guard& operator=(const Guard&) = delete;
        Guard(Guard&&)                 = delete;
        Guard& operator=(Guard&&)      = delete;
    private:
        RequestTracker* _t;
        std::string     _id;
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

    /// Serialise one in-flight request into its status-block shape.
    static nlohmann::json requestJson(const CurrentRequest& r);

    mutable std::mutex                                _mutex;
    std::unordered_map<std::string, CurrentRequest>   _requests;
};

} // namespace mimirmind::server