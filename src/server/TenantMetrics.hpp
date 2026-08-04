// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <nlohmann/json.hpp>

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

namespace mimirmind::runtime::serving {
class ContinuousBatcher;
}

namespace mimirmind::server {

/**
 * Per-tenant (per-API-key) usage accounting for admin evaluation.
 *
 * One aggregate row per `tenantId` (the label an API key carries). Fed
 * from the request path — every chat request records exactly one outcome
 * (success / quota-rejected / overload-rejected / error) plus, on success,
 * its token + latency + energy deltas. Read by the operator-only routes
 * `/v1/admin/tenants` (JSON) and `/metrics` (OpenMetrics text).
 *
 * ---- Persistence -------------------------------------------------------
 * Counters survive a restart: on construction the accumulated totals are
 * loaded from `persistPath` (a JSON file), and a background thread flushes
 * the (dirty) map back every few seconds plus once on shutdown. Writes are
 * atomic (temp file + rename) so a crash mid-flush never truncates the
 * store. An empty `persistPath` disables persistence (in-memory only).
 *
 * ---- Threading ---------------------------------------------------------
 * All public methods are thread-safe (internal mutex): recorders run on
 * the httplib worker threads, the snapshots on whichever thread serves the
 * admin route, and the flush thread reads under the same lock.
 */
class TenantMetrics {
public:
    /// `persistPath` empty => in-memory only (no load, no flush thread).
    /// Otherwise the prior totals are loaded immediately and a background
    /// flush thread is started.
    explicit TenantMetrics(std::string persistPath = {});
    ~TenantMetrics();

    TenantMetrics(const TenantMetrics&)            = delete;
    TenantMetrics& operator=(const TenantMetrics&) = delete;
    TenantMetrics(TenantMetrics&&)                 = delete;
    TenantMetrics& operator=(TenantMetrics&&)      = delete;

    // ---- Recording (request path) --------------------------------------

    /// A completed (200) request: add its prompt/completion token counts,
    /// energy (J), and prefill/decode wall-time (ms) to the tenant's totals.
    void recordSuccess(const std::string& tenant,
                       std::uint64_t promptTokens, std::uint64_t completionTokens,
                       double energyJoules, double prefillMs, double decodeMs);

    /// A request shed by the per-tenant admission quota (429).
    void recordQuotaRejected(const std::string& tenant);

    /// A request shed by the whole-server overload / thermal guard (503).
    void recordOverloadRejected(const std::string& tenant);

    /// A request that failed with a server error (500).
    void recordError(const std::string& tenant);

    // ---- Reading (admin routes) ----------------------------------------

    /// JSON for `GET /v1/admin/tenants`. When `batcher` is non-null each
    /// tenant row also carries its live `active_now` in-flight count.
    [[nodiscard]] nlohmann::json snapshotJson(
        const runtime::serving::ContinuousBatcher* batcher) const;

    /// OpenMetrics/Prometheus text exposition for `GET /metrics`.
    [[nodiscard]] std::string snapshotOpenMetrics(
        const runtime::serving::ContinuousBatcher* batcher) const;

    /// Force a synchronous write to `persistPath` (no-op without a path).
    void flush() const;

private:
    struct TenantStat {
        std::uint64_t requests_ok{0};
        std::uint64_t rejected_quota{0};
        std::uint64_t rejected_overload{0};
        std::uint64_t errors{0};
        std::uint64_t prompt_tokens{0};
        std::uint64_t completion_tokens{0};
        double        energy_joules{0.0};
        double        prefill_ms{0.0};
        double        decode_ms{0.0};
        std::int64_t  last_seen{0};   // unix seconds
    };

    /// Upsert the tenant, stamp last_seen=now, mark dirty. Caller holds `_mtx`.
    TenantStat& touchLocked(const std::string& tenant);

    void load();                 // read persistPath into the map (ctor)
    void flushLocked() const;    // write map to persistPath; caller holds `_mtx`
    void flushLoop();            // background thread body

    mutable std::mutex                            _mtx;
    std::unordered_map<std::string, TenantStat>   _byTenant;
    std::string                                   _persistPath;

    // Background flush thread + its wake/stop signalling. Only started when
    // `_persistPath` is non-empty. `_dirty` gates redundant writes.
    mutable bool                                  _dirty{false};
    bool                                          _running{false};
    std::condition_variable                       _cv;
    std::thread                                   _flusher;
};

} // namespace mimirmind::server
