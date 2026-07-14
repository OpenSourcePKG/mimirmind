// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <chrono>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace mimirmind::runtime {

/**
 * Reads Intel RAPL energy counters from /sys/class/powercap/intel-rapl*.
 * RAPL exposes monotonically-rising microjoule counters per energy
 * domain (package, core, uncore, dram, psys) — sampling them at two
 * timestamps gives Joules consumed in between, and dividing by elapsed
 * time gives average Watts.
 *
 * Visibility caveat: Docker masks /sys/devices/virtual/powercap by
 * default (Platypus side-channel mitigation). The mimirmind compose
 * file sets `security_opt: systempaths=unconfined` on the runtime
 * service to expose it. When that masking is in place this class
 * reports `available() == false` with an explanation rather than
 * throwing — power telemetry is optional, not a safety feature.
 *
 * Wrap handling: each RAPL counter wraps at `max_energy_range_uj`,
 * typically in the tens of GJ (weeks to months at typical idle).
 * `energyBetween()` handles a single wrap between two snapshots
 * transparently. Multi-wrap between samples is extraordinarily
 * unlikely at any sane polling rate and is treated as "delta = 0".
 */
class PowerMonitor {
public:
    /// Single counter snapshot — keep one around to compute energy
    /// deltas against a later snapshot.
    struct Snapshot {
        std::chrono::steady_clock::time_point taken_at{};
        /// Parallel to domainNames(). Empty if monitor is unavailable.
        std::vector<std::uint64_t> raw_energy_uj{};
    };

    /// `sysfsRoot` defaults to `/sys/class/powercap`. Override only in
    /// tests to point at a synthetic powercap tree on disk.
    explicit PowerMonitor(std::string_view sysfsRoot = "/sys/class/powercap");

    [[nodiscard]] bool available() const noexcept { return _available; }

    /// Human-readable reason the monitor is unavailable. Empty when
    /// `available()` is true.
    [[nodiscard]] std::string_view unavailableReason() const noexcept {
        return _unavailableReason;
    }

    /// Names of the discovered domains, parallel to Snapshot::raw_energy_uj.
    /// Examples: "package-0", "core", "uncore", "dram", "psys".
    [[nodiscard]] std::span<const std::string> domainNames() const noexcept {
        return _domainNames;
    }

    /// Take a fresh snapshot of all discovered domains. If the monitor
    /// is unavailable, returns a Snapshot whose `raw_energy_uj` is
    /// empty.
    [[nodiscard]] Snapshot snapshot() const;

    /// Joules consumed in each domain between `start` and `end`,
    /// handling counter wrap. Returns a vector parallel to
    /// domainNames(). On size mismatch or unavailable monitor, returns
    /// empty.
    [[nodiscard]] std::vector<double>
    energyBetween(const Snapshot& start, const Snapshot& end) const;

    /// Convenience: average Watts per domain across `start..end`.
    /// Same shape as energyBetween(); returns 0 for domains where the
    /// elapsed time is zero.
    [[nodiscard]] std::vector<double>
    averageWattsBetween(const Snapshot& start, const Snapshot& end) const;

private:
    struct Domain {
        std::string   name;
        std::string   energyUjPath;
        std::uint64_t maxRangeUj{0};
    };

    void probe(std::string_view sysfsRoot);

    std::vector<Domain>      _domains;
    std::vector<std::string> _domainNames;
    bool                     _available{false};
    std::string              _unavailableReason{};
};

} // namespace mimirmind::runtime