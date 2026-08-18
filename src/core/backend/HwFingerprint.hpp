// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <array>
#include <cstdint>
#include <string>

namespace mimirmind::core::backend {

/**
 * Stable hardware-identity fields for the probe fingerprint (M-Probe.1).
 * The same struct is built by `mimirmind-probe` (from its constructed
 * ComputeContext) and by the runtime (from InferenceEngine's ComputeContext)
 * so both derive an IDENTICAL fingerprint and the runtime can look up the
 * offline probe artefact keyed by it. Only stable identity goes in here —
 * never clock/bandwidth (jitter with governor state) or timestamps.
 */
struct HwProbeIdentity {
    bool          haveBackend{false};   // false => host-only fingerprint
    std::string   backendKind;          // "LevelZero" / "Hip" / "Cuda" / "none"
    std::string   deviceName;
    std::string   uuid;
    std::string   pci;                  // "0x8086:0x591b"
    std::uint32_t numComputeUnits{0};
    // MutableCommandLists, IntegerDotProduct, IpcHandleExport,
    // UnifiedMemoryHost, MatrixEngine — same order as kProbeFeatureNames.
    std::array<bool, 5> features{};
};

/// The five neutral feature flags, in the fixed order used by the fingerprint
/// and the probe artefact.
inline constexpr std::array<const char*, 5> kProbeFeatureNames = {
    "MutableCommandLists", "IntegerDotProduct", "IpcHandleExport",
    "UnifiedMemoryHost", "MatrixEngine"};

/// Host-side identity (Linux /proc + uname). RAM is bucketed to whole GiB so
/// small cross-boot jitter does not change the fingerprint.
struct HwHostInfo {
    std::string   cpuModel;
    unsigned      threads{0};
    std::uint64_t ramGiB{0};
    std::string   kernel;
};

class ComputeBackend;   // fwd

/// Build the fingerprint identity from a constructed backend. The single
/// source of truth so the probe tool and the runtime cannot drift.
[[nodiscard]] HwProbeIdentity identityFromBackend(const ComputeBackend& backend);

/// Gather host identity from /proc/cpuinfo, /proc/meminfo and uname.
[[nodiscard]] HwHostInfo gatherHostInfo();

/// Deterministic 16-hex-char (FNV-1a-64) fingerprint over the stable host +
/// device identity. A cache key, not a security digest.
[[nodiscard]] std::string computeHwFingerprint(const HwHostInfo&      host,
                                               const HwProbeIdentity& id);

} // namespace mimirmind::core::backend
