// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <optional>
#include <string>

namespace mimirmind::runtime {

/**
 * The actionable subset of an offline probe artefact (M-Probe.1), consumed
 * by the runtime at model load. Lookup-only: a match lets the runtime apply
 * the persisted kernel-variant decision and skip the online autotune bench;
 * a miss falls back to the normal online autotune (no behaviour change).
 */
struct ProbePicks {
    std::string fingerprint;   // as read from the artefact
    std::string modelId;       // model_id tag in the artefact
    std::string probeStatus;   // "fingerprint-only" | "swept-partial" | ...
    bool        hasAutotune{false};
    // Every GEMM-capable matmul QuantType resolved to matvec (GEMM never
    // picked). On a fingerprint match this is safe to apply as
    // features.gemm=Disable — same decision the online autotune would reach
    // on identical hardware, minus the multi-second bench.
    bool        gemmNeverForAllMatmul{false};
};

/**
 * Load configs/hw/{fingerprint}/probe-result.json under `dir` and extract the
 * actionable picks. Returns nullopt on any miss: no dir/file, a fingerprint
 * mismatch inside the file, or a parse error (logged, never throws). The
 * caller stays on its defaults on nullopt.
 */
[[nodiscard]] std::optional<ProbePicks>
loadProbePicks(const std::string& dir, const std::string& fingerprint);

} // namespace mimirmind::runtime
