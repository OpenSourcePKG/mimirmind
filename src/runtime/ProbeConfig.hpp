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

    // Layer-2 flag intents parsed from the profile `flags` block. Each is
    // populated ONLY for a flag entry marked "apply": true — the profile
    // author's explicit opt-in that the flag is validated-safe to auto-enable
    // on this hardware. nullopt = "the profile does not drive this flag", so
    // the runtime keeps its code-default / env value. Lossy or
    // precision-unvalidated flags stay apply:false (hence nullopt here) until
    // the DE-goldset gate (5.19 Increment C) clears them.
    std::optional<bool> applyPrefillCudnn;
    std::optional<bool> applyF32TcPrefill;
    std::optional<bool> applyCublasFp8Prefill;
    std::optional<bool> applyMmq;
    std::optional<bool> applyMmqTc;
    // 5.21.7/8 serving-prefill flags (backend getenv-owned -> applied via setenv
    // before the arch backend is constructed; explicit env wins).
    std::optional<bool> applyAttnCudnnPaged;
    std::optional<bool> applyMoeSiluFuse;
    // 5.18.9/5.18.13: register-staged batched-decode grouped MoE-GEMM (backend +
    // GpuOps getenv-owned; applied via setenv before the backend ctor). MODE-
    // valued, not a bool: 0=off, 1=m2reg@tileM2, 4=m4reg@tileM4 — the profile's
    // integer `value` is passed through verbatim (5.18.14).
    std::optional<int> applyMoeDecodeReg;
    // Mode-valued grouped-MoE prefill/decode path selector (0=blocked-only,
    // 3=FP4-TC-only). MODEL-DEPENDENT: 512-expert checkpoints (Qwen3-Coder-Next)
    // need the TC path (the blocked device-driven grouped kernels scale
    // catastrophically at 512 experts — seconds/forward), while it also drops
    // the blocked bank so peak memory is LOWER. Lives in the per-model overlay,
    // not the HW profile, because the right value differs per checkpoint on the
    // same GPU. nullopt = keep code-default / env.
    std::optional<int> applyGroupedMoe;
};

/**
 * Load configs/hw/{fingerprint}/probe-result.json under `dir` and extract the
 * actionable picks. Returns nullopt on any miss: no dir/file, a fingerprint
 * mismatch inside the file, or a parse error (logged, never throws). The
 * caller stays on its defaults on nullopt.
 */
[[nodiscard]] std::optional<ProbePicks>
loadProbePicks(const std::string& dir, const std::string& fingerprint);

/**
 * Load a per-model flag overlay from
 * configs/hw/{fingerprint}/model-overlays/{modelId}.json under `dir` and return its
 * `flags` picks (same schema as the HW profile's flags block: an entry drives
 * a flag only with "apply": true). This is the (hardware x model) key: the same
 * flag can be validated-safe on one checkpoint and unsafe on another running on
 * the same GPU (e.g. F32_TC_PREFILL, or GROUPED_MOE's 512-expert TC path).
 * Returns nullopt on any miss (no file / parse error), so the caller keeps the
 * HW-profile picks unchanged. `modelId` is sanitised to a bare filename.
 */
[[nodiscard]] std::optional<ProbePicks>
loadModelOverlay(const std::string& dir, const std::string& fingerprint,
                 const std::string& modelId);

/**
 * Overlay `over`'s set (non-nullopt) flag intents onto `base`, model-wins.
 * Only the Layer-2 flag optionals are merged; identity fields stay `base`'s.
 */
void mergeOverlay(ProbePicks& base, const ProbePicks& over);

} // namespace mimirmind::runtime
