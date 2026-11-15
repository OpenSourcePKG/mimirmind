// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "runtime/ProbeConfig.hpp"

#include <optional>
#include <string>

namespace mimirmind::core::backend { class ComputeContext; }
namespace mimirmind::compute {
class ComputeOps;
class ComputeMatmul;
}
namespace mimirmind::model { struct LlmConfig; }

namespace mimirmind::runtime {

/**
 * The probe profile resolved for one model load: the HW fingerprint, the
 * profile directory it was looked up under, and the actionable picks (with the
 * per-(HW x model) overlay merged in when one was requested and found).
 */
struct ResolvedProfile {
    std::string               fingerprint;
    std::string               dir;
    std::optional<ProbePicks> picks;          // nullopt = no profile deployed
    bool                      overlayMerged{false};
};

/**
 * Applies the Layer-2 flag intents of an M-Probe.1 / 5.19 per-HW profile (and
 * its per-model overlay) to a freshly loaded engine. Extracted from
 * InferenceEngine::finalizeLoad (roadmap 8.30.11.6).
 *
 * Three flag sinks, applied in this order:
 *   1. kernel setters on ComputeOps / ComputeMatmul (cuDNN / F32-TC / cuBLAS-FP8
 *      prefill, MMQ) — skipped when the backend reports an explicit env override;
 *   2. process env for flags the arch backend / loader read via getenv in their
 *      ctor (must run BEFORE the backend is constructed);
 *   3. per-model LlmConfig fields (sampling floors, honesty floor, Jinja) — not a
 *      process env, so co-resident models keep their own values.
 * In every sink an explicit env variable wins over the profile.
 */
class ProfileOverlayApplier {
public:
    ProfileOverlayApplier(compute::ComputeOps&    ops,
                          compute::ComputeMatmul& gmm,
                          model::LlmConfig&       config) noexcept;

    /// Fingerprint the host + backend and load the profile from
    /// `$MIMIRMIND_PROBE_DIR` (default /usr/local/share/mimirmind/configs).
    /// A non-empty `modelIdHint` also merges that model's overlay (model wins).
    [[nodiscard]] static ResolvedProfile
    resolve(const core::backend::ComputeContext& ctx,
            const std::string&                   modelIdHint);

    /// Apply every flag `picks` opts into (explicit env wins).
    void apply(const ProbePicks& picks);

private:
    void applyKernelSetters(const ProbePicks& picks);
    void applyBackendEnvFlags(const ProbePicks& picks);
    void applyModelConfig(const ProbePicks& picks);

    /// setenv(`env`, `value`) unless `env` is already set; logs `label` -> `shown`.
    static void exportEnvFlag(const char*        env,
                              const std::string& value,
                              const char*        label,
                              const std::string& shown);
    [[nodiscard]] static bool envSet(const char* env) noexcept;

    compute::ComputeOps&    _ops;
    compute::ComputeMatmul& _gmm;
    model::LlmConfig&       _config;
};

} // namespace mimirmind::runtime
