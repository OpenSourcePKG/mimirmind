// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/ProfileOverlayApplier.hpp"

#include "compute/ComputeMatmul.hpp"
#include "compute/ComputeOps.hpp"
#include "core/backend/ComputeContext.hpp"
#include "core/backend/HwFingerprint.hpp"
#include "core/log/Log.hpp"
#include "model/LlmConfig.hpp"

#include <cstdint>
#include <cstdlib>

namespace mimirmind::runtime {

ProfileOverlayApplier::ProfileOverlayApplier(compute::ComputeOps&    ops,
                                             compute::ComputeMatmul& gmm,
                                             model::LlmConfig&       config) noexcept
    : _ops(ops), _gmm(gmm), _config(config) {}

ResolvedProfile ProfileOverlayApplier::resolve(
    const core::backend::ComputeContext& ctx, const std::string& modelIdHint) {
    ResolvedProfile out;
    out.fingerprint = core::backend::computeHwFingerprint(
        core::backend::gatherHostInfo(),
        core::backend::identityFromBackend(ctx.backend()));
    const char* dirEnv = std::getenv("MIMIRMIND_PROBE_DIR");
    out.dir = dirEnv ? dirEnv : "/usr/local/share/mimirmind/configs";
    out.picks = loadProbePicks(out.dir, out.fingerprint);
    // Per-(HW x model) overlay: the same flag can be validated-safe on one
    // checkpoint and unsafe on another on the SAME GPU (e.g. F32_TC_PREFILL, or
    // GROUPED_MOE's 512-expert TC path). Model wins over the HW profile.
    if (out.picks && !modelIdHint.empty()) {
        if (auto over = loadModelOverlay(out.dir, out.fingerprint, modelIdHint)) {
            mergeOverlay(*out.picks, *over);
            out.overlayMerged = true;
        }
    }
    return out;
}

void ProfileOverlayApplier::apply(const ProbePicks& picks) {
    applyKernelSetters(picks);
    applyBackendEnvFlags(picks);
    applyModelConfig(picks);
}

bool ProfileOverlayApplier::envSet(const char* env) noexcept {
    return std::getenv(env) != nullptr;
}

void ProfileOverlayApplier::exportEnvFlag(const char*        env,
                                          const std::string& value,
                                          const char*        label,
                                          const std::string& shown) {
    if (envSet(env)) {
        return;   // explicit env wins
    }
    ::setenv(env, value.c_str(), 1);
    MM_LOG_INFO("probe", "  profile applied: {} -> {}", label, shown);
}

void ProfileOverlayApplier::applyKernelSetters(const ProbePicks& picks) {
    if (picks.applyPrefillCudnn && !_ops.prefillCudnnEnvOverridden()) {
        _ops.setPrefillCudnn(*picks.applyPrefillCudnn);
        MM_LOG_INFO("probe", "  profile applied: cuDNN SDPA prefill -> {}",
                    *picks.applyPrefillCudnn ? "on" : "off");
    }
    if (picks.applyF32TcPrefill && !_gmm.f32TcPrefillEnvOverridden()) {
        _gmm.setF32TcPrefill(*picks.applyF32TcPrefill);
        MM_LOG_INFO("probe", "  profile applied: F32-TC prefill -> {}",
                    *picks.applyF32TcPrefill ? "on" : "off");
    }
    if (picks.applyCublasFp8Prefill && !_gmm.cublasFp8PrefillEnvOverridden()) {
        _gmm.setCublasFp8Prefill(*picks.applyCublasFp8Prefill);
        MM_LOG_INFO("probe", "  profile applied: cuBLAS-FP8 prefill -> {}",
                    *picks.applyCublasFp8Prefill ? "on" : "off");
    }
    if (picks.applyMmq && !_gmm.mmqEnvOverridden()) {
        _gmm.setMmq(*picks.applyMmq);
        MM_LOG_INFO("probe", "  profile applied: MMQ (Q8_0 int8 GEMM) -> {}",
                    *picks.applyMmq ? "on" : "off");
    }
    if (picks.applyMmqTc && !_gmm.mmqEnvOverridden()) {
        _gmm.setMmqTc(*picks.applyMmqTc);
        MM_LOG_INFO("probe", "  profile applied: MMQ tensor-core -> {}",
                    *picks.applyMmqTc ? "on" : "off");
    }
}

void ProfileOverlayApplier::applyBackendEnvFlags(const ProbePicks& picks) {
    // These flags are getenv-owned by the arch backend / loader, which read them
    // in their ctor — so they are exported here, BEFORE the backend is picked.
    const auto onOff = [](bool b) { return std::string{b ? "on" : "off"}; };
    const auto bit   = [](bool b) { return std::string{b ? "1" : "0"}; };

    // 5.21.7/8 serving-prefill flags.
    if (picks.applyAttnCudnnPaged) {
        exportEnvFlag("MIMIRMIND_ATTN_CUDNN_PAGED", bit(*picks.applyAttnCudnnPaged),
                      "cuDNN paged prefill", onOff(*picks.applyAttnCudnnPaged));
    }
    if (picks.applyMoeSiluFuse) {
        exportEnvFlag("MIMIRMIND_MOE_SILU_FUSE", bit(*picks.applyMoeSiluFuse),
                      "MoE silu+quant fuse", onOff(*picks.applyMoeSiluFuse));
    }
    // 5.18.14: mode-valued (0=off, 1=m2reg@tileM2, 4=m4reg@tileM4) — the
    // profile integer goes through verbatim.
    if (picks.applyMoeDecodeReg) {
        const std::string mode = std::to_string(*picks.applyMoeDecodeReg);
        exportEnvFlag("MIMIRMIND_MOE_DECODE_REG", mode,
                      "MoE decode register-staged", "mode " + mode);
    }
    // Grouped-MoE path selector (model overlay only; 0=blocked-only,
    // 3=FP4-TC-only).
    if (picks.applyGroupedMoe) {
        const std::string mode = std::to_string(*picks.applyGroupedMoe);
        exportEnvFlag("MIMIRMIND_GROUPED_MOE", mode,
                      "grouped-MoE path", "mode " + mode);
    }
    // Prefill chunk size (model overlay). Larger chunks cut the per-chunk
    // paged-KV re-gather of the cuDNN prefill path (O(n^2) across chunks) while
    // staying memory-bounded; ServingSession reads it via getenv.
    if (picks.applyPrefillChunk) {
        const std::string c = std::to_string(*picks.applyPrefillChunk);
        exportEnvFlag("MIMIRMIND_PREFILL_CHUNK", c, "prefill chunk", c + " tokens");
    }
    // 5.18.10.3 batched GDN input-projection fuse: 4 narrow projections -> 2
    // wide GEMMs, cutting the weight-read-bound gdn.proj term.
    if (picks.applyGdnProjFuseBatch) {
        exportEnvFlag("MIMIRMIND_GDN_PROJ_FUSE_BATCH",
                      bit(*picks.applyGdnProjFuseBatch),
                      "GDN proj fuse (batch)", onOff(*picks.applyGdnProjFuseBatch));
    }
    // 5.18.30: decode MoE expert GEMM through CUTLASS NVFP4-TC (vs hand
    // m2reg/m4reg). +42% decode conc1/4/8, coherence goldset ship-safe.
    if (picks.applyGroupedMoeDecodeTc) {
        exportEnvFlag("MIMIRMIND_GROUPED_MOE_DECODE_TC",
                      bit(*picks.applyGroupedMoeDecodeTc),
                      "grouped-MoE decode TC", onOff(*picks.applyGroupedMoeDecodeTc));
    }
    // 5.18.30: shared-expert CUTLASS-TC threshold (decode-M when lowered).
    if (picks.applyShexpTcMinM) {
        const std::string m = std::to_string(*picks.applyShexpTcMinM);
        exportEnvFlag("MIMIRMIND_SHEXP_TC_MINM", m, "shexp TC min-M", m);
    }
}

void ProfileOverlayApplier::applyModelConfig(const ProbePicks& picks) {
    // Answer-floor temperature-lift cap (model overlay). Explicit
    // MIMIRMIND_ANSWER_FLOOR_TEMP_CAP still wins (read per-request in the
    // ChatCompletionHandler answer floor).
    if (picks.applyAnswerFloorTempCap &&
        !envSet("MIMIRMIND_ANSWER_FLOOR_TEMP_CAP")) {
        _config.answerFloorTempCap = *picks.applyAnswerFloorTempCap;
        MM_LOG_INFO("probe", "  profile applied: answer-floor temp cap -> {}",
                    _config.answerFloorTempCap);
    }
    // Model THINKING sampling: the vendor's reasoning preset, overriding the
    // generic generation_config temperature in the thinking floor (qwen3.6 ships
    // do_sample temp=1.0, too hot for a long reasoning chain on a 3B-active model
    // -> derails; card recommends 0.6). Explicit MIMIRMIND_THINKING_* still wins.
    if (picks.applyThinkingTemp && !envSet("MIMIRMIND_THINKING_TEMP")) {
        _config.thinkingTemp = *picks.applyThinkingTemp;
    }
    if (picks.applyThinkingTopP && !envSet("MIMIRMIND_THINKING_TOP_P")) {
        _config.thinkingTopP = *picks.applyThinkingTopP;
    }
    if (picks.applyThinkingTopK && !envSet("MIMIRMIND_THINKING_TOP_K")) {
        _config.thinkingTopK = static_cast<std::uint32_t>(*picks.applyThinkingTopK);
    }
    if (picks.applyThinkingTemp || picks.applyThinkingTopP ||
        picks.applyThinkingTopK) {
        MM_LOG_INFO("probe",
                    "  profile applied: thinking sampling -> temp={} "
                    "top_p={} top_k={}",
                    _config.thinkingTemp, _config.thinkingTopP,
                    _config.thinkingTopK);
    }
    // Honesty floor: default OFF in LlmConfig; an overlay can re-enable it per
    // checkpoint. Explicit MIMIRMIND_HONESTY_FLOOR still wins per-request.
    if (picks.applyHonestyFloor && !envSet("MIMIRMIND_HONESTY_FLOOR")) {
        _config.honestyFloor = *picks.applyHonestyFloor;
        MM_LOG_INFO("probe", "  profile applied: honesty floor -> {}",
                    _config.honestyFloor ? "on" : "off");
    }
    // Jinja chat-template renderer opt-in (8.24.6). Default OFF; a profile flips
    // it on once the family is prompt-token bit-parat (8.24.5). Explicit
    // MIMIRMIND_ENABLE_JINJA wins per-process (handled in ChatCompletionHandler).
    if (picks.applyEnableJinja && !envSet("MIMIRMIND_ENABLE_JINJA")) {
        _config.enableJinja = *picks.applyEnableJinja;
        MM_LOG_INFO("probe", "  profile applied: jinja renderer -> {}",
                    _config.enableJinja ? "on" : "off");
    }
}

} // namespace mimirmind::runtime
