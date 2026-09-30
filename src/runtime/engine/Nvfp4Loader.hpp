// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace mimirmind::runtime {
class InferenceEngine;
}
namespace mimirmind::core::safetensors {
class SafetensorsModel;
}
namespace mimirmind::core::cuda {
class CudaComputeContext;
}
namespace mimirmind::compute::cuda {
class CudaMaterializerOps;
}
namespace mimirmind::core::modelopt {
struct MaterializationStep;
class CompressedTensorsConfig;
}

namespace mimirmind::runtime::engine {

/**
 * The NVFP4 checkpoint load pipeline, extracted from InferenceEngine so the
 * engine's translation unit stays focused on generation. Runs on the engine
 * as a friend collaborator: reads config.json + tokenizer, uploads the
 * NVFP4/FP8 weights, dequantises them to BF16 on device, applies the
 * GatedDeltaNet value-head regroup, and re-quantises the MoE experts to
 * K-quants (and, gated off, the attention projections to Q8_0). It leaves the
 * engine with `_config`, `_tokenizer`, `_materializedBf16` and `_weights`
 * populated; the caller runs `finalizeLoad()` afterwards.
 *
 * CUDA-only — the body is compiled only under MIMIRMIND_HAVE_CUDA.
 */
class Nvfp4Loader {
public:
    /// Populate `engine` from the NVFP4 checkpoint at `checkpointDir`, taking
    /// the tokenizer from `tokenizerGguf`. Throws on a non-CUDA backend or a
    /// malformed checkpoint. Does NOT call finalizeLoad().
    ///
    /// `attachedSm` is the M-Munin.CUDA attach hook: when non-null, the
    /// safetensors shards are read from that already-open SafetensorsModel
    /// (reconstructed from shm memfd chunks) instead of from disk. The small
    /// text sidecars (config.json / hf_quant_config.json / tokenizer.json)
    /// are still read locally from `checkpointDir`. `attachedSm` must outlive
    /// the call.
    static void load(InferenceEngine&                     engine,
                     std::string_view                     checkpointDir,
                     std::string_view                     tokenizerGguf,
                     core::safetensors::SafetensorsModel* attachedSm = nullptr);

private:
    // Post-plan stages of load(), extracted verbatim so the pipeline reads as
    // a sequence of named passes. All are friends of InferenceEngine (as the
    // enclosing class is) and are compiled only under MIMIRMIND_HAVE_CUDA;
    // in a non-CUDA build load()'s stub never ODR-uses them.

    /// The compressed-tensors Gemma-4 (dense text tower) load path: its own
    /// config schema + NVFP4 name-triple, none of the qwen35moe post-passes.
    /// Populates the engine and returns; the caller then `return`s from load().
    static void loadGemma4(InferenceEngine&                     e,
                           const std::string&                   dir,
                           const std::string&                   configText,
                           std::string_view                     checkpointDir,
                           std::string_view                     tokenizerGguf,
                           core::safetensors::SafetensorsModel* attachedSm);

    /// GatedDeltaNet value-head regroup (HF -> GGUF layout), a pure element
    /// permutation of the vDim value channels + the per-head decay/beta tensors.
    static void regroupGatedDeltaNetValueHeads(
        InferenceEngine& e, core::cuda::CudaComputeContext& cudaCtx,
        compute::cuda::CudaMaterializerOps& devOps);

    /// MoE routed-expert bank repack (blocked-NVFP4 [+ additive FP4-TC sidecars]
    /// / FP4-TC-only / K-quant), with streaming source release to bound the peak.
    static void buildMoeExpertBanks(
        InferenceEngine&                                        e,
        const std::vector<core::modelopt::MaterializationStep>& steps,
        core::cuda::CudaComputeContext&                         cudaCtx,
        compute::cuda::CudaMaterializerOps&                     devOps);

    /// 5b'. Optional (MIMIRMIND_MTP_EHSWAP) MTP eh_proj concat-half swap — swaps
    /// the hnorm/enorm input halves of blk.<N>.nextn.eh_proj.weight in place.
    static void applyMtpEhProjSwap(InferenceEngine&                e,
                                   core::cuda::CudaComputeContext& cudaCtx);

    /// 5c. Optional (MIMIRMIND_NVFP4_Q8_PROJ) re-quant of the dense attention
    /// projections BF16 -> Q8_0. Default off (linear Q8_0 crushes FP8/NVFP4-origin
    /// log-distributed weights); kept for A/B.
    static void requantDenseAttnProjQ8_0(
        InferenceEngine& e, core::cuda::CudaComputeContext& cudaCtx,
        compute::cuda::CudaMaterializerOps& devOps);

    /// 5e. Optional (MIMIRMIND_NVFP4_ATTN_FP8) re-quant of the dense attention
    /// projections BF16 -> blocked-FP8 (E4M3), preserving the log format per
    /// 32-block. Default off; `gdn` restores the memory-saving path.
    static void requantDenseAttnProjFp8(
        InferenceEngine& e, core::cuda::CudaComputeContext& cudaCtx,
        compute::cuda::CudaMaterializerOps& devOps);

    /// 5.27 I-2 lever (a) — repack every single-source, NVFP4-sourced projection
    /// matching `keep` (empty = all) from its just-materialised BF16 image into
    /// the blocked-NVFP4 format (¼ the bytes), freeing the BF16 in place (RAII).
    /// Lossless. Was a load()-local [&] lambda (8.30.11.4); leaves the NVFP4
    /// source resident.
    static void repackDenseNvfp4(
        InferenceEngine&                                        e,
        const std::vector<core::modelopt::MaterializationStep>& steps,
        core::cuda::CudaComputeContext&                         cudaCtx,
        compute::cuda::CudaMaterializerOps&                     devOps,
        const std::function<bool(const std::string&)>&          keep,
        const char*                                             label);

    /// 5e-dense. Opt-in (MIMIRMIND_QWEN_DENSE_NVFP4_DECODE) dense-27B bandwidth
    /// lever: repack ALL single-source NVFP4 dense projections to blocked-NVFP4.
    static void applyDenseNvfp4DecodeLever(
        InferenceEngine&                                        e,
        const std::vector<core::modelopt::MaterializationStep>& steps,
        core::cuda::CudaComputeContext&                         cudaCtx,
        compute::cuda::CudaMaterializerOps&                     devOps,
        const core::modelopt::CompressedTensorsConfig&          ctCfg);

    /// 5f. Keep the MoE shared-expert (ffn_*_shexp) NVFP4 projections native
    /// blocked-NVFP4 (+ additive FP4-TC sidecars for the prefill grouped GEMM).
    static void repackSharedExpertsNvfp4(
        InferenceEngine&                                        e,
        const std::vector<core::modelopt::MaterializationStep>& steps,
        core::cuda::CudaComputeContext&                         cudaCtx,
        compute::cuda::CudaMaterializerOps&                     devOps);

    /// 5f-lmhead. Opt-in (MIMIRMIND_LMHEAD_NVFP4) dual-copy: keep BF16 lm_head
    /// and add a blocked-NVFP4 ".nv" sibling for single-user (nSeq<=maxT) decode.
    static void addLmHeadNvfp4Sibling(
        InferenceEngine&                                        e,
        const std::vector<core::modelopt::MaterializationStep>& steps,
        core::cuda::CudaComputeContext&                         cudaCtx,
        compute::cuda::CudaMaterializerOps&                     devOps);

    /// 5g. Opt-in (MIMIRMIND_DENSE_FP8_LOWM) M-dependent dual-copy: keep BF16
    /// dense projections and add a blocked-FP8 E4M3 ".fp8" variant for low-batch.
    static void addDenseFp8LowMVariants(
        InferenceEngine& e, core::cuda::CudaComputeContext& cudaCtx,
        compute::cuda::CudaMaterializerOps& devOps);
};

} // namespace mimirmind::runtime::engine
