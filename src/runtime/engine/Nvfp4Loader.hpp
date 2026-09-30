// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

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
        InferenceEngine& e, core::cuda::CudaComputeContext& cudaCtx);

    /// MoE routed-expert bank repack (blocked-NVFP4 [+ additive FP4-TC sidecars]
    /// / FP4-TC-only / K-quant), with streaming source release to bound the peak.
    static void buildMoeExpertBanks(
        InferenceEngine&                                        e,
        const std::vector<core::modelopt::MaterializationStep>& steps,
        core::cuda::CudaComputeContext&                         cudaCtx,
        compute::cuda::CudaMaterializerOps&                     devOps);
};

} // namespace mimirmind::runtime::engine
