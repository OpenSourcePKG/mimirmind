// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/modelopt/HfQuantConfig.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mimirmind::core::safetensors {
class SafetensorsModel;
}

namespace mimirmind::core::modelopt {

/// How one HF source tensor is turned into (part of) a GGUF BF16 tensor.
enum class SourceKind : std::uint8_t {
    Nvfp4,          ///< dequant packed U8 + F8 block scale + F32 global -> BF16
    Fp8,            ///< dequant F8_E4M3 + F32 per-tensor scale -> BF16
    Bf16Passthrough ///< already BF16 (norms, router, embed, conv1d, ssm gates) -> copy
};

/// Element-wise transform applied to a step's finished output buffer, after
/// all sources have been dequantised/widened.
enum class PostTransform : std::uint8_t {
    None,    ///< emit the materialised values as-is
    NegExp,  ///< y = -exp(y) over the F32 output; turns HF `A_log` into the
             ///< GGUF `ssm_a` (= -exp(A_log)) the DeltaNet decay gate expects
    AddOne   ///< y = y + 1 over the F32 output; the transformer RMSNorm
             ///< weights use the centred (1 + w) convention that llama.cpp
             ///< bakes into the GGUF norm tensors, so the runtime multiplies
             ///< by the stored weight directly. The GatedDeltaNet ssm_norm is
             ///< excluded — it uses the plain (uncentred) convention.
};

/// One HF tensor feeding a GGUF tensor. For a stacked expert tensor there are
/// N of these, each writing at its own `dstElemOffset`.
struct MaterializationSource {
    std::string   hfWeightName;  ///< full HF name of the `.weight`
    SourceKind    kind;
    std::uint64_t rows;          ///< out-features (or product for passthrough vectors)
    std::uint64_t in;            ///< in-features (1 for a 1-D passthrough)
    std::uint64_t dstElemOffset; ///< element offset into the GGUF tensor (0 unless stacking)
};

/// One GGUF tensor to build from one or more HF sources.
struct MaterializationStep {
    std::string                ggufName;    ///< e.g. "blk.3.attn_q.weight", "output.weight"
    std::vector<std::uint64_t> ggufDims;    ///< GGUF ne-order (reverse+squeeze of HF shape)
    std::uint64_t              totalElems;  ///< output elements in the tensor
    /// Output dtype: passthrough (unquantised) tensors materialise to F32 so
    /// the runtime's norm / SSM / bias kernels can `static_cast<const float*>`
    /// them; dequantised NVFP4/FP8 matmul weights stay BF16. Set per step
    /// because a plan mixes both.
    bool                       outF32{false};
    /// Element-wise fix-up applied once after the sources are materialised.
    /// Only NegExp is used (ssm_a = -exp(A_log)); it requires an F32 output.
    PostTransform              postTransform{PostTransform::None};
    std::vector<MaterializationSource> sources;
};

/// Architecture parameters needed to walk the layers (from config.json).
struct Qwen35MoeArch {
    int numLayers          = 40;
    int numExperts         = 256;
    int fullAttnInterval   = 4;
};

/**
 * Build the materialization plan: for every GGUF tensor the `qwen35moe`
 * backend needs, which HF tensor(s) it comes from, the dequant kind, the
 * expert-stacking offsets, and the GGUF dims. Pure — it only reads tensor
 * shapes/dtypes from `model` and schemes from `cfg`; it launches no kernels
 * and touches no device. The device executor walks the returned steps.
 *
 * Throws std::runtime_error if a required HF tensor is missing from the
 * checkpoint (a truncated/mismatched model), naming it.
 */
[[nodiscard]] std::vector<MaterializationStep>
planQwen35MoeMaterialization(const safetensors::SafetensorsModel& model,
                             const HfQuantConfig&                 cfg,
                             const Qwen35MoeArch&                 arch);

} // namespace mimirmind::core::modelopt