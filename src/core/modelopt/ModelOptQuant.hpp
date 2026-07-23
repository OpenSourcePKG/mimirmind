// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/safetensors/SafetensorsDtype.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

namespace mimirmind::core::modelopt {

/// Container-level element dtype, owned by the safetensors layer. Aliased
/// here so scheme descriptors can name it without a backwards dependency
/// from the container reader onto this interpretation layer.
using safetensors::SafetensorsDtype;

/**
 * Quantisation schemes emitted by NVIDIA ModelOpt into HuggingFace
 * `.safetensors` checkpoints (the `nvidia/...-NVFP4` and `unsloth/...-NVFP4`
 * family). These are the CUDA / Bragi weight formats — deliberately kept
 * OUT of `core::gguf::GgmlType`, which models ggml's GGUF types by their
 * on-disk numeric ids. A ModelOpt scheme is a safetensors concept: it is
 * NOT a single interleaved block buffer (as GGUF quants are) but a group
 * of separate tensors — a packed weight plus one or two scale sidecars.
 *
 * The per-module scheme is read at load time from `hf_quant_config.json`'s
 * `quantized_layers` map (see `schemeFromQuantAlgo`). A single checkpoint
 * mixes schemes: for `nvidia/Qwen3.6-35B-A3B-NVFP4` the MoE / shared-expert
 * FFN + lm_head are NVFP4 while every attention projection is FP8, so the
 * loader must resolve the scheme per module, not once per model.
 *
 * Verified against the on-disk safetensors index of that checkpoint
 * (2026-07-23). See the Synaipse note "NVFP4-Modellformat:
 * nvidia/Qwen3.6-35B-A3B-NVFP4 — Loader-Input".
 */
enum class ModelOptQuantScheme : std::uint8_t {
    /**
     * NVFP4 weights, 16-bit (bf16) activations — `W4A16_NVFP4`.
     *
     * Three tensors per weight:
     *   - `<name>.weight`         U8      [out, in/2]     2× E2M1 (fp4) per byte
     *   - `<name>.weight_scale`   F8_E4M3 [out, in/16]    per-16-element block scale
     *   - `<name>.weight_scale_2` F32     scalar          per-tensor global scale
     *
     * `_BLK16` is explicit in the name because the industry NVFP4 default
     * is 16-element blocks (not the 32-element OCP MX-FP4 variant). A16
     * means activations stay bf16 — there is no activation quantisation on
     * this path, which is exactly what makes it long-prefill quality-safe
     * where int8 W8A8 (MMQ) was not.
     */
    NVFP4_E2M1_BLK16 = 0,

    /**
     * FP8 E4M3 weights with a per-tensor scale — `FP8`.
     *
     * Tensors per weight:
     *   - `<name>.weight`       F8_E4M3 [out, in]   unpacked, one byte per element
     *   - `<name>.weight_scale` F32     scalar       per-tensor weight scale
     *   - `<name>.input_scale`  F32     scalar       per-tensor activation scale
     *
     * Unlike NVFP4 this IS activation-quantised (W8A8), but FP8-E4M3 has a
     * far wider dynamic range than int8, so outlier activations survive.
     */
    FP8_E4M3 = 1,
};

/**
 * Static description of a ModelOpt scheme's tensor layout. Everything the
 * loader needs to validate a weight's sidecar tensors and size its buffers
 * without hard-coding per-scheme logic at every call site.
 */
struct ModelOptSchemeInfo {
    std::string_view name;              ///< matches hf_quant_config `quant_algo`

    SafetensorsDtype weightDtype;       ///< dtype of `<name>.weight`
    /// Elements packed into one stored weight byte along the in-features
    /// axis: 2 for NVFP4 (two fp4 per byte), 1 for FP8 (unpacked).
    std::uint8_t     weightPackFactor;

    /// Micro-block size for the block scale, in elements along in-features.
    /// 0 means the scheme has no per-block scale (FP8 is per-tensor only).
    std::uint16_t    blockScaleGroupSize;
    SafetensorsDtype blockScaleDtype;   ///< dtype of `<name>.weight_scale` when blocked

    bool             hasBlockScale;     ///< `<name>.weight_scale` is per-block [out, in/group]
    bool             hasTensorWeightScale; ///< `<name>.weight_scale` is a per-tensor scalar
    bool             hasGlobalScale;    ///< `<name>.weight_scale_2` scalar present (NVFP4)
    bool             hasInputScale;     ///< `<name>.input_scale` scalar present (FP8)
};

/// Descriptor for a scheme. Total, stable, `noexcept`.
[[nodiscard]] const ModelOptSchemeInfo& schemeInfo(ModelOptQuantScheme s) noexcept;

/**
 * Map an `hf_quant_config.json` `quant_algo` string to a scheme.
 * Recognises `"W4A16_NVFP4"` and `"FP8"`. Returns `std::nullopt` for any
 * other value (e.g. `"MIXED_PRECISION"`, which is the top-level marker and
 * never appears on a leaf module, or an unsupported algo the loader must
 * reject explicitly).
 */
[[nodiscard]] std::optional<ModelOptQuantScheme>
schemeFromQuantAlgo(std::string_view quantAlgo) noexcept;

/// Number of stored weight bytes for one output row of length `inFeatures`
/// under `s`. `inFeatures` must be a multiple of `weightPackFactor` (and,
/// when blocked, of `blockScaleGroupSize`). Returns 0 on a bad divisor.
[[nodiscard]] std::size_t packedRowBytes(ModelOptQuantScheme s,
                                         std::size_t         inFeatures) noexcept;

/// Number of block-scale columns for one output row of length `inFeatures`
/// (i.e. `inFeatures / blockScaleGroupSize`). Returns 0 when the scheme has
/// no block scale or `inFeatures` is not a multiple of the group size.
[[nodiscard]] std::size_t blockScaleCols(ModelOptQuantScheme s,
                                         std::size_t         inFeatures) noexcept;

} // namespace mimirmind::core::modelopt