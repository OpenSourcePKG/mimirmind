// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/gguf/GgufTypes.hpp"
#include "core/l0/UsmHandle.hpp"
#include "runtime/arch/GemmaBaseBackend.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace mimirmind::runtime::arch {

using ::mimirmind::core::l0::UsmHandle;


/**
 * Gemma 4 E-Series (E4B / E2B) as shipped by the llama.cpp-compatible
 * GGUF conversion. The GGUF does NOT carry AltUp / Laurel weights (see
 * ggml-org/llama.cpp #22243) — the converter emits a simplified variant
 * that folds those pieces away and replaces them with a small pair of
 * model-level tensors (`per_layer_model_proj`, `per_layer_proj_norm`)
 * plus the standard per-block PLE gate/proj/post_norm trio.
 *
 * Per-forward, before the block loop starts:
 *
 *   proj = per_layer_model_proj(hidden_states)   // [T, d_model → num_layers*per_layer_dim]
 *   proj = per_layer_proj_norm(proj)             // RMSNorm on per_layer_dim
 *   embd = per_layer_token_embd(token_ids)       // [T, num_layers, per_layer_dim]
 *   per_layer_input = (proj + embd) * (1 / sqrt(2))
 *
 * Per block, inside runBlock (in addition to the standard Dense-Gemma
 * attention + SwiGLU-GELU FFN):
 *
 *   h_ple = inp_gate(h)                          // [d_model → per_layer_dim]
 *   h_ple = SiLU(h_ple) * per_layer_input[block] // fused via siluMulAsync
 *   h_ple = proj(h_ple)                          // [per_layer_dim → d_model]
 *   h_ple = rmsnorm(h_ple, post_norm)            // plain w * rmsnorm(x)
 *   h += h_ple
 *   h *= layer_output_scale
 *
 * References:
 *   - alandao.net "Gemma 4 E2B & PLE Research Notes" (combine formula)
 *   - Google AI E-Series architecture overview
 *   - ggml-org/llama.cpp Issue #22243 (simplified conversion)
 */
class Gemma4E4BBackend final : public GemmaBaseBackend {
public:
    Gemma4E4BBackend(const model::LlmConfig&        config,
                     const core::gguf::WeightsMap&       weights,
                     const model::FusedQkvWeights*  fusedQkv,
                     compute::GpuOps&               ops,
                     compute::GpuMatmul&            gmm,
                     runtime::OpProfiler&           opProfiler);

    void runBlock(std::size_t   blockIdx,
                  float*        x,
                  std::size_t   T,
                  KvCache&      cache,
                  BlockBuffers& s,
                  bool          traceBlock0) override;

    /// Precompute the per-layer-input tensor for every token about to
    /// run through the block chain:
    ///   1) dequant `per_layer_token_embd` into `_pleBuf`
    ///   2) matmul `per_layer_model_proj` on `hiddenStates` into
    ///      `_pleProjBuf`, then RMSNorm with `per_layer_proj_norm`
    ///   3) combine into `_pleBuf` as `(proj + embd) * 1/sqrt(2)`
    /// runBlock's PLE branch then treats `_pleBuf[layer]` as a plain
    /// contiguous [T × per_layer_dim] slice.
    void prepareForward(std::span<const std::int32_t> tokIds,
                        const float*                  hiddenStates,
                        std::size_t                   T) override;

private:
    /// Grow the PLE-slice USM scratches (`_pleBuf`, `_pleProjBuf`,
    /// `_pleGateBuf`) to hold `T` tokens. Idempotent.
    void ensurePleCapacity(std::size_t T);

    /// One-time quantization of the BF16 `per_layer_model_proj.weight`
    /// tensor into a Q8_0-formatted blob in USM. The GPU matmul kernels
    /// don't have a BF16 path, and CPU fallback is too slow (~100 ms per
    /// decode step at [10752, 2560]). Q8_0 is ~half the size (30 MiB
    /// vs 55 MiB) and dispatches through the existing Q8_0 vec/gemm
    /// kernels.
    void requantizeModelProjToQ8_0(const core::gguf::GgufTensor& src);

    // --- Static geometry -------------------------------------------------

    std::size_t      _perLayerDim{0};   // 256 for E4B — from inp_gate dims

    // --- PLE-embedding-table pointer + geometry --------------------------

    const void*      _pleTablePtr{nullptr};
    core::gguf::GgmlType  _pleTableType{core::gguf::GgmlType::F32};
    std::size_t      _pleBytesPerBlock{0};
    std::size_t      _vocabSize{0};

    // --- per_layer_model_proj tensor after Q8_0 requantization -----------

    /// Owned Q8_0-quantized copy of per_layer_model_proj.weight. Layout
    /// [N=num_layers*per_layer_dim, K=d_model], one row per output
    /// element, K/32 Q8_0 blocks per row. Null when the model doesn't
    /// carry the tensor (defensive; disables PLE).
    UsmHandle        _projQ8;
    std::size_t      _projQ8Bytes{0};

    /// M8.K.Q8_0-Reorder — separate USM buffer holding `_projQ8` in
    /// scales-then-quants layout (see kernels/matmul_q8_0_vec_reorder
    /// .cl and Q8_0::reorderRow). Only populated when
    /// GpuOps::q8_0ReorderMode() != Disable at load time; stays empty
    /// otherwise. Dispatch at injectPerLayerInputs picks the reorder
    /// path only for M=1 (decode) since matmul_q8_0_vec_reorder is a
    /// matvec kernel; M>1 (prefill) always uses the native `_projQ8`
    /// through GpuMatmul's GEMM dispatch. Dual-copy costs ~size of
    /// `_projQ8` extra USM (~23 MiB for E4B) but avoids un-reordering
    /// on every prefill call.
    UsmHandle        _projQ8Reorder;
    std::size_t      _projQ8ReorderBytes{0};

    /// F32 pointer to per_layer_proj_norm.weight in USM. Owned by the
    /// weights map — this is just a non-owning cache.
    const float*     _projNorm{nullptr};

    // --- Per-forward-pass scratch ----------------------------------------

    /// Dequantized PLE embedding slices, layout [num_layers, T, per_layer_dim].
    /// Row L is contiguous [T * per_layer_dim] float span — exactly what
    /// siluMulAsync consumes as its `up` operand.
    UsmHandle        _pleBuf;
    std::size_t      _pleBufCapT{0};

    /// Output of per_layer_model_proj matmul + rmsnorm, layout
    /// [T, num_layers, per_layer_dim]. Combined into `_pleBuf` after
    /// scaling by 1/sqrt(2).
    UsmHandle        _pleProjBuf;
    std::size_t      _pleProjBufCapT{0};

    /// Post-inp_gate scratch [maxT, per_layer_dim] — siluMulAsync writes
    /// its output here (gate = SiLU(gate) * per_layer_input[layer]).
    UsmHandle        _pleGateBuf;
    std::size_t      _pleGateBufCapT{0};

    /// Number of tokens `prepareForward` last wrote — how far runBlock
    /// can safely index into `_pleBuf` for the current forward pass.
    std::size_t      _pleActiveT{0};

    /// One-shot flag so the PLE first-slice sanity log fires only once.
    bool             _pleDumpDone{false};
};

} // namespace mimirmind::runtime::arch