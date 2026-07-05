#pragma once

#include "model/GgufTypes.hpp"
#include "runtime/UsmHandle.hpp"
#include "runtime/arch/GemmaBaseBackend.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace mimirmind::runtime::arch {

/**
 * Gemma 4 E-Series (E4B / E2B) variant — Matryoshka Transformer with
 * Per-Layer Embeddings (PLE).
 *
 * Structurally identical to `Gemma4DenseBackend` for attention + the main
 * SwiGLU-GELU FFN. What's new is the PLE-injection branch at the tail of
 * every block:
 *
 *   h_ple = inp_gate  @ h                          // [d_model → per_layer_dim=256]
 *   h_ple = GELU(h_ple)
 *   h_ple *= per_layer_embd[block][token]          // element-wise
 *   h_ple = proj @ h_ple                           // [per_layer_dim → d_model]
 *   h_ple = rmsnorm(h_ple, post_norm)              // Gemma-plain w * rmsnorm(x)
 *   h += h_ple
 *   h *= layer_output_scale
 *
 * The per_layer_token_embd.weight table is a big [num_layers*per_layer_dim,
 * vocab_size] block-quantized weight, one column per token. Every forward
 * pass we dequant-into a USM scratch of shape [num_layers, T, per_layer_dim]
 * so per-layer slices are contiguous — that's what `prepareForward` does.
 * Then per block, `runBlock` treats its layer's slice as a plain vector
 * and reuses `geluMulAsync(gate, slice, ...)` for the fused GELU + PLE-mul.
 *
 * References:
 *   - Google AI "Gemma 3n model overview" (PLE mechanics)
 *   - Sebastian Raschka's LLM Architecture Gallery — Per-Layer Embeddings
 *   - ggml-org/llama.cpp Issue #22243 (missing PLE in llama.cpp forward graph)
 */
class Gemma4E4BBackend final : public GemmaBaseBackend {
public:
    Gemma4E4BBackend(const model::LlmConfig&        config,
                     const model::WeightsMap&       weights,
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

    /// Extract the per-layer-embedding slice for every token about to run
    /// through the block chain. Dequantizes `per_layer_token_embd.weight`
    /// on the CPU into `_pleBuf` — the block layouts align neatly (one
    /// Q-K block per (token, layer) since per_layer_dim == 256 == block
    /// size). Called from InferenceEngine before prefill / decode / verify.
    void prepareForward(std::span<const std::int32_t> tokIds) override;

private:
    /// Grow the PLE-slice USM scratch to hold `T` tokens. Idempotent.
    void ensurePleCapacity(std::size_t T);

    // --- PLE table + geometry --------------------------------------------

    /// Points at the per_layer_token_embd.weight blob in USM. When null
    /// the model doesn't carry PLE (shouldn't happen for E-series but
    /// we degrade gracefully — runBlock skips the PLE branch).
    const void*      _pleTablePtr{nullptr};
    model::GgmlType  _pleTableType{model::GgmlType::F32};

    std::size_t      _perLayerDim{0};   // 256 for E4B — from inp_gate dims
    std::size_t      _pleBytesPerBlock{0};
    std::size_t      _vocabSize{0};

    // --- Per-forward-pass scratch ----------------------------------------

    /// Dequantized PLE slices in layout [num_layers, T, per_layer_dim].
    /// Row L is a contiguous [T * per_layer_dim] float span, exactly
    /// what `geluMulAsync` consumes as its `up` operand.
    UsmHandle        _pleBuf;
    std::size_t      _pleBufCapT{0};

    /// Post-inp_gate scratch [maxT, per_layer_dim] — geluMulAsync writes
    /// its output here (gate = GELU(gate) * PLE_slice). Grows with T.
    UsmHandle        _pleGateBuf;
    std::size_t      _pleGateBufCapT{0};

    /// Number of tokens `prepareForward` last wrote — how far runBlock
    /// can safely index into `_pleBuf` for the current forward pass.
    std::size_t      _pleActiveT{0};
};

} // namespace mimirmind::runtime::arch