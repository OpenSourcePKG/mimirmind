// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "compute/ComputeBuffer.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace mimirmind::compute {
class ComputeOps;
class ComputeMatmul;
} // namespace mimirmind::compute

namespace mimirmind::runtime::encoder::laya {

class LayaEncoderModel;

/**
 * Runs the ModernBERT backbone of the Laya model (LayaEncoderModel) forward.
 * Sibling of EncoderRunner (XLM-RoBERTa); shares the backend-neutral
 * compute::ComputeOps / compute::ComputeMatmul plumbing but implements the
 * ModernBERT geometry:
 *   - pre-LN (norm before attention and MLP) + a final norm, all LayerNorm
 *     WITHOUT bias (a zeroed beta buffer feeds the bias'd layerNormAsync);
 *   - RoPE on Q/K per layer (split/NEOX, ropeInPlaceAsync), with the global vs
 *     local theta chosen per layer; NO learned absolute/token-type embeddings;
 *   - fused QKV (attn.Wqkv, split [q|k|v] per row) and GeGLU MLP (mlp.Wi =
 *     [input|gate], out = gelu(input) * gate, built from the existing
 *     geluMulAsync after de-interleaving the halves);
 *   - layer 0's attn_norm is Identity (absent weight -> skipped).
 *
 * Attention: global layers use the full-window attentionEncoderAsync; sliding
 * layers use attentionEncoderWindowedAsync with radius localAttention/2 (=64),
 * matching ModernBERT's alternating attention — exact at any sequence length.
 */
class LayaEncoderRunner {
public:
    LayaEncoderRunner(const LayaEncoderModel& model, compute::ComputeOps& ops,
                      compute::ComputeMatmul& matmul);

    /// Pooled sentence embedding [hidden]: mean over tokens (laya's
    /// classifier_pooling="mean") or the CLS row, per config. Not L2-normalised
    /// (the Laya scorer/act_head consume the raw pooled vector).
    [[nodiscard]] std::vector<float> embed(std::span<const std::int32_t> inputIds);

    /// Full final hidden state [T*hidden] on host (row-major), for parity
    /// against the python oracle's last_hidden_state. `Tout` receives T.
    [[nodiscard]] std::vector<float>
    lastHiddenHost(std::span<const std::int32_t> inputIds, std::size_t& Tout);

    /// Final hidden state [T*hidden] left ON DEVICE (row-major, already flushed),
    /// for the decision head (LayaDecisionRunner) to consume without a round-trip.
    [[nodiscard]] compute::ComputeBuffer
    lastHiddenDevice(std::span<const std::int32_t> inputIds, std::size_t& Tout) {
        return encode(inputIds, Tout);
    }

private:
    /// Shared embeddings + N-layer ModernBERT body. Returns the final [T*H]
    /// hidden state on device; flushes before returning (its scratch is freed).
    [[nodiscard]] compute::ComputeBuffer
    encode(std::span<const std::int32_t> inputIds, std::size_t& Tout);

    const LayaEncoderModel& _m;
    compute::ComputeOps&    _ops;
    compute::ComputeMatmul& _mm;
};

} // namespace mimirmind::runtime::encoder::laya
