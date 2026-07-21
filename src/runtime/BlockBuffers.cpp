// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/BlockBuffers.hpp"

#include "compute/ComputeOps.hpp"

#include <algorithm>
#include <cstdint>

namespace mimirmind::runtime {

BlockBuffers allocBlockBuffers(compute::ComputeOps&    ops,
                               const model::LlmConfig& config,
                               std::size_t             maxT,
                               std::size_t             maxSeq,
                               std::size_t             qDimMax,
                               std::size_t             kvDimMax,
                               bool                    withFusedQkv,
                               bool                    withKvFp32Scratch,
                               bool                    withQGate) {
    BlockBuffers b{};
    b.maxT    = maxT;
    b.maxSeq  = maxSeq;
    b.d_model = config.embeddingLength;
    b.q_dim   = qDimMax;
    b.kv_dim  = kvDimMax;
    b.ff_dim  = config.feedForwardLength;

    // FFN scratch width. For dense models this is `feedForwardLength`. For
    // MoE models the per-routed-expert (`expertFeedForwardLength`) and the
    // shared-expert (`expertSharedFeedForwardLength`) intermediate widths
    // can exceed the dense `ff_dim` (Qwen3-Next `qwen35moe` is pure MoE, so
    // its dense `ff_dim` need not bound either). Size the gate/up + MoE
    // compact buffers on the max so no per-expert matmul overruns them.
    const std::size_t ffScratch = std::max({
        static_cast<std::size_t>(config.feedForwardLength),
        static_cast<std::size_t>(config.expertFeedForwardLength),
        static_cast<std::size_t>(config.expertSharedFeedForwardLength)});

    const std::size_t qBytes            = maxT * b.q_dim   * sizeof(float);
    const std::size_t normBytes         = maxT * b.d_model * sizeof(float);
    const std::size_t attnOutBytes      = maxT * b.q_dim   * sizeof(float);
    const std::size_t projOutBytes      = maxT * b.d_model * sizeof(float);
    const std::size_t gateOutBytes      = maxT * ffScratch * sizeof(float);
    const std::size_t upOutBytes        = maxT * ffScratch * sizeof(float);
    const std::size_t scoreScratchBytes = maxSeq           * sizeof(float);
    const std::size_t matmulScratchBytes =
        std::max({b.d_model, b.q_dim, ffScratch}) * sizeof(float);

    b.qBuf          = ops.allocate(qBytes);
    b.normBuf       = ops.allocate(normBytes);
    b.attnOut       = ops.allocate(attnOutBytes);
    b.projOut       = ops.allocate(projOutBytes);
    b.gateOut       = ops.allocate(gateOutBytes);
    b.upOut         = ops.allocate(upOutBytes);
    b.matmulScratch = ops.allocate(matmulScratchBytes);
    b.scoreScratch  = ops.allocate(scoreScratchBytes);

    if (withFusedQkv) {
        // Q + K + V fused output. Widest layer: Q width + two KV widths.
        const std::size_t fusedBytes =
            maxT * (b.q_dim + 2 * b.kv_dim) * sizeof(float);
        b.qkvFusedScratch = ops.allocate(fusedBytes);
    }

    if (withQGate) {
        // Qwen3-Next full-attention fused [Q|gate] projection + the split
        // gate. qGateFused holds the 2*q_dim matmul output; gateScratch
        // holds the de-interleaved gate that survives until the
        // post-attention sigmoid multiply.
        b.qGateFused  = ops.allocate(maxT * 2 * b.q_dim * sizeof(float));
        b.gateScratch = ops.allocate(maxT *     b.q_dim * sizeof(float));
    }

    if (withKvFp32Scratch) {
        // Persistent fp32 K/V staging for the Q8_0 KV path. Sized for
        // the widest layer's kv_dim; every backend's Q8_0 branch writes
        // rmsnorm_qkv + RoPE output here and then invokes
        // kv_quant_commit_q8_0 to finalise into the cache slot.
        const std::size_t kvFp32Bytes =
            maxT * b.kv_dim * sizeof(float);
        b.kvKFp32Scratch = ops.allocate(kvFp32Bytes);
        b.kvVFp32Scratch = ops.allocate(kvFp32Bytes);
    }

    if (config.expertCount > 0) {
        const std::size_t moeBytes = maxT * b.d_model * sizeof(float);
        b.moeAccumBuf  = ops.allocate(moeBytes);
        b.expertOutBuf = ops.allocate(moeBytes);

        // Expert-grouping scratch (M5i.F). Per-block worst-case row
        // count = maxT * expertUsedCount (K_top). Feed-forward per
        // expert is inferred from the block layout at run time —
        // BlockBuffers is sized on ff_dim (dense path A) which is a
        // safe upper bound: for Gemma 4 26B-A4B ff_dim == ffPerExpert.
        const std::size_t nRowsMax  = maxT * config.expertUsedCount;
        const std::size_t xBytes    = nRowsMax * b.d_model   * sizeof(float);
        const std::size_t gateBytes = nRowsMax * ffScratch   * sizeof(float);
        b.moeXCompact    = ops.allocate(xBytes);
        b.moeGateCompact = ops.allocate(gateBytes);
        b.moeUpCompact   = ops.allocate(gateBytes);
        b.moeDownCompact = ops.allocate(xBytes);

        // M-MoE.Fused-Decode — routing scratches. `blockCount *
        // expertUsedCount` slots so each layer owns its own K-tuple
        // across the recorded command stream.
        const std::size_t routeSlots =
            config.blockCount * config.expertUsedCount;
        const std::size_t idxBytes = routeSlots * sizeof(std::int32_t);
        const std::size_t kwBytes  = routeSlots * sizeof(float);
        b.moeExpIdxScratch = ops.allocate(idxBytes);
        b.moeKwScratch     = ops.allocate(kwBytes);
    }
    return b;
}

} // namespace mimirmind::runtime