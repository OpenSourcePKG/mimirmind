// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "runtime/arch/ArchBackend.hpp"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace mimirmind::compute {
class ComputeMatmul;
class ComputeOps;
} // namespace mimirmind::compute

namespace mimirmind::core::gguf {
class WeightsMap;
struct GgufTensor;
} // namespace mimirmind::core::gguf

namespace mimirmind::model {
class FusedQkvWeights;
struct LlmConfig;
} // namespace mimirmind::model

namespace mimirmind::runtime::arch {

/**
 * Qwen3-Next / Qwen3.5-MoE (`qwen35moe`) decoder block — the hybrid
 * linear-attention + full-attention MoE architecture (Bragi target, see
 * research/qwen3next-gated-deltanet-recon-2026-07-21).
 *
 * Layer topology is interleaved: every `full_attention_interval`-th layer
 * is a full (softmax) attention layer, the rest are GatedDeltaNet linear-
 * attention layers (`config.isRecurrentLayer(blockIdx)`). Every layer runs
 * the same MoE FFN (routed experts + gated shared expert).
 *
 * M-Q3N.2 scope: the FULL-attention layers only.
 *
 *   x -> rmsNorm(attn_norm)
 *     -> attn_q proj  = [Q | gate] per head  (splitHeadPair)
 *     -> QK-norm(Q,K) -> IMRoPE(Q,K) -> GQA attention
 *     -> attn * sigmoid(gate)  (output gate)  -> attn_output proj
 *     -> attn residual
 *     -> rmsNorm(attn_post_norm)
 *     -> MoE FFN (routed top-K softmax + gated shared expert)
 *     -> FFN residual
 *
 * The recurrent (GatedDeltaNet) layers throw until M-Q3N.3 lands the
 * conv1d + delta-rule kernels. Reaching a full-attention block end-to-end
 * therefore needs input injection (parity harness) until then.
 */
class Qwen35MoeBackend final : public ArchBackend {
public:
    Qwen35MoeBackend(const model::LlmConfig&       config,
                     const core::gguf::WeightsMap& weights,
                     const model::FusedQkvWeights* fusedQkv,
                     compute::ComputeOps&          ops,
                     compute::ComputeMatmul&       gmm,
                     runtime::OpProfiler&          opProfiler,
                     bool                          moeGroupEnabled     = true,
                     bool                          moeFusedDownEnabled = false);

    void runBlock(std::size_t   blockIdx,
                  float*        x,
                  std::size_t   T,
                  KvCache&      cache,
                  BlockBuffers& buffers,
                  bool          traceBlock0) override;

    [[nodiscard]] bool        scalesEmbedding()   const noexcept override { return false; }
    [[nodiscard]] const char* name()              const noexcept override { return "qwen35moe"; }
    [[nodiscard]] bool        needsQGateScratch() const noexcept override { return true; }
    [[nodiscard]] bool        needsSsmScratch()   const noexcept override;

    [[nodiscard]] std::vector<std::size_t>
        kvDimPerLayer() const override;
    [[nodiscard]] std::pair<std::size_t, std::size_t>
        maxQKVDims() const override;

private:
    /// Full-attention layer forward (the "easy" 1-in-interval layers).
    void runFullAttentionBlock(std::size_t   blockIdx,
                               float*        x,
                               std::size_t   T,
                               KvCache&      cache,
                               BlockBuffers& s,
                               bool          diag);

    /// GatedDeltaNet linear-attention layer forward (M-Q3N.3.2). Runs the
    /// conv1d → delta-rule recurrence → gated norm → out projection, then
    /// the shared MoE FFN. The recurrent state (s.ssmStatePtr /
    /// ssmConvStatePtr) is per-sequence (owned by SsmState) and persists
    /// across decode steps; it is zeroed at sequence start (cache.length()==0).
    void runLinearBlock(std::size_t   blockIdx,
                        float*        x,
                        std::size_t   T,
                        KvCache&      cache,
                        BlockBuffers& s,
                        bool          diag);

    /// MoE FFN shared by full-attention (and, later, linear) layers:
    /// routed top-K softmax experts + gated shared expert. Reads its input
    /// from `moeInput` [T, d_model], writes the block-summed result into
    /// `moeAccumBuf`.
    void runMoeFfn(std::size_t         blockIdx,
                   const float*        moeInput,
                   std::size_t         T,
                   BlockBuffers&       s);

    const model::LlmConfig&       _config;
    const core::gguf::WeightsMap& _weights;
    const model::FusedQkvWeights* _fusedQkv{nullptr};
    compute::ComputeOps&          _ops;
    compute::ComputeMatmul&       _gmm;
    runtime::OpProfiler&          _op;

    bool _moeGroupEnabled;
    bool _moeFusedDownEnabled;

    // Diagnostic: when MIMIRMIND_SSM_TRACE is set, log per-linear-layer
    // recurrent-state / output norms and per-block residual-stream norms
    // each forward. Localises the M-Q3N.3 length-degeneration bug (state
    // saturation) without an external reference. No-op / zero cost when off.
    bool _ssmTrace{false};

    // MIMIRMIND_Q8_DP4A: route the Q8_0 shared-expert GEMVs through the
    // dp4a (int8) path at T=1 decode (M-Q3N.4e). Experimental A/B toggle
    // for the perf measurement; default off.
    bool _q8Dp4a{false};

    // MIMIRMIND_GDN_CHUNK: use the chunked GatedDeltaNet prefill (K0->K1->K2)
    // for T>1 instead of the sequential AR loop (M-Q3N.4). Parity-equivalent
    // (cuda_parity 10/10); env-gated for A/B until it is the default. Decode
    // (T==1) always uses the AR recurrence.
    bool _gdnChunk{false};

    /// Host-side L2 norm + max|.| of a compute buffer, after a sync. Only
    /// called on the diagnostic trace path.
    void traceNorm(const char* tag, std::size_t blockIdx,
                   std::size_t pos, const float* p, std::size_t n) const;

    // IMRoPE dimension sections (config.ropeSections), padded to 4 int32
    // so the op always receives a valid 4-element pointer. Zeroed when the
    // model ships no sections (degenerates to plain RoPE).
    std::int32_t _ropeSections[4]{0, 0, 0, 0};

    // Router scratch, hoisted so steady-state runBlock does no allocation.
    std::vector<std::int32_t> _topKIdx;      // [T*K]
    std::vector<float>        _topKWeight;   // [T*K]
};

} // namespace mimirmind::runtime::arch