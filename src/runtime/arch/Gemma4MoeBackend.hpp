// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "runtime/arch/GemmaBaseBackend.hpp"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace mimirmind::runtime::arch {

/**
 * Gemma 4 MoE variant — the 26B-A4B-it target (30 blocks, 128 experts,
 * top-8). Adds the hybrid FFN choreography on top of `GemmaBaseBackend`'s
 * shared attention pipeline: dense Path A + MoE Path B + combine +
 * `post_ffw_norm`, then residual + `layer_output_scale`.
 *
 * Tensor requires beyond the shared attention set:
 *   - post_ffw_norm_1, pre_ffw_norm_2, post_ffw_norm_2, post_ffw_norm
 *   - ffn_gate, ffn_up, ffn_down, ffn_norm (Path A dense weights)
 *   - ffn_gate_inp.scale, ffn_gate_inp.weight (router)
 *   - ffn_gate_up_exps.weight, ffn_down_exps.weight, ffn_down_exps.scale
 *   - layer_output_scale
 */
class Gemma4MoeBackend final : public GemmaBaseBackend {
public:
    Gemma4MoeBackend(const model::LlmConfig&        config,
                     const core::gguf::WeightsMap&       weights,
                     const model::FusedQkvWeights*  fusedQkv,
                     compute::ComputeOps&               ops,
                     compute::ComputeMatmul&            gmm,
                     runtime::OpProfiler&           opProfiler,
                     bool                           moeGroupEnabled     = true,
                     bool                           moeFusedDownEnabled = false);

    void runBlock(std::size_t   blockIdx,
                  float*        x,
                  std::size_t   T,
                  KvCache&      cache,
                  BlockBuffers& s,
                  bool          traceBlock0) override;

private:
    /// `features.moeGroup` at construction — routes T>1 through the
    /// expert-grouped batched dispatch path in runBlock(). Off falls back
    /// to per-token dispatch even during prefill.
    bool _moeGroupEnabled;

    /// `features.moeFusedDown != Disable` at construction. Enables the
    /// fused-K down-projection kernel for the T=1 decode path (M-MoE
    /// prototype). Effective only when the kernel actually loaded on
    /// this iGPU AND `ffn_down_exps.weight` is Q6_K; the backend
    /// checks both at first use and falls back silently otherwise.
    bool _moeFusedDownEnabled;

    // ---- Per-block routing scratch (reused across runBlock) ----------
    //
    // These held std::vectors that were heap-allocated and freed on every
    // runBlock() — per block, per token step. The router pick buffers
    // (_topKIdx/_topKWeight) churned on every call; the expert-grouping
    // buffers additionally allocated `nExperts` (128) nested vectors each
    // prefill pass. Hoisted to members so steady-state runBlock() does
    // zero routing-side allocation: outer containers are sized once, the
    // hot path clear()/resize()es (capacity is retained). Single-threaded
    // per engine, same as GemmaBaseBackend's own scratch — the serving
    // loop serialises all calls on one instance.
    std::vector<std::int32_t> _topKIdx;      // [T*K] router expert picks
    std::vector<float>        _topKWeight;   // [T*K] router weights

    // Expert-grouped prefill (T>1) permutation buffers.
    std::vector<std::vector<std::pair<std::size_t, float>>> _expertTokens;
    std::vector<std::size_t>  _expertOffset; // [nExperts+1] prefix sum
    std::vector<std::size_t>  _gatherToken;  // [T*K] compact-row → token
    std::vector<float>        _rowWeight;    // [T*K] compact-row weight
};

} // namespace mimirmind::runtime::arch