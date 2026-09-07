// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "compute/ComputeBuffer.hpp"
#include "runtime/arch/Qwen3_5MoeBackend.hpp"
#include "runtime/nvfp4/PleNgramTable.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace mimirmind::runtime::arch {

/**
 * Qwen4-Exp (`qwen4_exp`) — Qwen3.8-Flash-Next (blazux/qwen3.8-Flash-DGX).
 *
 * Architecturally the SAME hybrid backbone as Qwen3_5MoeBackend — GatedDeltaNet
 * linear attention + periodic full attention (interval 4), routed MoE (512
 * experts, top-10) + gated shared expert, and an MTP head — so it derives from
 * Qwen3_5MoeBackend and reuses the whole routed-expert + serving + MTP path.
 *
 * Over that backbone the model adds two components (roadmap 5.27):
 *   - Hyper-Connections: learned multi-stream residual mixing (per-layer
 *     attn/mlp hyper_connection + a top-level mixer), replacing the plain
 *     residual add. Wired in I-3.
 *   - PLE per-layer n-gram embeddings: a ~48 GiB FP8 sparse lookup (~16 rows /
 *     token) fed into designated layers, served off-VRAM via host mmap. I-4.
 *
 * I-1 (this commit) is the SEAM only: the ctor forwards to Qwen3_5MoeBackend
 * and nothing else is overridden, so a qwen4_exp checkpoint loads + runs the
 * MoE backbone unchanged (Hyper-Connections + PLE not yet active). This lets
 * the factory select the arch and the config/loader path light up before the
 * two new components land.
 */
class Qwen4ExpBackend : public Qwen3_5MoeBackend {
public:
    Qwen4ExpBackend(const model::LlmConfig&       config,
                    const core::gguf::WeightsMap& weights,
                    const model::FusedQkvWeights* fusedQkv,
                    compute::ComputeOps&          ops,
                    compute::ComputeMatmul&       gmm,
                    runtime::OpProfiler&          opProfiler,
                    bool                          moeGroupEnabled     = true,
                    bool                          moeFusedDownEnabled = false);

    /// I-3: this arch keeps a 4-stream residual state, so the driver must NOT
    /// apply the plain final output_norm — the top-level hyper_connection_mixer
    /// (below) collapses the streams AND replaces that norm.
    [[nodiscard]] bool usesHyperConnections() const noexcept override { return true; }

    /// I-3: collapse the 4 hyper-connection streams (built up across the block
    /// loop) into the final d_model hidden state via the top-level
    /// hyper_connection_mixer (GatedResidual, use_combine=false). Writes `out`
    /// [T, d_model], ready for lm_head. Replaces the driver's output_norm.
    /// Must be called after the last block's forward, before lm_head.
    void collapseHyperStreams(std::size_t T, BlockBuffers& s, float* out) override;

    /// I-4: hand the engine-owned n-gram table + the ple gguf block index (and
    /// the model eos) to the backend after load. Precomputes the hashing tables.
    void setPleTable(nvfp4::PleNgramTable&& table, int pleGgufLayer,
                     std::int64_t eosTokenId);

    /// I-4: capture the current forward's token ids (for the n-gram hash).
    /// Prefill (T>1) resets the rolling 2-token context; decode (T=1) carries it.
    void prepareForward(std::span<const std::int32_t> tokIds,
                        const float* hiddenStates, std::size_t T) override;

protected:
    /// I-4: at the ple layer, inject the PLE n-gram features into the stream state.
    void blockEnter(std::size_t blockIdx, float* x, std::size_t T,
                    BlockBuffers& s) override;
    // I-3 residual-stream seams: replace the plain norm/residual with the
    // per-layer attn/mlp GatedResidual over the [T, hc*d] stream state.
    void blockInputNorm(std::size_t blockIdx, const float* x, std::size_t T,
                        const float* normWeight, BlockBuffers& s, float* normBuf,
                        bool isAttn) override;
    void blockResidualAdd(std::size_t blockIdx, float* x, const float* moduleOut,
                          std::size_t T, BlockBuffers& s, bool isAttn) override;

private:
    // GatedResidual over the stream state `_hcStreams` -> `mixed` [T, d_model].
    // `combine` (per-layer) also computes the injection weights into `_hcInj`;
    // the mixer (combine=false) skips them. `normWeight` is the hc_norm.
    void hcGatedResidual(std::size_t blockIdx, std::size_t T, const float* normWeight,
                         const std::string& modulePrefix, bool combine,
                         BlockBuffers& s, float* mixed);
    void growHcScratch(std::size_t T);

    compute::ComputeBuffer _hcStreams;   ///< [T, hc*d] residual streams (per forward)
    compute::ComputeBuffer _hcNormed;    ///< [T, hc*d] grouped-RMSNorm output
    compute::ComputeBuffer _hcW1;        ///< [T, lowrank] input-mix bottleneck
    compute::ComputeBuffer _hcW2;        ///< [T, hc*d] input-mix gate
    compute::ComputeBuffer _hcInj;       ///< [T, hc] injection weights
    std::size_t            _hcCapT{0};   ///< current scratch capacity in rows

    // --- I-4 PLE ------------------------------------------------------------
    void pleForward(std::size_t T, BlockBuffers& s);
    void growPleScratch(std::size_t T);
    void computeNgramIds(std::size_t T, std::vector<std::int64_t>& outIds) const;

    nvfp4::PleNgramTable       _pleTable;
    int                        _pleGgufLayer{-1};
    std::int64_t               _pleEos{0};
    // Hashing tables (precomputed in setPleTable): the reference's
    // layer_multipliers, per-head prime vocab sizes and offsets.
    std::vector<std::int64_t>  _pleMult;
    std::vector<std::int64_t>  _pleHeadVocab;
    std::vector<std::int64_t>  _pleHeadOff;
    int                        _pleNgramSize{0};
    int                        _pleHeadsPerNgram{0};
    int                        _pleNgramHeads{0};
    int                        _pleEmbedDim{0};
    // Per-forward token capture + rolling 2-token n-gram context.
    std::vector<std::int32_t>  _pleTokens;
    std::array<std::int32_t, 2> _pleCtx{};
    bool                       _pleCtxInit{false};
    // Host + device scratch.
    std::vector<float>         _pleEmbHost;
    std::vector<std::int64_t>  _pleIdsHost;
    compute::ComputeBuffer _pleEmb, _pleKey, _pleVal, _pleKeyN, _pleQryN;
    compute::ComputeBuffer _pleGated, _pleGvn, _pleConvOut;
    std::size_t                _pleCapT{0};
};

} // namespace mimirmind::runtime::arch
