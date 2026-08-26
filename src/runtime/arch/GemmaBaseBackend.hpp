// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "compute/ComputeBuffer.hpp"

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

namespace mimirmind::runtime {
class BlockBuffers;
class KvCache;
class OpProfiler;
} // namespace mimirmind::runtime

namespace mimirmind::runtime::arch {

/**
 * Shared behaviour for the Gemma family of transformer backends.
 *
 * Owns the attention block choreography, per-layer info, RoPE-factors
 * loading, and parity-dump plumbing. Concrete backends inherit and
 * only implement the block-level FFN sequence in their runBlock —
 * everything from `attn_norm` through `x += post_attention_norm(W_o @ attn(...))`
 * is factored into `runAttentionSection`.
 *
 * Not derived from `ArchBackend`. The public `Gemma4Backend` facade owns
 * one of these as its concrete implementation and forwards the
 * `ArchBackend` interface to it. That keeps the factory (`createArchBackend`)
 * dispatching purely on the model architecture string ("gemma4"), while
 * the dense-vs-MoE choice stays inside the facade.
 *
 * Designed to also accept a future `Gemma3Backend` derived class once we
 * know the exact Gemma-3 attention/norm layout — most of the choreography
 * is already Gemma-common (Q/K-norm, layer-output-scale, SWA-vs-full split,
 * proportional RoPE on full layers).
 */
class GemmaBaseBackend {
public:
    virtual ~GemmaBaseBackend() = default;

    GemmaBaseBackend(const GemmaBaseBackend&)            = delete;
    GemmaBaseBackend& operator=(const GemmaBaseBackend&) = delete;

    /// Variant-specific block execution. Each derived class calls
    /// `runAttentionSection` for the shared prefix, then does its own
    /// FFN sequence, then a final residual + `layer_output_scale`.
    virtual void runBlock(std::size_t   blockIdx,
                          float*        x,
                          std::size_t   T,
                          KvCache&      cache,
                          BlockBuffers& s,
                          bool          traceBlock0) = 0;

    /// M-L0.Batch Phase 1 — synchronized batched decode of `nSeq`
    /// lock-step sequences (each contributes exactly one row of `x`,
    /// T=1 per sequence). `caches[i]` is sequence i's own single-sequence
    /// KvCache; all must sit at their pre-forward length (the batch
    /// harness commits each once after the whole block chain). The
    /// dominant matmuls (Q/O projection, dense FFN, MoE) run at M=nSeq so
    /// each weight read amortizes across the batch (the GEMV→GEMM lever
    /// the decode roofline identified); attention + RoPE stay per-sequence
    /// because each sits at a different position. Default throws —
    /// only the MoE variant implements it in Phase 1.
    virtual void runBlockBatched(std::size_t                blockIdx,
                                 float*                     x,
                                 std::size_t                nSeq,
                                 std::span<KvCache* const>  caches,
                                 BlockBuffers&              s,
                                 bool                       diag) {
        (void)blockIdx; (void)x; (void)nSeq; (void)caches; (void)s; (void)diag;
        throw std::runtime_error(
            "runBlockBatched: batched decode not implemented for this backend");
    }

    /// True when this variant implements runBlockBatched (Gemma 4 MoE in
    /// Phase 1). The Gemma4Backend facade forwards this to ArchBackend.
    [[nodiscard]] virtual bool supportsBatchedDecode() const noexcept {
        return false;
    }

    /// Forwarded from the ArchBackend facade so `Gemma4E4BBackend` can
    /// prefetch its per-layer-embedding (PLE) slices and run the
    /// per_layer_model_proj chain before the block chain runs. Default
    /// no-op — Dense and MoE variants ignore it.
    virtual void prepareForward(std::span<const std::int32_t> /*tokIds*/,
                                const float*                  /*hiddenStates*/,
                                std::size_t                   /*T*/) {}

    /// M-CLR.MoE Increment 3 — mirrors ArchBackend::moeDecodeClrSafe().
    /// True when this variant's decode block runs fully device-side expert
    /// dispatch (no host routing read), making it CLR-capturable. The
    /// facade Gemma4Backend forwards its ArchBackend override here. Dense
    /// variants keep the default false.
    [[nodiscard]] virtual bool moeDecodeClrSafe() const noexcept {
        return false;
    }

    [[nodiscard]] std::vector<std::size_t> kvDimPerLayer() const;
    [[nodiscard]] std::vector<std::size_t> kvSourceLayerPerLayer() const;
    [[nodiscard]] std::pair<std::size_t, std::size_t> maxQKVDims() const;

    void setParityDumpPrefix(std::string prefix) noexcept {
        _parityDumpPrefix = std::move(prefix);
    }

protected:
    GemmaBaseBackend(const model::LlmConfig&        config,
                     const core::gguf::WeightsMap&       weights,
                     const model::FusedQkvWeights*  fusedQkv,
                     compute::ComputeOps&               ops,
                     compute::ComputeMatmul&            gmm,
                     runtime::OpProfiler&           opProfiler);

    /// Per-layer config snapshot, resolved once at construction. All Gemma-4
    /// variants use per-layer head_dim / kv_heads (SWA layers can differ from
    /// full layers), per-layer RoPE base, and an optional "alt attention"
    /// flag for full layers that omit `attn_v.weight` (V = raw K projection).
    struct LayerInfo {
        bool        isSwa;            // SWA vs full-attention layer
        bool        altAttention;     // V derived from raw K (no attn_v.weight)
        std::size_t headDim;          // head_dim for this layer
        std::size_t nHeads;           // Q head count (always config.headCount)
        std::size_t nKvHeads;         // KV head count for this layer
        std::size_t qDim;             // nHeads * headDim
        std::size_t kvDim;            // nKvHeads * headDim
        float       ropeBase;         // SWA vs full rope base

        // Gemma 4 shared-KV mechanism. `ownsKv` is false for the trailing
        // `config.sharedKvLayers` layers — they skip the K/V projection +
        // norm + RoPE + cache-write, and attention instead reads from
        // `kvSourceLayer`'s cache (which was written by an earlier
        // full-KV layer during this same prefill pass).
        bool        ownsKv;
        std::size_t kvSourceLayer;    // = blockIdx when ownsKv, else the reuse source
    };

    /// Fill `_layers` from `_config` + `_weights`. Called from constructor.
    void buildLayerInfos();

    /// Populate `_ropeFreqsForFullAttn` from `rope_freqs.weight` if the model
    /// supplies proportional RoPE. Silent when absent. Called from constructor.
    void loadRopeFreqs();

    /// Per-block tensor lookup that throws with a labelled message so a
    /// caller can identify which backend variant failed to bind. `clsName`
    /// prefixes the runtime_error text.
    const core::gguf::GgufTensor* requireTensor(std::size_t blockIdx,
                                           const char* suffix,
                                           const char* clsName) const;

    /// Write a per-stage parity dump when `_parityDumpPrefix` is set. No-op
    /// otherwise. Format matches `llama-parity-dump` so parity-diff pairs
    /// files by name. Syncs the GPU queue before reading so the USM region
    /// is settled.
    void dumpStage(const char* stage,
                   std::size_t blockIdx,
                   const float* p,
                   std::size_t Trow,
                   std::size_t dim) const;

    /// Runs the pre-attention rmsNorm, Q/K/V projection (fused or split),
    /// per-tensor Q/K/V norms, RoPE, attention (GPU), output projection,
    /// `post_attention_norm`, and residual-add. On return `x` holds
    /// `sa_out = inpL + post_attention_norm(W_o @ attn(...))`.
    ///
    /// `diag` gates one MM_LOG_INFO per stage; derived class computes it as
    /// `(blockIdx == 0 && cache.length() == 0 && traceBlock0)` and passes it in.
    void runAttentionSection(std::size_t   blockIdx,
                             float*        x,
                             std::size_t   T,
                             KvCache&      cache,
                             BlockBuffers& s,
                             bool          diag);

    /// Batched (synchronized-decode) counterpart of runAttentionSection.
    /// `x` is [nSeq, d_model] (row i = sequence i's hidden state, T=1
    /// each); `caches[i]` is sequence i's KvCache, all at the same
    /// pre-forward length within this pass. Batches the position-
    /// independent work — pre-norm, Q projection, Q-norm, O projection,
    /// post-attention norm — at M=nSeq, and loops per sequence only for
    /// the position-dependent K/V projection, K-norm, RoPE (per-seq
    /// startPos) and attention. On return `x` is untouched and
    /// `s.projOut` holds `attn_post_norm(W_o @ attn(...))` per row for the
    /// caller's fused residual+ffn_norm — identical contract to the
    /// single-sequence path.
    ///
    /// Phase 1 supports F32 KV only (throws otherwise) and mirrors every
    /// Gemma-4 attention variant the single-seq path handles: own-KV vs
    /// shared-KV (`kvSourceLayer`), altAttention (V = raw K), SWA vs full
    /// (sliding-window clamp + proportional RoPE factors). K/V projection
    /// runs at M=1 per sequence — batching it needs a K/V staging +
    /// scatter, deferred to a later increment (K/V proj is a small share
    /// of the attention-section matmul time).
    void runAttentionSectionBatched(std::size_t                blockIdx,
                                    float*                     x,
                                    std::size_t                nSeq,
                                    std::span<KvCache* const>  caches,
                                    BlockBuffers&              s,
                                    bool                       diag);

    // Shared runtime state -----------------------------------------------

    const model::LlmConfig&        _config;
    const core::gguf::WeightsMap&       _weights;
    const model::FusedQkvWeights*  _fusedQkv{nullptr};
    compute::ComputeOps&               _ops;
    compute::ComputeMatmul&            _gmm;
    runtime::OpProfiler&           _op;

    std::vector<LayerInfo>    _layers;

    /// USM pointer to the global `rope_freqs.weight` (F32 [head_dim/2])
    /// used as `freq_factors` for full-attention layers only. Points either at
    /// the loaded rope_freqs.weight tensor OR, when the checkpoint ships none
    /// but the config has partial_rotary_factor<1 (gemma4 NVFP4), at the
    /// synthesised p-RoPE factors in `_ropeFreqsComputed`.
    const float*              _ropeFreqsForFullAttn{nullptr};
    /// Backing store for the synthesised p-RoPE freq_factors (see loadRopeFreqs).
    compute::ComputeBuffer    _ropeFreqsComputed{};

    /// Active when InferenceEngine reads `diagnostics.parityDump` from
    /// config.json and passes the value via setParityDumpPrefix().
    /// Empty = disabled.
    std::string               _parityDumpPrefix{};
};

} // namespace mimirmind::runtime::arch