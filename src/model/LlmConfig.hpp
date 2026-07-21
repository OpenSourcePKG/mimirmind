// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/gguf/GgufReader.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mimirmind::model {

using ::mimirmind::core::gguf::GgufReader;
using ::mimirmind::core::gguf::GgufTensor;
using ::mimirmind::core::gguf::MetadataValue;
using ::mimirmind::core::gguf::GgmlType;
using ::mimirmind::core::gguf::GgufArray;
using ::mimirmind::core::gguf::GgufValueType;
using ::mimirmind::core::gguf::typeInfo;

/**
 * Architecture-agnostic config for a decoder-only LLM stored in GGUF.
 * Works for any architecture that exposes the standard llama.cpp metadata
 * keys (gemma, gemma2, gemma3, gemma4, llama, mistral, qwen2, phi3, ...).
 * The `architecture` field disambiguates downstream code paths that care
 * about per-arch quirks (Gemma 3 sliding window, Llama RoPE scaling, ...).
 *
 * Required keys throw if missing; optional keys fall back to sensible
 * defaults and are logged at INFO so it's visible *which* defaults
 * kicked in.
 *
 * Reference: github.com/ggerganov/llama.cpp src/llama-model.cpp.
 */
struct LlmConfig {
    std::string architecture;

    // Required — throws if missing from metadata.
    std::uint32_t blockCount        {0};
    std::uint32_t contextLength     {0};
    std::uint32_t embeddingLength   {0};
    std::uint32_t feedForwardLength {0};
    std::uint32_t headCount         {0};
    std::uint32_t headCountKv       {0};

    // Optional — defaults applied + logged.
    std::uint32_t keyLength         {0};        // 0 = derive embedding/head_count
    std::uint32_t valueLength       {0};        // 0 = same as keyLength
    std::uint32_t keyLengthSwa      {0};        // SWA-layer key length (Gemma 4)
    std::uint32_t valueLengthSwa    {0};        // SWA-layer value length (Gemma 4)
    float         rmsNormEps        {1e-6F};
    // Explicit attention softmax scale (GGUF `<arch>.attention.scale`).
    // 0 = unset → attention uses the default 1/sqrt(head_dim). Qwen3-Next
    // (`qwen35moe`) ships this key; llama.cpp:
    // `kq_scale = f_attention_scale ? f_attention_scale : 1/sqrt(head_dim)`.
    float         attentionScale    {0.0F};
    float         ropeFreqBase      {10000.0F}; // global / full-attention layers
    float         ropeFreqBaseSwa   {10000.0F}; // sliding-window layers (Gemma 3/4)
    std::uint32_t slidingWindow     {0};        // 0 = disabled (Gemma 3+ uses 4096)

    // Gemma 3/4 sliding-window-attention pattern. Empty = all layers are
    // full attention. Otherwise size == blockCount and entry b is true if
    // block b is SWA, false if it's a global-attention block.
    std::vector<bool> slidingWindowPattern{};

    // Per-layer KV head count. Empty = uniform headCountKv across layers
    // (Qwen / Llama). On Gemma 4 it's typically e.g. 8 for SWA layers and
    // 2 for the (sparser) full-attention layers.
    std::vector<std::uint32_t> headCountKvPerLayer{};

    // MoE (Gemma 4). 0 = dense model (no expert routing).
    std::uint32_t expertCount       {0};        // total experts per block
    std::uint32_t expertUsedCount   {0};        // top-K experts activated per token

    // MoE expert feed-forward lengths + gating scale. Populated when the
    // model ships them (Qwen3-Next `qwen35moe`; also some Gemma 4 MoE).
    // `expertFeedForwardLength` (GGUF `<arch>.expert_feed_forward_length`)
    // is the per-routed-expert intermediate width; `expertSharedFeedForwardLength`
    // (`<arch>.expert_shared_feed_forward_length`) sizes the always-on shared
    // expert. Zero ⇒ derive from `feedForwardLength` at use-site.
    // `expertWeightsScale` (`<arch>.expert_weights_scale`) multiplies the
    // (renormalised top-K softmax) router weights. Zero ⇒ no extra scale.
    std::uint32_t expertFeedForwardLength       {0};
    std::uint32_t expertSharedFeedForwardLength {0};
    float         expertWeightsScale            {0.0F};

    // --- Qwen3-Next / GatedDeltaNet hybrid (qwen35moe) -------------------
    // Linear-attention ("gated delta net") SSM hyperparameters. All zero on
    // non-hybrid archs. GGUF keys `<arch>.ssm.{conv_kernel,inner_size,
    // state_size,time_step_rank,group_count}`. For Qwen3.6-35B-A3B:
    //   conv_kernel=4, inner_size=4096, state_size=128, time_step_rank=32,
    //   group_count=16. See research/qwen3next-gated-deltanet-recon-2026-07-21.
    std::uint32_t ssmConvKernel   {0};  // ssm_d_conv  — causal conv1d width
    std::uint32_t ssmInnerSize    {0};  // ssm_d_inner — = value_dim (H_v*S_v)
    std::uint32_t ssmStateSize    {0};  // ssm_d_state — GatedDeltaNet head_dim
    std::uint32_t ssmTimeStepRank {0};  // ssm_dt_rank — num v-heads (H_v)
    std::uint32_t ssmGroupCount   {0};  // ssm_n_group — num k-heads  (H_k)

    // NextN / MTP speculative-decode head. >0 ⇒ the model appends
    // `nextnPredictLayers` dense-attention MTP block(s) past the main stack
    // (`blk.<blockCount>.nextn.*`). Not executed in the main forward pass;
    // stored here so the spec-decode path (M-Q3N.5+) can find them.
    std::uint32_t nextnPredictLayers {0};

    // Interleaved-MRoPE (IMRoPE) dimension sections. Empty on archs without
    // mRoPE. Length 4 when present (Qwen3-Next / Qwen2.5-VL). GGUF key
    // `<arch>.rope.dimension_sections`.
    std::vector<std::int32_t> ropeSections{};

    // Per-layer "is this a recurrent (GatedDeltaNet linear-attention)
    // layer?" mask. Empty = every layer is standard attention (all non-hybrid
    // archs). When present, size == blockCount; entry b is true if block b is
    // a linear-attention layer, false if it's a full (softmax) attention
    // layer. Derived from `<arch>.attention.recurrent_layers` (bool array)
    // when present, else synthesised from `<arch>.full_attention_interval`
    // (default 4) via `(b+1) % interval != 0` — the llama.cpp qwen35moe
    // convention (every `interval`-th layer is full attention).
    std::vector<bool> recurrentLayerPattern{};

    // Gemma 4 "shared KV layers": last N layers reuse an earlier layer's
    // K/V cache instead of computing their own. GGUF key
    // `<arch>.attention.shared_kv_layers`. 0 = every layer computes its
    // own K/V (the historical default we've been assuming).
    // Populated for E4B (=18) and 26B-A4B (=some value); zero elsewhere.
    // Consumers should treat `blockCount - sharedKvLayers` as the
    // "n_layer_kv_from_start" threshold — layers >= this reuse an
    // earlier layer's K/V per the llama.cpp reuse callback
    // `n_kv_from_start - (is_swa(il) ? 2 : 1)`.
    std::uint32_t sharedKvLayers    {0};

    // Gemma 4 final-logit softcap. GGUF key
    // `<arch>.final_logit_softcapping`. When > 0 the sampler applies
    // `cap * tanh(logit / cap)` to the output logits before penalties,
    // temperature scaling, and softmax — matches llama.cpp's placement
    // inside the compute graph (`src/models/gemma4.cpp`). Argmax-invariant,
    // so greedy decode is unaffected in bit-identical terms; sampling
    // (temperature > 0) picks a strictly different distribution shape.
    // Attention-logit softcap (Gemma 2) was dropped in Gemma 4 — no
    // corresponding field here. Populated for gemma4 (typically 30.0);
    // zero for archs that don't declare the key.
    float         finalLogitSoftcap {0.0F};

    [[nodiscard]] std::uint32_t headDim() const noexcept {
        if (keyLength > 0) {
            return keyLength;
        }
        return headCount > 0 ? embeddingLength / headCount : 0;
    }

    /// Effective attention softmax scale for a given head_dim. Returns the
    /// explicit `attentionScale` when the model ships it (>0), else the
    /// conventional `1/sqrt(head_dim)`. Matches llama.cpp's kq_scale pick.
    [[nodiscard]] float attentionScaleFor(std::size_t headDim_) const noexcept {
        if (attentionScale > 0.0F) {
            return attentionScale;
        }
        return headDim_ > 0
                   ? 1.0F / std::sqrt(static_cast<float>(headDim_))
                   : 0.0F;
    }

    /// Per-layer head_dim. For Gemma 4 returns `keyLengthSwa` on SWA
    /// layers and `keyLength` (full) on global-attention layers. For
    /// every other arch (uniform attention) it just returns `headDim()`.
    [[nodiscard]] std::uint32_t headDim(std::size_t blockIdx) const noexcept {
        if (blockIdx < slidingWindowPattern.size() && keyLengthSwa > 0) {
            return slidingWindowPattern[blockIdx] ? keyLengthSwa : keyLength;
        }
        return headDim();
    }

    /// Per-layer KV head count. Reads `headCountKvPerLayer[blockIdx]` if
    /// the model ships a per-layer array, else falls back to the global
    /// `headCountKv` value.
    [[nodiscard]] std::uint32_t headCountKvFor(std::size_t blockIdx) const noexcept {
        if (blockIdx < headCountKvPerLayer.size()) {
            return headCountKvPerLayer[blockIdx];
        }
        return headCountKv;
    }

    /// True iff block `blockIdx` is a GatedDeltaNet linear-attention layer.
    /// False for every layer on non-hybrid archs (empty pattern).
    [[nodiscard]] bool isRecurrentLayer(std::size_t blockIdx) const noexcept {
        return blockIdx < recurrentLayerPattern.size() &&
               recurrentLayerPattern[blockIdx];
    }

    /// True iff this is a hybrid arch with at least one recurrent
    /// (linear-attention) layer — i.e. the SSM path is active.
    [[nodiscard]] bool isHybridRecurrent() const noexcept {
        return !recurrentLayerPattern.empty();
    }

    /// GatedDeltaNet dims, derived from the SSM hyperparameters. All return
    /// 0 on non-hybrid archs. `ssmConvDim` is the depthwise-conv channel
    /// count `key_dim*2 + value_dim = d_inner + 2*n_group*d_state` (the
    /// `ssm_conv1d.weight` dim[1]).
    [[nodiscard]] std::uint32_t ssmHeadDim()   const noexcept { return ssmStateSize; }
    [[nodiscard]] std::uint32_t ssmNumKHeads() const noexcept { return ssmGroupCount; }
    [[nodiscard]] std::uint32_t ssmNumVHeads() const noexcept { return ssmTimeStepRank; }
    [[nodiscard]] std::uint32_t ssmConvDim()   const noexcept {
        return ssmInnerSize + 2U * ssmGroupCount * ssmStateSize;
    }

    /// Recurrent SSM state elements per linear-attention layer per sequence:
    /// the `[head_dim, head_dim]` memory matrix per v-head, i.e.
    /// `S_v * S_v * H_v = ssm_d_state^2 * ssm_dt_rank`. For 35B-A3B this is
    /// 128*128*32 = 512Ki floats = 2 MiB (F32) per linear layer per sequence.
    /// Used to size the SSM state pool (M-Q3N.3). Zero on non-hybrid archs.
    [[nodiscard]] std::size_t ssmStateElemsPerLayer() const noexcept {
        return std::size_t{ssmStateSize} * ssmStateSize * ssmTimeStepRank;
    }

    /// Causal-conv rolling state elements per linear-attention layer per
    /// sequence: `(conv_kernel - 1) * conv_dim`. Zero on non-hybrid archs.
    [[nodiscard]] std::size_t ssmConvStateElemsPerLayer() const noexcept {
        return ssmConvKernel == 0
                   ? 0
                   : std::size_t{ssmConvKernel - 1U} * ssmConvDim();
    }

    /// KV-cache bytes required per newly-decoded token, summed across
    /// all layers. Formula: `sum_{b=0..blockCount-1} 2 * n_kv(b) * head_dim(b) * dtypeBytes`
    /// where the leading `2` covers K + V. Respects per-layer variations
    /// (Gemma 4 mixed SWA/full head_dim, per-layer n_kv).
    ///
    /// `dtypeBytes` is the size of one element in the runtime KV dtype:
    ///   4 = F32 (current default)
    ///   2 = FP16 (M10.2 Phase 0)
    ///   1 = Q8_0 approximation (per-element cost incl. block scale is
    ///       closer to 1.0625 for GGUF-Q8_0; caller should round up when
    ///       using this for capacity planning)
    ///
    /// Used by BatchCapacityProbe to estimate memory-limited batch size
    /// and by KvCache to size preallocations. Never throws.
    [[nodiscard]] std::size_t kvBytesPerToken(std::size_t dtypeBytes = 2) const noexcept {
        std::size_t total = 0;
        for (std::size_t b = 0; b < blockCount; ++b) {
            const std::size_t nKv = headCountKvFor(b);
            const std::size_t hd  = headDim(b);
            // 2 = K + V. Per-block product cannot exceed a few KB even for
            // the largest models, so plain size_t multiplication is safe.
            total += 2U * nKv * hd * dtypeBytes;
        }
        return total;
    }

    /// Populate from the reader's metadata. Throws on missing required
    /// keys; logs every value found at INFO.
    void parseFromGguf(const GgufReader& reader);
};

} // namespace mimirmind::model