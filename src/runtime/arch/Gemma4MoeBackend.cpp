// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/arch/Gemma4MoeBackend.hpp"

#include "compute/l0/GpuMatmul.hpp"
#include "compute/l0/GpuOps.hpp"
#include "core/gpu/l0/CommandQueue.hpp"
#include "compute/MoeRouting.hpp"
#include "compute/QuantType.hpp"
#include "compute/QuantTypeRegistry.hpp"
#include "core/gguf/GgufReader.hpp"
#include "core/gguf/GgufTypes.hpp"
#include "model/LlmConfig.hpp"
#include "core/gguf/WeightsMap.hpp"
#include "runtime/BlockBuffers.hpp"
#include "runtime/KvCache.hpp"
#include "core/log/Log.hpp"
#include "runtime/OpProfiler.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace mimirmind::runtime::arch {

Gemma4MoeBackend::Gemma4MoeBackend(const model::LlmConfig&        config,
                                   const core::gguf::WeightsMap&       weights,
                                   const model::FusedQkvWeights*  fusedQkv,
                                   compute::l0::GpuOps&               ops,
                                   compute::l0::GpuMatmul&            gmm,
                                   runtime::OpProfiler&           opProfiler,
                                   bool                           moeGroupEnabled,
                                   bool                           moeFusedDownEnabled)
    : GemmaBaseBackend{config, weights, fusedQkv, ops, gmm, opProfiler},
      _moeGroupEnabled{moeGroupEnabled},
      _moeFusedDownEnabled{moeFusedDownEnabled} {
    MM_LOG_INFO("gemma4-moe",
                "Gemma4MoeBackend ready — blocks={} d_model={} ff={} "
                "experts={} top_k={} swa head_dim={} kv={}, "
                "full head_dim={} kv={} moeGroup={} moeFusedDown={}",
                _config.blockCount, _config.embeddingLength,
                _config.feedForwardLength,
                _config.expertCount, _config.expertUsedCount,
                _config.keyLengthSwa,
                _config.headCountKvFor(0),
                _config.keyLength,
                _layers.empty() ? 0 : _layers.front().nKvHeads,
                _moeGroupEnabled, _moeFusedDownEnabled);
}

void Gemma4MoeBackend::runBlock(std::size_t   blockIdx,
                                float*        x,
                                std::size_t   T,
                                KvCache&      cache,
                                BlockBuffers& s,
                                bool          traceBlock0) {
    namespace cmp = mimirmind::compute;

    const bool diag = (blockIdx == 0 && cache.length() == 0 && traceBlock0);
    auto trace = [&](const char* tag) {
        if (diag) MM_LOG_INFO("blkdiag-g4m", "blk0 {}", tag);
    };
    trace("enter (moe)");

    // Shared attention section. On return `x` holds
    // sa_out = inpL + post_attention_norm(W_o @ attn(...)).
    runAttentionSection(blockIdx, x, T, cache, s, diag);

    // FFN tensors (Path A dense weights + MoE router + expert bank +
    // per-side / combined norms + layer output scale).
    const auto* ffnNorm     = requireTensor(blockIdx, "ffn_norm.weight",           "Gemma4MoeBackend");
    const auto* ffnGate     = requireTensor(blockIdx, "ffn_gate.weight",           "Gemma4MoeBackend");
    const auto* ffnUp       = requireTensor(blockIdx, "ffn_up.weight",             "Gemma4MoeBackend");
    const auto* ffnDown     = requireTensor(blockIdx, "ffn_down.weight",           "Gemma4MoeBackend");
    const auto* ffwPost1    = requireTensor(blockIdx, "post_ffw_norm_1.weight",    "Gemma4MoeBackend");
    const auto* ffwPost     = requireTensor(blockIdx, "post_ffw_norm.weight",      "Gemma4MoeBackend");
    const auto* outScale    = requireTensor(blockIdx, "layer_output_scale.weight", "Gemma4MoeBackend");
    const auto* preNorm2    = requireTensor(blockIdx, "pre_ffw_norm_2.weight",     "Gemma4MoeBackend");
    const auto* postNorm2   = requireTensor(blockIdx, "post_ffw_norm_2.weight",    "Gemma4MoeBackend");
    const auto* routerScale = requireTensor(blockIdx, "ffn_gate_inp.scale",        "Gemma4MoeBackend");
    const auto* routerW     = requireTensor(blockIdx, "ffn_gate_inp.weight",       "Gemma4MoeBackend");
    const auto* expGateUp   = requireTensor(blockIdx, "ffn_gate_up_exps.weight",   "Gemma4MoeBackend");
    const auto* expDown     = requireTensor(blockIdx, "ffn_down_exps.weight",      "Gemma4MoeBackend");
    const auto* expDownScale= requireTensor(blockIdx, "ffn_down_exps.scale",       "Gemma4MoeBackend");

    const std::size_t d_model  = s.d_model;
    const std::size_t ff_dim   = s.ff_dim;

    float* const normBuf       = s.normBuf.as<float>();
    float* const attnOutBuf    = s.attnOut.as<float>();
    float* const projOutBuf    = s.projOut.as<float>();
    float* const gateOutBuf    = s.gateOut.as<float>();
    float* const upOutBuf      = s.upOut.as<float>();
    float* const matmulScratch = s.matmulScratch.as<float>();
    float* const moeAccumBuf   = s.moeAccumBuf.as<float>();
    float* const expertOutBuf  = s.expertOutBuf.as<float>();

    // --- FFN path A — dense SwiGLU with GELU ---------------------------
    // Fused attn-residual + ffn_norm: runAttentionSection left
    // `projOutBuf = attn_post_norm(attn_out)` for us to fold in here.

    _op.mark(runtime::OpProfiler::Cat::NORM);
    trace("attn residual + ffn_norm (fused)");
    _ops.addRmsNormAsync(x, projOutBuf, T, d_model,
                         static_cast<const float*>(ffnNorm->usmPtr),
                         _config.rmsNormEps,
                         normBuf);
    dumpStage("attn_out", blockIdx, x, T, d_model);

    // M5f.4: FFN gate + up read normBuf, write disjoint outputs.
    _op.mark(runtime::OpProfiler::Cat::MATMUL);
    trace("FFN gate+up proj (unordered)");
    {
        runtime::UnorderedScope u{_ops.queue()};
        _gmm.matmulAsync(ffnGate->type, ffnGate->usmPtr, ff_dim, d_model,
                         normBuf, T, gateOutBuf, matmulScratch);
        _gmm.matmulAsync(ffnUp->type, ffnUp->usmPtr, ff_dim, d_model,
                         normBuf, T, upOutBuf, matmulScratch);
    }

    _op.mark(runtime::OpProfiler::Cat::ACTIVATION);
    trace("GELU + mul (fused)");
    _ops.geluMulAsync(gateOutBuf, upOutBuf, T * ff_dim);

    _op.mark(runtime::OpProfiler::Cat::MATMUL);
    trace("FFN down proj");
    _gmm.matmul(ffnDown->type, ffnDown->usmPtr, d_model, ff_dim,
                gateOutBuf, T,
                projOutBuf, matmulScratch);

    _op.mark(runtime::OpProfiler::Cat::NORM);
    trace("post_ffw_norm_1 (path A post)");
    _ops.rmsNormAsync(projOutBuf, T, d_model,
                      static_cast<const float*>(ffwPost1->usmPtr),
                      _config.rmsNormEps,
                      projOutBuf);            // in-place
    dumpStage("ffn_mlp", blockIdx, projOutBuf, T, d_model);

    // --- Path B — MoE -------------------------------------------------

    // M5f.4: two rmsNorms on the same input x with different weights and
    // different output buffers — fully independent, can pipeline.
    _op.mark(runtime::OpProfiler::Cat::NORM);
    trace("path B: pre_ffw_norm_2 + router rmsNorm (unordered)");
    {
        runtime::UnorderedScope u{_ops.queue()};
        _ops.rmsNormAsync(x, T, d_model,
                          static_cast<const float*>(preNorm2->usmPtr),
                          _config.rmsNormEps,
                          normBuf);
        _ops.rmsNormAsync(x, T, d_model,
                          static_cast<const float*>(routerScale->usmPtr),
                          _config.rmsNormEps,
                          attnOutBuf);
    }
    const float invSqrtDm = 1.0F /
        std::sqrt(static_cast<float>(d_model));
    _ops.mulScalarAsync(attnOutBuf, invSqrtDm, T * d_model);

    const std::size_t nExperts = _config.expertCount;
    const std::size_t K        = _config.expertUsedCount;
    _op.mark(runtime::OpProfiler::Cat::ROUTER);
    trace("path B: router matmul (CPU)");
    _gmm.matmul(routerW->type, routerW->usmPtr,
                nExperts, d_model,
                attnOutBuf, T,
                upOutBuf, matmulScratch);

    std::vector<std::int32_t> topKIdx(T * K);
    std::vector<float>        topKWeight(T * K);
    cmp::moeTopKRoute(upOutBuf, T, nExperts, K,
                      topKIdx.data(), topKWeight.data());

    _op.mark(runtime::OpProfiler::Cat::RESIDUAL);
    trace("path B: zero accumulator");
    _ops.mulScalarAsync(moeAccumBuf, 0.0F, T * d_model);

    const std::size_t gateUpFused = expGateUp->dimensions.size() >= 2
                                      ? expGateUp->dimensions[1] : 0;
    const std::size_t ffPerExpert = gateUpFused / 2;
    if (gateUpFused == 0 || (gateUpFused % 2) != 0) {
        throw std::runtime_error(
            "Gemma4MoeBackend: ffn_gate_up_exps has unexpected fused dim " +
            std::to_string(gateUpFused));
    }

    const compute::QuantType* const qtGateUp = compute::quantType(expGateUp->type);
    const compute::QuantType* const qtDown   = compute::quantType(expDown->type);
    if (qtGateUp == nullptr || qtDown == nullptr) {
        throw std::runtime_error(
            "Gemma4MoeBackend: expert weight type(s) not in QuantType registry");
    }

    const std::size_t expertBytesGateUp =
        gateUpFused * (d_model / qtGateUp->blockElements()) * qtGateUp->blockBytes();
    const std::size_t expertBytesDown =
        d_model * (ffPerExpert / qtDown->blockElements()) * qtDown->blockBytes();

    auto* const expGateUpBase =
        static_cast<const std::uint8_t*>(expGateUp->usmPtr);
    auto* const expDownBase =
        static_cast<const std::uint8_t*>(expDown->usmPtr);
    const float* const expDownScalePtr =
        static_cast<const float*>(expDownScale->usmPtr);

    // M5i.F: Expert-grouped dispatch for prefill (T > 1). Groups the
    // T*K_top per-token expert selections by expert so each expert's
    // matmul runs on a batch of M = n_routed rows instead of M=1. The
    // gate/up matmuls are split into two calls (one per half of the
    // fused weight rows) so we can reuse the existing plain-flat
    // geluMulAsync — a batched activation-with-stride kernel would
    // save one launch per expert but adds a new kernel to maintain.
    //
    // Decode (T == 1) still walks the per-token loop below: with only
    // top-K work items there's no batching opportunity and the compact
    // scratch write-back would just add overhead.
    const bool useMoeGrouping = (T > 1) && _moeGroupEnabled;

    _op.mark(runtime::OpProfiler::Cat::MATMUL);
    if (useMoeGrouping) {
        trace("path B: expert-grouped dispatch");

        // Half-of-fused byte offset — points at the "up" rows in each
        // per-expert gate_up_exps.weight block.
        const std::size_t gateBytesHalf = ffPerExpert *
            (d_model / qtGateUp->blockElements()) * qtGateUp->blockBytes();

        // CPU-side permutation. Build per-expert (tokenIdx, weight)
        // lists, expertOffset prefix-sum, and the flat gather+weight
        // arrays used by the scatter step below.
        const std::size_t nRows = T * K;
        std::vector<std::vector<std::pair<std::size_t, float>>>
            expertTokens(nExperts);
        for (std::size_t t = 0; t < T; ++t) {
            for (std::size_t k = 0; k < K; ++k) {
                const std::size_t e =
                    static_cast<std::size_t>(topKIdx[t * K + k]);
                expertTokens[e].emplace_back(t, topKWeight[t * K + k]);
            }
        }
        std::vector<std::size_t> expertOffset(nExperts + 1, 0);
        for (std::size_t e = 0; e < nExperts; ++e) {
            expertOffset[e + 1] =
                expertOffset[e] + expertTokens[e].size();
        }
        // Invariant — every (token, top-k slot) contributes exactly one
        // compact row, so the sum of per-expert token counts must be
        // T*K_top. If this trips it means moeTopKRoute produced an
        // expert index out of [0, nExperts) or the loop above skipped a
        // pair — either way we'd overwrite past the scratch bounds.
        if (expertOffset[nExperts] != nRows) {
            throw std::runtime_error(
                "Gemma4MoeBackend MoE grouping: expertOffset[nExperts]=" +
                std::to_string(expertOffset[nExperts]) +
                " != T*K_top=" + std::to_string(nRows) +
                " (routing produced out-of-range expert index?)");
        }

        std::vector<std::size_t> gatherToken(nRows);
        std::vector<float>       rowWeight(nRows);
        for (std::size_t e = 0; e < nExperts; ++e) {
            const float scale = expDownScalePtr[e];
            const std::size_t off = expertOffset[e];
            for (std::size_t i = 0; i < expertTokens[e].size(); ++i) {
                gatherToken[off + i] = expertTokens[e][i].first;
                rowWeight[off + i]   =
                    expertTokens[e][i].second * scale;
            }
        }

        float* const xComp    = s.moeXCompact.as<float>();
        float* const gateComp = s.moeGateCompact.as<float>();
        float* const upComp   = s.moeUpCompact.as<float>();
        float* const downComp = s.moeDownCompact.as<float>();

        // Gather X → compact rows. normBuf holds the path-B input
        // [T, d_model]; sync so the CPU memcpy sees settled memory.
        _gmm.sync();
        for (std::size_t i = 0; i < nRows; ++i) {
            const std::size_t t = gatherToken[i];
            std::memcpy(xComp + i * d_model,
                        normBuf + t * d_model,
                        d_model * sizeof(float));
        }

        // Zero the accumulator; every touched token gets its
        // contributions summed into it, untouched tokens stay 0.
        _ops.mulScalarAsync(moeAccumBuf, 0.0F, T * d_model);

        // Per-expert batched matmuls. Skip experts with no routed
        // tokens; they'd dispatch a matmul with M=0 which the kernel
        // handles by returning early but the launch overhead isn't
        // free.
        trace("path B: per-expert batched matmuls");
        for (std::size_t e = 0; e < nExperts; ++e) {
            const std::size_t nRoutedE = expertTokens[e].size();
            if (nRoutedE == 0) continue;
            const std::size_t off = expertOffset[e];

            const auto* Wgu =
                static_cast<const std::uint8_t*>(expGateUpBase) +
                e * expertBytesGateUp;
            const void* Wd  = expDownBase + e * expertBytesDown;

            // Gate rows: first half of the fused gate_up weight block.
            _gmm.matmulAsync(expGateUp->type, Wgu,
                             ffPerExpert, d_model,
                             xComp + off * d_model, nRoutedE,
                             gateComp + off * ffPerExpert,
                             matmulScratch);
            // Up rows: second half, offset in bytes.
            _gmm.matmulAsync(expGateUp->type, Wgu + gateBytesHalf,
                             ffPerExpert, d_model,
                             xComp + off * d_model, nRoutedE,
                             upComp + off * ffPerExpert,
                             matmulScratch);

            // gelu(gate) * up, in place into gateComp region.
            _ops.geluMulAsync(gateComp + off * ffPerExpert,
                              upComp   + off * ffPerExpert,
                              nRoutedE * ffPerExpert);

            // Down: gate_activated @ W_d[e]  →  downComp region.
            _gmm.matmulAsync(expDown->type, Wd,
                             d_model, ffPerExpert,
                             gateComp + off * ffPerExpert, nRoutedE,
                             downComp + off * d_model,
                             matmulScratch);
        }

        // Scatter-accumulate. Each compact row contributes
        //   accum[t] += weight[i] * downComp[i]
        // where t is the token that produced that expert selection.
        // scaledAddResidualAsync appends one kernel per row —
        // same launch count as the pre-grouping path's inner
        // scaled-add, so no regression on that op.
        trace("path B: scatter-accumulate");
        for (std::size_t i = 0; i < nRows; ++i) {
            const std::size_t t = gatherToken[i];
            _ops.scaledAddResidualAsync(
                moeAccumBuf + t * d_model,
                downComp    + i * d_model,
                rowWeight[i],
                d_model);
        }
    } else {
        // M-MoE.Fused-Decode — enable the fused-K down path when all
        // preconditions line up: toggle on, kernel loaded for this
        // expert quant type on this iGPU, T == 1 (decode), scratches
        // allocated. Otherwise fall through to the sequential per-expert
        // dispatch below. 26B-A4B mixes types (gate_up=Q6_K,
        // ffn_down=Q8_0); Q4_K/Q5_K expert downs are future-model
        // candidates that just need their own kernel variant.
        const bool useMoeFusedDown =
            _moeFusedDownEnabled &&
            _gmm.moeDownFusedKAvailable(expDown->type) &&
            T == 1 &&
            s.moeExpIdxScratch.get() != nullptr &&
            s.moeKwScratch.get()     != nullptr &&
            s.moeGateCompact.get()   != nullptr;

        if (useMoeFusedDown) {
            trace("path B: per-token expert dispatch (fused-K down)");
            float* const pathBIn = normBuf;         // T == 1
            float* const accumT  = moeAccumBuf;
            float* const gateActAll = s.moeGateCompact.as<float>();

            // Per-expert gate_up + gelu_mul into strided slots — kept
            // separate so the fused down kernel sees [K, ffPer].
            for (std::size_t k = 0; k < K; ++k) {
                const std::size_t e =
                    static_cast<std::size_t>(topKIdx[k]);
                const void* Wgu = expGateUpBase + e * expertBytesGateUp;

                _gmm.matmulAsync(expGateUp->type, Wgu,
                                 gateUpFused, d_model,
                                 pathBIn, 1,
                                 gateOutBuf, matmulScratch);
                _ops.geluMulAsync(gateOutBuf, gateOutBuf + ffPerExpert,
                                  ffPerExpert);
                // Copy the ffPerExpert activations into the K-strided
                // slot the fused kernel expects. Kept on the queue for
                // ordering vs the next iteration's gate_up.
                _ops.queue().appendMemoryCopy(
                    gateActAll + k * ffPerExpert,
                    gateOutBuf,
                    ffPerExpert * sizeof(float));
            }

            // Populate this layer's routing scratch. Direct writes to
            // host-visible USM — no memcpy path needed on UMA.
            auto* const expIdxSlot =
                s.moeExpIdxScratch.as<std::int32_t>() +
                blockIdx * K;
            auto* const kwSlot =
                s.moeKwScratch.as<float>() +
                blockIdx * K;
            for (std::size_t k = 0; k < K; ++k) {
                const std::size_t e =
                    static_cast<std::size_t>(topKIdx[k]);
                expIdxSlot[k] = static_cast<std::int32_t>(e);
                kwSlot[k]     = topKWeight[k] * expDownScalePtr[e];
            }

            trace("path B: fused-K down dispatch");
            _gmm.moeDownFusedKAsync(
                expDown->type,
                gateActAll,
                expDownBase,
                expIdxSlot,
                kwSlot,
                accumT,
                ffPerExpert,
                d_model,
                K,
                expertBytesDown);
        } else {
            trace("path B: per-token expert dispatch");
            for (std::size_t t = 0; t < T; ++t) {
                float* const pathBInT  = normBuf      + t * d_model;
                float* const accumT    = moeAccumBuf  + t * d_model;
                for (std::size_t k = 0; k < K; ++k) {
                    const std::size_t e =
                        static_cast<std::size_t>(topKIdx[t * K + k]);
                    const float       routerWeight = topKWeight[t * K + k];

                    const void* Wgu = expGateUpBase + e * expertBytesGateUp;
                    const void* Wd  = expDownBase   + e * expertBytesDown;

                    _gmm.matmulAsync(expGateUp->type, Wgu,
                                     gateUpFused, d_model,
                                     pathBInT, 1,
                                     gateOutBuf, matmulScratch);

                    _ops.geluMulAsync(gateOutBuf, gateOutBuf + ffPerExpert,
                                      ffPerExpert);

                    // M5i.F prep: async instead of sync. Auto-barrier after
                    // the append keeps ordering vs the following
                    // scaledAddResidual, and expertOutBuf isn't read by the
                    // CPU inside this loop. Removes T*K_top syncs per MoE
                    // block per prefill call.
                    _gmm.matmulAsync(expDown->type, Wd,
                                     d_model, ffPerExpert,
                                     gateOutBuf, 1,
                                     expertOutBuf, matmulScratch);

                    // M9.6.4: fused scale-and-accumulate. expertOutBuf is
                    // overwritten by the next iteration's down-projection
                    // so there's no downstream reader of the post-scale
                    // buffer — safe to do dst[i] += scale * src[i] in one
                    // kernel instead of two passes.
                    const float combined = routerWeight * expDownScalePtr[e];
                    _ops.scaledAddResidualAsync(accumT, expertOutBuf, combined,
                                                d_model);
                }
            }
        }
    }

    _op.mark(runtime::OpProfiler::Cat::NORM);
    trace("path B: post_ffw_norm_2");
    _ops.rmsNormAsync(moeAccumBuf, T, d_model,
                      static_cast<const float*>(postNorm2->usmPtr),
                      _config.rmsNormEps,
                      moeAccumBuf);
    dumpStage("ffn_moe", blockIdx, moeAccumBuf, T, d_model);

    _op.mark(runtime::OpProfiler::Cat::RESIDUAL);
    trace("combined = pathA_out + pathB_out");
    _ops.addResidualAsync(moeAccumBuf, projOutBuf, T * d_model);
    dumpStage("ffn_moe_combined", blockIdx, moeAccumBuf, T, d_model);

    _op.mark(runtime::OpProfiler::Cat::NORM);
    trace("post_ffw_norm (combined)");
    _ops.rmsNormAsync(moeAccumBuf, T, d_model,
                      static_cast<const float*>(ffwPost->usmPtr),
                      _config.rmsNormEps,
                      moeAccumBuf);
    dumpStage("ffn_post_norm", blockIdx, moeAccumBuf, T, d_model);

    _op.mark(runtime::OpProfiler::Cat::RESIDUAL);
    trace("ffn residual");
    _ops.addResidualAsync(x, moeAccumBuf, T * d_model);

    const float scaleVal = *static_cast<const float*>(outScale->usmPtr);
    _op.mark(runtime::OpProfiler::Cat::ACTIVATION);
    trace("layer_output_scale");
    _ops.mulScalarAsync(x, scaleVal, T * d_model);
    dumpStage("out_scaled", blockIdx, x, T, d_model);
    dumpStage("l_out",      blockIdx, x, T, d_model);

    // Close the last phase before returning so its time lands in the
    // accumulator. Cheap no-op when profiling is disabled.
    _op.finish();
}

} // namespace mimirmind::runtime::arch