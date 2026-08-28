// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/arch/Gemma4MoeBackend.hpp"

#include "compute/ComputeMatmul.hpp"
#include "compute/ComputeOps.hpp"
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
#include "runtime/perf/OpProfiler.hpp"

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
                                   compute::ComputeOps&               ops,
                                   compute::ComputeMatmul&            gmm,
                                   runtime::OpProfiler&           opProfiler,
                                   bool                           moeGroupEnabled,
                                   bool                           moeFusedDownEnabled)
    : GemmaBaseBackend{config, weights, fusedQkv, ops, gmm, opProfiler},
      _moeGroupEnabled{moeGroupEnabled},
      _moeFusedDownEnabled{moeFusedDownEnabled} {
    // Size the per-expert grouping bucket once — the count is fixed for
    // the model's lifetime, so runBlock() only ever clear()s the inner
    // vectors (retaining their capacity) instead of reallocating 128 of
    // them each prefill pass.
    _expertTokens.resize(_config.expertCount);
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
    const bool diag = (blockIdx == 0 && cache.length() == 0 && traceBlock0);
    auto trace = [&](const char* tag) {
        if (diag) MM_LOG_INFO("blkdiag-g4m", "blk0 {}", tag);
    };
    trace("enter (moe)");
    _ops.profileSection("g4.attn");   // MIMIRMIND_DECODE_PROFILE section

    // Shared attention section. On return `x` holds
    // sa_out = inpL + post_attention_norm(W_o @ attn(...)).
    runAttentionSection(blockIdx, x, T, cache, s, diag);

    // FFN/MoE tail — position-independent, shared verbatim with the
    // batched decode path (runBlockBatched calls this with T = nSeq).
    // Everything from ffn_norm onward depends only on the row count T,
    // never on sequence position.
    runFfnMoeSection(blockIdx, x, T, s, diag);
}

void Gemma4MoeBackend::runFfnMoeSection(std::size_t   blockIdx,
                                        float*        x,
                                        std::size_t   T,
                                        BlockBuffers& s,
                                        bool          diag) {
    namespace cmp = mimirmind::compute;

    auto trace = [&](const char* tag) {
        if (diag) MM_LOG_INFO("blkdiag-g4m", "blk0 {}", tag);
    };

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
    _ops.profileSection("g4.pathA");   // MIMIRMIND_DECODE_PROFILE section
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
        compute::UnorderedScope u{_ops};
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
    _ops.profileSection("g4.moe.route");   // MIMIRMIND_DECODE_PROFILE section
    trace("path B: pre_ffw_norm_2 + router rmsNorm (unordered)");
    {
        compute::UnorderedScope u{_ops};
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

    // Reused scratch — resize() retains capacity, so no allocation in
    // steady state. moeTopKRoute writes all T*K entries, so the
    // (uninitialised) grown tail is fully overwritten before any read.
    _topKIdx.resize(T * K);
    _topKWeight.resize(T * K);

    // M-CLR.MoE Increment 1 — opt-in device-side top-K routing. The
    // kernel is algebraically identical to cmp::moeTopKRoute (wScale=1;
    // the global softmax denominator cancels against the kept-mass
    // renormaliser). We copy the result back into the host pick buffers,
    // so the downstream expert-dispatch path is byte-for-byte unchanged —
    // this increment only establishes routing parity ahead of the
    // device-side dispatch that will let MoE run under L0-CLR. Default
    // off; any launch failure falls back to the host path silently.
    static const bool kDeviceTopK =
        std::getenv("MIMIRMIND_MOE_DEVICE_TOPK") != nullptr;

    // M-CLR.MoE Increment 2 — can the entire expert dispatch run
    // device-side this block? When yes, the routing never round-trips to
    // the host: the device top-K result feeds the device gate_up (Q6_K) +
    // fused-K down kernels directly, so no host op reads _topKIdx (and the
    // memcpy-back below is skipped). That is the precondition for
    // Command-List-Replay capture of the MoE decode block. Requires the
    // fused-decode preconditions plus both device kernels for these
    // weight types; otherwise fall back to the host dispatch path.
    // MIMIRMIND_MOE_DEVICE_TOPK is the master switch for the whole
    // device-MoE decode feature, so the device dispatch rides on kernel
    // availability alone — it does not require the separate host-side
    // features.moeFusedDown config toggle (that governs the host fused-K
    // path). The fused-K down kernels are loaded unconditionally by
    // GpuMatmul, so moeDownFusedKAvailable() reflects driver support.
    const bool wantDeviceDispatch =
        kDeviceTopK &&
        T == 1 &&
        _gmm.moeDownFusedKAvailable(expDown->type) &&
        expGateUp->type == core::gguf::GgmlType::Q6_K &&
        _ops.moeGateUpFusedKGeluAvailable(d_model) &&
        s.moeGateCompact.get() != nullptr;

    bool deviceTopKDone = false;
    if (kDeviceTopK) {
        try {
            const std::size_t need = T * K;
            if (_devTopKIdx.bytes() < need * sizeof(std::int32_t)) {
                _devTopKIdx = _ops.allocate(need * sizeof(std::int32_t));
            }
            if (_devTopKWeight.bytes() < need * sizeof(float)) {
                _devTopKWeight = _ops.allocate(need * sizeof(float));
            }
            _ops.moeTopKRouteDeviceAsync(upOutBuf,
                                         _devTopKIdx.as<std::int32_t>(),
                                         _devTopKWeight.as<float>(),
                                         T, nExperts, K, 1.0F);
            // The host pick buffers are only read when the dispatch stays
            // on the host (prefill grouping, or a missing device kernel).
            // Under full device dispatch we skip the flush + copy so that
            // no host op touches the routing — the CLR-capture win. The
            // device kernels read _devTopKIdx/_devTopKWeight directly and
            // are ordered after the top-K launch on the same queue.
            if (!wantDeviceDispatch) {
                _ops.flush();
                std::memcpy(_topKIdx.data(), _devTopKIdx.as<std::int32_t>(),
                            need * sizeof(std::int32_t));
                std::memcpy(_topKWeight.data(), _devTopKWeight.as<float>(),
                            need * sizeof(float));
            }
            deviceTopKDone = true;
            static bool announced = false;
            if (!announced) {
                announced = true;
                MM_LOG_INFO("gemma4moe",
                            "MoE device top-K routing active "
                            "(MIMIRMIND_MOE_DEVICE_TOPK)");
            }
        } catch (const std::exception&) {
            trace("path B: device top-K unavailable, host fallback");
        }
    }
    if (!deviceTopKDone) {
        cmp::moeTopKRoute(upOutBuf, T, nExperts, K,
                          _topKIdx.data(), _topKWeight.data());
    }
    // Only take the device dispatch path if the device top-K actually ran
    // (a launch failure above leaves the host _topKIdx authoritative).
    const bool useDeviceDispatch = wantDeviceDispatch && deviceTopKDone;

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
        trace("path B (device-grouped): build + gather + per-expert GEMM + scatter");

        // Half-of-fused byte offset — points at the "up" rows in each
        // per-expert gate_up_exps.weight block.
        const std::size_t gateBytesHalf = ffPerExpert *
            (d_model / qtGateUp->blockElements()) * qtGateUp->blockBytes();

        // M-Gemma4MoE.Prefill: device-driven grouped dispatch. The host
        // permutation + host gather (a _gmm.sync() + ~T*K*d_model float
        // memcpy) + host scatter (T*K per-row kernel launches) were the
        // dominant prefill cost (host-driven grouped-MoE loses to fused-K
        // on GB10). Replaced with the device moeGroupBuild / moeGatherRows
        // / moeScatterExpertOut ops (mirrors Qwen3_5MoeBackend). Only the
        // per-expert launch bounds (nExperts+1 ints) cross to the host.
        const std::size_t nRows = T * K;   // one compact row per assignment

        auto growBuf = [&](compute::ComputeBuffer& b, std::size_t bytes) {
            if (b.bytes() < bytes) b = _ops.allocate(bytes);
        };
        growBuf(_grpExpIdx,    nRows * sizeof(std::int32_t));
        growBuf(_grpKw,        nRows * sizeof(float));
        growBuf(_grpExpOffset, (nExperts + 1) * sizeof(std::int32_t));
        growBuf(_grpRowSrcTok, nRows * sizeof(std::int32_t));
        growBuf(_grpRowKw,     nRows * sizeof(float));
        growBuf(_grpAsnToRow,  nRows * sizeof(std::int32_t));

        // Routing (expIdx/kw) on the device: reuse the device top-K result
        // if it ran, else upload the host pick buffers (small: T*K each).
        const std::int32_t* expIdxDev;
        const float*        kwDev;
        if (deviceTopKDone) {
            expIdxDev = _devTopKIdx.as<std::int32_t>();
            kwDev     = _devTopKWeight.as<float>();
        } else {
            _ops.uploadToDevice(_grpExpIdx.as<std::int32_t>(), _topKIdx.data(),
                                nRows * sizeof(std::int32_t));
            _ops.uploadToDevice(_grpKw.as<float>(), _topKWeight.data(),
                                nRows * sizeof(float));
            expIdxDev = _grpExpIdx.as<std::int32_t>();
            kwDev     = _grpKw.as<float>();
        }

        // Device counting-sort: group the T*K assignments by expert →
        // expOffset (prefix sum), rowSrcTok (compact-row → token), rowKw,
        // asnToRow (assignment → compact-row).
        _ops.profileSection("g4.moe.build");   // MIMIRMIND_DECODE_PROFILE section
        _ops.moeGroupBuildAsync(expIdxDev, kwDev,
                                _grpExpOffset.as<std::int32_t>(),
                                _grpRowSrcTok.as<std::int32_t>(),
                                _grpRowKw.as<float>(),
                                _grpAsnToRow.as<std::int32_t>(),
                                nRows, nExperts, K);

        float* const xComp    = s.moeXCompact.as<float>();
        float* const gateComp = s.moeGateCompact.as<float>();
        float* const upComp   = s.moeUpCompact.as<float>();
        float* const downComp = s.moeDownCompact.as<float>();

        // Device gather activations into per-expert-contiguous rows.
        _ops.profileSection("g4.moe.gather");   // MIMIRMIND_DECODE_PROFILE section
        _ops.moeGatherRowsAsync(normBuf, _grpRowSrcTok.as<std::int32_t>(),
                                xComp, d_model, nRows);

        // The ONE small D2H per MoE layer: per-expert launch bounds.
        _grpOffsetHost.resize(nExperts + 1);
        _ops.flush();
        _ops.readbackToHost(_grpOffsetHost.data(),
                            _grpExpOffset.as<std::int32_t>(),
                            (nExperts + 1) * sizeof(std::int32_t));

        // Zero the accumulator (the scatter sums each token's K rows in).
        _ops.mulScalarAsync(moeAccumBuf, 0.0F, T * d_model);

        _ops.profileSection("g4.moe.gemm");   // MIMIRMIND_DECODE_PROFILE section
        trace("path B (device-grouped): per-expert batched matmuls");
        for (std::size_t e = 0; e < nExperts; ++e) {
            const std::int32_t off = _grpOffsetHost[e];
            const std::int32_t end = _grpOffsetHost[e + 1];
            const std::size_t  Me  = static_cast<std::size_t>(end - off);
            if (Me == 0) continue;
            const std::size_t offRows = static_cast<std::size_t>(off);

            const auto* Wgu =
                static_cast<const std::uint8_t*>(expGateUpBase) +
                e * expertBytesGateUp;
            const void* Wd  = expDownBase + e * expertBytesDown;

            // Gate rows: first half of the fused gate_up weight block.
            _gmm.matmulAsync(expGateUp->type, Wgu,
                             ffPerExpert, d_model,
                             xComp + offRows * d_model, Me,
                             gateComp + offRows * ffPerExpert,
                             matmulScratch);
            // Up rows: second half, offset in bytes.
            _gmm.matmulAsync(expGateUp->type, Wgu + gateBytesHalf,
                             ffPerExpert, d_model,
                             xComp + offRows * d_model, Me,
                             upComp + offRows * ffPerExpert,
                             matmulScratch);

            // gelu(gate) * up, in place into gateComp region.
            _ops.geluMulAsync(gateComp + offRows * ffPerExpert,
                              upComp   + offRows * ffPerExpert,
                              Me * ffPerExpert);

            // Down: gate_activated @ W_d[e]  →  downComp region.
            _gmm.matmulAsync(expDown->type, Wd,
                             d_model, ffPerExpert,
                             gateComp + offRows * ffPerExpert, Me,
                             downComp + offRows * d_model,
                             matmulScratch);

            // Fold the per-expert down scale into the grouped output so the
            // device scatter below can use the plain router weight (kwDev).
            // expDownScalePtr is USM (host-readable on the integrated GB10).
            _ops.mulScalarAsync(downComp + offRows * d_model,
                                expDownScalePtr[e], Me * d_model);
        }

        // Device scatter-accumulate over the T*K assignments:
        //   accum[token(a)] += kw[a] * downComp[asnToRow[a]]
        // (the per-expert down scale is already folded into downComp).
        _ops.profileSection("g4.moe.scatter");   // MIMIRMIND_DECODE_PROFILE section
        trace("path B (device-grouped): device scatter-accumulate");
        _ops.moeScatterExpertOutAsync(downComp, _grpAsnToRow.as<std::int32_t>(),
                                      kwDev, moeAccumBuf, d_model, T, K);
    } else if (useDeviceDispatch) {
        // M-CLR.MoE Increment 2 — fully device-side expert dispatch for
        // T=1 decode. The device gate_up kernel reads the router pick
        // expIdx[k] from _devTopKIdx (no host read) and folds downScale[e]
        // into the gate activation; the fused-K down kernel then takes the
        // raw router weight (_devTopKWeight) as kw. No host op touches the
        // routing between top-K and the accumulator, so the whole block is
        // Command-List-Replay-capturable (Increment 3).
        trace("path B: device expert dispatch (gate_up + fused-K down)");
        static bool dispatchAnnounced = false;
        if (!dispatchAnnounced) {
            dispatchAnnounced = true;
            MM_LOG_INFO("gemma4moe",
                        "MoE device expert dispatch active — gate_up "
                        "(Q6_K) + fused-K down, no host routing read");
        }
        float* const gateActAll = s.moeGateCompact.as<float>();
        const auto* const devIdx = _devTopKIdx.as<std::int32_t>();

        // gate_up: gateActAll[k, f] = gelu(gate_f · x) * (up_f · x) * scale[e]
        _ops.moeGateUpFusedKGeluAsync(normBuf, expGateUpBase, devIdx,
                                      expDownScalePtr, gateActAll,
                                      d_model, ffPerExpert, K,
                                      expertBytesGateUp);

        // down: moeAccumBuf[n] += sum_k weight[k] * (W_down[e_k] · gateAct_k)
        _gmm.moeDownFusedKAsync(expDown->type, gateActAll, expDownBase,
                                devIdx, _devTopKWeight.as<float>(),
                                moeAccumBuf, ffPerExpert, d_model, K,
                                expertBytesDown);
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

        // llama.cpp-style warp32 MMVQ (int8 dp4a, per-32-block quantised
        // activation) for the per-token expert loop — opt-in, GB10/CUDA
        // only. Measured moe.expertloop at only ~7-8% of the 273 GB/s
        // roofline for the plain fp32 vec path (see lesson
        // moe-fuseddown-toggle-neutral-gb10-2026-08-28); this replaces
        // BOTH the gate_up (Q6_K) and down (Q8_0) matmul with the DP4A
        // warp32 vec kernels instead. Lossy (int8 activation quant) —
        // unlike every other path in this loop, output is not expected
        // to be bit-identical to the fp32 baseline. Requires K dims that
        // satisfy both kernels' block guards (Q6_K needs d_model % 256,
        // Q8_0/DP4A activation blocks need ffPerExpert % 32) so a future
        // model with incompatible dims falls back silently instead of
        // throwing from inside matmulDp4aAsync.
        static const bool kExpertLoopDp4a =
            std::getenv("MIMIRMIND_MOE_EXPERTLOOP_DP4A") != nullptr;
        const bool useExpertLoopDp4a =
            kExpertLoopDp4a &&
            T == 1 &&
            d_model % 256 == 0 &&
            ffPerExpert % 32 == 0 &&
            _gmm.dp4aAvailable(expGateUp->type) &&
            _gmm.dp4aAvailable(expDown->type) &&
            s.moeXq8GateUp.get()      != nullptr &&
            s.moeXq8GateUpScale.get() != nullptr &&
            s.moeXq8Down.get()        != nullptr &&
            s.moeXq8DownScale.get()   != nullptr;

        if (useExpertLoopDp4a) {
            _ops.profileSection("g4.moe.expertloop");   // MIMIRMIND_DECODE_PROFILE section
            trace("path B: per-token expert dispatch (DP4A warp32)");
            static bool dp4aAnnounced = false;
            if (!dp4aAnnounced) {
                dp4aAnnounced = true;
                MM_LOG_INFO("gemma4moe",
                            "MoE expert-loop DP4A warp32 active "
                            "(MIMIRMIND_MOE_EXPERTLOOP_DP4A) — lossy int8 "
                            "activation quant, not bit-identical to fp32");
            }

            float* const pathBIn = normBuf;         // T == 1
            float* const accumT  = moeAccumBuf;

            auto* const xq8Gate      = s.moeXq8GateUp.as<std::int8_t>();
            auto* const xq8GateScale = s.moeXq8GateUpScale.as<float>();
            auto* const xq8Down      = s.moeXq8Down.as<std::int8_t>();
            auto* const xq8DownScale = s.moeXq8DownScale.as<float>();

            // Shared across all K experts (identical input) — quantised
            // once per token instead of once per expert.
            _ops.xQuantQ8_1BlocksAsync(pathBIn, xq8Gate, xq8GateScale,
                                       d_model);

            for (std::size_t k = 0; k < K; ++k) {
                const std::size_t e =
                    static_cast<std::size_t>(_topKIdx[k]);
                const float routerWeight = _topKWeight[k];

                const void* Wgu = expGateUpBase + e * expertBytesGateUp;
                const void* Wd  = expDownBase   + e * expertBytesDown;

                _ops.profileSection("g4.moe.gateup");   // MIMIRMIND_DECODE_PROFILE section
                _gmm.matmulDp4aAsync(expGateUp->type, xq8Gate, xq8GateScale,
                                     Wgu, gateUpFused, d_model, 1,
                                     gateOutBuf);

                _ops.geluMulAsync(gateOutBuf, gateOutBuf + ffPerExpert,
                                  ffPerExpert);

                // Per-expert activation (differs each k) — quantised
                // fresh before the down DP4A matmul.
                _ops.xQuantQ8_1BlocksAsync(gateOutBuf, xq8Down, xq8DownScale,
                                           ffPerExpert);

                _ops.profileSection("g4.moe.down");   // MIMIRMIND_DECODE_PROFILE section
                _gmm.matmulDp4aAsync(expDown->type, xq8Down, xq8DownScale,
                                     Wd, d_model, ffPerExpert, 1,
                                     expertOutBuf);

                const float combined = routerWeight * expDownScalePtr[e];
                _ops.scaledAddResidualAsync(accumT, expertOutBuf, combined,
                                            d_model);
            }
        } else if (useMoeFusedDown) {
            trace("path B: per-token expert dispatch (fused-K down)");
            float* const pathBIn = normBuf;         // T == 1
            float* const accumT  = moeAccumBuf;
            float* const gateActAll = s.moeGateCompact.as<float>();

            // Per-expert gate_up + gelu_mul into strided slots — kept
            // separate so the fused down kernel sees [K, ffPer].
            for (std::size_t k = 0; k < K; ++k) {
                const std::size_t e =
                    static_cast<std::size_t>(_topKIdx[k]);
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
                _ops.appendMemoryCopy(
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
                    static_cast<std::size_t>(_topKIdx[k]);
                expIdxSlot[k] = static_cast<std::int32_t>(e);
                kwSlot[k]     = _topKWeight[k] * expDownScalePtr[e];
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
            _ops.profileSection("g4.moe.expertloop");   // MIMIRMIND_DECODE_PROFILE section
            trace("path B: per-token expert dispatch");
            for (std::size_t t = 0; t < T; ++t) {
                float* const pathBInT  = normBuf      + t * d_model;
                float* const accumT    = moeAccumBuf  + t * d_model;
                for (std::size_t k = 0; k < K; ++k) {
                    const std::size_t e =
                        static_cast<std::size_t>(_topKIdx[t * K + k]);
                    const float       routerWeight = _topKWeight[t * K + k];

                    const void* Wgu = expGateUpBase + e * expertBytesGateUp;
                    const void* Wd  = expDownBase   + e * expertBytesDown;

                    _ops.profileSection("g4.moe.gateup");   // MIMIRMIND_DECODE_PROFILE section
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
                    _ops.profileSection("g4.moe.down");   // MIMIRMIND_DECODE_PROFILE section
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
    _ops.profileSection("g4.combine");   // MIMIRMIND_DECODE_PROFILE section
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

    // MIMIRMIND_DECODE_PROFILE: dump the g4.* section breakdown once per
    // forward. This backend is served single-session (co-resident engine),
    // so ServingSession's per-step profileStepEnd never fires for it —
    // trigger it here on the last block. No-op when profiling is disabled.
    if (blockIdx + 1 == _config.blockCount) {
        _ops.profileStepEnd();
    }
}

void Gemma4MoeBackend::runBlockBatched(std::size_t                blockIdx,
                                       float*                     x,
                                       std::size_t                nSeq,
                                       std::span<KvCache* const>  caches,
                                       BlockBuffers&              s,
                                       bool                       diag) {
    // Batched attention section (per-seq caches), then the shared FFN/MoE
    // tail at T=nSeq. The tail's T==1 fast paths (device dispatch, fused-K
    // down) stay off for nSeq>1 — it falls to the expert-grouping /
    // per-token dispatch, which is correct for a batch of rows.
    runAttentionSectionBatched(blockIdx, x, nSeq, caches, s, diag);
    runFfnMoeSection(blockIdx, x, nSeq, s, diag);
}

bool Gemma4MoeBackend::moeDecodeClrSafe() const noexcept {
    // M-CLR.MoE Increment 3 — CLR-safe when the whole decode block runs
    // device-side expert dispatch (Increment 2), so no host op reads the
    // routing between the router matmul and the accumulator. Verified on
    // NUC / Xe-LPG (26B-A4B-it-Q6_K): with device dispatch + the two CLR
    // record fixes this depends on (the F32 GPU router matmul so the router
    // is recorded instead of a stale host fallback — see
    // kernels/matmul_f32_vec.cl — and the replay-path embedding-scale flush
    // in InferenceEngine), record/replay output is bit-identical to
    // immediate mode. Gated behind MIMIRMIND_MOE_DEVICE_TOPK (the device
    // dispatch master switch) + kernel availability, evaluated per block so
    // no layer can silently fall back to the host routing path mid-replay.
    static const bool kDeviceTopK =
        std::getenv("MIMIRMIND_MOE_DEVICE_TOPK") != nullptr;
    if (!kDeviceTopK ||
        !_ops.moeGateUpFusedKGeluAvailable(_config.embeddingLength)) {
        return false;
    }
    try {
        for (std::size_t b = 0; b < _config.blockCount; ++b) {
            const auto* eg =
                requireTensor(b, "ffn_gate_up_exps.weight", "Gemma4MoeBackend");
            const auto* ed =
                requireTensor(b, "ffn_down_exps.weight", "Gemma4MoeBackend");
            if (eg->type != core::gguf::GgmlType::Q6_K ||
                !_gmm.moeDownFusedKAvailable(ed->type)) {
                return false;
            }
        }
    } catch (...) {
        return false;
    }
    return true;
}

} // namespace mimirmind::runtime::arch