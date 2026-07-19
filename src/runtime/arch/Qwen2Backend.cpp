// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/arch/Qwen2Backend.hpp"

#include "compute/ComputeMatmul.hpp"
#include "compute/ComputeOps.hpp"
#include "model/FusedQkvWeights.hpp"
#include "core/gguf/GgufTypes.hpp"
#include "model/LlmConfig.hpp"
#include "core/gguf/WeightsMap.hpp"
#include "runtime/BlockBuffers.hpp"
#include "runtime/KvCache.hpp"
#include "core/log/Log.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace mimirmind::runtime::arch {

Qwen2Backend::Qwen2Backend(const model::LlmConfig&        config,
                           const core::gguf::WeightsMap&       weights,
                           const model::FusedQkvWeights*  fusedQkv,
                           compute::ComputeOps&               ops,
                           compute::ComputeMatmul&            gmm,
                           runtime::OpProfiler&           opProfiler)
    : _config{config}, _weights{weights}, _fusedQkv{fusedQkv},
      _ops{ops}, _gmm{gmm}, _op{opProfiler} {
    MM_LOG_INFO("qwen2", "Qwen2Backend ready — blocks={} d_model={} ff={} "
                         "heads={} kv={}",
                _config.blockCount, _config.embeddingLength,
                _config.feedForwardLength, _config.headCount,
                _config.headCountKv);
}

std::vector<std::size_t> Qwen2Backend::kvDimPerLayer() const {
    const std::size_t kvDim = _config.headCountKv * _config.headDim();
    return std::vector<std::size_t>(_config.blockCount, kvDim);
}

std::pair<std::size_t, std::size_t> Qwen2Backend::maxQKVDims() const {
    const std::size_t qDim  = _config.headCount   * _config.headDim();
    const std::size_t kvDim = _config.headCountKv * _config.headDim();
    return {qDim, kvDim};
}

void Qwen2Backend::runBlock(std::size_t   blockIdx,
                            float*        x,
                            std::size_t   T,
                            KvCache&      cache,
                            BlockBuffers& s,
                            bool          traceBlock0) {
    const bool diag = (blockIdx == 0 && cache.length() == 0 && traceBlock0);
    auto trace = [&](const char* tag) {
        if (diag) {
            MM_LOG_INFO("blkdiag", "blk0 {}", tag);
        }
    };

    trace("enter");

    const auto& w = _weights;
    const auto* attnNorm = w.findBlock(blockIdx, "attn_norm.weight");
    const auto* qW       = w.findBlock(blockIdx, "attn_q.weight");
    const auto* qB       = w.findBlock(blockIdx, "attn_q.bias");
    const auto* kW       = w.findBlock(blockIdx, "attn_k.weight");
    const auto* kB       = w.findBlock(blockIdx, "attn_k.bias");
    const auto* vW       = w.findBlock(blockIdx, "attn_v.weight");
    const auto* vB       = w.findBlock(blockIdx, "attn_v.bias");
    const auto* oW       = w.findBlock(blockIdx, "attn_output.weight");

    const auto* ffnNorm = w.findBlock(blockIdx, "ffn_norm.weight");
    const auto* ffnGate = w.findBlock(blockIdx, "ffn_gate.weight");
    const auto* ffnUp   = w.findBlock(blockIdx, "ffn_up.weight");
    const auto* ffnDown = w.findBlock(blockIdx, "ffn_down.weight");

    if (diag) {
        auto t = [](const core::gguf::GgufTensor* p) {
            return p == nullptr ? "MISSING" : core::gguf::typeInfo(p->type).name.data();
        };
        MM_LOG_INFO("blkdiag",
                    "blk0 lookups: attn_norm={} qW={} kW={} vW={} oW={} "
                    "ffn_norm={} ffn_gate={} ffn_up={} ffn_down={} qB={} kB={} vB={}",
                    t(attnNorm), t(qW), t(kW), t(vW), t(oW),
                    t(ffnNorm), t(ffnGate), t(ffnUp), t(ffnDown),
                    t(qB), t(kB), t(vB));
    }

    if (attnNorm == nullptr || qW == nullptr || kW == nullptr ||
        vW == nullptr || oW == nullptr ||
        ffnNorm == nullptr || ffnGate == nullptr ||
        ffnUp == nullptr || ffnDown == nullptr) {
        throw std::runtime_error(
            "Qwen2Backend: transformer block " + std::to_string(blockIdx) +
            " missing a tensor");
    }

    const std::size_t d_model  = s.d_model;
    const std::size_t q_dim    = s.q_dim;
    const std::size_t kv_dim   = _config.headCountKv * _config.headDim();
    const std::size_t ff_dim   = s.ff_dim;
    const std::size_t head_dim = _config.headDim();
    const std::size_t curLen   = cache.length();
    const std::size_t totalLen = curLen + T;

    float* const normBuf       = s.normBuf.as<float>();
    float* const qBuf          = s.qBuf.as<float>();
    float* const attnOutBuf    = s.attnOut.as<float>();
    float* const projOutBuf    = s.projOut.as<float>();
    float* const gateOutBuf    = s.gateOut.as<float>();
    float* const upOutBuf      = s.upOut.as<float>();
    float* const matmulScratch = s.matmulScratch.as<float>();
    // scoreScratch was the CPU-attention softmax row buffer; the GPU
    // attention kernel keeps the score row in SLM, so it's unused here.
    (void)s.scoreScratch;

    trace("attn rmsNorm");
    _ops.rmsNormAsync(x, T, d_model,
                      static_cast<const float*>(attnNorm->usmPtr),
                      _config.rmsNormEps,
                      normBuf);

    auto projectAsync = [&](const core::gguf::GgufTensor* W,
                            std::size_t N, float* dst) {
        _gmm.matmulAsync(W->type, W->usmPtr, N, d_model,
                         normBuf, T, dst, matmulScratch);
    };
    auto addBiasIf = [&](const core::gguf::GgufTensor* B,
                         std::size_t N, float* dst) {
        if (B != nullptr && B->type == core::gguf::GgmlType::F32) {
            _ops.addBiasAsync(dst, T, N,
                              static_cast<const float*>(B->usmPtr));
        }
    };

    // M10.2 Commit 5: KV slots are typed void* so the same call sites
    // service both f32 and fp16 storage; ops methods branch internally
    // on `cache.dtype()`.
    void* kSlot = cache.writeSlotK(blockIdx);
    void* vSlot = cache.writeSlotV(blockIdx);
    // M-CLR.2 Wave 3: qkv_split receives the cache BASE + curLen so its
    // pointer arg is stable across replays. Other kernels here still use
    // the per-token slot; refactoring rope-K is Wave 3b per the ADR.
    void* kBase = const_cast<void*>(cache.baseK(blockIdx));
    void* vBase = const_cast<void*>(cache.baseV(blockIdx));
    const auto kvDtype = cache.dtype();

    // M10.2 Phase 1a Commit 5: Q8_0 KV routes rmsnorm_qkv + RoPE through
    // an fp32 workspace, then folds the results into 32-elem Q8_0 blocks
    // via `kv_quant_commit_q8_0`. The workspace pointers replace kBase /
    // vBase for the pre-quantise pipeline; the actual cache slots stay
    // Q8_0-typed and are only written by the commit kernel below.
    const bool q8Path = (kvDtype == KvDtype::Q8_0);
    float* const kFp32Scratch = q8Path ? s.kvKFp32Scratch.as<float>() : nullptr;
    float* const vFp32Scratch = q8Path ? s.kvVFp32Scratch.as<float>() : nullptr;
    void* const kStagingBase  = q8Path
        ? static_cast<void*>(kFp32Scratch)
        : kBase;
    void* const vStagingBase  = q8Path
        ? static_cast<void*>(vFp32Scratch)
        : vBase;
    // Under Q8_0 the staging buffers hold exactly the T new rows starting
    // at row 0 (no history to skip), and every write kernel that consumes
    // them must be told KvDtype::F32 + writeOffset=0. The `curLen` value
    // is still passed to K-RoPE via startPos for correct positional
    // angles — only the *destination* pointer arithmetic changes.
    const auto stagingKvDtype  = q8Path ? KvDtype::F32 : kvDtype;
    const std::size_t stagingWriteOffset = q8Path ? 0 : curLen;
    const std::size_t stagingWriteStride = q8Path ? 0 : kv_dim;

    // M5i.B: Fused Q+K+V — single matmul into a staging buffer, then a
    // scatter kernel routes the sub-ranges into qBuf/kSlot/vSlot. Bias
    // adds run on the split outputs as usual, no change there.
    const model::FusedQkvWeights::Block* fBlk =
        (_fusedQkv != nullptr) ? _fusedQkv->find(blockIdx) : nullptr;

    if (fBlk != nullptr) {
        trace("Q+K+V projections (fused matmul + split)");
        float* const qkvFused = s.qkvFusedScratch.as<float>();
        const std::size_t Nfused =
            fBlk->Nq + fBlk->Nkv * (fBlk->hasV ? 2 : 1);
        _gmm.matmulAsync(fBlk->type, fBlk->usmPtr, Nfused, d_model,
                         normBuf, T, qkvFused, matmulScratch);
        // M10.2 Phase 1a Commit 5: Q8_0 scatters K/V into the fp32
        // staging buffers (writeOffset=0, dtype=F32). The subsequent
        // rope-K + kv_quant_commit_q8_0 sequence walks the staging
        // buffers and finalises the current-row Q8_0 blocks into the
        // cache slot.
        _ops.qkvSplitAsync(qkvFused, qBuf, kStagingBase, vStagingBase,
                           T, fBlk->Nq, fBlk->Nkv, fBlk->hasV,
                           stagingWriteOffset,
                           stagingKvDtype,
                           /*useStagingSlot=*/q8Path);
    } else {
        // M5f.4: Q/K/V projections write disjoint buffers and depend only
        // on normBuf (already published by the rmsnorm above). They can
        // pipeline freely inside the unordered scope; the scope's pop
        // emits a single barrier that the bias adds + RoPE read against.
        // M10.2 Commit 5: matmul writes fp32 directly into K/V slots —
        // only correct when the KV cache is fp32. fp16-KV / Q8_0-KV
        // models must ship fused-QKV weights (the engine-side guard in
        // Commit 8 / Phase 1a Commit 2 enforces this); the K/V write
        // destination below is the fp32 staging under Q8_0 so a future
        // guard bypass corrupts the *staging* rather than the packed
        // cache slot.
        trace("Q+K+V projections (matmulAsync, unordered)");
        {
            compute::UnorderedScope u{_ops};
            projectAsync(qW, q_dim,  qBuf);
            projectAsync(kW, kv_dim,
                         q8Path ? kFp32Scratch
                                : static_cast<float*>(kSlot));
            projectAsync(vW, kv_dim,
                         q8Path ? vFp32Scratch
                                : static_cast<float*>(vSlot));
        }
    }

    // QKV bias adds — each adds to its own buffer, independent of the
    // others. The bias add does read its own projection output, so
    // this scope's preceding barrier (from the Q/K/V projection pop
    // above) is what makes the chain safe.
    // M10.2 Commit 5: addBiasAsync writes fp32; on non-F32-KV models the
    // K/V bias tensors must be null (Qwen 2.5+ style, Gemma family) —
    // the engine-side setKvDtype guard rejects any block that carries
    // attn_k/v.bias so this branch is safe. Under Q8_0 the K/V bias
    // destinations point at the fp32 staging (kFp32Scratch/vFp32Scratch)
    // for the same defence-in-depth reason as the projection above.
    trace("QKV bias add (async, unordered)");
    {
        compute::UnorderedScope u{_ops};
        addBiasIf(qB, q_dim,  qBuf);
        addBiasIf(kB, kv_dim,
                  q8Path ? kFp32Scratch
                         : static_cast<float*>(kSlot));
        addBiasIf(vB, kv_dim,
                  q8Path ? vFp32Scratch
                         : static_cast<float*>(vSlot));
    }

    // RoPE on Q and K — independent buffers, no V-dependency.
    // M-CLR.2 Wave 3b: K-rope uses cache BASE + kv_dim stride so the
    // kernel writes into the current slot at startPos*kv_dim internally.
    // M10.2 Commit 5: K-rope dispatches through the fp16-aware variant
    // when the cache is fp16 (rotation stays fp32 in registers, only
    // load/store change). Q-rope always targets the fp32 workspace, so
    // it explicitly passes KvDtype::F32.
    // M10.2 Phase 1a Commit 5: under Q8_0 K-rope targets the fp32
    // staging (kFp32Scratch, T rows starting at row 0). startPos still
    // carries `curLen` for correct positional angles, but the write-
    // offset stride is 0 because there's no history in front of row 0.
    trace("RoPE Q+K (async, unordered)");
    {
        compute::UnorderedScope u{_ops};
        _ops.ropeInPlaceAsync(qBuf, T,
                              _config.headCount,   head_dim, curLen,
                              _config.ropeFreqBase);
        _ops.ropeInPlaceAsync(kStagingBase, T,
                              _config.headCountKv, head_dim, curLen,
                              _config.ropeFreqBase, stagingWriteStride,
                              stagingKvDtype);
    }

    // M10.2 Phase 1a Commit 5: fold the fp32 K/V staging rows into 32-
    // element Q8_0 blocks inside the actual cache slots. Two dispatches
    // because the layout matches [T, kv_dim] fp32 → [T, kv_dim/32, 34 B]
    // Q8_0. Same immediate/replay semantics as ropeInPlaceAsync — the
    // shared curLenSlot() drives the row offset.
    if (q8Path) {
        trace("KV commit Q8_0 (K + V)");
        _ops.kvQuantCommitQ8Async(kFp32Scratch,
                                  static_cast<void*>(kBase),
                                  T, kv_dim, curLen);
        _ops.kvQuantCommitQ8Async(vFp32Scratch,
                                  static_cast<void*>(vBase),
                                  T, kv_dim, curLen);
    }

    // M5f.3: attention is on the GPU. No sync needed before it — the
    // command-list barriers chain RoPE → attention. O-projection
    // (synchronous matmul below) flushes the queue so attnOutBuf is
    // visible to it by the time it executes.
    trace("attention (GPU)");
    const float attnScale = 1.0F /
        std::sqrt(static_cast<float>(head_dim));
    _ops.attentionAsync(qBuf,
                        cache.baseK(blockIdx),
                        cache.baseV(blockIdx),
                        T, totalLen,
                        _config.headCount, _config.headCountKv, head_dim,
                        curLen, attnScale,
                        attnOutBuf,
                        /*slidingWindow=*/0,
                        kvDtype);

    trace("O projection (matmulAsync — in-stream ordered, no sync needed)");
    _gmm.matmulAsync(oW->type, oW->usmPtr, d_model, q_dim,
                     attnOutBuf, T,
                     projOutBuf, matmulScratch);

    trace("attn residual + ffn rmsNorm (fused)");
    _ops.addRmsNormAsync(x, projOutBuf, T, d_model,
                         static_cast<const float*>(ffnNorm->usmPtr),
                         _config.rmsNormEps,
                         normBuf);

    // M5f.4: FFN gate and up are independent — both read normBuf, write
    // to different output buffers. silu_mul (the next op) reads BOTH so
    // the pop's barrier is what protects it.
    trace("FFN gate+up (matmulAsync, unordered)");
    {
        compute::UnorderedScope u{_ops};
        _gmm.matmulAsync(ffnGate->type, ffnGate->usmPtr, ff_dim, d_model,
                         normBuf, T, gateOutBuf, matmulScratch);
        _gmm.matmulAsync(ffnUp->type, ffnUp->usmPtr, ff_dim, d_model,
                         normBuf, T, upOutBuf, matmulScratch);
    }

    trace("FFN silu+mul (async, fused)");
    _ops.siluMulAsync(gateOutBuf, upOutBuf, T * ff_dim);

    trace("FFN down (matmulAsync — in-stream ordered, no sync needed)");
    _gmm.matmulAsync(ffnDown->type, ffnDown->usmPtr, d_model, ff_dim,
                     gateOutBuf, T,
                     projOutBuf, matmulScratch);

    trace("ffn residual (async, exit)");
    _ops.addResidualAsync(x, projOutBuf, T * d_model);
    // No sync here — the next block's rmsNormAsync (or sampleNext's
    // final-norm) reads x on the GPU through the same queue, so
    // command-list ordering covers the hand-off.
}

} // namespace mimirmind::runtime::arch