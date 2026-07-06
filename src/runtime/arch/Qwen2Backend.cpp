#include "runtime/arch/Qwen2Backend.hpp"

#include "compute/GpuMatmul.hpp"
#include "compute/GpuOps.hpp"
#include "model/FusedQkvWeights.hpp"
#include "model/GgufTypes.hpp"
#include "model/LlmConfig.hpp"
#include "model/WeightsMap.hpp"
#include "runtime/BlockBuffers.hpp"
#include "runtime/KvCache.hpp"
#include "runtime/Log.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace mimirmind::runtime::arch {

Qwen2Backend::Qwen2Backend(const model::LlmConfig&        config,
                           const model::WeightsMap&       weights,
                           const model::FusedQkvWeights*  fusedQkv,
                           compute::GpuOps&               ops,
                           compute::GpuMatmul&            gmm,
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
        auto t = [](const model::GgufTensor* p) {
            return p == nullptr ? "MISSING" : model::typeInfo(p->type).name.data();
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

    auto projectAsync = [&](const model::GgufTensor* W,
                            std::size_t N, float* dst) {
        _gmm.matmulAsync(W->type, W->usmPtr, N, d_model,
                         normBuf, T, dst, matmulScratch);
    };
    auto addBiasIf = [&](const model::GgufTensor* B,
                         std::size_t N, float* dst) {
        if (B != nullptr && B->type == model::GgmlType::F32) {
            _ops.addBiasAsync(dst, T, N,
                              static_cast<const float*>(B->usmPtr));
        }
    };

    float* kSlot = cache.writeSlotK(blockIdx);
    float* vSlot = cache.writeSlotV(blockIdx);
    // M-CLR.2 Wave 3: qkv_split receives the cache BASE + curLen so its
    // pointer arg is stable across replays. Other kernels here still use
    // the per-token slot; refactoring rope-K is Wave 3b per the ADR.
    float* kBase = const_cast<float*>(cache.baseK(blockIdx));
    float* vBase = const_cast<float*>(cache.baseV(blockIdx));

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
        _ops.qkvSplitAsync(qkvFused, qBuf, kBase, vBase,
                           T, fBlk->Nq, fBlk->Nkv, fBlk->hasV, curLen);
    } else {
        // M5f.4: Q/K/V projections write disjoint buffers and depend only
        // on normBuf (already published by the rmsnorm above). They can
        // pipeline freely inside the unordered scope; the scope's pop
        // emits a single barrier that the bias adds + RoPE read against.
        trace("Q+K+V projections (matmulAsync, unordered)");
        {
            runtime::UnorderedScope u{_ops.queue()};
            projectAsync(qW, q_dim,  qBuf);
            projectAsync(kW, kv_dim, kSlot);
            projectAsync(vW, kv_dim, vSlot);
        }
    }

    // QKV bias adds — each adds to its own buffer, independent of the
    // others. The bias add does read its own projection output, so
    // this scope's preceding barrier (from the Q/K/V projection pop
    // above) is what makes the chain safe.
    trace("QKV bias add (async, unordered)");
    {
        runtime::UnorderedScope u{_ops.queue()};
        addBiasIf(qB, q_dim,  qBuf);
        addBiasIf(kB, kv_dim, kSlot);
        addBiasIf(vB, kv_dim, vSlot);
    }

    // RoPE on Q and K — independent buffers, no V-dependency.
    // M-CLR.2 Wave 3b: K-rope uses cache BASE + kv_dim stride so the
    // kernel writes into the current slot at startPos*kv_dim internally.
    trace("RoPE Q+K (async, unordered)");
    {
        runtime::UnorderedScope u{_ops.queue()};
        _ops.ropeInPlaceAsync(qBuf, T,
                              _config.headCount,   head_dim, curLen,
                              _config.ropeFreqBase);
        _ops.ropeInPlaceAsync(kBase, T,
                              _config.headCountKv, head_dim, curLen,
                              _config.ropeFreqBase, kv_dim);
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
                        attnOutBuf);

    trace("O projection (matmul)");
    _gmm.matmul(oW->type, oW->usmPtr, d_model, q_dim,
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
        runtime::UnorderedScope u{_ops.queue()};
        _gmm.matmulAsync(ffnGate->type, ffnGate->usmPtr, ff_dim, d_model,
                         normBuf, T, gateOutBuf, matmulScratch);
        _gmm.matmulAsync(ffnUp->type, ffnUp->usmPtr, ff_dim, d_model,
                         normBuf, T, upOutBuf, matmulScratch);
    }

    trace("FFN silu+mul (async, fused)");
    _ops.siluMulAsync(gateOutBuf, upOutBuf, T * ff_dim);

    trace("FFN down (matmul)");
    _gmm.matmul(ffnDown->type, ffnDown->usmPtr, d_model, ff_dim,
                gateOutBuf, T,
                projOutBuf, matmulScratch);

    trace("ffn residual (async, exit)");
    _ops.addResidualAsync(x, projOutBuf, T * d_model);
    // No sync here — the next block's rmsNormAsync (or sampleNext's
    // final-norm) reads x on the GPU through the same queue, so
    // command-list ordering covers the hand-off.
}

} // namespace mimirmind::runtime::arch