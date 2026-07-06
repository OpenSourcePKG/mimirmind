#include "runtime/arch/GemmaBaseBackend.hpp"

#include "compute/GpuMatmul.hpp"
#include "compute/GpuOps.hpp"
#include "model/FusedQkvWeights.hpp"
#include "model/GgufReader.hpp"
#include "model/GgufTypes.hpp"
#include "model/LlmConfig.hpp"
#include "model/WeightsMap.hpp"
#include "runtime/BlockBuffers.hpp"
#include "runtime/KvCache.hpp"
#include "runtime/Log.hpp"
#include "runtime/OpProfiler.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

namespace mimirmind::runtime::arch {

GemmaBaseBackend::GemmaBaseBackend(const model::LlmConfig&        config,
                                   const model::WeightsMap&       weights,
                                   const model::FusedQkvWeights*  fusedQkv,
                                   compute::GpuOps&               ops,
                                   compute::GpuMatmul&            gmm,
                                   runtime::OpProfiler&           opProfiler)
    : _config{config}, _weights{weights}, _fusedQkv{fusedQkv},
      _ops{ops}, _gmm{gmm}, _op{opProfiler} {
    buildLayerInfos();
    loadRopeFreqs();
}

void GemmaBaseBackend::buildLayerInfos() {
    _layers.reserve(_config.blockCount);

    std::size_t swaCount = 0, fullCount = 0, altCount = 0;
    std::string pattern;
    pattern.reserve(_config.blockCount);

    for (std::size_t b = 0; b < _config.blockCount; ++b) {
        LayerInfo li{};
        li.isSwa = (b < _config.slidingWindowPattern.size())
                     ? _config.slidingWindowPattern[b]
                     : true; // default to SWA if pattern missing
        li.headDim  = _config.headDim(b);
        li.nHeads   = _config.headCount;
        li.nKvHeads = _config.headCountKvFor(b);
        li.qDim     = li.nHeads   * li.headDim;
        li.kvDim    = li.nKvHeads * li.headDim;
        li.ropeBase = li.isSwa ? _config.ropeFreqBaseSwa
                                : _config.ropeFreqBase;

        // Alternative attention: layer omits attn_v.weight, so V is
        // taken from the raw K projection. Every gemma4 layer must
        // still own attn_k.weight (the assumption that some layers
        // share K from earlier ones was wrong).
        const bool hasV = _weights.findBlock(b, "attn_v.weight") != nullptr;
        const bool hasK = _weights.findBlock(b, "attn_k.weight") != nullptr;
        if (!hasK) {
            throw std::runtime_error(
                "gemma: block " + std::to_string(b) +
                " missing attn_k.weight — model is malformed");
        }
        li.altAttention = !hasV;

        // Own-KV vs reuse. First `n_layer_kv_from_start` layers compute
        // and store their own K/V. Trailing `sharedKvLayers` layers reuse
        // an earlier layer via `n_kv_from_start - (is_swa ? 2 : 1)` per
        // llama.cpp/src/llama-model.cpp:2160.
        const std::size_t nKvStart =
            _config.sharedKvLayers > 0
                ? _config.blockCount - _config.sharedKvLayers
                : _config.blockCount;
        if (b < nKvStart) {
            li.ownsKv        = true;
            li.kvSourceLayer = b;
        } else {
            li.ownsKv        = false;
            const std::size_t offset = li.isSwa ? 2 : 1;
            li.kvSourceLayer =
                nKvStart > offset ? nKvStart - offset : 0;
        }

        _layers.push_back(li);
        pattern.push_back(li.ownsKv
                              ? (li.isSwa ? (li.altAttention ? 'a' : 's')
                                          : (li.altAttention ? 'A' : 'F'))
                              : (li.isSwa ? 'x' : 'X'));
        if (li.isSwa) ++swaCount; else ++fullCount;
        if (li.altAttention) ++altCount;
    }

    std::size_t reuseCount = 0;
    for (const auto& li : _layers) if (!li.ownsKv) ++reuseCount;

    MM_LOG_INFO("gemma",
                "layer map: {} ({} SWA, {} full, {} alt-attn V=K, "
                "{} reuse-KV [x/X])",
                pattern, swaCount, fullCount, altCount, reuseCount);
}

void GemmaBaseBackend::loadRopeFreqs() {
    if (const auto* rf = _weights.find("rope_freqs.weight");
        rf != nullptr && rf->type == model::GgmlType::F32) {
        // ggml_rope_ext expects [head_dim/2]; the 26B-A4B GGUF stores 256
        // floats (full head_dim/2 = 256). Accept any tensor whose element
        // count is at least the largest layer's halfDim and use the first
        // halfDim entries per layer.
        std::size_t maxHalfDim = 0;
        for (const auto& li : _layers) {
            maxHalfDim = std::max(maxHalfDim, li.headDim / 2);
        }
        if (rf->nelements >= maxHalfDim) {
            _ropeFreqsForFullAttn = static_cast<const float*>(rf->usmPtr);
            MM_LOG_INFO("gemma",
                        "proportional RoPE enabled — rope_freqs.weight "
                        "has {} float(s), using first {} for full-attn layers",
                        rf->nelements, maxHalfDim);
        } else {
            MM_LOG_WARN("gemma",
                        "rope_freqs.weight has {} float(s) < halfDim={} — "
                        "proportional RoPE disabled, full-attn layers will "
                        "use plain RoPE", rf->nelements, maxHalfDim);
        }
    } else {
        MM_LOG_INFO("gemma",
                    "no rope_freqs.weight — full-attn layers use plain RoPE");
    }
}

std::vector<std::size_t> GemmaBaseBackend::kvDimPerLayer() const {
    std::vector<std::size_t> out;
    out.reserve(_layers.size());
    for (const auto& li : _layers) {
        out.push_back(li.kvDim);
    }
    return out;
}

std::vector<std::size_t> GemmaBaseBackend::kvSourceLayerPerLayer() const {
    // Identity when the model doesn't use shared K/V — matches the
    // ArchBackend default. Avoids an unnecessary allocation on 26B-A4B
    // and on any dense Gemma-4 that has no reuse layers.
    if (_config.sharedKvLayers == 0) return {};
    std::vector<std::size_t> out;
    out.reserve(_layers.size());
    for (const auto& li : _layers) {
        out.push_back(li.kvSourceLayer);
    }
    return out;
}

std::pair<std::size_t, std::size_t> GemmaBaseBackend::maxQKVDims() const {
    std::size_t qMax = 0, kvMax = 0;
    for (const auto& li : _layers) {
        qMax  = std::max(qMax,  li.qDim);
        kvMax = std::max(kvMax, li.kvDim);
    }
    return {qMax, kvMax};
}

const model::GgufTensor*
GemmaBaseBackend::requireTensor(std::size_t blockIdx,
                                const char* suffix,
                                const char* clsName) const {
    const auto* t = _weights.findBlock(blockIdx, suffix);
    if (t == nullptr) {
        throw std::runtime_error(
            std::string{clsName} + ": missing tensor blk." +
            std::to_string(blockIdx) + "." + suffix);
    }
    return t;
}

void GemmaBaseBackend::dumpStage(const char* stage,
                                 std::size_t blockIdx,
                                 const float* p,
                                 std::size_t Trow,
                                 std::size_t dim) const {
    if (_parityDumpPrefix.empty()) {
        return;
    }
    _gmm.sync();
    const std::string fname =
        _parityDumpPrefix + "-blk" + std::to_string(blockIdx) +
        "-" + stage + ".bin";
    std::ofstream f(fname, std::ios::binary);
    if (!f) {
        return;
    }
    const std::uint32_t header[3] = {
        static_cast<std::uint32_t>(blockIdx),
        static_cast<std::uint32_t>(Trow),
        static_cast<std::uint32_t>(dim),
    };
    f.write(reinterpret_cast<const char*>(header), sizeof(header));
    f.write(reinterpret_cast<const char*>(p),
            static_cast<std::streamsize>(Trow * dim * sizeof(float)));
}

void GemmaBaseBackend::runAttentionSection(std::size_t   blockIdx,
                                           float*        x,
                                           std::size_t   T,
                                           KvCache&      cache,
                                           BlockBuffers& s,
                                           bool          diag) {
    auto trace = [&](const char* tag) {
        if (diag) MM_LOG_INFO("blkdiag-g", "blk0 {}", tag);
    };

    const auto& li = _layers[blockIdx];

    const auto* attnNorm = requireTensor(blockIdx, "attn_norm.weight",         "GemmaBase");
    const auto* qW       = requireTensor(blockIdx, "attn_q.weight",            "GemmaBase");
    const auto* qNorm    = requireTensor(blockIdx, "attn_q_norm.weight",       "GemmaBase");
    const auto* oW       = requireTensor(blockIdx, "attn_output.weight",       "GemmaBase");
    const auto* attnPost = requireTensor(blockIdx, "post_attention_norm.weight", "GemmaBase");

    // K/V weights are only needed when this layer owns its K/V cache.
    // Shared-KV layers (Gemma 4 E4B: 18 trailing) skip the K/V projection
    // entirely and read from `kvSourceLayer`'s cache during attention.
    const model::GgufTensor* kW    = nullptr;
    const model::GgufTensor* kNorm = nullptr;
    const model::GgufTensor* vW    = nullptr;
    if (li.ownsKv) {
        kW    = requireTensor(blockIdx, "attn_k.weight",      "GemmaBase");
        kNorm = requireTensor(blockIdx, "attn_k_norm.weight", "GemmaBase");
        // vW is optional — altAttention layers derive V from the raw K.
        if (!li.altAttention) {
            vW = requireTensor(blockIdx, "attn_v.weight", "GemmaBase");
        }
    }

    const std::size_t d_model  = s.d_model;
    const std::size_t q_dim    = li.qDim;
    const std::size_t kv_dim   = li.kvDim;
    const std::size_t head_dim = li.headDim;
    const std::size_t curLen   = cache.length();
    const std::size_t totalLen = curLen + T;

    float* const normBuf       = s.normBuf.as<float>();
    float* const qBuf          = s.qBuf.as<float>();
    float* const attnOutBuf    = s.attnOut.as<float>();
    float* const projOutBuf    = s.projOut.as<float>();
    float* const matmulScratch = s.matmulScratch.as<float>();
    // scoreScratch was the CPU-attention softmax row buffer; the GPU
    // attention kernel keeps the score row in SLM, so it's unused here.
    (void)s.scoreScratch;

    // K/V slots. Own-KV layers write to their own cache. Shared-KV
    // layers leave the writeSlot NULL — attention will read from the
    // source layer's `baseK` / `baseV`.
    float* const kSlot = li.ownsKv ? cache.writeSlotK(blockIdx) : nullptr;
    float* const vSlot = li.ownsKv ? cache.writeSlotV(blockIdx) : nullptr;

    // --- pre-attention RMSNorm ----------------------------------------

    _op.mark(runtime::OpProfiler::Cat::NORM);
    trace("attn rmsNorm");
    _ops.rmsNormAsync(x, T, d_model,
                      static_cast<const float*>(attnNorm->usmPtr),
                      _config.rmsNormEps,
                      normBuf);
    dumpStage("attn_norm", blockIdx, normBuf, T, d_model);

    auto projectAsync = [&](const model::GgufTensor* W,
                            std::size_t N, float* dst) {
        _gmm.matmulAsync(W->type, W->usmPtr, N, d_model,
                         normBuf, T, dst, matmulScratch);
    };

    // M5i.B: Fused Q+K+V — one matmul into a staging buffer, then a
    // scatter kernel routes the sub-ranges into qBuf/kSlot/vSlot. The
    // fused block is only registered when the layer's V is separate
    // (altAttention layers stay on the split path since the V used
    // downstream is the *raw* K projection, not W_v @ X). Fused QKV
    // is also skipped for shared-KV layers — no K/V to compute.
    const model::FusedQkvWeights::Block* fBlk =
        (_fusedQkv != nullptr && !li.altAttention && li.ownsKv)
            ? _fusedQkv->find(blockIdx)
            : nullptr;

    _op.mark(runtime::OpProfiler::Cat::MATMUL);
    if (fBlk != nullptr) {
        trace("Q+K+V projections (fused matmul + split)");
        float* const qkvFused = s.qkvFusedScratch.as<float>();
        const std::size_t Nfused =
            fBlk->Nq + fBlk->Nkv * (fBlk->hasV ? 2 : 1);
        _gmm.matmulAsync(fBlk->type, fBlk->usmPtr, Nfused, d_model,
                         normBuf, T, qkvFused, matmulScratch);
        _ops.qkvSplitAsync(qkvFused, qBuf, kSlot, vSlot,
                           T, fBlk->Nq, fBlk->Nkv, fBlk->hasV);
    } else if (li.ownsKv) {
        // M5f.4: Q/K/V projections write disjoint buffers. The pop inserts
        // a single barrier so the norms below see all three matmul outputs.
        trace("Q+K+V projections (matmulAsync, unordered)");
        {
            runtime::UnorderedScope u{_ops.queue()};
            projectAsync(qW, q_dim, qBuf);
            projectAsync(kW, kv_dim, kSlot);
            if (!li.altAttention) {
                projectAsync(vW, kv_dim, vSlot);
            }
        }
    } else {
        // Shared-KV layer: only compute Q. K/V are read from the source
        // layer's cache during attentionAsync below (populated earlier
        // in this same forward pass by that layer's own compute).
        trace("Q-only projection (shared-KV layer)");
        projectAsync(qW, q_dim, qBuf);
    }

    if (li.altAttention && li.ownsKv) {
        // V = raw K projection. We need a copy of K *before* K-norm and
        // RoPE so V keeps its raw layout. The unordered pop above
        // already issued a barrier; flush so the host memcpy reads a
        // settled USM region.
        trace("alt-attn V = raw K (memcpy)");
        _gmm.sync();
        std::memcpy(vSlot, kSlot, T * kv_dim * sizeof(float));
    }

    // Q-norm always runs. K-norm + V-norm only when the layer owns K/V.
    // Own-KV layers fuse all three norms into a single dispatch so the
    // decode-step doesn't pay two extra kernel-launch overheads per
    // block (~40 μs each on Xe-LPG). Shared-KV layers keep the plain
    // Q-only call — there's no fusion opportunity with a single row-set.
    _op.mark(runtime::OpProfiler::Cat::NORM);
    if (li.ownsKv) {
        trace("Q+K+V norms (rmsNorm fused)");
        _ops.rmsNormQkvAsync(
            qBuf,  static_cast<const float*>(qNorm->usmPtr),
            kSlot, static_cast<const float*>(kNorm->usmPtr),
            vSlot,
            T * li.nHeads, T * li.nKvHeads, head_dim,
            _config.rmsNormEps);
    } else {
        trace("Q-norm only (shared-KV layer)");
        _ops.rmsNormAsync(qBuf, T * li.nHeads, head_dim,
                          static_cast<const float*>(qNorm->usmPtr),
                          _config.rmsNormEps,
                          qBuf);
    }

    // RoPE Q always runs. RoPE K only when the layer owns K/V.
    _op.mark(runtime::OpProfiler::Cat::ATTENTION);
    if (li.ownsKv) {
        trace("RoPE Q+K (unordered)");
        runtime::UnorderedScope u{_ops.queue()};
        if (!li.isSwa && _ropeFreqsForFullAttn != nullptr) {
            _ops.ropeInPlaceWithFactorsAsync(qBuf, _ropeFreqsForFullAttn, T,
                                             li.nHeads, head_dim, curLen,
                                             li.ropeBase);
            _ops.ropeInPlaceWithFactorsAsync(kSlot, _ropeFreqsForFullAttn, T,
                                             li.nKvHeads, head_dim, curLen,
                                             li.ropeBase);
        } else {
            _ops.ropeInPlaceAsync(qBuf, T, li.nHeads, head_dim, curLen,
                                  li.ropeBase);
            _ops.ropeInPlaceAsync(kSlot, T, li.nKvHeads, head_dim, curLen,
                                  li.ropeBase);
        }
    } else {
        trace("RoPE Q only (shared-KV layer)");
        if (!li.isSwa && _ropeFreqsForFullAttn != nullptr) {
            _ops.ropeInPlaceWithFactorsAsync(qBuf, _ropeFreqsForFullAttn, T,
                                             li.nHeads, head_dim, curLen,
                                             li.ropeBase);
        } else {
            _ops.ropeInPlaceAsync(qBuf, T, li.nHeads, head_dim, curLen,
                                  li.ropeBase);
        }
    }
    if (li.ownsKv) {
        dumpStage("Kcur_pos",    blockIdx, kSlot, T, kv_dim);
        dumpStage("Vcur_normed", blockIdx, vSlot, T, kv_dim);
    }
    dumpStage("Qcur_pos", blockIdx, qBuf, T, q_dim);

    // M5f.3: attention on the GPU. Gemma 4's f_attention_scale = 1.0
    // (gemma4.cpp:11), so we pass scale=1.0 directly — no sqrt(head_dim)
    // pre-scale needed anymore.
    // For shared-KV layers, cache.baseK/V(kvSourceLayer) redirects
    // attention to the K/V cache written by an earlier layer during
    // this same prefill pass.
    _op.mark(runtime::OpProfiler::Cat::ATTENTION);
    trace(li.ownsKv ? "attention (GPU, scale=1, own K/V)"
                    : "attention (GPU, scale=1, reuse K/V)");
    _ops.attentionAsync(qBuf,
                        cache.baseK(li.kvSourceLayer),
                        cache.baseV(li.kvSourceLayer),
                        T, totalLen,
                        li.nHeads, li.nKvHeads, head_dim,
                        curLen, /*scale=*/1.0F,
                        attnOutBuf);

    _op.mark(runtime::OpProfiler::Cat::MATMUL);
    trace("O projection");
    _gmm.matmul(oW->type, oW->usmPtr, d_model, q_dim,
                attnOutBuf, T,
                projOutBuf, matmulScratch);

    _op.mark(runtime::OpProfiler::Cat::NORM);
    trace("attn_post_norm");
    _ops.rmsNormAsync(projOutBuf, T, d_model,
                      static_cast<const float*>(attnPost->usmPtr),
                      _config.rmsNormEps,
                      projOutBuf);            // in-place
    dumpStage("attn_post_norm", blockIdx, projOutBuf, T, d_model);

    _op.mark(runtime::OpProfiler::Cat::RESIDUAL);
    trace("attn residual");
    _ops.addResidualAsync(x, projOutBuf, T * d_model);
    // x now holds sa_out = inpL + attn_post_norm(attn_out).
    dumpStage("attn_out", blockIdx, x, T, d_model);
}

} // namespace mimirmind::runtime::arch