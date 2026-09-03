// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/arch/Qwen3_5Backend.hpp"

#include "compute/ComputeMatmul.hpp"
#include "compute/ComputeOps.hpp"
#include "core/modelopt/BlockScaleSwizzle.hpp" // E-d.4b swizzledBlockScaleBytes

#include <algorithm>
#include "compute/Embedding.hpp"
#include "compute/MoeRouting.hpp"
#include "compute/QuantType.hpp"
#include "compute/QuantTypeRegistry.hpp"
#include "core/gguf/GgufTypes.hpp"
#include "core/gguf/WeightsMap.hpp"
#include "core/log/Log.hpp"
#include "model/LlmConfig.hpp"
#include "runtime/BlockBuffers.hpp"
#include "runtime/KvCache.hpp"
#include "runtime/serving/PagedKvPool.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <span>
#include <stdexcept>
#include <string>

namespace mimirmind::runtime::arch {

namespace {

// Track B: token threshold at/above which a shared-expert projection is routed
// through the FP4-TC grouped GEMM (nExp=1) instead of the dense blocked kernel.
// Below it (decode, small chunks) the blocked kGemmMaxM=16 kernel wins on
// launch overhead; the TC path only pays off once the M/16 weight re-reads it
// avoids dominate. Matches the routed-MoE "prefill takes TC" M>=64 crossover.
constexpr std::size_t kShexpTcMinM = 64;

// Block geometry (elements, bytes) for an expert weight type. NVFP4_BLK is a
// runtime-only blocked format (32-element super-blocks of 20 bytes) not in the
// QuantType registry, so handle it explicitly; everything else goes through the
// registry. Returns {0,0} for an unregistered non-NVFP4 type (caller checks).
inline std::pair<std::size_t, std::size_t>
moeBlockGeom(core::gguf::GgmlType type) {
    if (type == core::gguf::GgmlType::NVFP4_BLK) {
        return {32, 20};
    }
    if (type == core::gguf::GgmlType::NVFP4_TC) {
        return {32, 16};   // plain E2M1 nibbles: 16 B / 32 elems (decode uses TC banks)
    }
    const compute::QuantType* qt = compute::quantType(type);
    return qt != nullptr ? std::pair<std::size_t, std::size_t>{qt->blockElements(),
                                                               qt->blockBytes()}
                         : std::pair<std::size_t, std::size_t>{0, 0};
}

const core::gguf::GgufTensor&
requireBlock(const core::gguf::WeightsMap& w, std::size_t blockIdx,
             std::string_view suffix) {
    const auto* t = w.findBlock(blockIdx, suffix);
    if (t == nullptr) {
        throw std::runtime_error(
            "Qwen3_5Backend: block " + std::to_string(blockIdx) +
            " missing tensor '" + std::string(suffix) + "'");
    }
    return *t;
}

// M-dependent dense-weight pick (M-Cuda dense-FP8-lowM). At low batch
// (seqCount <= maxT) prefer the blocked-FP8 E4M3 ".fp8" variant — decode is
// memory-bound there and FP8 halves the always-read dense traffic (+~15%
// single-request, bit-coherent). At high batch use the BF16 copy — compute-
// bound, where the TF32 tensor-core GEMM wins. Falls back to BF16 when maxT==0
// or the model carries no ".fp8" variant (dual-copy not loaded).
const core::gguf::GgufTensor&
pickDense(const core::gguf::WeightsMap& w, std::size_t blockIdx,
          std::string_view suffix, std::size_t seqCount, std::size_t maxT) {
    if (maxT > 0 && seqCount <= maxT) {
        std::string fp8Suffix(suffix);
        fp8Suffix += ".fp8";
        if (const auto* v = w.findBlock(blockIdx, fp8Suffix)) {
            return *v;
        }
    }
    return requireBlock(w, blockIdx, suffix);
}

} // namespace

Qwen3_5Backend::Qwen3_5Backend(const model::LlmConfig&       config,
                                   const core::gguf::WeightsMap& weights,
                                   const model::FusedQkvWeights* fusedQkv,
                                   compute::ComputeOps&          ops,
                                   compute::ComputeMatmul&       gmm,
                                   runtime::OpProfiler&          opProfiler)
    : _config{config}, _weights{weights}, _fusedQkv{fusedQkv},
      _ops{ops}, _gmm{gmm}, _op{opProfiler} {
    // M-dependent dense FP8: read the max batch at which the FP8 ".fp8" dense
    // variants are preferred (0 = off / no dual-copy). Must match the loader's
    // MIMIRMIND_DENSE_FP8_LOWM (which decides whether the variants exist).
    if (const char* m = std::getenv("MIMIRMIND_DENSE_FP8_LOWM")) {
        const long v = std::strtol(m, nullptr, 10);
        _denseFp8MaxT = v > 0 ? static_cast<std::size_t>(v) : 0;
    }
    // Routed-MoE grouping / shared-expert-TC / decode-variant env flags moved to
    // the MoE subclass ctor (Qwen3_5MoeBackend) — they only drive the routed
    // expert path, which the base does not own.
    if (const char* p = std::getenv("MIMIRMIND_PAGED_V1")) {
        _forcePagedV1 = (p[0] == '1' && p[1] == '\0');
    }
    // GDN-Inc 1: vLLM-style fused GDN input projections (qkvz + ba), nSeq==1.
    if (const char* pf = std::getenv("MIMIRMIND_GDN_PROJ_FUSE")) {
        _gdnProjFuse = (pf[0] == '1' && pf[1] == '\0');
    }
    // 5.18.10.3: batched (M>1) projection fuse — opt-in, coherence-gated.
    if (const char* pfb = std::getenv("MIMIRMIND_GDN_PROJ_FUSE_BATCH")) {
        _gdnProjFuseBatch = (pfb[0] == '1' && pfb[1] == '\0');
    }
    // GDN-Inc 2 / 2b are bit-identical and cost no memory, so they default ON;
    // MIMIRMIND_GDN_GATE_FUSE=0 / MIMIRMIND_GDN_PREP_FUSE=0 roll back.
    if (const char* gf = std::getenv("MIMIRMIND_GDN_GATE_FUSE")) {
        _gdnGateFuse = !(gf[0] == '0' && gf[1] == '\0');
    }
    if (const char* pcf = std::getenv("MIMIRMIND_GDN_PREP_FUSE")) {
        _gdnPrepFuse = !(pcf[0] == '0' && pcf[1] == '\0');
    }
    // 5.18.10.2: batched conv-state pack/save (bit-identical) — default ON.
    if (const char* cbp = std::getenv("MIMIRMIND_GDN_CONV_BATCHPACK")) {
        _gdnConvBatchPack = !(cbp[0] == '0' && cbp[1] == '\0');
    }
    _gdnQkvzW.resize(_config.blockCount);
    _gdnBaW.resize(_config.blockCount);
    if (const char* w13 = std::getenv("MIMIRMIND_MOE_W13_FUSE")) {
        _moeW13Fuse = (w13[0] == '1' && w13[1] == '\0');
    }
    _moeW13W.resize(_config.blockCount);
    _ssmTrace = (std::getenv("MIMIRMIND_SSM_TRACE") != nullptr);
    if (const char* d = std::getenv("MIMIRMIND_SSM_DUMP")) {
        _ssmDump    = true;
        _ssmDumpDir = d;
        if (const char* pp = std::getenv("MIMIRMIND_SSM_DUMP_POS")) {
            _ssmDumpPos = std::strtol(pp, nullptr, 10);
        }
        MM_LOG_INFO("qwen35moe",
                    "MIMIRMIND_SSM_DUMP active — dir='{}' pos={} (directional "
                    "per-block residual dump)",
                    _ssmDumpDir, _ssmDumpPos);
    }
    if (const char* d = std::getenv("MIMIRMIND_GDN_DUMP")) {
        _gdnDump    = true;
        _gdnDumpDir = d;
        if (const char* b = std::getenv("MIMIRMIND_GDN_DUMP_BLK")) {
            _gdnDumpBlk = static_cast<std::size_t>(std::strtol(b, nullptr, 10));
        }
        MM_LOG_INFO("qwen35moe",
                    "MIMIRMIND_GDN_DUMP active — dir='{}' blk={} (recurrence "
                    "in/out isolation dump)",
                    _gdnDumpDir, _gdnDumpBlk);
    }
    // Chunked GatedDeltaNet prefill auto-gate (M-Q3N.4). See the header for the
    // precedence rules; MIN_T (the A/B sweep knob) overrides the coarse flag.
    _gdnChunkMinT = kGdnChunkMinTDefault;
    if (std::getenv("MIMIRMIND_GDN_CHUNK") != nullptr) {
        _gdnChunkMinT = 2;  // force chunk for every prefill (T > 1)
    }
    if (const char* mt = std::getenv("MIMIRMIND_GDN_CHUNK_MIN_T")) {
        const long v = std::strtol(mt, nullptr, 10);
        if (v >= 2) {
            _gdnChunkMinT = static_cast<std::size_t>(v);
        } else {
            MM_LOG_WARN("qwen35moe",
                        "MIMIRMIND_GDN_CHUNK_MIN_T='{}' ignored (need >= 2)", mt);
        }
    }
    for (std::size_t i = 0; i < 4; ++i) {
        _ropeSections[i] = i < _config.ropeSections.size()
                               ? _config.ropeSections[i]
                               : 0;
    }
    std::size_t recurrent = 0;
    _fullAttnDense.assign(_config.blockCount,
                          std::numeric_limits<std::size_t>::max());
    std::size_t denseFull = 0;
    for (std::size_t b = 0; b < _config.blockCount; ++b) {
        if (_config.isRecurrentLayer(b)) {
            ++recurrent;
        } else {
            _fullAttnDense[b] = denseFull++;   // dense PagedKvPool layer index
        }
    }
    MM_LOG_INFO("qwen35moe",
                "Qwen3_5Backend ready — blocks={} ({} full / {} linear) "
                "d_model={} heads={} kv={} head_dim={} experts={}/{} "
                "ff_exp={} ff_shexp={} sections=[{},{},{},{}]",
                _config.blockCount, _config.blockCount - recurrent, recurrent,
                _config.embeddingLength, _config.headCount, _config.headCountKv,
                _config.headDim(), _config.expertCount, _config.expertUsedCount,
                _config.expertFeedForwardLength,
                _config.expertSharedFeedForwardLength,
                _ropeSections[0], _ropeSections[1],
                _ropeSections[2], _ropeSections[3]);
    if (_gdnChunkMinT == kGdnChunkMinTDefault) {
        MM_LOG_INFO("qwen35moe", "GatedDeltaNet prefill: AR (chunk disabled)");
    } else {
        MM_LOG_INFO("qwen35moe",
                    "GatedDeltaNet prefill: chunked (C=64) for T>={}", _gdnChunkMinT);
    }
}

std::vector<std::size_t> Qwen3_5Backend::kvDimPerLayer() const {
    // Full-attention layers own a KV cache of nKvHeads*headDim. The
    // recurrent (GatedDeltaNet) layers keep an SSM state instead of KV;
    // M-Q3N.2 sizes them the same (unused) and M-Q3N.3 replaces that with
    // a dedicated SSM state pool + zero KV.
    const std::size_t kvDim = _config.headCountKv * _config.headDim();
    return std::vector<std::size_t>(_config.blockCount, kvDim);
}

std::pair<std::size_t, std::size_t> Qwen3_5Backend::maxQKVDims() const {
    const std::size_t qDim  = _config.headCount   * _config.headDim();
    const std::size_t kvDim = _config.headCountKv * _config.headDim();
    return {qDim, kvDim};
}

bool Qwen3_5Backend::needsSsmScratch() const noexcept {
    return _config.isHybridRecurrent();
}

void Qwen3_5Backend::traceNorm(const char* tag, std::size_t blockIdx,
                                 std::size_t pos, const float* p,
                                 std::size_t n) const {
    _gmm.sync();  // p is a unified-memory pointer; readable after sync.
    double sumSq = 0.0;
    float  maxAbs = 0.0F;
    for (std::size_t i = 0; i < n; ++i) {
        const float v = p[i];
        sumSq += static_cast<double>(v) * static_cast<double>(v);
        maxAbs = std::fmax(maxAbs, std::fabs(v));
    }
    MM_LOG_INFO("ssm-trace", "pos={} blk={} {} l2={:.5g} max={:.5g}",
                pos, blockIdx, tag, std::sqrt(sumSq), maxAbs);
}

void Qwen3_5Backend::traceDump(const char* tag, std::size_t blockIdx,
                                 std::size_t pos, const float* p,
                                 std::size_t n) const {
    if (_ssmDumpPos >= 0 && pos != static_cast<std::size_t>(_ssmDumpPos)) {
        return;
    }
    _gmm.sync();  // p is a unified-memory pointer; readable after sync.
    const std::string path = _ssmDumpDir + "/pos" + std::to_string(pos) +
                             "-blk" + std::to_string(blockIdx) + "-" + tag +
                             ".bin";
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) {
        MM_LOG_WARN("ssm-dump", "cannot open '{}' for write", path);
        return;
    }
    f.write(reinterpret_cast<const char*>(p),
            static_cast<std::streamsize>(n * sizeof(float)));
}

void Qwen3_5Backend::configureHiddenTap(std::span<const std::size_t> tapLayers,
                                          std::span<float* const>      tapDst) {
    if (tapLayers.size() != tapDst.size()) {
        throw std::runtime_error(
            "Qwen3_5Backend::configureHiddenTap: tapLayers/tapDst size mismatch");
    }
    if (tapLayers.empty()) {
        clearHiddenTap();
        return;
    }
    _hiddenTapSlot.assign(_config.blockCount, -1);
    _hiddenTapDst.assign(tapDst.begin(), tapDst.end());
    for (std::size_t k = 0; k < tapLayers.size(); ++k) {
        const std::size_t l = tapLayers[k];
        if (l >= _config.blockCount) {
            clearHiddenTap();
            throw std::runtime_error(
                "Qwen3_5Backend::configureHiddenTap: tap layer index out of range");
        }
        if (tapDst[k] == nullptr) {
            clearHiddenTap();
            throw std::runtime_error(
                "Qwen3_5Backend::configureHiddenTap: null tap sink pointer");
        }
        _hiddenTapSlot[l] = static_cast<int>(k);
    }
}

std::size_t Qwen3_5Backend::gdnVHeads()     const noexcept { return _config.ssmNumVHeads(); }
std::size_t Qwen3_5Backend::gdnStateSize()  const noexcept { return _config.ssmStateSize; }
std::size_t Qwen3_5Backend::gdnConvDim()    const noexcept { return _config.ssmConvDim(); }
std::size_t Qwen3_5Backend::gdnConvKernel() const noexcept { return _config.ssmConvKernel; }
std::size_t Qwen3_5Backend::layerCount()    const noexcept { return _config.blockCount; }
bool Qwen3_5Backend::isRecurrent(std::size_t b) const noexcept {
    return _config.isRecurrentLayer(b);
}

void Qwen3_5Backend::configureGdnCapture(
        std::span<const std::size_t> recurBlocks,
        std::span<float* const>      kSinks,
        std::span<float* const>      vSinks,
        std::span<float* const>      gSinks,
        std::span<float* const>      bSinks,
        std::span<float* const>      convSinks,
        std::size_t                  maxT) {
    _gdnCapSlot.assign(_config.blockCount, -1);
    _gdnCapK.assign(kSinks.begin(), kSinks.end());
    _gdnCapV.assign(vSinks.begin(), vSinks.end());
    _gdnCapG.assign(gSinks.begin(), gSinks.end());
    _gdnCapB.assign(bSinks.begin(), bSinks.end());
    _gdnCapConv.assign(convSinks.begin(), convSinks.end());
    _gdnCapMaxT = maxT;
    for (std::size_t slot = 0; slot < recurBlocks.size(); ++slot) {
        const std::size_t b = recurBlocks[slot];
        if (b < _config.blockCount) {
            _gdnCapSlot[b] = static_cast<int>(slot);
        }
    }
}

void Qwen3_5Backend::runBlock(std::size_t   blockIdx,
                                float*        x,
                                std::size_t   T,
                                KvCache&      cache,
                                BlockBuffers& s,
                                bool          traceBlock0) {
    const bool diag = (blockIdx == 0 && cache.length() == 0 && traceBlock0);

    if (_config.isRecurrentLayer(blockIdx)) {
        runLinearBlock(blockIdx, x, T, cache, s, diag);
    } else {
        runFullAttentionBlock(blockIdx, x, T, cache, s, diag);
    }

    // M-Cuda.DFlash Phase 2 — hidden-state tap. Copy the residual after this
    // block into the caller's sink for the tapped layers (position-major,
    // row = absolute position base+r). CLR-safe device copy on the compute
    // stream; prod-inert when unconfigured. `base` = cache.length() is this
    // forward's first absolute position (the same base the ssm dump uses).
    if (!_hiddenTapDst.empty()) {
        const int slot = _hiddenTapSlot[blockIdx];
        if (slot >= 0) {
            const std::size_t base = cache.length();
            _ops.appendMemoryCopy(
                _hiddenTapDst[static_cast<std::size_t>(slot)] + base * s.d_model,
                x, T * s.d_model * sizeof(float));
        }
    }

    if (_ssmTrace || _ssmDump) {
        const std::size_t base    = cache.length();
        const std::size_t last    = (T > 0 ? T - 1 : 0);
        const std::size_t lastPos = base + last;
        const char* kind = _config.isRecurrentLayer(blockIdx) ? "xout(lin)"
                                                              : "xout(full)";
        if (_ssmTrace) {
            traceNorm(kind, blockIdx, lastPos, x, T * s.d_model);
        }
        if (_ssmDump) {
            // Directional dump of the residual stream after this block — the
            // vector to diff for prefill-vs-decode localisation. With a target
            // position set (MIMIRMIND_SSM_DUMP_POS), dump that absolute
            // position's row whenever it falls inside this call's T-window (so
            // a T=N prefill dumps the same position a T=1 decode step does);
            // otherwise dump the last row every call (decode-trajectory mode).
            if (_ssmDumpPos >= 0) {
                const std::size_t tgt = static_cast<std::size_t>(_ssmDumpPos);
                if (tgt >= base && tgt < base + T) {
                    traceDump("xout", blockIdx, tgt,
                              x + (tgt - base) * s.d_model, s.d_model);
                }
            } else {
                // Dump every row at its absolute position, so a single T=N
                // prefill pass captures the same positions a T=1 decode
                // trajectory does — the basis for the prefill-vs-decode diff.
                for (std::size_t r = 0; r < T; ++r) {
                    traceDump("xout", blockIdx, base + r,
                              x + r * s.d_model, s.d_model);
                }
            }
        }
    }
}

void Qwen3_5Backend::runFullAttentionBlock(std::size_t   blockIdx,
                                             float*        x,
                                             std::size_t   T,
                                             KvCache&      cache,
                                             BlockBuffers& s,
                                             bool          diag,
                                             std::size_t   kvLayerIdx) {
    // Weight-block index (blockIdx) vs KV-cache layer index (kvL) are
    // decoupled so the MTP module can read blk.<mtp> weights while writing
    // into a private 1-layer KvCache at layer 0.
    const std::size_t kvL =
        (kvLayerIdx == std::numeric_limits<std::size_t>::max()) ? blockIdx
                                                                : kvLayerIdx;
    auto trace = [&](const char* tag) {
        if (diag) MM_LOG_INFO("blkdiag-q35", "blk {} {}", blockIdx, tag);
    };
    trace("enter (full-attn)");

    const auto kvDtype = cache.dtype();
    // FP16 and Q8_0 KV both use the fp32-staging redirect: K/V project into an
    // fp32 scratch, rmsnorm + IMRoPE run in fp32 there, then a single commit
    // kernel casts (fp16) or block-quantises (Q8_0) each row into the packed
    // cache. RoPE cannot run in-place on fp16/Q8_0 storage, so the staging is
    // mandatory for both. The plain F32 path writes the cache slot in place and
    // is unchanged. This mirrors the q8Path/fp16Path already live in
    // Qwen2Backend/GemmaBaseBackend — same building blocks (kvKFp32Scratch,
    // kv_quant_commit_q8_0 / kv_commit_fp16), just wired through the IMRoPE path.
    const bool fp16Path   = (kvDtype == KvDtype::FP16);
    const bool q8Path     = (kvDtype == KvDtype::Q8_0);
    const bool stagedPath = fp16Path || q8Path;
    if (kvDtype != KvDtype::F32 && !stagedPath) {
        throw std::runtime_error(
            "Qwen3_5Backend: only KvDtype::F32, FP16 or Q8_0 is supported "
            "(FP16/Q8_0 via fp32-staging redirect; other dtypes e.g. FP8_E4M3 "
            "are not wired for IMRoPE)");
    }

    const auto& w    = _weights;
    const auto& attnNorm = requireBlock(w, blockIdx, "attn_norm.weight");
    const auto& qW       = pickDense(w, blockIdx, "attn_q.weight", T, _denseFp8MaxT);
    const auto& kW       = pickDense(w, blockIdx, "attn_k.weight", T, _denseFp8MaxT);
    const auto& vW       = pickDense(w, blockIdx, "attn_v.weight", T, _denseFp8MaxT);
    const auto& qNorm    = requireBlock(w, blockIdx, "attn_q_norm.weight");
    const auto& kNorm    = requireBlock(w, blockIdx, "attn_k_norm.weight");
    const auto& oW       = pickDense(w, blockIdx, "attn_output.weight", T, _denseFp8MaxT);
    const auto& attnPost = requireBlock(w, blockIdx, "post_attention_norm.weight");

    const std::size_t d_model  = s.d_model;
    const std::size_t head_dim = _config.headDim();
    const std::size_t nHeads   = _config.headCount;
    const std::size_t nKvHeads = _config.headCountKv;
    const std::size_t q_dim    = nHeads   * head_dim;
    const std::size_t kv_dim   = nKvHeads * head_dim;
    const std::size_t curLen   = cache.length();
    const std::size_t totalLen = curLen + T;

    float* const normBuf       = s.normBuf.as<float>();
    float* const qBuf          = s.qBuf.as<float>();
    float* const gateBuf       = s.gateScratch.as<float>();
    float* const qGateFused    = s.qGateFused.as<float>();
    float* const attnOutBuf    = s.attnOut.as<float>();
    float* const projOutBuf    = s.projOut.as<float>();
    float* const matmulScratch = s.matmulScratch.as<float>();

    void* const kSlot = cache.writeSlotK(kvL);
    void* const vSlot = cache.writeSlotV(kvL);
    void* const kBase = const_cast<void*>(cache.baseK(kvL));
    void* const vBase = const_cast<void*>(cache.baseV(kvL));

    // Staging redirect (fp16 + Q8_0): the K/V pre-attention pipeline
    // (projection, QK-norm, IMRoPE) runs on an fp32 scratch [T, kv_dim] at row
    // 0; the commit kernel below then casts (fp16) or block-quantises (Q8_0)
    // into the packed cache at the curLen offset. The plain F32 path keeps
    // writing the cache slot in place (kStaging == kBase, F32, curLen offset).
    float* const kFp32Scratch = stagedPath ? s.kvKFp32Scratch.as<float>() : nullptr;
    float* const vFp32Scratch = stagedPath ? s.kvVFp32Scratch.as<float>() : nullptr;
    void* const kStaging = stagedPath ? static_cast<void*>(kFp32Scratch) : kBase;
    void* const vStaging = stagedPath ? static_cast<void*>(vFp32Scratch) : vBase;
    const auto  stagingKvDtype    = stagedPath ? KvDtype::F32 : kvDtype;
    const std::size_t stagingWriteOffset = stagedPath ? 0 : curLen;
    const std::size_t stagingWriteStride = stagedPath ? 0 : kv_dim;

    // --- pre-attention RMSNorm ---------------------------------------
    _ops.profileSection("attn");   // prefill full-attention layer (DECODE_PROFILE)
    trace("attn rmsNorm");
    _ops.rmsNormAsync(x, T, d_model,
                      static_cast<const float*>(attnNorm.usmPtr),
                      _config.rmsNormEps, normBuf);

    // --- Q(+gate) / K / V projections --------------------------------
    // attn_q fuses query + per-head output gate: output width is 2*q_dim
    // laid out [Q_h | gate_h] per head. splitHeadPair de-interleaves it
    // into qBuf (Q, roped + attended) and gateBuf (raw gate, applied as a
    // sigmoid after attention).
    trace("Q|gate / K / V projections");
    {
        compute::UnorderedScope u{_ops};
        _gmm.matmulAsync(qW.type, qW.usmPtr, 2 * q_dim, d_model,
                         normBuf, T, qGateFused, matmulScratch);
        _gmm.matmulAsync(kW.type, kW.usmPtr, kv_dim, d_model,
                         normBuf, T,
                         stagedPath ? kFp32Scratch : static_cast<float*>(kSlot),
                         matmulScratch);
        _gmm.matmulAsync(vW.type, vW.usmPtr, kv_dim, d_model,
                         normBuf, T,
                         stagedPath ? vFp32Scratch : static_cast<float*>(vSlot),
                         matmulScratch);
    }

    trace("split Q|gate");
    _ops.splitHeadPairAsync(qGateFused, qBuf, gateBuf, T, nHeads, head_dim);

    // --- QK-norm (per-head RMS over head_dim) + V passthrough --------
    trace("QK-norm");
    _ops.rmsNormQkvAsync(
        qBuf,     static_cast<const float*>(qNorm.usmPtr),
        kStaging, static_cast<const float*>(kNorm.usmPtr),
        vStaging,
        T * nHeads, T * nKvHeads, head_dim,
        _config.rmsNormEps,
        /*writeOffset=*/stagingWriteOffset, kv_dim,
        stagingKvDtype, /*useStagingSlot=*/stagedPath);

    // --- IMRoPE on Q and K -------------------------------------------
    trace("IMRoPE Q+K");
    {
        compute::UnorderedScope u{_ops};
        _ops.mropeInPlaceAsync(qBuf, T, nHeads, head_dim, curLen,
                               _config.ropeFreqBase, _ropeSections);
        // startPos stays curLen (correct positional angles); the write stride
        // is 0 under fp16 so IMRoPE targets the fp32 scratch row 0.
        _ops.mropeInPlaceAsync(kStaging, T, nKvHeads, head_dim, curLen,
                               _config.ropeFreqBase, _ropeSections,
                               stagingWriteStride, stagingKvDtype);
    }

    // --- Commit: fold the roped/normed fp32 K/V scratch into the packed cache
    // at the curLen write offset. FP16 = single cast per element; Q8_0 =
    // per-32-element-block absmax + quantise. Same immediate/replay semantics
    // as the Qwen2Backend q8Path. F32 wrote in place, so no commit needed.
    if (fp16Path) {
        trace("KV commit fp16 (K + V)");
        _ops.kvCommitFp16Async(kFp32Scratch, kBase, T, kv_dim, curLen);
        _ops.kvCommitFp16Async(vFp32Scratch, vBase, T, kv_dim, curLen);
    } else if (q8Path) {
        trace("KV commit Q8_0 (K + V)");
        _ops.kvQuantCommitQ8Async(kFp32Scratch, kBase, T, kv_dim, curLen);
        _ops.kvQuantCommitQ8Async(vFp32Scratch, vBase, T, kv_dim, curLen);
    }

    // --- GQA attention -----------------------------------------------
    trace("attention");
    const float attnScale = _config.attentionScaleFor(head_dim);
    _ops.attentionAsync(qBuf, cache.baseK(kvL), cache.baseV(kvL),
                        T, totalLen, nHeads, nKvHeads, head_dim,
                        curLen, attnScale, attnOutBuf,
                        /*slidingWindow=*/0, kvDtype);

    // --- per-head output gate: attn *= sigmoid(gate) -----------------
    trace("output sigmoid gate");
    _ops.sigmoidGateMulAsync(attnOutBuf, gateBuf, T, q_dim, /*gateDim=*/q_dim);

    // --- O projection + attn residual --------------------------------
    trace("O projection");
    _gmm.matmulAsync(oW.type, oW.usmPtr, d_model, q_dim,
                     attnOutBuf, T, projOutBuf, matmulScratch);

    trace("attn residual");
    _ops.addResidualAsync(x, projOutBuf, T * d_model);   // x = x + attn_out

    // --- post-attention norm -> MoE FFN -> FFN residual --------------
    trace("post_attention_norm");
    _ops.rmsNormAsync(x, T, d_model,
                      static_cast<const float*>(attnPost.usmPtr),
                      _config.rmsNormEps, normBuf);

    trace("FFN");
    // Polymorphic FFN seam: the concrete subclass supplies the FFN over the
    // post-attention-norm activation, writing its result into s.moeAccumBuf
    // (Qwen3_5MoeBackend = routed experts + shared expert; Qwen3_5DenseBackend
    // = plain SwiGLU). Both write moeAccumBuf so this residual add is shared.
    runFfn(blockIdx, normBuf, T, s);

    trace("ffn residual");
    _ops.addResidualAsync(x, s.moeAccumBuf.as<float>(), T * d_model);
}

void Qwen3_5Backend::runLinearBlock(std::size_t   blockIdx,
                                      float*        x,
                                      std::size_t   T,
                                      KvCache&      cache,
                                      BlockBuffers& s,
                                      bool          diag) {
    auto trace = [&](const char* tag) {
        if (diag) MM_LOG_INFO("blkdiag-q35", "blk {} lin {}", blockIdx, tag);
    };
    trace("enter (linear/GatedDeltaNet)");

    const auto& w = _weights;
    const auto& attnNorm  = requireBlock(w, blockIdx, "attn_norm.weight");
    const auto& qkvW      = pickDense(w, blockIdx, "attn_qkv.weight", T, _denseFp8MaxT);
    const auto& gateW     = pickDense(w, blockIdx, "attn_gate.weight", T, _denseFp8MaxT);
    const auto& betaW     = requireBlock(w, blockIdx, "ssm_beta.weight");
    const auto& alphaW    = requireBlock(w, blockIdx, "ssm_alpha.weight");
    const auto& ssmA      = requireBlock(w, blockIdx, "ssm_a");
    const auto& ssmDt     = requireBlock(w, blockIdx, "ssm_dt.bias");
    const auto& convW     = requireBlock(w, blockIdx, "ssm_conv1d.weight");
    const auto& ssmNormW  = requireBlock(w, blockIdx, "ssm_norm.weight");
    const auto& ssmOutW   = pickDense(w, blockIdx, "ssm_out.weight", T, _denseFp8MaxT);
    const auto& attnPost  = requireBlock(w, blockIdx, "post_attention_norm.weight");

    const std::size_t d_model   = s.d_model;
    const std::size_t S         = _config.ssmStateSize;      // head_dim
    const std::size_t hK        = _config.ssmNumKHeads();
    const std::size_t hV        = _config.ssmNumVHeads();
    const std::size_t valueDim  = _config.ssmInnerSize;      // = hV * S
    const std::size_t convDim    = _config.ssmConvDim();
    const std::size_t dConv     = _config.ssmConvKernel;
    const std::size_t keyDim    = S * hK;
    const std::size_t stateElems     = _config.ssmStateElemsPerLayer();
    const std::size_t convStateElems = _config.ssmConvStateElemsPerLayer();
    const float       eps       = _config.rmsNormEps;

    // Persistent per-layer recurrent state (survives across decode steps);
    // zeroed only at sequence start (curLen == 0).
    const bool isSeqStart = (cache.length() == 0);

    float* const normBuf   = s.normBuf.as<float>();
    float* const qkvMixed  = s.ssmQkvMixed.as<float>();
    float* const convInput = s.ssmConvInput.as<float>();
    float* const zBuf      = s.ssmZ.as<float>();
    float* const qBuf      = s.ssmQ.as<float>();
    float* const kBuf      = s.ssmK.as<float>();
    float* const vBuf      = s.ssmV.as<float>();
    float* const deltaOut  = s.ssmDeltaOut.as<float>();
    float* const alphaBuf  = s.ssmAlpha.as<float>();
    float* const betaBuf   = s.ssmBeta.as<float>();
    float* const gateBuf   = s.ssmGate.as<float>();
    // Persistent per-sequence recurrent state, bound by the engine into
    // BlockBuffers from the per-sequence SsmState (survives BlockBuffers
    // reallocation). Slab-aware per-layer stride (BlockBuffers convention,
    // see BlockBuffers.hpp): the layer base steps by ssmSlabNSeq*elems so a
    // multi-sequence slab (serving) and the single-sequence path share one
    // indexing rule. ssmSlabNSeq defaults to 1, where this reduces to
    // blockIdx*elems (bit-identical to the historical single-session path).
    // ServingSession::prefillSlot pre-offsets ssmStatePtr/ssmConvStatePtr by
    // slot*elems so this T>1 path targets that slot's slab slice.
    const std::size_t slabNSeq = (s.ssmSlabNSeq != 0) ? s.ssmSlabNSeq : 1;
    float* const stateBuf  = s.ssmStatePtr     + blockIdx * (slabNSeq * stateElems);
    float* const convState = s.ssmConvStatePtr + blockIdx * (slabNSeq * convStateElems);
    float* const projOut   = s.projOut.as<float>();
    float* const matmulScr = s.matmulScratch.as<float>();

    // --- pre-attention RMSNorm ---------------------------------------
    _ops.profileSection("gdn.proj");   // prefill GDN sub-split (DECODE_PROFILE)
    trace("attn rmsNorm");
    _ops.rmsNormAsync(x, T, d_model,
                      static_cast<const float*>(attnNorm.usmPtr), eps, normBuf);

    // --- projections (all read normBuf, disjoint outputs) ------------
    trace("qkv / gate / beta / alpha projections");
    {
        compute::UnorderedScope u{_ops};
        _gmm.matmulAsync(qkvW.type,   qkvW.usmPtr,   convDim,  d_model, normBuf, T, qkvMixed, matmulScr);
        _gmm.matmulAsync(gateW.type,  gateW.usmPtr,  valueDim, d_model, normBuf, T, zBuf,     matmulScr);
        _gmm.matmulAsync(betaW.type,  betaW.usmPtr,  hV,       d_model, normBuf, T, betaBuf,  matmulScr);
        _gmm.matmulAsync(alphaW.type, alphaW.usmPtr, hV,       d_model, normBuf, T, alphaBuf, matmulScr);
    }

    // beta = sigmoid(beta); gLog = -exp(ssm_a) * softplus(alpha + ssm_dt).
    trace("beta sigmoid + decay gate");
    _ops.sigmoidInPlaceAsync(betaBuf, T * hV);
    _ops.deltanetGateAsync(alphaBuf,
                           static_cast<const float*>(ssmA.usmPtr),
                           static_cast<const float*>(ssmDt.usmPtr),
                           gateBuf, T, hV);

    // --- causal conv1d + silu ----------------------------------------
    // conv_input = concat(conv_state[d_conv-1], qkv_mixed[T]) along time.
    // conv_state persists across decode steps (rolling tail); it is zeroed
    // only at sequence start. After the conv, the last (d_conv-1) rows of
    // conv_input become the new conv_state. conv output reuses qkvMixed.
    _ops.profileSection("gdn.conv");   // prefill GDN sub-split (conv1d + gather + l2)
    trace("conv1d (rolling state-concat + silu)");
    const std::size_t stateRows = (dConv > 0 ? dConv - 1 : 0);
    if (isSeqStart) {
        _ops.mulScalarAsync(convState, 0.0F, convStateElems);
    }
    // conv_input = [conv_state | qkv_mixed]
    _ops.appendMemoryCopy(convInput, convState,
                          convStateElems * sizeof(float));
    _ops.appendMemoryCopy(convInput + stateRows * convDim, qkvMixed,
                          T * convDim * sizeof(float));
    _ops.causalConv1dSiluAsync(convInput,
                               static_cast<const float*>(convW.usmPtr),
                               qkvMixed, T, convDim, dConv);      // conv out -> qkvMixed
    // Save the trailing (d_conv-1) rows as the next conv_state.
    _ops.appendMemoryCopy(convState, convInput + T * convDim,
                          convStateElems * sizeof(float));

    // --- split conv into q / k / v (+ GQA repeat H_k -> H_v) ---------
    trace("gather q/k/v");
    _ops.gatherHeadsFromChannelsAsync(qkvMixed, qBuf, T, 0,          hK, hV, S, convDim);
    _ops.gatherHeadsFromChannelsAsync(qkvMixed, kBuf, T, keyDim,     hK, hV, S, convDim);
    _ops.gatherHeadsFromChannelsAsync(qkvMixed, vBuf, T, 2 * keyDim, hV, hV, S, convDim);

    // --- L2-norm q, k over head_dim ----------------------------------
    trace("L2-norm q,k");
    _ops.l2NormInPlaceAsync(qBuf, T * hV, S, eps);
    _ops.l2NormInPlaceAsync(kBuf, T * hV, S, eps);

    // GDN ReplaySSM verify capture: stash the recurrence inputs (post-L2norm k,
    // v, gLog=gateBuf, beta) + the conv input ([conv_state | qkv_mixed], which
    // survives the conv since its output went to qkvMixed) so a DFlash partial
    // accept can fold the accepted prefix without a trunk re-forward. Placed
    // BEFORE the recurrence branch so chunked-prefill in-place mutation can't
    // touch the cached k/beta. Skipped for prefill (T > maxT); prod-inert when
    // unconfigured (one empty-check per recurrent block). One T=K+1 verify call
    // => the whole window lands at ring offset 0.
    if (!_gdnCapSlot.empty() && T <= _gdnCapMaxT) {
        const int capSlot = _gdnCapSlot[blockIdx];
        if (capSlot >= 0) {
            const std::size_t sl = static_cast<std::size_t>(capSlot);
            _ops.appendMemoryCopy(_gdnCapK[sl],    kBuf,     T * hV * S * sizeof(float));
            _ops.appendMemoryCopy(_gdnCapV[sl],    vBuf,     T * hV * S * sizeof(float));
            _ops.appendMemoryCopy(_gdnCapG[sl],    gateBuf,  T * hV * sizeof(float));
            _ops.appendMemoryCopy(_gdnCapB[sl],    betaBuf,  T * hV * sizeof(float));
            _ops.appendMemoryCopy(_gdnCapConv[sl], convInput,
                                  (stateRows + T) * convDim * sizeof(float));
        }
    }

    // --- gated delta-rule recurrence (persistent state) -------------
    // state zeroed only at sequence start; decode steps evolve it in place.
    _ops.profileSection("gdn.recur");   // prefill GDN sub-split (delta-rule chunk/AR)
    trace("delta-rule recurrence");
    if (isSeqStart) {
        _ops.mulScalarAsync(stateBuf, 0.0F, stateElems);
    }
    if (T > 1 && T >= _gdnChunkMinT) {
        // Chunked prefill: K0 (cumgate) -> K1 (kkt inverse) -> K2 (forward).
        // Parity-equivalent to the AR recurrence (cuda_parity 10/10) but
        // parallel across the chunk instead of T sequential steps. gateBuf is
        // gLog; K0 turns it into gCum. Chunk size C = 64. Gated on T so short
        // prefills (where chunking is correct but not faster) keep the AR path.
        const std::size_t cChunk = 64;
        float* const gCum = s.ssmGCum.as<float>();
        float* const a0   = s.ssmA0.as<float>();
        _ops.profileSection("gdn.k0");   // chunk cumgate (recur sub-split)
        _ops.deltanetChunkCumGateAsync(gateBuf, gCum, T, hV, cChunk);
        _ops.profileSection("gdn.k1");   // KKT triangular-inverse (recur sub-split)
        _ops.deltanetKktSolveInverseAsync(kBuf, betaBuf, a0, T, hV, S, cChunk);
        _ops.profileSection("gdn.k2");   // chunk forward (recur sub-split)
        _ops.deltanetChunkForwardAsync(qBuf, kBuf, vBuf, gCum, betaBuf, a0,
                                       stateBuf, deltaOut, T, hV, S, cChunk);
    } else {
        _ops.gatedDeltaNetRecurrentAsync(qBuf, kBuf, vBuf, gateBuf, betaBuf,
                                         stateBuf, deltaOut, T, hV, S);
    }

    if (_ssmTrace) {
        // For T>1 (prefill) the state reflects the last token; for decode
        // (T==1) cache.length() is the position of the token just added.
        const std::size_t pos = cache.length() + (T > 0 ? T - 1 : 0);
        traceNorm("gate",  blockIdx, pos, gateBuf, T * hV);
        traceNorm("state", blockIdx, pos, stateBuf, stateElems);
        traceNorm("dnet",  blockIdx, pos, deltaOut, T * valueDim);
    }

    if (_gdnDump && blockIdx == _gdnDumpBlk) {
        // Isolate the recurrence: dump its exact in/out tensors so the fp64 HF
        // reference can be run on identical inputs. q,k are L2-normed here (as
        // the kernel consumes them); glog=gateBuf, beta=betaBuf. One prefill.
        _gmm.sync();
        auto dumpT = [&](const char* tag, const float* p, std::size_t n) {
            std::ofstream f(_gdnDumpDir + "/blk" + std::to_string(blockIdx) +
                            "-" + tag + ".bin",
                            std::ios::binary | std::ios::trunc);
            if (f) {
                f.write(reinterpret_cast<const char*>(p),
                        static_cast<std::streamsize>(n * sizeof(float)));
            }
        };
        dumpT("q",    qBuf,     T * hV * S);
        dumpT("k",    kBuf,     T * hV * S);
        dumpT("v",    vBuf,     T * hV * S);
        dumpT("glog", gateBuf,  T * hV);
        dumpT("beta", betaBuf,  T * hV);
        dumpT("dnet", deltaOut, T * valueDim);
        MM_LOG_INFO("gdn-dump", "blk{} dumped q/k/v/glog/beta/dnet T={} hV={} S={}",
                    blockIdx, T, hV, S);
    }

    // --- gated output norm: ssm_norm(out) * silu(z) ------------------
    // rmsNorm(out) over head_dim -> qBuf (reused as norm buffer), then
    // siluMul(z, n) = silu(z) * n, in place into zBuf.
    _ops.profileSection("gdn.out");   // prefill GDN sub-split (ssm_norm+out+resid)
    trace("gated ssm_norm x silu(z)");
    _ops.rmsNormAsync(deltaOut, T * hV, S,
                      static_cast<const float*>(ssmNormW.usmPtr), eps, qBuf);
    _ops.siluMulAsync(zBuf, qBuf, T * valueDim);

    // --- output projection ssm_out -----------------------------------
    trace("ssm_out projection");
    _gmm.matmulAsync(ssmOutW.type, ssmOutW.usmPtr, d_model, valueDim,
                     zBuf, T, projOut, matmulScr);

    // --- attn residual + post-attn-norm -> MoE FFN -> FFN residual ---
    trace("attn residual");
    _ops.addResidualAsync(x, projOut, T * d_model);

    if (diag) {
        const std::size_t posPA = cache.length() + (T > 0 ? T - 1 : 0);
        traceNorm("postattn", blockIdx, posPA, x, T * d_model);
    }

    trace("post_attention_norm");
    _ops.rmsNormAsync(x, T, d_model,
                      static_cast<const float*>(attnPost.usmPtr), eps, normBuf);

    trace("FFN");
    // Polymorphic FFN seam: the concrete subclass supplies the FFN over the
    // post-attention-norm activation, writing its result into s.moeAccumBuf
    // (Qwen3_5MoeBackend = routed experts + shared expert; Qwen3_5DenseBackend
    // = plain SwiGLU). Both write moeAccumBuf so this residual add is shared.
    runFfn(blockIdx, normBuf, T, s);

    trace("ffn residual");
    _ops.addResidualAsync(x, s.moeAccumBuf.as<float>(), T * d_model);
}


} // namespace mimirmind::runtime::arch