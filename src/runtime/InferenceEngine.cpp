#include "runtime/InferenceEngine.hpp"

#include "compute/Activations.hpp"
#include "compute/Attention.hpp"
#include "compute/Embedding.hpp"
#include "compute/Matmul.hpp"
#include "compute/Norm.hpp"
#include "compute/Rope.hpp"
#include "model/GgufTypes.hpp"
#include "runtime/KvCache.hpp"
#include "runtime/Log.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace mimirmind::runtime {

namespace {

/// One-shot diagnostic: log the suffix names + quant types of one
/// transformer block. Truth for the architecture handler.
void logBlockTensorInventory(const model::GgufReader& reader,
                             std::size_t              blockIdx) {
    const std::string prefix = "blk." + std::to_string(blockIdx) + ".";
    std::size_t hits = 0;
    for (const auto& t : reader.tensors()) {
        if (t.name.compare(0, prefix.size(), prefix) != 0) {
            continue;
        }
        ++hits;
        std::string dims;
        for (std::size_t i = 0; i < t.dimensions.size(); ++i) {
            if (i > 0) {
                dims += ',';
            }
            dims += std::to_string(t.dimensions[i]);
        }
        MM_LOG_INFO("inventory",
                    "  {} type={} dims=[{}] bytes={}",
                    t.name, model::typeInfo(t.type).name,
                    dims, t.nbytes);
    }
    MM_LOG_INFO("inventory", "block {} has {} tensor(s)", blockIdx, hits);
}

} // namespace

InferenceEngine::InferenceEngine()
    : _ctx{},
      _allocator{_ctx},
      _queue{_ctx},
      _gmm{_ctx, _queue},
      _ops{_ctx, _queue} {
    MM_LOG_INFO("engine", "InferenceEngine: probing USM limits");
    _allocator.probeLimits();

    // Opt-in block-0 trace. Anything non-empty / non-"0" enables it.
    if (const char* env = std::getenv("MIMIRMIND_TRACE_BLOCK0")) {
        std::string_view v{env};
        if (!v.empty() && v != "0" && v != "false" && v != "off") {
            _traceBlock0 = true;
            MM_LOG_INFO("engine",
                        "MIMIRMIND_TRACE_BLOCK0={} — block-0 trace enabled "
                        "for first forward only",
                        env);
        }
    }
}

InferenceEngine::~InferenceEngine() = default;

const model::WeightsMap& InferenceEngine::weights() const {
    if (!_weights.has_value()) {
        throw std::runtime_error("InferenceEngine: no model loaded");
    }
    return *_weights;
}

void InferenceEngine::loadModel(std::string_view ggufPath) {
    if (_modelLoaded) {
        throw std::runtime_error("InferenceEngine: model already loaded");
    }

    MM_LOG_INFO("engine", "loadModel: opening '{}'", ggufPath);
    _reader.open(ggufPath);

    _config.parseFromGguf(_reader);
    _tokenizer.loadFromGguf(_reader);

    MM_LOG_INFO("engine", "loadModel: copying tensors into USM");
    _reader.loadTensors(_allocator);

    _weights.emplace(_reader);

    // One-shot architecture diagnostic.
    logBlockTensorInventory(_reader, 0);

    // Pin the internal architecture tag so generate()'s hot-path
    // dispatch can switch on an enum instead of string-compare.
    if (_config.architecture == "qwen2") {
        _arch = Architecture::Qwen2;
    } else if (_config.architecture == "gemma4") {
        _arch = Architecture::Gemma4;
    } else {
        _arch = Architecture::Unsupported;
    }

    // Gemma 4 KV-sharing pattern. Some blocks omit attn_k/v.weight and
    // read K/V from an earlier block's cache. Walk the blocks once and
    // build the per-block source map.
    if (_arch == Architecture::Gemma4) {
        _gemma4KvSource.assign(_config.blockCount, 0);
        std::optional<std::size_t> lastKv;
        std::string pattern;
        pattern.reserve(_config.blockCount);
        for (std::size_t b = 0; b < _config.blockCount; ++b) {
            const bool hasOwn =
                _weights->findBlock(b, "attn_k.weight") != nullptr &&
                _weights->findBlock(b, "attn_v.weight") != nullptr;
            if (hasOwn) {
                _gemma4KvSource[b] = b;
                lastKv             = b;
                pattern.push_back('K');
            } else if (lastKv) {
                _gemma4KvSource[b] = *lastKv;
                pattern.push_back('.');
            } else {
                throw std::runtime_error(
                    "gemma4: block " + std::to_string(b) +
                    " has no K/V and no earlier source — model is malformed");
            }
        }
        MM_LOG_INFO("engine",
                    "gemma4 KV-share pattern: {} ({} owns, {} shared)",
                    pattern,
                    std::count(pattern.begin(), pattern.end(), 'K'),
                    std::count(pattern.begin(), pattern.end(), '.'));
    }

    _modelLoaded = true;
    MM_LOG_INFO("engine",
                "loadModel: ready — arch={} blocks={} d_model={} ff={} heads={} kv={}",
                _config.architecture, _config.blockCount,
                _config.embeddingLength, _config.feedForwardLength,
                _config.headCount, _config.headCountKv);
}

InferenceEngine::BlockBuffers
InferenceEngine::allocBlockBuffers(std::size_t maxT, std::size_t maxSeq) {
    BlockBuffers b{};
    b.maxT    = maxT;
    b.maxSeq  = maxSeq;
    b.d_model = _config.embeddingLength;
    b.q_dim   = _config.headCount * _config.headDim();
    b.ff_dim  = _config.feedForwardLength;

    const std::size_t qBytes            = maxT * b.q_dim   * sizeof(float);
    const std::size_t normBytes         = maxT * b.d_model * sizeof(float);
    const std::size_t attnOutBytes      = maxT * b.q_dim   * sizeof(float);
    const std::size_t projOutBytes      = maxT * b.d_model * sizeof(float);
    const std::size_t gateOutBytes      = maxT * b.ff_dim  * sizeof(float);
    const std::size_t upOutBytes        = maxT * b.ff_dim  * sizeof(float);
    const std::size_t scoreScratchBytes = maxSeq           * sizeof(float);
    const std::size_t matmulScratchBytes =
        std::max({b.d_model, b.q_dim, b.ff_dim}) * sizeof(float);

    b.qBuf          = UsmHandle{_allocator, qBytes};
    b.normBuf       = UsmHandle{_allocator, normBytes};
    b.attnOut       = UsmHandle{_allocator, attnOutBytes};
    b.projOut       = UsmHandle{_allocator, projOutBytes};
    b.gateOut       = UsmHandle{_allocator, gateOutBytes};
    b.upOut         = UsmHandle{_allocator, upOutBytes};
    b.matmulScratch = UsmHandle{_allocator, matmulScratchBytes};
    b.scoreScratch  = UsmHandle{_allocator, scoreScratchBytes};

    // MoE scratch (gemma4 Path B). Skip allocation for dense models so
    // the per-call setup cost stays the same as Qwen2.
    if (_config.expertCount > 0) {
        const std::size_t moeBytes = maxT * b.d_model * sizeof(float);
        b.moeAccumBuf  = UsmHandle{_allocator, moeBytes};
        b.expertOutBuf = UsmHandle{_allocator, moeBytes};
    }
    return b;
}

void InferenceEngine::runQwen2Block(std::size_t   blockIdx,
                                    float*        x,
                                    std::size_t   T,
                                    KvCache&      cache,
                                    BlockBuffers& s) {
    namespace cmp = mimirmind::compute;

    // Block-0 INFO trace fires only on the very first call after load
    // (`cache.length() == 0` plus the one-shot _traceBlock0 latch). Cheap
    // crash-localiser for new architectures; muted at request scale.
    const bool diag = (blockIdx == 0 && cache.length() == 0 && _traceBlock0);
    auto trace = [&](const char* tag) {
        if (diag) {
            MM_LOG_INFO("blkdiag", "blk0 {}", tag);
        }
    };

    trace("enter");

    const auto& w = *_weights;
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
            "transformer block " + std::to_string(blockIdx) +
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
    float* const scoreScratch  = s.scoreScratch.as<float>();

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

    // Q/K/V projections + bias adds all on the same queue. The bias
    // adds depend on the matmul output but happen in append order, so
    // GPU-side ordering is automatic. Single sync at the end so RoPE
    // (CPU, until M5f.2) sees the final values.
    trace("Q projection (matmulAsync)");
    projectAsync(qW, q_dim,  qBuf);
    trace("K projection (matmulAsync)");
    projectAsync(kW, kv_dim, kSlot);
    trace("V projection (matmulAsync)");
    projectAsync(vW, kv_dim, vSlot);
    trace("QKV bias add (async)");
    addBiasIf(qB, q_dim,  qBuf);
    addBiasIf(kB, kv_dim, kSlot);
    addBiasIf(vB, kv_dim, vSlot);

    trace("RoPE Q+K (async)");
    _ops.ropeInPlaceAsync(qBuf, T,
                          _config.headCount,   head_dim, curLen,
                          _config.ropeFreqBase);
    _ops.ropeInPlaceAsync(kSlot, T,
                          _config.headCountKv, head_dim, curLen,
                          _config.ropeFreqBase);
    trace("QKV+bias+RoPE sync");
    _gmm.sync();

    trace("attention");
    cmp::multiHeadAttention(
        qBuf,
        cache.baseK(blockIdx),
        cache.baseV(blockIdx),
        T, totalLen,
        _config.headCount, _config.headCountKv, head_dim,
        curLen,
        scoreScratch,
        attnOutBuf);

    trace("O projection (matmul)");
    _gmm.matmul(oW->type, oW->usmPtr, d_model, q_dim,
                attnOutBuf, T,
                projOutBuf, matmulScratch);

    trace("attn residual (async)");
    _ops.addResidualAsync(x, projOutBuf, T * d_model);

    trace("ffn rmsNorm (async)");
    _ops.rmsNormAsync(x, T, d_model,
                      static_cast<const float*>(ffnNorm->usmPtr),
                      _config.rmsNormEps,
                      normBuf);

    trace("FFN gate+up (matmulAsync)");
    _gmm.matmulAsync(ffnGate->type, ffnGate->usmPtr, ff_dim, d_model,
                     normBuf, T, gateOutBuf, matmulScratch);
    _gmm.matmulAsync(ffnUp->type, ffnUp->usmPtr, ff_dim, d_model,
                     normBuf, T, upOutBuf, matmulScratch);

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

void InferenceEngine::runGemma4Block(std::size_t   blockIdx,
                                     float*        x,
                                     std::size_t   T,
                                     KvCache&      cache,
                                     BlockBuffers& s) {
    namespace cmp = mimirmind::compute;

    // M8.3 + M8.5 — partial gemma4 block forward.
    // Implements: attn-norm, Q/K/V proj, Q-K-norm BEFORE RoPE, RoPE,
    // attention, O proj, attn_post_norm, dense FFN path A with GELU,
    // multi-norm choreography (post_ffw_norm_1, post_ffw_norm),
    // layer_output_scale.
    // Skips: sliding-window attention (M8.4), MoE path B (M8.6),
    // and any explicit f_attention_scale (multiHeadAttention does
    // 1/sqrt(head_dim) internally — adequate for the structure smoke,
    // parity polish lands in M8.7).
    //
    // Output is expected to be garbage because path B is ~80% of the
    // FFN compute and is currently skipped. The point is "forward runs
    // end-to-end without crashing" — concrete baseline for M8.6.

    const bool diag = (blockIdx == 0 && cache.length() == 0 && _traceBlock0);
    auto trace = [&](const char* tag) {
        if (diag) MM_LOG_INFO("blkdiag-g4", "blk0 {}", tag);
    };
    trace("enter");

    const auto& w = *_weights;
    auto require = [&](const char* suffix) {
        const auto* t = w.findBlock(blockIdx, suffix);
        if (t == nullptr) {
            throw std::runtime_error(
                "runGemma4Block: missing tensor blk." +
                std::to_string(blockIdx) + "." + suffix);
        }
        return t;
    };

    const auto* attnNorm    = require("attn_norm.weight");
    const auto* qW          = require("attn_q.weight");
    const auto* qNorm       = require("attn_q_norm.weight");
    const auto* oW          = require("attn_output.weight");
    const auto* attnPost    = require("post_attention_norm.weight");
    const auto* ffnNorm     = require("ffn_norm.weight");
    const auto* ffnGate     = require("ffn_gate.weight");
    const auto* ffnUp       = require("ffn_up.weight");
    const auto* ffnDown     = require("ffn_down.weight");
    const auto* ffwPost1    = require("post_ffw_norm_1.weight");
    const auto* ffwPost     = require("post_ffw_norm.weight");
    const auto* outScale    = require("layer_output_scale.weight");

    // Path B (MoE) tensors. Required for gemma4 26B-A4B; every block has them.
    const auto* preNorm2    = require("pre_ffw_norm_2.weight");
    const auto* postNorm2   = require("post_ffw_norm_2.weight");
    const auto* routerScale = require("ffn_gate_inp.scale");
    const auto* routerW     = require("ffn_gate_inp.weight");
    const auto* expGateUp   = require("ffn_gate_up_exps.weight");
    const auto* expDown     = require("ffn_down_exps.weight");
    // Per-expert scalar applied to the down projection output. Mirrors
    // build_lora_mm_id(down_exps, ..., down_exps_s) in
    // llama.cpp/src/llama-graph.cpp: res *= w_s[selected_expert].
    const auto* expDownScale = require("ffn_down_exps.scale");

    // KV-sharing: only blocks that own K/V do the K/V projection +
    // K-norm + K-RoPE. Other blocks read K/V from the source block's
    // cache slot (already RoPE'd by the source block earlier in this
    // forward pass).
    const std::size_t kvLayer  = _gemma4KvSource[blockIdx];
    const bool        hasOwnKv = (kvLayer == blockIdx);
    const auto* kW    = hasOwnKv ? require("attn_k.weight")      : nullptr;
    const auto* vW    = hasOwnKv ? require("attn_v.weight")      : nullptr;
    const auto* kNorm = hasOwnKv ? require("attn_k_norm.weight") : nullptr;

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
    float* const scoreScratch  = s.scoreScratch.as<float>();
    float* const moeAccumBuf   = s.moeAccumBuf.as<float>();
    float* const expertOutBuf  = s.expertOutBuf.as<float>();

    // --- pre-attention RMSNorm ----------------------------------------

    trace("attn rmsNorm");
    _ops.rmsNormAsync(x, T, d_model,
                           static_cast<const float*>(attnNorm->usmPtr),
                           _config.rmsNormEps,
                           normBuf);

    // --- Q projection (always own; only K/V are shared across layers) --

    auto projectAsync = [&](const model::GgufTensor* W,
                            std::size_t N, float* dst) {
        _gmm.matmulAsync(W->type, W->usmPtr, N, d_model,
                         normBuf, T, dst, matmulScratch);
    };

    trace("Q projection");
    projectAsync(qW, q_dim,  qBuf);

    // --- K/V projections only if this block owns them ------------------

    if (hasOwnKv) {
        float* const kSlot = cache.writeSlotK(blockIdx);
        float* const vSlot = cache.writeSlotV(blockIdx);
        trace("K projection");
        projectAsync(kW, kv_dim, kSlot);
        trace("V projection");
        projectAsync(vW, kv_dim, vSlot);

        // K-norm BEFORE RoPE (per-head, like Q-norm below).
        trace("K-norm");
        _ops.rmsNormAsync(kSlot, T * _config.headCountKv, head_dim,
                               static_cast<const float*>(kNorm->usmPtr),
                               _config.rmsNormEps,
                               kSlot);              // in-place

        // V-norm: bare RMSNorm over head_dim, no learned weight. Gemma 4
        // pushes V through `ggml_rms_norm` before it enters the KV cache
        // (see ggml-org/llama.cpp src/models/gemma4.cpp ~line 256).
        trace("V-norm (no weight)");
        _ops.rmsNormNoWeightAsync(vSlot, T * _config.headCountKv, head_dim,
                                  _config.rmsNormEps,
                                  vSlot);            // in-place

        // RoPE on the freshly-written K rows.
        trace("RoPE K");
        _ops.ropeInPlaceAsync(kSlot, T,
                              _config.headCountKv, head_dim, curLen,
                              _config.ropeFreqBase);
    } else {
        trace("KV reused from earlier layer");
    }

    // --- Q-K-norm BEFORE RoPE (per-head RMSNorm over head_dim) ---------
    //
    // The trick: virtual reshape from [T, n_heads, head_dim] to
    // [T*n_heads, head_dim]. Our row-RMSNorm kernel processes one row
    // (head_dim elements) per workgroup, weight is broadcast across
    // rows — exactly what we want for per-head normalisation.

    trace("Q-norm");
    _ops.rmsNormAsync(qBuf, T * _config.headCount, head_dim,
                           static_cast<const float*>(qNorm->usmPtr),
                           _config.rmsNormEps,
                           qBuf);                   // in-place

    trace("RoPE Q");
    _ops.ropeInPlaceAsync(qBuf, T,
                          _config.headCount,   head_dim, curLen,
                          _config.ropeFreqBase);

    // Gemma 4 uses f_attention_scale = 1.0 (see gemma4.cpp line ~11):
    // attention scores are NOT divided by sqrt(head_dim). Our CPU
    // multiHeadAttention always divides by 1/sqrt(headDim) internally,
    // so we pre-scale Q by sqrt(head_dim) here. The internal divide
    // cancels and the effective score scale becomes 1.0.
    {
        const float qScale = std::sqrt(static_cast<float>(head_dim));
        trace("Q pre-scale sqrt(head_dim) (gemma4 f_attention_scale=1.0)");
        _ops.mulScalarAsync(qBuf, qScale, T * q_dim);
    }

    // --- Attention is still CPU (M8.4 will replace this) ---------------

    trace("QKV sync (before CPU attention)");
    _gmm.sync();

    // K/V come from the source layer's cache. For own-KV blocks that's
    // the same as `blockIdx`; for shared blocks it points back at the
    // closest earlier own-KV layer (which has already RoPE'd this same
    // forward's new K rows by the time we get here).
    trace("multiHeadAttention (CPU)");
    cmp::multiHeadAttention(
        qBuf,
        cache.baseK(kvLayer),
        cache.baseV(kvLayer),
        T, totalLen,
        _config.headCount, _config.headCountKv, head_dim,
        curLen,
        scoreScratch,
        attnOutBuf);

    // --- O projection (Q8_0 -> CPU) ------------------------------------

    trace("O projection");
    _gmm.matmul(oW->type, oW->usmPtr, d_model, q_dim,
                attnOutBuf, T,
                projOutBuf, matmulScratch);

    // --- attn_post_norm + attention residual ---------------------------

    trace("attn_post_norm");
    _ops.rmsNormAsync(projOutBuf, T, d_model,
                           static_cast<const float*>(attnPost->usmPtr),
                           _config.rmsNormEps,
                           projOutBuf);            // in-place

    trace("attn residual");
    _ops.addResidualAsync(x, projOutBuf, T * d_model);
    // x now holds sa_out = inpL + attn_post_norm(attn_out).

    // --- FFN path A — dense SwiGLU with GELU ---------------------------

    trace("ffn_norm (path A pre)");
    _ops.rmsNormAsync(x, T, d_model,
                           static_cast<const float*>(ffnNorm->usmPtr),
                           _config.rmsNormEps,
                           normBuf);

    trace("FFN gate proj");
    _gmm.matmulAsync(ffnGate->type, ffnGate->usmPtr, ff_dim, d_model,
                     normBuf, T, gateOutBuf, matmulScratch);
    trace("FFN up proj");
    _gmm.matmulAsync(ffnUp->type, ffnUp->usmPtr, ff_dim, d_model,
                     normBuf, T, upOutBuf, matmulScratch);

    trace("GELU + mul (fused)");
    _ops.geluMulAsync(gateOutBuf, upOutBuf, T * ff_dim);

    trace("FFN down proj");
    _gmm.matmul(ffnDown->type, ffnDown->usmPtr, d_model, ff_dim,
                gateOutBuf, T,
                projOutBuf, matmulScratch);

    trace("post_ffw_norm_1 (path A post)");
    _ops.rmsNormAsync(projOutBuf, T, d_model,
                           static_cast<const float*>(ffwPost1->usmPtr),
                           _config.rmsNormEps,
                           projOutBuf);            // in-place

    // --- Path B — MoE (M8.6) ------------------------------------------
    //
    // pathB_in    = RMSNorm(x, pre_ffw_norm_2)
    // router_in   = RMSNorm(x, ffn_gate_inp.scale) * 1/sqrt(d_model)
    // logits      = ffn_gate_inp.weight · router_in            -> [T, n_exp]
    // weights, e  = softmax(top_K(logits))                     (CPU)
    // pathB[t]    = Σ_e weights[t,e] *
    //               (down_exps[e] · gelu_swiglu(gate_up_exps[e] · pathB_in[t]))
    // pathB_out   = RMSNorm(pathB, post_ffw_norm_2)

    // Step 1: pathB_in (overwrites normBuf — pathA already consumed it).
    trace("path B: pre_ffw_norm_2");
    _ops.rmsNormAsync(x, T, d_model,
                           static_cast<const float*>(preNorm2->usmPtr),
                           _config.rmsNormEps,
                           normBuf);

    // Step 2: router input = RMSNorm(x, ffn_gate_inp.scale) * 1/√d_model.
    // attnOutBuf is unused from here on; reuse it for routerIn (size
    // T*d_model fits because attnOut is sized T*q_dim with q_dim >= d_model).
    trace("path B: router rmsNorm + scale");
    _ops.rmsNormAsync(x, T, d_model,
                      static_cast<const float*>(routerScale->usmPtr),
                      _config.rmsNormEps,
                      attnOutBuf);
    const float invSqrtDm = 1.0F /
        std::sqrt(static_cast<float>(d_model));
    _ops.mulScalarAsync(attnOutBuf, invSqrtDm, T * d_model);

    // Step 3: router projection. ffn_gate_inp is F32, so this goes to
    // CPU fallback inside _gmm.matmul (which also syncs the queue).
    // Output layout: [T, n_experts] floats into upOutBuf (large enough:
    // upOut is sized T*ff_dim, with ff_dim >= n_experts here).
    const std::size_t nExperts = _config.expertCount;
    const std::size_t K        = _config.expertUsedCount;
    trace("path B: router matmul (CPU)");
    _gmm.matmul(routerW->type, routerW->usmPtr,
                nExperts, d_model,
                attnOutBuf, T,
                upOutBuf, matmulScratch);

    // Step 4: CPU softmax + top-K per token. The previous matmul.sync()
    // means upOutBuf is up-to-date for the CPU read.
    std::vector<std::int32_t> topKIdx(T * K);
    std::vector<float>        topKWeight(T * K);
    {
        std::vector<float> probs(nExperts);
        for (std::size_t t = 0; t < T; ++t) {
            const float* row = upOutBuf + t * nExperts;
            // Numerically-stable softmax.
            float maxL = row[0];
            for (std::size_t e = 1; e < nExperts; ++e) {
                if (row[e] > maxL) maxL = row[e];
            }
            double sum = 0.0;
            for (std::size_t e = 0; e < nExperts; ++e) {
                probs[e] = std::exp(row[e] - maxL);
                sum += static_cast<double>(probs[e]);
            }
            const float invSum = static_cast<float>(1.0 / sum);
            for (auto& p : probs) p *= invSum;

            // Top-K by probability.
            std::vector<std::size_t> idx(nExperts);
            std::iota(idx.begin(), idx.end(), 0);
            std::partial_sort(idx.begin(),
                              idx.begin() + static_cast<std::ptrdiff_t>(K),
                              idx.end(),
                              [&](std::size_t a, std::size_t b) {
                                  return probs[a] > probs[b];
                              });
            // Renormalise selected weights so the kept K probs sum to 1.
            double kept = 0.0;
            for (std::size_t k = 0; k < K; ++k) {
                kept += static_cast<double>(probs[idx[k]]);
            }
            const float invKept = kept > 0.0 ? static_cast<float>(1.0 / kept) : 1.0F;
            for (std::size_t k = 0; k < K; ++k) {
                topKIdx[t * K + k]    = static_cast<std::int32_t>(idx[k]);
                topKWeight[t * K + k] = probs[idx[k]] * invKept;
            }
        }
    }

    // Step 5: zero the per-block MoE accumulator.
    trace("path B: zero accumulator");
    _ops.mulScalarAsync(moeAccumBuf, 0.0F, T * d_model);

    // Step 6: per-token, per-expert dispatch. Expert pointer offsets
    // mirror the GGUF storage layout for 3D quantised tensors.
    //
    //   gate_up_exps: Q6_K [K=d_model, N=gate_up_fused, n_exp]
    //                  per-expert bytes = N * (K/256) * 210
    //   down_exps:    Q8_0 [K=ff_per_expert, N=d_model, n_exp]
    //                  per-expert bytes = N * (K/32)  * 34
    //
    // For gemma4 26B-A4B: gate_up_fused=1408, ff_per_expert=704.
    const std::size_t gateUpFused = expGateUp->dimensions.size() >= 2
                                      ? expGateUp->dimensions[1] : 0;
    const std::size_t ffPerExpert = gateUpFused / 2;
    if (gateUpFused == 0 || (gateUpFused % 2) != 0) {
        throw std::runtime_error(
            "runGemma4Block: ffn_gate_up_exps has unexpected fused dim " +
            std::to_string(gateUpFused));
    }

    constexpr std::size_t kQ6KBlockElems = 256;
    constexpr std::size_t kQ6KBlockBytes = 210;
    constexpr std::size_t kQ8_0BlockElems = 32;
    constexpr std::size_t kQ8_0BlockBytes = 34;

    const std::size_t expertBytesGateUp =
        gateUpFused * (d_model / kQ6KBlockElems) * kQ6KBlockBytes;
    const std::size_t expertBytesDown =
        d_model * (ffPerExpert / kQ8_0BlockElems) * kQ8_0BlockBytes;

    auto* const expGateUpBase =
        static_cast<const std::uint8_t*>(expGateUp->usmPtr);
    auto* const expDownBase =
        static_cast<const std::uint8_t*>(expDown->usmPtr);
    const float* const expDownScalePtr =
        static_cast<const float*>(expDownScale->usmPtr);

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

            // 6a. gate+up: [1, gate_up_fused] = Q6_K [d_model, fused] @ x
            _gmm.matmulAsync(expGateUp->type, Wgu,
                             gateUpFused, d_model,
                             pathBInT, 1,
                             gateOutBuf, matmulScratch);

            // 6b. fused gelu * up: split halves in place — first half is
            //     gate, second half is up. Output goes into gate slot.
            _ops.geluMulAsync(gateOutBuf, gateOutBuf + ffPerExpert,
                              ffPerExpert);

            // 6c. down: [1, d_model] = Q8_0 [ff_per_expert, d_model] @ silu_out
            _gmm.matmul(expDown->type, Wd,
                        d_model, ffPerExpert,
                        gateOutBuf, 1,
                        expertOutBuf, matmulScratch);

            // 6d. scale by per-expert down scalar AND router weight, then
            //     accumulate. ffn_down_exps.scale[e] is the per-expert
            //     `w_s` from build_lora_mm_id (see llama-graph.cpp:1411-
            //     1418): res *= w_s[selected_expert]. The matmul above
            //     synced via _gmm.matmul(...) so reading USM-host F32 is
            //     safe here.
            const float combined = routerWeight * expDownScalePtr[e];
            _ops.mulScalarAsync(expertOutBuf, combined, d_model);
            _ops.addResidualAsync(accumT, expertOutBuf, d_model);
        }
    }

    // Step 7: post-norm of path B output.
    trace("path B: post_ffw_norm_2");
    _ops.rmsNormAsync(moeAccumBuf, T, d_model,
                           static_cast<const float*>(postNorm2->usmPtr),
                           _config.rmsNormEps,
                           moeAccumBuf);

    // --- Combine paths + post_ffw_norm + residual ----------------------

    trace("combined = pathA_out + pathB_out");
    _ops.addResidualAsync(moeAccumBuf, projOutBuf, T * d_model);

    trace("post_ffw_norm (combined)");
    _ops.rmsNormAsync(moeAccumBuf, T, d_model,
                           static_cast<const float*>(ffwPost->usmPtr),
                           _config.rmsNormEps,
                           moeAccumBuf);

    trace("ffn residual");
    _ops.addResidualAsync(x, moeAccumBuf, T * d_model);
    // x = sa_out + post_ffw_norm(pathA_out + pathB_out)

    // --- layer_output_scale (F32 scalar per block) ---------------------
    //
    // layer_output_scale.weight is a USM-shared F32[1] — read directly
    // from CPU. Weights are loaded once at startup and never mutated,
    // so no GPU↔CPU fence is needed.

    const float scaleVal = *static_cast<const float*>(outScale->usmPtr);
    trace("layer_output_scale");
    _ops.mulScalarAsync(x, scaleVal, T * d_model);
    // No sync here — the next block's rmsNormAsync (or sampleNext's
    // final-norm) reads x on the GPU through the same queue.
}

std::int32_t
InferenceEngine::sampleNext(const float*                   hidden,
                            std::size_t                    vocab_lm,
                            const model::GgufTensor&       outNorm,
                            const model::GgufTensor&       lmHead,
                            float*                         normScratch,
                            float*                         logits,
                            float*                         matmulScratch,
                            const compute::SamplingParams& sampling) {
    // Final-norm runs on the same queue as the residual-add that
    // produced `hidden`. The subsequent _gmm.matmul flushes and syncs,
    // so CPU argmax sees a fully-resolved logits buffer.
    //
    // Both Qwen and Gemma 4 use plain `w * rms_norm(x)` at runtime:
    // Gemma 3 stored (1+w) in HF and the converter shifts w_gguf =
    // w_hf + 1; Gemma 4's converter (gemma.py:621-623) returns
    // norm_shift = 0.0 because the HF reference uses standard weight
    // (init at 1.0) — so the GGUF weight is already the multiplicative
    // scale.
    _ops.rmsNormAsync(
        hidden, 1, _config.embeddingLength,
        static_cast<const float*>(outNorm.usmPtr),
        _config.rmsNormEps,
        normScratch);

    _gmm.matmul(
        lmHead.type, lmHead.usmPtr,
        vocab_lm, _config.embeddingLength,
        normScratch, 1,
        logits, matmulScratch);

    return _sampler.sample(
        std::span<const float>{logits, vocab_lm}, sampling);
}

std::vector<std::int32_t>
InferenceEngine::generate(std::span<const std::int32_t> promptIds,
                          const GenerateParams&         params,
                          const TokenCallback&          onToken,
                          GenerateStats*                outStats) {
    namespace cmp = mimirmind::compute;
    using clock = std::chrono::steady_clock;

    if (!_modelLoaded) {
        throw std::runtime_error("InferenceEngine::generate: no model loaded");
    }
    if (promptIds.empty()) {
        throw std::runtime_error("InferenceEngine::generate: empty prompt");
    }

    // Architecture-specific block forward is dispatched per-layer
    // below via _arch. Refuse early for genuinely unknown arches so
    // the failure mode is clear (vs. running with a wrong handler).
    if (_arch == Architecture::Unsupported) {
        throw std::runtime_error(
            "generate: architecture '" + _config.architecture +
            "' is not recognised. Model is loaded into USM but the "
            "engine has no handler for it. See "
            "Memory/mimirmind/research/m8-gemma4-staging.md for the "
            "list of supported architectures.");
    }

    const auto* tokEmb = _weights->find("token_embd.weight");
    if (tokEmb == nullptr) {
        tokEmb = _weights->find("tok_embeddings.weight");
    }
    const auto* outNorm = _weights->find("output_norm.weight");
    const auto* lmHead  = _weights->find("output.weight");
    if (lmHead == nullptr) {
        lmHead = _weights->find("token_embd.weight");
    }

    if (tokEmb == nullptr) {
        throw std::runtime_error("generate: token embedding tensor missing");
    }
    if (outNorm == nullptr ||
        outNorm->type != model::GgmlType::F32) {
        throw std::runtime_error(
            "generate: output_norm.weight missing or not F32");
    }
    if (lmHead == nullptr) {
        throw std::runtime_error("generate: lm_head tensor missing");
    }

    const std::size_t Tp        = promptIds.size();
    const std::size_t maxNew    = params.maxNewTokens;
    const std::size_t vocab_lm  = lmHead->dimensions.size() >= 2
                                    ? lmHead->dimensions[1]
                                    : _tokenizer.vocabSize();
    const std::size_t vocab_emb = tokEmb->dimensions.size() >= 2
                                    ? tokEmb->dimensions[1]
                                    : _tokenizer.vocabSize();
    const std::size_t cacheMax  = Tp + maxNew + 4;
    const std::size_t maxT      = std::max<std::size_t>(Tp, 1);
    const std::size_t d_model   = _config.embeddingLength;

    KvCache cache(_allocator, _config.blockCount, cacheMax,
                  _config.headCountKv, _config.headDim());

    BlockBuffers buffers = allocBlockBuffers(maxT, cacheMax);

    const std::size_t xBytes         = maxT * d_model * sizeof(float);
    const std::size_t normFinalBytes = d_model * sizeof(float);
    const std::size_t logitsBytes    = vocab_lm * sizeof(float);
    const std::size_t logitsScBytes  = d_model * sizeof(float);

    UsmHandle xBufH     {_allocator, xBytes};
    UsmHandle normFinalH{_allocator, normFinalBytes};
    UsmHandle logitsH   {_allocator, logitsBytes};
    UsmHandle logitsScH {_allocator, logitsScBytes};

    float* const xBuf      = xBufH.as<float>();
    float* const normFinal = normFinalH.as<float>();
    float* const logits    = logitsH.as<float>();
    float* const logitsSc  = logitsScH.as<float>();

    // Per-architecture block forward, picked once per generate() to
    // keep the inner loops simple. Throws from inside the variant if
    // the handler is a skeleton (Gemma4 today).
    auto runBlock = [this](std::size_t blockIdx, float* xb, std::size_t Tb,
                           KvCache& cb, BlockBuffers& sb) {
        switch (_arch) {
            case Architecture::Qwen2:       runQwen2Block (blockIdx, xb, Tb, cb, sb); break;
            case Architecture::Gemma4:      runGemma4Block(blockIdx, xb, Tb, cb, sb); break;
            case Architecture::Unsupported:
                throw std::runtime_error("runBlock: unsupported architecture");
        }
    };

    // Gemma family scales the token embedding by sqrt(d_model) before
    // it enters the first block — the per-token vectors are otherwise
    // in the ~0.05 range and attention/FFN expects them at unit-ish
    // scale. Qwen/Llama don't do this. Centralise here so prefill and
    // decode paths share the same scaling.
    const bool        gemmaEmbedScale = (_arch == Architecture::Gemma4);
    const float       embedScale =
        gemmaEmbedScale ? std::sqrt(static_cast<float>(d_model)) : 1.0F;
    auto scaleEmbeddingIfGemma = [&](float* dst, std::size_t T) {
        if (gemmaEmbedScale && T > 0) {
            _ops.mulScalarAsync(dst, embedScale, T * d_model);
        }
    };

    // -- Prefill ---------------------------------------------------------

    const auto preT0 = clock::now();
    cmp::embeddingLookup(
        tokEmb->type, tokEmb->usmPtr,
        d_model, vocab_emb,
        promptIds, xBuf);
    scaleEmbeddingIfGemma(xBuf, Tp);

    for (std::uint32_t b = 0; b < _config.blockCount; ++b) {
        runBlock(b, xBuf, Tp, cache, buffers);
    }
    cache.commit(Tp);
    const auto preT1 = clock::now();
    const double preMs =
        std::chrono::duration<double, std::milli>(preT1 - preT0).count();

    _traceBlock0 = false;  // diagnostic done; mute for further calls

    // Reseed the sampler per generate() call so deterministic seeds
    // produce reproducible streams. seed == 0 ⇒ random_device internally.
    _sampler.reseed(params.sampling.seed);

    auto isStop = [&](std::int32_t id) -> bool {
        if (id == _tokenizer.eosId()) {
            return true;
        }
        for (auto s : params.stopIds) {
            if (id == s) {
                return true;
            }
        }
        return false;
    };

    // Sample first new token from the last prefill row.
    const float* lastRow = xBuf + (Tp - 1) * d_model;
    std::int32_t nextId = sampleNext(lastRow, vocab_lm,
                                     *outNorm, *lmHead,
                                     normFinal, logits, logitsSc,
                                     params.sampling);

    std::vector<std::int32_t> generated;
    generated.reserve(maxNew);
    generated.push_back(nextId);

    bool aborted = false;
    if (onToken && !onToken(nextId)) {
        aborted = true;
    }

    // -- Decode loop -----------------------------------------------------

    const auto decT0 = clock::now();
    bool hitStop = false;

    for (std::size_t step = 1;
         !aborted && step < maxNew && cache.length() < cacheMax;
         ++step)
    {
        if (isStop(nextId)) {
            hitStop = true;
            break;
        }

        std::array<std::int32_t, 1> oneId{nextId};
        cmp::embeddingLookup(
            tokEmb->type, tokEmb->usmPtr,
            d_model, vocab_emb,
            oneId, xBuf);
        scaleEmbeddingIfGemma(xBuf, 1);

        for (std::uint32_t b = 0; b < _config.blockCount; ++b) {
            runBlock(b, xBuf, 1, cache, buffers);
        }
        cache.commit(1);

        nextId = sampleNext(xBuf, vocab_lm,
                            *outNorm, *lmHead,
                            normFinal, logits, logitsSc,
                            params.sampling);
        generated.push_back(nextId);

        if (onToken && !onToken(nextId)) {
            aborted = true;
        }
    }

    const auto decT1 = clock::now();
    const double decMs =
        std::chrono::duration<double, std::milli>(decT1 - decT0).count();

    if (outStats != nullptr) {
        outStats->promptTokens    = Tp;
        outStats->generatedTokens = generated.size();
        outStats->prefillMs       = preMs;
        outStats->decodeMs        = decMs;
        outStats->hitStop         = hitStop;
    }

    return generated;
}

} // namespace mimirmind::runtime