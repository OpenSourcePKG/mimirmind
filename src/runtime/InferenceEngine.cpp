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
#include <stdexcept>
#include <string>

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
      _gmm{_ctx} {
    MM_LOG_INFO("engine", "InferenceEngine: probing USM limits");
    _allocator.probeLimits();
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
    return b;
}

void InferenceEngine::runTransformerBlock(std::size_t   blockIdx,
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
    cmp::rmsNorm(x, T, d_model,
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
            cmp::addBias(dst, T, N,
                         static_cast<const float*>(B->usmPtr));
        }
    };

    float* kSlot = cache.writeSlotK(blockIdx);
    float* vSlot = cache.writeSlotV(blockIdx);

    // Q/K/V are independent and all needed before RoPE — batch into one
    // GPU submission, sync once, then run the CPU bias adds.
    trace("Q projection (matmulAsync)");
    projectAsync(qW, q_dim,  qBuf);
    trace("K projection (matmulAsync)");
    projectAsync(kW, kv_dim, kSlot);
    trace("V projection (matmulAsync)");
    projectAsync(vW, kv_dim, vSlot);
    trace("QKV sync");
    _gmm.sync();
    trace("QKV bias add");
    addBiasIf(qB, q_dim,  qBuf);
    addBiasIf(kB, kv_dim, kSlot);
    addBiasIf(vB, kv_dim, vSlot);

    trace("RoPE Q+K");
    cmp::applyRopeInPlace(qBuf, T,
                          _config.headCount,   head_dim, curLen,
                          _config.ropeFreqBase);
    cmp::applyRopeInPlace(kSlot, T,
                          _config.headCountKv, head_dim, curLen,
                          _config.ropeFreqBase);

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

    trace("attn residual");
    cmp::addResidual(x, projOutBuf, T * d_model);

    trace("ffn rmsNorm");
    cmp::rmsNorm(x, T, d_model,
                 static_cast<const float*>(ffnNorm->usmPtr),
                 _config.rmsNormEps,
                 normBuf);

    trace("FFN gate+up (matmulAsync)");
    _gmm.matmulAsync(ffnGate->type, ffnGate->usmPtr, ff_dim, d_model,
                     normBuf, T, gateOutBuf, matmulScratch);
    _gmm.matmulAsync(ffnUp->type, ffnUp->usmPtr, ff_dim, d_model,
                     normBuf, T, upOutBuf, matmulScratch);
    trace("FFN gate+up sync");
    _gmm.sync();

    trace("FFN silu+mul");
    cmp::siluInPlace(gateOutBuf, T * ff_dim);
    cmp::mulInPlace(gateOutBuf, upOutBuf, T * ff_dim);

    trace("FFN down (matmul)");
    _gmm.matmul(ffnDown->type, ffnDown->usmPtr, d_model, ff_dim,
                gateOutBuf, T,
                projOutBuf, matmulScratch);

    trace("ffn residual + exit");
    cmp::addResidual(x, projOutBuf, T * d_model);
}

std::int32_t
InferenceEngine::sampleArgmax(const float*             hidden,
                              std::size_t              vocab_lm,
                              const model::GgufTensor& outNorm,
                              const model::GgufTensor& lmHead,
                              float*                   normScratch,
                              float*                   logits,
                              float*                   matmulScratch) {
    compute::rmsNorm(
        hidden, 1, _config.embeddingLength,
        static_cast<const float*>(outNorm.usmPtr),
        _config.rmsNormEps,
        normScratch);

    _gmm.matmul(
        lmHead.type, lmHead.usmPtr,
        vocab_lm, _config.embeddingLength,
        normScratch, 1,
        logits, matmulScratch);

    std::int32_t best  = 0;
    float        bestV = logits[0];
    for (std::size_t i = 1; i < vocab_lm; ++i) {
        if (logits[i] > bestV) {
            bestV = logits[i];
            best  = static_cast<std::int32_t>(i);
        }
    }
    return best;
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

    // -- Prefill ---------------------------------------------------------

    const auto preT0 = clock::now();
    cmp::embeddingLookup(
        tokEmb->type, tokEmb->usmPtr,
        d_model, vocab_emb,
        promptIds, xBuf);

    for (std::uint32_t b = 0; b < _config.blockCount; ++b) {
        runTransformerBlock(b, xBuf, Tp, cache, buffers);
    }
    cache.commit(Tp);
    const auto preT1 = clock::now();
    const double preMs =
        std::chrono::duration<double, std::milli>(preT1 - preT0).count();

    _traceBlock0 = false;  // diagnostic done; mute for further calls

    // Sample first new token from the last prefill row.
    const float* lastRow = xBuf + (Tp - 1) * d_model;
    std::int32_t nextId = sampleArgmax(lastRow, vocab_lm,
                                       *outNorm, *lmHead,
                                       normFinal, logits, logitsSc);

    std::vector<std::int32_t> generated;
    generated.reserve(maxNew);
    generated.push_back(nextId);

    bool aborted = false;
    if (onToken && !onToken(nextId)) {
        aborted = true;
    }

    // -- Decode loop -----------------------------------------------------

    const auto decT0 = clock::now();
    bool hitEos = false;

    for (std::size_t step = 1;
         !aborted && step < maxNew && cache.length() < cacheMax;
         ++step)
    {
        if (nextId == _tokenizer.eosId()) {
            hitEos = true;
            break;
        }

        std::array<std::int32_t, 1> oneId{nextId};
        cmp::embeddingLookup(
            tokEmb->type, tokEmb->usmPtr,
            d_model, vocab_emb,
            oneId, xBuf);

        for (std::uint32_t b = 0; b < _config.blockCount; ++b) {
            runTransformerBlock(b, xBuf, 1, cache, buffers);
        }
        cache.commit(1);

        nextId = sampleArgmax(xBuf, vocab_lm,
                              *outNorm, *lmHead,
                              normFinal, logits, logitsSc);
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
        outStats->hitEos          = hitEos;
    }

    return generated;
}

} // namespace mimirmind::runtime