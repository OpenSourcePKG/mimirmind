#include "runtime/InferenceEngine.hpp"

#include "compute/Embedding.hpp"
#include "model/GgufTypes.hpp"
#include "runtime/BlockBuffers.hpp"
#include "runtime/KvCache.hpp"
#include "runtime/Log.hpp"
#include "runtime/arch/ArchBackend.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
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

    // One-shot architecture diagnostic. For gemma4 also dump the first
    // shared-KV block (block 5 in the standard 26B-A4B layout) — we
    // need to see if its Q tensor has a different output dim than block 0,
    // which would tell us full-attention layers use head_dim_full while
    // SWA layers use head_dim_swa.
    logBlockTensorInventory(_reader, 0);
    if (_config.architecture == "gemma4" && _config.blockCount > 5) {
        logBlockTensorInventory(_reader, 5);
    }

    // Pick the arch backend now that weights are available. Returns
    // nullptr for unsupported architectures so generate() can refuse
    // gracefully with the original architecture string in the error.
    _backend = arch::createArchBackend(
        _config.architecture, _config, *_weights, _ops, _gmm);

    _modelLoaded = true;
    MM_LOG_INFO("engine",
                "loadModel: ready — arch={} blocks={} d_model={} ff={} heads={} kv={}",
                _config.architecture, _config.blockCount,
                _config.embeddingLength, _config.feedForwardLength,
                _config.headCount, _config.headCountKv);
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

    if (_backend == nullptr) {
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

    KvCache cache(_allocator, cacheMax, _backend->kvDimPerLayer());

    const auto [qDimMax, kvDimMax] = _backend->maxQKVDims();
    BlockBuffers buffers = allocBlockBuffers(_allocator, _config,
                                             maxT, cacheMax,
                                             qDimMax, kvDimMax);

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

    // Gemma family scales the token embedding by sqrt(d_model) before
    // it enters the first block — the per-token vectors are otherwise
    // in the ~0.05 range and attention/FFN expects them at unit-ish
    // scale. Qwen/Llama don't do this. Backend tells us.
    const bool  embedScaleEnabled = _backend->scalesEmbedding();
    const float embedScale = embedScaleEnabled
        ? std::sqrt(static_cast<float>(d_model))
        : 1.0F;
    auto scaleEmbeddingIfNeeded = [&](float* dst, std::size_t T) {
        if (embedScaleEnabled && T > 0) {
            _ops.mulScalarAsync(dst, embedScale, T * d_model);
        }
    };

    // -- Prefill ---------------------------------------------------------
    //
    // Optional parity-test dump. `MIMIRMIND_PARITY_DUMP=PREFIX` makes
    // the backend write PREFIX-blk{N}-<stage>.bin at multiple stages
    // inside each block during prefill. Format matches llama-parity-dump.

    if (const char* dumpPrefix = std::getenv("MIMIRMIND_PARITY_DUMP")) {
        _backend->setParityDumpPrefix(dumpPrefix);
    }

    const auto preT0 = clock::now();
    cmp::embeddingLookup(
        tokEmb->type, tokEmb->usmPtr,
        d_model, vocab_emb,
        promptIds, xBuf);
    scaleEmbeddingIfNeeded(xBuf, Tp);

    for (std::uint32_t b = 0; b < _config.blockCount; ++b) {
        _backend->runBlock(b, xBuf, Tp, cache, buffers, _traceBlock0);
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
        scaleEmbeddingIfNeeded(xBuf, 1);

        for (std::uint32_t b = 0; b < _config.blockCount; ++b) {
            _backend->runBlock(b, xBuf, 1, cache, buffers, false);
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