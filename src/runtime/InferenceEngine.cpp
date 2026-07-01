#include "runtime/InferenceEngine.hpp"

#include "compute/Embedding.hpp"
#include "model/GgufTypes.hpp"
#include "runtime/Lcp.hpp"
#include "runtime/Log.hpp"
#include "runtime/GpuClockGovernor.hpp"
#include "runtime/PowerMonitor.hpp"
#include "runtime/SystemMonitor.hpp"
#include "runtime/ThermalGuard.hpp"
#include "runtime/arch/ArchBackend.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
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
      _ops{_ctx, _allocator, _queue} {
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

    // Per-decode-token NDJSON telemetry sink. Opens the file truncating;
    // the engine never rotates so a long-running process keeps appending.
    // Operators should rotate externally if they care.
    if (const char* path = std::getenv("MIMIRMIND_TRACE_DECODE_FILE")) {
        if (path[0] != '\0') {
            _decodeTrace = std::fopen(path, "w");
            if (_decodeTrace != nullptr) {
                MM_LOG_INFO("engine",
                            "MIMIRMIND_TRACE_DECODE_FILE={} — per-token "
                            "decode trace enabled (wall_ms, cap_mhz, pkg_c)",
                            path);
            } else {
                MM_LOG_WARN("engine",
                            "MIMIRMIND_TRACE_DECODE_FILE={} — could not "
                            "open for writing, decode trace disabled",
                            path);
            }
        }
    }
}

InferenceEngine::~InferenceEngine() {
    if (_decodeTrace != nullptr) {
        std::fclose(_decodeTrace);
        _decodeTrace = nullptr;
    }
}

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

    // M5i.B: Try to fuse per-block attn_q/k/v weights so the QKV
    // projections can dispatch as one matmul. The class no-ops when
    // MIMIRMIND_DISABLE_FUSED_QKV is set or when a block doesn't
    // qualify (missing tensors, mismatched types).
    _fusedQkv = std::make_unique<model::FusedQkvWeights>(
        *_weights, _allocator, _config.blockCount);

    // M5i.D: Load-time self-tests of the GPU compute path. selfTest
    // verifies every non-matmul GPU op against a CPU reference on a
    // tiny fixed input — catches broken SPV loads and driver bugs on
    // unfamiliar iGPU µarchs. autotune runs a matvec-vs-GEMM micro-
    // bench (with its own parity gate) and pins the dispatch decision
    // per QuantType. Honours MIMIRMIND_DISABLE_GEMM / _FORCE_GEMM.
    _ops.selfTest(_allocator);
    _gmm.autotune(_allocator, _config.embeddingLength);

    // Pick the arch backend now that weights are available. Returns
    // nullptr for unsupported architectures so generate() can refuse
    // gracefully with the original architecture string in the error.
    _backend = arch::createArchBackend(
        _config.architecture, _config, *_weights, _fusedQkv.get(),
        _ops, _gmm);

    _modelLoaded = true;
    // Defensive: a previous model's KV state must not survive into the
    // new model. The current API throws on loadModel-while-loaded so
    // this is theoretical, but it keeps the invariant local.
    resetCache();
    MM_LOG_INFO("engine",
                "loadModel: ready — arch={} blocks={} d_model={} ff={} heads={} kv={}",
                _config.architecture, _config.blockCount,
                _config.embeddingLength, _config.feedForwardLength,
                _config.headCount, _config.headCountKv);
}

void InferenceEngine::resetCache() noexcept {
    if (_kvCache != nullptr) {
        _kvCache->reset();
    }
    _cachedTokens.clear();
}

void InferenceEngine::ensureCapacity(std::size_t maxT, std::size_t Tp,
                                     std::size_t maxNew,
                                     std::size_t vocab_lm, std::size_t d_model) {
    // KvCache + cachedTokens: lifetime-critical. Allocate once at the
    // configured _maxContextTokens; never realloc on request growth.
    // This is the change that makes multi-turn prefix reuse actually
    // work — previously a growing prompt or max_tokens would trip the
    // realloc-and-reset path and lose every cached token.
    if (_kvCache == nullptr) {
        _kvCache = std::make_unique<KvCache>(
            _allocator, _maxContextTokens, _backend->kvDimPerLayer());
        MM_LOG_INFO("kvcache",
                    "pre-allocated for {} tokens (set via "
                    "MIMIRMIND_MAX_CONTEXT_TOKENS)",
                    _maxContextTokens);
    }

    // Hard cap: a request that doesn't fit gets a clear error rather
    // than silently overflowing the cache. Operator can raise
    // MIMIRMIND_MAX_CONTEXT_TOKENS if the workload needs it.
    if (Tp + maxNew + 4 > _maxContextTokens) {
        throw std::runtime_error(
            "generate: request needs " + std::to_string(Tp + maxNew + 4) +
            " tokens of KV (prompt " + std::to_string(Tp) +
            " + max_new " + std::to_string(maxNew) +
            " + slack 4) but the engine is configured for "
            + std::to_string(_maxContextTokens) +
            " — raise MIMIRMIND_MAX_CONTEXT_TOKENS or shrink the request");
    }

    // BlockBuffers + scratch: purely transient, safe to realloc on
    // demand. scoreScratch inside BlockBuffers is sized to the cache
    // capacity (not the request chunk) so attention can scan the full
    // current KV length even when the chunk is just 1 decode token.
    const bool needScratchGrow =
        !_blockBuffers.has_value() ||
        maxT      > _cacheMaxT ||
        vocab_lm  > _cacheVocabLm;

    if (!needScratchGrow) {
        return;
    }

    const auto [qDimMax, kvDimMax] = _backend->maxQKVDims();
    const bool withFusedQkv =
        _fusedQkv != nullptr && _fusedQkv->anyFused();
    _blockBuffers = allocBlockBuffers(_allocator, _config,
                                      maxT, _maxContextTokens,
                                      qDimMax, kvDimMax,
                                      withFusedQkv);

    _xBufH      = UsmHandle{_allocator, maxT      * d_model  * sizeof(float)};
    _normFinalH = UsmHandle{_allocator, d_model   * sizeof(float)};
    _logitsH    = UsmHandle{_allocator, vocab_lm  * sizeof(float)};
    _logitsScH  = UsmHandle{_allocator, d_model   * sizeof(float)};

    _cacheMaxT     = maxT;
    _cacheVocabLm  = vocab_lm;
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
InferenceEngine::generate(std::span<const std::int32_t>   promptIds,
                          const GenerateParams&           params,
                          const TokenCallback&            onToken,
                          GenerateStats*                  outStats,
                          const PrefillCallback&          onPrefillDone,
                          const PrefillProgressCallback&  onPrefillProgress) {
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

    // M9.2 thermal admission. Throws ThermalLimitExceeded if any hard
    // limit in the configured profile is currently breached; ApiServer
    // turns that into HTTP 503 + Retry-After. Skips silently when no
    // guard is installed (op chose to run unprotected).
    if (_thermalGuard != nullptr) {
        _thermalGuard->checkAdmission();
    }

    // Snapshot RAPL counters at the start so generate() can report how
    // much energy the request consumed. No-op when the monitor is
    // unavailable or absent.
    PowerMonitor::Snapshot powerStart{};
    if (_powerMonitor != nullptr && _powerMonitor->available()) {
        powerStart = _powerMonitor->snapshot();
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
    const std::size_t maxT      = std::max<std::size_t>(Tp, 1);
    const std::size_t d_model   = _config.embeddingLength;

    ensureCapacity(maxT, Tp, maxNew, vocab_lm, d_model);
    KvCache&      cache   = *_kvCache;
    BlockBuffers& buffers = *_blockBuffers;

    float* const xBuf      = _xBufH     .as<float>();
    float* const normFinal = _normFinalH.as<float>();
    float* const logits    = _logitsH   .as<float>();
    float* const logitsSc  = _logitsScH .as<float>();

    // --- M9.1 prefix cache ----------------------------------------------
    //
    // _cachedTokens holds the ids whose K/V state is currently sitting in
    // `cache` from a previous generate() call. Compute how many leading
    // tokens of the new prompt match — those tokens can re-use the
    // existing KV rows.
    //
    // Clamp to Tp - 1: even on a perfect prefix match we still need to
    // re-run prefill for the final prompt token, because sampleNext()
    // reads its hidden state directly from xBuf (the cache only stores
    // K/V, not the hidden state that feeds the lm-head).
    std::size_t lcp = longestCommonPrefix(promptIds,
                                          std::span<const std::int32_t>{_cachedTokens});
    if (lcp >= Tp) {
        lcp = Tp - 1;
    }
    cache.truncate(lcp);
    const std::size_t prefillStart = lcp;
    const std::size_t prefillCount = Tp - lcp;

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

    std::vector<std::int32_t> generated;
    double                    preMs   = 0.0;
    double                    decMs   = 0.0;
    bool                      hitStop = false;
    bool                      aborted = false;

    try {
        const auto preT0 = clock::now();
        const auto prefillIds = promptIds.subspan(prefillStart, prefillCount);
        cmp::embeddingLookup(
            tokEmb->type, tokEmb->usmPtr,
            d_model, vocab_emb,
            prefillIds, xBuf);
        scaleEmbeddingIfNeeded(xBuf, prefillCount);

        for (std::uint32_t b = 0; b < _config.blockCount; ++b) {
            _backend->runBlock(b, xBuf, prefillCount, cache, buffers,
                               _traceBlock0);
            if (onPrefillProgress) {
                const auto now = clock::now();
                const double elapsedMs =
                    std::chrono::duration<double, std::milli>(now - preT0)
                        .count();
                onPrefillProgress(PrefillProgress{
                    static_cast<std::size_t>(b) + 1,
                    static_cast<std::size_t>(_config.blockCount),
                    elapsedMs,
                });
            }
        }
        cache.commit(prefillCount);
        const auto preT1 = clock::now();
        preMs =
            std::chrono::duration<double, std::milli>(preT1 - preT0).count();

        if (onPrefillDone) {
            onPrefillDone(PrefillDone{Tp, prefillCount, preMs});
        }

        _traceBlock0 = false;  // diagnostic done; mute for further calls

        // Reseed the sampler per generate() call so deterministic seeds
        // produce reproducible streams. seed == 0 ⇒ random_device.
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

        // Sample first new token from the last prefill row. xBuf only
        // holds the freshly-prefilled suffix, so the last row sits at
        // (prefillCount - 1) * d_model — not (Tp - 1).
        const float* lastRow = xBuf + (prefillCount - 1) * d_model;
        std::int32_t nextId = sampleNext(lastRow, vocab_lm,
                                         *outNorm, *lmHead,
                                         normFinal, logits, logitsSc,
                                         params.sampling);

        generated.reserve(maxNew);
        generated.push_back(nextId);

        if (onToken && !onToken(nextId)) {
            aborted = true;
        }

        // -- Decode loop -------------------------------------------------

        const auto decT0 = clock::now();

        // Inter-token thermal pacing — consult guard every kPaceWindow
        // tokens so /sys reads don't dominate the inner loop. Window of
        // 4 keeps overhead under a millisecond per token at ~145 ms/tok
        // decode while still reacting to a fast temperature climb
        // within ~500 ms.
        constexpr std::size_t kPaceWindow = 4;
        // The GPU clock governor adjusts the iGPU max-freq cap; this
        // happens at a slower cadence than the per-token pacing
        // because a fresh sysfs write costs ~200 µs and reaches the
        // hardware on the next dispatch. 8 tokens at ~145 ms each is
        // ~1.2 s between adjustments — well within the package
        // thermal time constant.
        constexpr std::size_t kGovernorWindow = 8;

        for (std::size_t step = 1;
             !aborted && step < maxNew && cache.length() < _maxContextTokens;
             ++step)
        {
            if (isStop(nextId)) {
                hitStop = true;
                break;
            }

            if (_thermalGuard != nullptr && (step % kPaceWindow) == 0) {
                const auto pause = _thermalGuard->paceForCurrentReading();
                if (pause.count() > 0) {
                    std::this_thread::sleep_for(pause);
                }
            }

            if (_gpuGovernor != nullptr && _governorMonitor != nullptr &&
                (step % kGovernorWindow) == 0) {
                (void)_gpuGovernor->tick(*_governorMonitor);
            }

            const auto tokT0 = clock::now();

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

            // Per-token trace, only when the env-controlled sink is open.
            // sampleNext sync-waits on the lmHead matmul so tokT1 is real
            // wall-time including the whole layer chain.
            if (_decodeTrace != nullptr) {
                const auto tokT1 = clock::now();
                const double tokMs = std::chrono::duration<double, std::milli>(
                    tokT1 - tokT0).count();
                std::uint32_t cap = 0;
                double pkg = -1.0;
                if (_gpuGovernor != nullptr) {
                    cap = _gpuGovernor->currentCapMhz();
                }
                if (_governorMonitor != nullptr) {
                    const auto r = _governorMonitor->read();
                    if (r.package_temp_c.has_value()) {
                        pkg = static_cast<double>(*r.package_temp_c);
                    }
                }
                std::fprintf(_decodeTrace,
                             "{\"tok\":%zu,\"wall_ms\":%.3f,"
                             "\"cap_mhz\":%u,\"pkg_c\":%.1f}\n",
                             step, tokMs, cap, pkg);
            }

            if (onToken && !onToken(nextId)) {
                aborted = true;
            }
        }

        if (_decodeTrace != nullptr) {
            std::fflush(_decodeTrace);
        }

        const auto decT1 = clock::now();
        decMs =
            std::chrono::duration<double, std::milli>(decT1 - decT0).count();
    } catch (...) {
        // Mid-flight failure leaves the KV state partially written.
        // Discarding the cache is cheap and keeps the next call honest.
        resetCache();
        throw;
    }

    // -- Update the prefix cache so the next generate() can resume -----
    //
    // Maintain the invariant `_cachedTokens.size() == cache.length()`.
    // cache.length() at this point is: prefillStart + prefillCount +
    // (decode steps that committed). The first sampled token is not
    // yet committed; each subsequent loop iteration commits exactly one
    // token (the previous step's sample) before sampling the next one.
    {
        const std::size_t finalLen   = cache.length();
        const std::size_t genFromCache = (finalLen > Tp) ? (finalLen - Tp) : 0;
        const std::size_t take       = std::min(genFromCache, generated.size());

        _cachedTokens.clear();
        _cachedTokens.reserve(finalLen);
        _cachedTokens.insert(_cachedTokens.end(),
                             promptIds.begin(), promptIds.end());
        _cachedTokens.insert(_cachedTokens.end(),
                             generated.begin(),
                             generated.begin() +
                                 static_cast<std::ptrdiff_t>(take));
    }

    if (outStats != nullptr) {
        outStats->promptTokens    = Tp;
        outStats->generatedTokens = generated.size();
        outStats->cachedTokens    = lcp;
        outStats->prefillMs       = preMs;
        outStats->decodeMs        = decMs;
        outStats->hitStop         = hitStop;

        if (_powerMonitor != nullptr && _powerMonitor->available() &&
            !powerStart.raw_energy_uj.empty()) {
            const auto powerEnd = _powerMonitor->snapshot();
            const auto joules   = _powerMonitor->energyBetween(powerStart, powerEnd);
            // The first discovered domain is the package socket (intel-rapl:0).
            // Report that as the headline figure. Operators who want the
            // per-sub-domain split can scrape /v1/system/status.
            if (!joules.empty()) {
                outStats->packageJoules = joules.front();
            }
        }
    }

    return generated;
}

} // namespace mimirmind::runtime