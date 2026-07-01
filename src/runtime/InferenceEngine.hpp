#pragma once

#include "compute/GpuMatmul.hpp"
#include "compute/GpuOps.hpp"
#include "compute/Sampling.hpp"
#include "model/GgufReader.hpp"
#include "model/LlmConfig.hpp"
#include "model/Tokenizer.hpp"
#include "model/FusedQkvWeights.hpp"
#include "model/WeightsMap.hpp"
#include "runtime/BlockBuffers.hpp"
#include "runtime/CommandQueue.hpp"
#include "runtime/KvCache.hpp"
#include "runtime/L0Context.hpp"
#include "runtime/UsmAllocator.hpp"
#include "runtime/UsmHandle.hpp"

namespace mimirmind::runtime {
class GpuClockGovernor;
class PowerMonitor;
class SystemMonitor;
class ThermalGuard;
} // namespace mimirmind::runtime

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace mimirmind::runtime {

namespace arch {
class ArchBackend;
} // namespace arch

/**
 * Sampling + generation knobs. Default is greedy/argmax — sampling.temperature
 * <= 0 keeps the decode loop bit-identical to plain argmax.
 *
 * `stopIds` is an additional list of token ids that terminate the loop
 * (besides the tokenizer's EOS). M7c populates it with chat-template
 * markers like Qwen's `<|im_end|>`.
 */
struct GenerateParams {
    std::size_t                  maxNewTokens{256};
    compute::SamplingParams      sampling{};
    std::vector<std::int32_t>    stopIds{};
};

/**
 * Timings + counters from one generate() call. Optional output param.
 */
struct GenerateStats {
    std::size_t promptTokens{0};
    std::size_t generatedTokens{0};
    /// Prompt tokens that were already in the KV-cache from a previous
    /// generate() call (longest-common-prefix hit). Their prefill cost
    /// is skipped. `promptTokens - cachedTokens` is what actually ran
    /// through the transformer this turn.
    std::size_t cachedTokens{0};
    double      prefillMs{0.0};
    double      decodeMs{0.0};
    /// True if the decode loop broke because of a stop token (tokenizer
    /// EOS, chat-template stop, or user-supplied stop). False means we
    /// hit `maxNewTokens`.
    bool        hitStop{false};

    /// Energy consumed by the CPU package (in Joules) over the full
    /// generate() call. Only populated when a PowerMonitor is installed
    /// and the kernel exposes RAPL counters; otherwise stays 0.
    double      packageJoules{0.0};
};

/**
 * Long-lived inference state: Level-Zero context, USM allocator, GPU
 * matmul dispatcher, plus the currently loaded model (GGUF tensors,
 * config, tokenizer, weights index, per-arch backend). Constructed once,
 * serves many generate() calls (one at a time — KV cache is built per
 * call for now).
 *
 * Lifetime + ownership:
 *   ctor: brings up L0 + USM (probeLimits) + GpuMatmul. Does NOT load
 *         a model — call loadModel(path) for that.
 *   dtor: releases everything in LIFO order.
 *
 * Not thread-safe. One generate() at a time.
 */
class InferenceEngine {
public:
    /// Called per emitted decode token. Returning false aborts the loop.
    using TokenCallback = std::function<bool(std::int32_t id)>;
    /// Payload delivered to the prefill-done callback. `promptTokens` is
    /// the full prompt length; `prefilledTokens` is what actually ran
    /// through the model this call (`promptTokens` minus any prefix
    /// cache-hit). `prefillMs` is the wall time of the prefill phase.
    struct PrefillDone {
        std::size_t promptTokens;
        std::size_t prefilledTokens;
        double      prefillMs;
    };

    /// Called once, immediately after prefill and before the first decode
    /// token is sampled. Used by the streaming server to emit a
    /// `prefill_done` SSE event so a client can flip its UX from "reading
    /// your prompt" to "answering" without waiting for the first token.
    using PrefillCallback = std::function<void(const PrefillDone&)>;

    /// Payload delivered to the per-block prefill-progress callback.
    /// `blocksDone` is 1-indexed count of transformer blocks that just
    /// finished (so it walks 1..blocksTotal). `elapsedMs` measures wall
    /// time since the start of the prefill phase — clients can use it to
    /// project remaining time via linear extrapolation.
    struct PrefillProgress {
        std::size_t blocksDone;
        std::size_t blocksTotal;
        double      elapsedMs;
    };

    /// Called after every transformer block during prefill so a
    /// streaming client can render a progress bar for long prompts.
    /// Callbacks are cheap by design — the server should still rate-
    /// limit outgoing SSE events since a 34-block model on a short
    /// prompt fires this 34 times in a few hundred milliseconds.
    using PrefillProgressCallback = std::function<void(const PrefillProgress&)>;

    InferenceEngine();
    ~InferenceEngine();

    InferenceEngine(const InferenceEngine&)            = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;
    InferenceEngine(InferenceEngine&&)                 = delete;
    InferenceEngine& operator=(InferenceEngine&&)      = delete;

    /// Parse + load a GGUF model into USM. Throws if a model is already
    /// loaded (no hot-swap in M7a) or the file is malformed.
    void loadModel(std::string_view ggufPath);

    [[nodiscard]] bool modelLoaded() const noexcept { return _modelLoaded; }

    /**
     * Prefill `promptIds` then decode up to `params.maxNewTokens` tokens
     * via greedy argmax. Stops early on EOS or when `onToken` returns
     * false. Returns the generated ids (without the prompt).
     */
    std::vector<std::int32_t>
    generate(std::span<const std::int32_t>   promptIds,
             const GenerateParams&           params,
             const TokenCallback&            onToken            = {},
             GenerateStats*                  outStats           = nullptr,
             const PrefillCallback&          onPrefillDone      = {},
             const PrefillProgressCallback&  onPrefillProgress  = {});

    /// Drop the persistent KV-cache so the next generate() starts from
    /// scratch. Cache *buffers* stay allocated — only the logical length
    /// and the cached-token bookkeeping are cleared. Used by tests, by
    /// loadModel(), and as the recovery path when a generate() fails
    /// partway through.
    void resetCache() noexcept;

    /// Number of token ids currently cached (i.e. how long an exact
    /// prefix the next request could potentially skip-prefill).
    [[nodiscard]] std::size_t cachedTokenCount() const noexcept {
        return _cachedTokens.size();
    }

    /// Maximum total prompt + max_new_tokens any single generate() call
    /// can hold in the persistent KV-cache. The cache is sized to this
    /// number once and never reallocated for growth — that's what makes
    /// multi-turn prefix reuse actually work, since growing requests
    /// would otherwise reset the cache between turns.
    ///
    /// Setting a new value AFTER the cache is already allocated takes
    /// effect on the next loadModel()/resetCache cycle. Setting it
    /// before the first generate() is the normal path.
    void setMaxContextTokens(std::size_t n) noexcept { _maxContextTokens = n; }
    [[nodiscard]] std::size_t maxContextTokens() const noexcept {
        return _maxContextTokens;
    }

    /// Install (or remove with nullptr) the thermal guard the decode
    /// loop should consult. The engine does not own the guard. If set,
    /// generate() will call checkAdmission() before prefill (which may
    /// throw ThermalLimitExceeded) and pace the decode loop based on
    /// paceForCurrentReading() between every few tokens.
    void setThermalGuard(ThermalGuard* guard) noexcept { _thermalGuard = guard; }
    [[nodiscard]] ThermalGuard* thermalGuard() const noexcept { return _thermalGuard; }

    /// Install (or remove with nullptr) the power monitor. The engine
    /// does not own it. When set, generate() snapshots RAPL counters
    /// before prefill and after decode, and reports packageJoules in
    /// GenerateStats. No-op when monitor is unavailable.
    void setPowerMonitor(PowerMonitor* monitor) noexcept { _powerMonitor = monitor; }
    [[nodiscard]] PowerMonitor* powerMonitor() const noexcept { return _powerMonitor; }

    /// Install (or remove with nullptr) the GPU clock governor + the
    /// system monitor it reads temperatures from. Both must be
    /// non-null for the governor to tick during decode. The engine
    /// owns neither. When set, every kGovernorTickWindow decode tokens
    /// the engine reads package temp from the monitor and lets the
    /// governor adjust the iGPU max-freq cap toward its target.
    void setGpuClockGovernor(GpuClockGovernor* governor,
                             SystemMonitor*    monitor) noexcept {
        _gpuGovernor    = governor;
        _governorMonitor = monitor;
    }
    [[nodiscard]] GpuClockGovernor* gpuClockGovernor() const noexcept {
        return _gpuGovernor;
    }

    // --- Accessors (used by smoke path + diagnostics) -------------------

    [[nodiscard]] L0Context&               ctx()              noexcept { return _ctx; }
    [[nodiscard]] const L0Context&         ctx()        const noexcept { return _ctx; }
    [[nodiscard]] UsmAllocator&            allocator()        noexcept { return _allocator; }
    [[nodiscard]] const UsmAllocator&      allocator()  const noexcept { return _allocator; }
    [[nodiscard]] compute::GpuMatmul&      gpuMatmul()        noexcept { return _gmm; }

    [[nodiscard]] const model::GgufReader& reader()    const noexcept { return _reader; }
    [[nodiscard]] const model::LlmConfig&  config()    const noexcept { return _config; }
    [[nodiscard]] const model::Tokenizer&  tokenizer() const noexcept { return _tokenizer; }
    [[nodiscard]] const model::WeightsMap& weights()   const;

private:
    /// Compute logits over the last hidden state row via final-norm +
    /// lm_head, then draw one token id using `_sampler` and `params`.
    std::int32_t sampleNext(const float*                  hidden,
                            std::size_t                   vocab_lm,
                            const model::GgufTensor&      outNorm,
                            const model::GgufTensor&      lmHead,
                            float*                        normScratch,
                            float*                        logits,
                            float*                        matmulScratch,
                            const compute::SamplingParams& sampling);

    /// Allocate (lazily on first call) the persistent KV-cache at the
    /// configured `_maxContextTokens`. Validates that the request fits.
    /// Reallocates the chunk-sized scratch (BlockBuffers, xBuf, logits)
    /// only when the request grew — these are state-free and can be
    /// thrown out without losing cached tokens.
    void ensureCapacity(std::size_t maxT, std::size_t Tp, std::size_t maxNew,
                       std::size_t vocab_lm, std::size_t d_model);

    L0Context                          _ctx;
    UsmAllocator                       _allocator;
    CommandQueue                       _queue;
    compute::GpuMatmul                 _gmm;
    compute::GpuOps                    _ops;
    compute::Sampler                   _sampler{};

    model::GgufReader                  _reader;
    model::LlmConfig                   _config;
    model::Tokenizer                   _tokenizer;
    std::optional<model::WeightsMap>       _weights;
    std::unique_ptr<model::FusedQkvWeights> _fusedQkv;
    std::unique_ptr<arch::ArchBackend>      _backend;
    bool                               _modelLoaded{false};

    // --- Persistent inference state (M9.1 prefix cache) -----------------
    // KvCache + BlockBuffers + scratch buffers stay around across
    // generate() calls so the second call into a conversation can reuse
    // the KV state for the matching prefix. _cachedTokens mirrors what
    // sits in the cache so the next generate() can compute the LCP.
    std::unique_ptr<KvCache>           _kvCache;
    std::optional<BlockBuffers>        _blockBuffers;
    UsmHandle                          _xBufH;
    UsmHandle                          _normFinalH;
    UsmHandle                          _logitsH;
    UsmHandle                          _logitsScH;
    std::size_t                        _maxContextTokens{8192}; // see setMaxContextTokens
    std::size_t                        _cacheMaxT    {0};   // max prompt-chunk scratch was sized for
    std::size_t                        _cacheVocabLm {0};   // lm-head vocab the logits buf fits
    std::vector<std::int32_t>          _cachedTokens;

    // Optional non-owning thermal guard. Engine consults it once before
    // prefill (admission) and every few decode tokens (pacing).
    ThermalGuard*                      _thermalGuard{nullptr};
    // Optional non-owning power monitor. Engine snapshots counters
    // around each generate() call to populate GenerateStats.packageJoules.
    PowerMonitor*                      _powerMonitor{nullptr};
    // Optional non-owning GPU clock governor + system monitor for
    // dynamic iGPU frequency control during decode.
    GpuClockGovernor*                  _gpuGovernor{nullptr};
    SystemMonitor*                     _governorMonitor{nullptr};

    // One-shot block-0 trace. Default off so production serve mode is
    // quiet; set MIMIRMIND_TRACE_BLOCK0=1 to enable when bringing up a
    // new architecture handler. Flips to false after the first call so
    // even an enabled trace only fires once per process.
    bool                               _traceBlock0{false};

    // Per-decode-token telemetry. When MIMIRMIND_TRACE_DECODE_FILE is
    // set, the engine opens an NDJSON sink at startup and writes one
    // line per decoded token: {tok, wall_ms, cap_mhz, pkg_c}. Used to
    // diagnose perf regressions (M5f.5 postmortem motivated this) and
    // to validate optimizations after the fact instead of guessing.
    // nullptr when the env var is unset — zero overhead in production.
    std::FILE*                         _decodeTrace{nullptr};
};

} // namespace mimirmind::runtime