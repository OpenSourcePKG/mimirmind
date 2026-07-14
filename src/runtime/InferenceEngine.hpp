#pragma once

#include "compute/GpuMatmul.hpp"
#include "compute/GpuOps.hpp"
#include "compute/Sampling.hpp"
#include "core/gguf/GgufReader.hpp"
#include "model/LlmConfig.hpp"
#include "model/Tokenizer.hpp"
#include "model/FusedQkvWeights.hpp"
#include "core/gguf/WeightsMap.hpp"
#include "core/ipc/TensorManifest.hpp"
#include "runtime/BlockBuffers.hpp"
#include "runtime/CommandQueue.hpp"
#include "runtime/KvCache.hpp"
#include "core/l0/L0Context.hpp"
#include "runtime/OpProfiler.hpp"
#include "core/l0/UsmAllocator.hpp"
#include "core/l0/UsmHandle.hpp"

namespace mimirmind::core::config {
struct Config;
}

namespace mimirmind::runtime {
class FanController;
class GpuClockGovernor;
class PerfRegressionDetector;
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

using ::mimirmind::core::l0::L0Context;
using ::mimirmind::core::l0::UsmAllocator;
using ::mimirmind::core::l0::UsmHandle;
using ::mimirmind::core::config::Config;

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

    /// M9.11.4 — populated by SpeculativeDecoder when the spec-dec
    /// loop engages; stay at 0 on the target-only fall-through path.
    /// `specDecRounds`   how many draft+verify rounds ran
    /// `specDecDrafted`  total tokens the draft produced across all rounds
    /// `specDecAccepted` how many of those the target accepted (excludes
    ///                   the target's own bonus/recovery sample)
    std::size_t specDecRounds{0};
    std::size_t specDecDrafted{0};
    std::size_t specDecAccepted{0};
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
    ///
    /// Return value: `true` to keep going, `false` to abort the prefill
    /// at the next block barrier (M7g). Abort is best-effort — the
    /// currently running block finishes first. The engine then rolls
    /// back the partial KV state, skips the decode phase, and returns
    /// an empty result set with `outStats->prefillMs` populated.
    using PrefillProgressCallback = std::function<bool(const PrefillProgress&)>;

    /// `cfg` is stored by reference and consulted for the runtime section
    /// (spvDir → L0Context, usmProbeTotalGib → UsmAllocator), the feature
    /// section (flashPrefill → GpuOps, gemm/gemmMinM/gemmV2/dp4a → GpuMatmul
    /// autotune, fusedQkv → FusedQkvWeights), and the diagnostics section
    /// (traceBlock0, traceDecodeFile). Per-model runtime overrides are
    /// applied by the operator via `setKvDtype`/`setMaxContextTokens`
    /// after resolving `cfg.effectiveRuntime(modelId)`.
    explicit InferenceEngine(const Config& cfg);
    ~InferenceEngine();

    InferenceEngine(const InferenceEngine&)            = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;
    InferenceEngine(InferenceEngine&&)                 = delete;
    InferenceEngine& operator=(InferenceEngine&&)      = delete;

    /// Parse + load a GGUF model into USM. Throws if a model is already
    /// loaded (no hot-swap in M7a) or the file is malformed.
    void loadModel(std::string_view ggufPath);

    /**
     * Attached-mode load for the M-Munin flow. Opens `ggufPath` only for
     * header parsing (LlmConfig + tokenizer + fingerprint verification),
     * then constructs the WeightsMap via
     * `WeightsMap::fromAttachedChunked(manifest, chunkBases)`. Every
     * tensor's `usmPtr` is `chunkBases[chunkIndex] + chunkOffset` —
     * Munin-owned USM, no copy. Does not call `_reader.loadTensors` —
     * that is what the deploy-downtime win is about.
     *
     * `manifest.modelFingerprint` is what the peer Munin advertised for
     * the model. This method recomputes the fingerprint locally from
     * `ggufPath` and refuses the attach if they disagree — see M-Munin
     * ADR "Modell-Fingerprint" for the reasoning ("Munin hat Q6, Worker
     * erwartet Q8"). Throws on mismatch, with both hashes in the
     * message.
     *
     * Throws if a model is already loaded on this engine.
     */
    void loadModelAttached(std::string_view                        ggufPath,
                           const core::ipc::TensorManifest&        manifest,
                           std::span<void* const>                  chunkBases);

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

    /**
     * M9.11.3 — batched forward pass for speculative-decode verification.
     *
     * Runs a single forward over `newTokens` starting from the current
     * KV-cache position, and returns one logits vector per input
     * position. The accept/reject logic (M9.11.4) consumes these to
     * decide how much of the draft's speculation to commit.
     *
     * Precondition: the KV cache is at position `cachedTokenCount()` —
     * i.e. a prior generate() has run through prefill (and possibly a
     * decode step) so the target's committed prefix already lives in
     * the cache. The N speculative tokens ride on top of it.
     *
     * Semantics:
     * - runBlock() writes provisional K/V rows at positions [len, len+N)
     *   but this method does NOT commit — the caller decides how many
     *   to accept and calls `commitVerified(k)` (or nothing to discard
     *   all N). Overwrites on the next forwardVerify() replace stale
     *   rows.
     * - Skips thermal / fan / perf-detector telemetry — verify is a
     *   sub-primitive inside a larger generate() call.
     * - Per-position logits: rmsNorm(final) + lmHead matmul is looped
     *   N times against M=1. The block matmuls dominate cost, so a
     *   batched lm-head path is not worth the extra scratch buffer.
     *
     * Throws if the KV cache would overflow (`cache.length() + N` past
     * the configured max), or if the model isn't loaded, or if
     * newTokens is empty.
     */
    std::vector<std::vector<float>> forwardVerify(
        std::span<const std::int32_t> newTokens);

    /**
     * M9.11.3 companion — commit `k` of the N speculative rows that
     * `forwardVerify()` just wrote provisionally. `k` must be
     * `<= last_verified_N`. Also appends the newly-committed token ids
     * to `_cachedTokens` so the prefix-cache stays consistent.
     *
     * Kept separate from `forwardVerify` so the accept logic can
     * inspect the returned logits before deciding how far to commit.
     */
    void commitVerified(std::span<const std::int32_t> acceptedTokens);

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

    /// M10.2 Phase 0 — configure KV cache element dtype. Same lifecycle
    /// caveat as setMaxContextTokens(): call once before the first
    /// generate() (i.e. before ensureCapacity() lazy-constructs the
    /// KvCache). Changing the value after the cache is built is a no-op
    /// until the cache is destroyed and rebuilt.
    ///
    /// M10.2 Phase 0 Commit 8 — when `dtype == FP16` this method
    /// validates the loaded model against the F32-only paths that
    /// remain after Commit 5: every own-KV block must have a
    /// fused-QKV entry (so the K/V projection routes through
    /// `qkv_split_fp16` instead of raw fp32 matmul), and no block may
    /// carry an `attn_k.bias` or `attn_v.bias` tensor (the fp32
    /// `addBiasAsync` would corrupt fp16 slots). Throws
    /// `std::runtime_error` with the failing block index and reason
    /// on violation. Requires `loadModel()` to have run first.
    /// F32 always passes; kept `noexcept`-behaviourally by early-exit.
    void setKvDtype(KvDtype dtype);
    [[nodiscard]] KvDtype kvDtype() const noexcept { return _kvDtype; }

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

    /// Install (or remove with nullptr) the in-process perf-regression
    /// detector. Non-owning. When set, generate() feeds it one Sample
    /// per decode token and calls onRunComplete() at the end. The
    /// detector persists a rolling baseline to a JSON file and exposes
    /// current/baseline p50 + last alert through /v1/system/status.
    void setPerfRegressionDetector(PerfRegressionDetector* d) noexcept {
        _perfDetector = d;
    }
    [[nodiscard]] PerfRegressionDetector* perfRegressionDetector() const noexcept {
        return _perfDetector;
    }

    /// Install (or remove with nullptr) the chassis-fan controller.
    /// Non-owning. When set, generate() calls boost() at the start of
    /// each request and releaseToAuto() at the end — proactive cooling
    /// so the GPU clock governor has headroom to hold high caps during
    /// sustained decode. No-op when the controller reports unavailable.
    void setFanController(FanController* c) noexcept {
        _fanController = c;
    }
    [[nodiscard]] FanController* fanController() const noexcept {
        return _fanController;
    }

    // --- Accessors (used by smoke path + diagnostics) -------------------

    [[nodiscard]] L0Context&               ctx()              noexcept { return _ctx; }
    [[nodiscard]] const L0Context&         ctx()        const noexcept { return _ctx; }
    [[nodiscard]] UsmAllocator&            allocator()        noexcept { return _allocator; }
    [[nodiscard]] const UsmAllocator&      allocator()  const noexcept { return _allocator; }
    [[nodiscard]] compute::GpuMatmul&      gpuMatmul()        noexcept { return _gmm; }
    [[nodiscard]] const compute::GpuMatmul& gpuMatmul()  const noexcept { return _gmm; }
    [[nodiscard]] const compute::GpuOps&   gpuOps()     const noexcept { return _ops; }
    [[nodiscard]] const model::FusedQkvWeights* fusedQkv() const noexcept {
        return _fusedQkv.get();
    }

    [[nodiscard]] const core::gguf::GgufReader& reader()    const noexcept { return _reader; }
    [[nodiscard]] const model::LlmConfig&  config()    const noexcept { return _config; }
    [[nodiscard]] const model::Tokenizer&  tokenizer() const noexcept { return _tokenizer; }
    [[nodiscard]] const core::gguf::WeightsMap& weights()   const;

private:
    /// Compute logits over the last hidden state row via final-norm +
    /// lm_head, then draw one token id using `_sampler` and `params`.
    /// `recentTokens` — ordered oldest-to-newest history that the
    /// sampler subspans internally for penalty accounting (M7f).
    std::int32_t sampleNext(const float*                   hidden,
                            std::size_t                    vocab_lm,
                            const core::gguf::GgufTensor&       outNorm,
                            const core::gguf::GgufTensor&       lmHead,
                            float*                         normScratch,
                            float*                         logits,
                            float*                         matmulScratch,
                            std::span<const std::int32_t>  recentTokens,
                            const compute::SamplingParams& sampling);

    /// Allocate (lazily on first call) the persistent KV-cache at the
    /// configured `_maxContextTokens`. Validates that the request fits.
    /// Reallocates the chunk-sized scratch (BlockBuffers, xBuf, logits)
    /// only when the request grew — these are state-free and can be
    /// thrown out without losing cached tokens.
    void ensureCapacity(std::size_t maxT, std::size_t Tp, std::size_t maxNew,
                       std::size_t vocab_lm, std::size_t d_model);

    /// Shared tail of loadModel / loadModelAttached. Runs after
    /// `_weights` is populated (standalone: from `_reader`; attached:
    /// from IPC-imported tensors). Builds FusedQkvWeights, autotunes,
    /// picks the arch backend, sets `_modelLoaded` and logs the ready
    /// line.
    void finalizeLoad();

    // Held by reference for the whole process lifetime. Provided by main().
    const Config&                      _cfg;
    L0Context                          _ctx;
    UsmAllocator                       _allocator;
    CommandQueue                       _queue;
    // M8.H.3: _ops is constructed BEFORE _gmm so GpuMatmul can hold
    // a reference to it (the DP4A dispatch path calls
    // GpuOps::xQuantI8Async to fill the internal Xq/Xscale scratch).
    compute::GpuOps                    _ops;
    compute::GpuMatmul                 _gmm;
    // M8.K.0: diagnostic per-op timer. Constructed after _queue so it
    // can hold a reference. Off by default; opt in via
    // `diagnostics.traceOpTimes: true` in config.json.
    OpProfiler                         _opProfiler;
    compute::Sampler                   _sampler{};

    core::gguf::GgufReader                  _reader;
    model::LlmConfig                   _config;
    model::Tokenizer                   _tokenizer;
    std::optional<core::gguf::WeightsMap>       _weights;
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
    KvDtype                            _kvDtype{KvDtype::F32};    // see setKvDtype
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
    // Optional non-owning perf-regression detector. Fed per decode
    // token; consulted from /v1/system/status.
    PerfRegressionDetector*            _perfDetector{nullptr};
    // Optional non-owning fan controller. When present, generate()
    // boosts the fan on entry and releases to auto on exit — proactive
    // cooling for sustained-decode headroom.
    FanController*                     _fanController{nullptr};

    // One-shot block-0 trace. Default off so production serve mode is
    // quiet; enable via `diagnostics.traceBlock0: true` when bringing
    // up a new architecture handler. Flips to false after the first
    // call so even an enabled trace only fires once per process.
    bool                               _traceBlock0{false};

    // Per-decode-token telemetry. When `diagnostics.traceDecodeFile`
    // is a non-empty path, the engine opens an NDJSON sink at startup
    // and writes one line per decoded token: {tok, wall_ms, cap_mhz,
    // pkg_c}. Used to diagnose perf regressions (M5f.5 postmortem
    // motivated this) and to validate optimizations after the fact
    // instead of guessing. nullptr when unset — zero overhead in
    // production.
    std::FILE*                         _decodeTrace{nullptr};
};

} // namespace mimirmind::runtime