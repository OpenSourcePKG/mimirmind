#pragma once

#include "compute/GpuMatmul.hpp"
#include "compute/GpuOps.hpp"
#include "compute/Sampling.hpp"
#include "model/GgufReader.hpp"
#include "model/LlmConfig.hpp"
#include "model/Tokenizer.hpp"
#include "model/WeightsMap.hpp"
#include "runtime/BlockBuffers.hpp"
#include "runtime/CommandQueue.hpp"
#include "runtime/KvCache.hpp"
#include "runtime/L0Context.hpp"
#include "runtime/UsmAllocator.hpp"
#include "runtime/UsmHandle.hpp"

#include <cstddef>
#include <cstdint>
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
    generate(std::span<const std::int32_t> promptIds,
             const GenerateParams&         params,
             const TokenCallback&          onToken  = {},
             GenerateStats*                outStats = nullptr);

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

    /// Allocate or grow the persistent KV-cache, BlockBuffers, and the
    /// reusable xBuf/normFinal/logits/logitsSc scratch buffers so they
    /// fit a request of `Tp` prompt tokens with a cache capacity of
    /// `cacheMax`. Invalidates `_cachedTokens` (the data is being
    /// thrown away if the buffers grow). No-op if the existing
    /// allocations already fit.
    void ensureCapacity(std::size_t Tp, std::size_t cacheMax,
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
    std::optional<model::WeightsMap>   _weights;
    std::unique_ptr<arch::ArchBackend> _backend;
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
    std::size_t                        _cacheCapacity{0};   // cacheMax it was sized for
    std::size_t                        _cacheMaxT    {0};   // max prompt-chunk it was sized for
    std::size_t                        _cacheVocabLm {0};   // lm-head vocab the logits buf fits
    std::vector<std::int32_t>          _cachedTokens;

    // One-shot block-0 trace. Default off so production serve mode is
    // quiet; set MIMIRMIND_TRACE_BLOCK0=1 to enable when bringing up a
    // new architecture handler. Flips to false after the first call so
    // even an enabled trace only fires once per process.
    bool                               _traceBlock0{false};
};

} // namespace mimirmind::runtime