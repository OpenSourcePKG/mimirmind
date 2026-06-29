#pragma once

#include "compute/GpuMatmul.hpp"
#include "compute/GpuOps.hpp"
#include "compute/Sampling.hpp"
#include "model/GgufReader.hpp"
#include "model/LlmConfig.hpp"
#include "model/Tokenizer.hpp"
#include "model/WeightsMap.hpp"
#include "runtime/CommandQueue.hpp"
#include "runtime/L0Context.hpp"
#include "runtime/UsmAllocator.hpp"
#include "runtime/UsmHandle.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace mimirmind::runtime {

class KvCache;

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
 * config, tokenizer, weights index). Constructed once, serves many
 * generate() calls (one at a time — KV cache is built per call for now).
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
    /// Per-call transformer-block scratch in USM.
    struct BlockBuffers {
        std::size_t maxT{0};
        std::size_t maxSeq{0};
        std::size_t d_model{0};
        std::size_t q_dim{0};
        std::size_t ff_dim{0};

        UsmHandle qBuf;          // [maxT, q_dim]
        UsmHandle normBuf;       // [maxT, d_model]
        UsmHandle attnOut;       // [maxT, q_dim]
        UsmHandle projOut;       // [maxT, d_model]
        UsmHandle gateOut;       // [maxT, ff_dim]
        UsmHandle upOut;         // [maxT, ff_dim]
        UsmHandle matmulScratch; // max(d_model, q_dim, ff_dim)
        UsmHandle scoreScratch;  // [maxSeq]
    };

    BlockBuffers allocBlockBuffers(std::size_t maxT, std::size_t maxSeq);

    /// Internal architecture tag. Set at loadModel based on
    /// _config.architecture; controls which runXxxBlock variant
    /// generate() dispatches per layer.
    enum class Architecture {
        Qwen2,       ///< Qwen2 / Qwen2.5 / llama-family decoder block
        Gemma4,      ///< Gemma 4 26B-A4B-it — hybrid dense+MoE, Q-K-norm
        Unsupported, ///< Loader works but forward is not implemented
    };

    /// Qwen2 / Qwen2.5 / llama-family block forward. RMSNorm → Q/K/V
    /// proj → RoPE → causal multi-head attention (GQA) → O proj →
    /// residual → RMSNorm → SwiGLU FFN (gate * silu(up)) → residual.
    void runQwen2Block(std::size_t   blockIdx,
                       float*        x,
                       std::size_t   T,
                       KvCache&      cache,
                       BlockBuffers& s);

    /// Gemma 4 hybrid dense+MoE block forward — skeleton, throws.
    /// Real implementation lands incrementally in M8.3 (Q-K-norm +
    /// multi-norm choreography), M8.4 (sliding-window attention),
    /// M8.5 (dense FFN), M8.6 (MoE routing + experts), M8.7
    /// (integration + CPU‖GPU split). See
    /// Memory/mimirmind/research/m8-gemma4-staging.md.
    void runGemma4Block(std::size_t   blockIdx,
                        float*        x,
                        std::size_t   T,
                        KvCache&      cache,
                        BlockBuffers& s);

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
    Architecture                       _arch{Architecture::Unsupported};
    bool                               _modelLoaded{false};

    // One-shot block-0 trace. Default off so production serve mode is
    // quiet; set MIMIRMIND_TRACE_BLOCK0=1 to enable when bringing up a
    // new architecture handler. Flips to false after the first call so
    // even an enabled trace only fires once per process.
    bool                               _traceBlock0{false};
};

} // namespace mimirmind::runtime