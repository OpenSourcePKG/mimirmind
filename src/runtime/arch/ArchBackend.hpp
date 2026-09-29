// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/gguf/GgufTypes.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace mimirmind::compute {
class ComputeMatmul;
class ComputeOps;
} // namespace mimirmind::compute

namespace mimirmind::runtime {
class OpProfiler;
} // namespace mimirmind::runtime

namespace mimirmind::core::gguf {
class WeightsMap;
struct GgufTensor;
} // namespace mimirmind::core::gguf

namespace mimirmind::model {
class FusedQkvWeights;
struct LlmConfig;
} // namespace mimirmind::model

namespace mimirmind::runtime {
class KvCache;
struct BlockBuffers;
} // namespace mimirmind::runtime

namespace mimirmind::runtime::arch {

using ::mimirmind::core::gguf::GgufTensor;
using ::mimirmind::core::gguf::GgmlType;
using ::mimirmind::core::gguf::WeightsMap;
using ::mimirmind::core::gguf::typeInfo;

/// Load-time-stable capability flags an architecture backend reports to the
/// engine. Replaces the former eight individual `virtual bool` capability
/// queries (8.30.8): a backend now populates ONE struct in
/// `computeCapabilities()` instead of overriding N predicate virtuals. Each
/// field mirrors the semantics of the query it replaces; the defaults match
/// the old default-return of each removed virtual.
struct ArchCapabilities {
    /// `runBlockBatched` is implemented (synchronized batched decode; L0
    /// Gemma 4 MoE). The engine checks this before allocating per-sequence state.
    bool supportsBatchedDecode = false;
    /// Token embedding must be scaled by sqrt(d_model) before block 0 (Gemma).
    bool scalesEmbedding = false;
    /// FP16-KV writes route through an fp32 staging redirect + kv_commit_fp16,
    /// so a raw fp32 K/V matmul never lands in an fp16 slot.
    bool supportsFp16KvStaging = false;
    /// Needs the per-head fused [Q|gate] scratch (Qwen3-Next full-attn gate).
    bool needsQGateScratch = false;
    /// Needs the GatedDeltaNet linear-layer scratch (Qwen3-Next hybrid-recurrent).
    bool needsSsmScratch = false;
    /// Keeps a multi-stream Hyper-Connections residual state (qwen4_exp); the
    /// driver folds the streams via `collapseHyperStreams` before lm_head.
    bool usesHyperConnections = false;
    /// MoE decode block runs fully device-side expert dispatch — no host
    /// routing read — so it is Command-List-Replay-capturable.
    bool moeDecodeClrSafe = false;
    /// Dense decode writes K/V through a replay-stable destination. Default
    /// true; a backend that can fall onto the unfused per-token-slot path
    /// (mixed-quant QKV that FusedQkvWeights refuses to fuse) reports false so
    /// InferenceEngine keeps decode in immediate mode.
    bool decodeQkvClrSafe = true;
};

/**
 * Architecture-specific block forward + per-call hooks.
 *
 * One subclass per supported architecture lives under src/runtime/arch/.
 * InferenceEngine owns exactly one (picked via createArchBackend() at
 * loadModel time) and delegates the per-layer work to it.
 *
 * Backends hold non-owning references to LlmConfig / WeightsMap / GpuOps
 * / GpuMatmul that the engine owns. Constructor injection keeps the
 * runtime hot-path (runBlock) free of lookups.
 */
class ArchBackend {
public:
    virtual ~ArchBackend() = default;

    ArchBackend(const ArchBackend&)            = delete;
    ArchBackend& operator=(const ArchBackend&) = delete;
    ArchBackend(ArchBackend&&)                 = delete;
    ArchBackend& operator=(ArchBackend&&)      = delete;

    /// Run one transformer block in place on `x`. Calls are async on the
    /// shared command queue — the caller flushes before reading on CPU.
    virtual void runBlock(std::size_t   blockIdx,
                          float*        x,
                          std::size_t   T,
                          KvCache&      cache,
                          BlockBuffers& buffers,
                          bool          traceBlock0) = 0;

    /// M-L0.Batch Phase 1 — run one transformer block for `nSeq`
    /// lock-step decode sequences at once (each contributes one row of
    /// `x`, T=1). `caches[i]` is sequence i's own KvCache; all sit at
    /// their pre-forward length (the caller commits each once after the
    /// whole block chain). Batches the position-independent matmuls at
    /// M=nSeq and loops per sequence only for attention/RoPE. Default
    /// throws — only backends that implement synchronized batched decode
    /// override it. `supportsBatchedDecode()` reports availability.
    virtual void runBlockBatched(std::size_t                blockIdx,
                                 float*                     x,
                                 std::size_t                nSeq,
                                 std::span<KvCache* const>  caches,
                                 BlockBuffers&              buffers,
                                 bool                       diag) {
        (void)blockIdx; (void)x; (void)nSeq; (void)caches;
        (void)buffers; (void)diag;
        throw std::runtime_error(
            "runBlockBatched: synchronized batched decode not supported by "
            "this architecture backend");
    }

    /// The backend's load-time-stable capability flags (8.30.8). Computed
    /// once, lazily, on first call — after full construction, so a Qwen
    /// subclass reports its most-derived values (a constructor cannot call
    /// the most-derived `computeCapabilities`). Thread-safe: `std::call_once`
    /// guards the single computation, and the result is immutable afterwards,
    /// so concurrent serving threads read it without a lock. Prefer caching
    /// the reference locally when reading several flags.
    [[nodiscard]] const ArchCapabilities& capabilities() const noexcept {
        std::call_once(_capsOnce, [this] { _caps = computeCapabilities(); });
        return _caps;
    }

    /// KV-cache row width per layer (nKvHeads(l) * headDim(l)). Used by
    /// InferenceEngine to size the KV cache. Length must == blockCount.
    [[nodiscard]] virtual std::vector<std::size_t>
        kvDimPerLayer() const = 0;

    /// Per-layer K/V source for cache aliasing. Entry L is the layer
    /// whose K/V buffer layer L reads/writes. Identity (L == L) means
    /// layer L owns its own cache slot. Any entry < L means the backend
    /// wants layer L to alias an earlier layer's buffer (Gemma 4 E4B
    /// shared-KV). Default = identity — returns {} which KvCache treats
    /// as "every layer owns its cache". Backends that use shared K/V
    /// override this so InferenceEngine skips the per-layer allocation
    /// for aliased layers.
    [[nodiscard]] virtual std::vector<std::size_t>
        kvSourceLayerPerLayer() const { return {}; }

    /// Maximum hidden-state dim across layers for any of: Q output, KV
    /// output. BlockBuffers is sized for this so scratch survives the
    /// largest layer. Returns a pair {qDimMax, kvDimMax}.
    [[nodiscard]] virtual std::pair<std::size_t, std::size_t>
        maxQKVDims() const = 0;

    /// Short identifier for logs ("qwen2", "gemma4").
    [[nodiscard]] virtual const char* name() const noexcept = 0;

    /// 5.27 I-3: collapse the Hyper-Connections streams built up over the block
    /// loop into `out` [T, d_model] (the top-level mixer; replaces output_norm).
    /// No-op default for every non-HC arch. Only called when
    /// `usesHyperConnections()` is true.
    virtual void collapseHyperStreams(std::size_t /*T*/, BlockBuffers& /*s*/,
                                      float* /*out*/) {}

    /// Enable per-stage parity dumps. PREFIX is the same string carried by
    /// `diagnostics.parityDump` in config.json: each stage writes a file at
    ///   <prefix>-blk{N}-<stage>.bin
    /// matching the layout llama-parity-dump produces. Empty string =
    /// disabled (default). Default impl is no-op; backends that wire
    /// intermediate dumps override.
    virtual void setParityDumpPrefix(const std::string& /*prefix*/) noexcept {}

    /// Give the backend a heads-up about the token ids AND the freshly
    /// looked-up token embeddings that are about to run through the
    /// block chain in the next `runBlock` sequence. Called once per
    /// forward pass — before prefill, before every decode step, and
    /// before `forwardVerify`. Called AFTER `embeddingLookup` +
    /// `scaleEmbeddingIfNeeded`, so `hiddenStates` is the exact tensor
    /// that block 0 will consume.
    ///
    /// Non-E-series backends have no per-token per-layer state, so the
    /// default is a no-op. `Gemma4E4BBackend` overrides this to
    /// pre-fetch PLE slices AND run the per_layer_model_proj chain on
    /// `hiddenStates`, combining them into the per-layer-input scratch
    /// that `runBlock` slices per layer.
    ///
    /// Both the span and pointer refer to caller-owned memory that stays
    /// valid for the duration of the block-chain call. The backend
    /// copies whatever it needs synchronously here.
    virtual void prepareForward(std::span<const std::int32_t> /*tokIds*/,
                                const float*                  /*hiddenStates*/,
                                std::size_t                   /*T*/) {}

    /// 5.27 I-9a: reset any per-sequence forward context the backend carries
    /// across steps (qwen4_exp: the PLE rolling n-gram context). Called at a
    /// sequence start (resetCache / a fresh batched run) so a prior generation
    /// cannot contaminate the next. Default no-op.
    virtual void resetForwardContext() {}

    /// 5.27.11.2: batched (per-slot) forward context for the PLE n-gram seam.
    /// tokIds[nSeq] is one token per active slot this step; isSeqStart[nSeq]
    /// marks slots beginning a new sequence (reset their own rolling context).
    /// Selects the per-slot PLE path for the batched forward. Default no-op.
    virtual void prepareForwardBatched(std::span<const std::int32_t> /*tokIds*/,
                                       std::span<const std::uint8_t> /*isSeqStart*/,
                                       std::size_t                   /*nSeq*/) {}

    /// 5.27.11.2: single-slot prefill forward context (qwen4_exp PLE n-gram). A
    /// prefill chunk of `tokens` for one serving slot; `seqStart` (startPos==0)
    /// resets that slot's rolling context. Uses the slot's own context around the
    /// single-session n-gram path (prefill goes through single-session runBlock).
    /// Default no-op.
    virtual void prepareForwardSlot(std::size_t                   /*slot*/,
                                    std::span<const std::int32_t> /*tokens*/,
                                    bool                          /*seqStart*/) {}

protected:
    ArchBackend() = default;

    /// Build this backend's capability flags (8.30.8). Replaces the former
    /// eight individual capability virtuals. Called once, lazily, by
    /// `capabilities()` after full construction — so a Qwen subclass reports
    /// its most-derived values (a constructor cannot). Must be a pure
    /// function of load-time-stable state (config / fused-QKV / delegated
    /// impl); it runs at most once per backend instance.
    [[nodiscard]] virtual ArchCapabilities computeCapabilities() const = 0;

private:
    mutable ArchCapabilities _caps{};
    mutable std::once_flag   _capsOnce;
};

/// True iff `architecture` matches one of the backends `createArchBackend`
/// can build. Pure name comparison — no model / GPU dependencies. Used by
/// the loader for early-fail diagnostics and by unit tests.
///
/// Inline so it can be linked into pure-CPU test binaries without dragging
/// in Qwen2Backend / Gemma4Backend implementations.
[[nodiscard]] inline bool
isSupportedArchitecture(std::string_view architecture) noexcept {
    return architecture == "qwen2" || architecture == "llama" ||
           architecture == "gemma4" || architecture == "qwen35moe" ||
           architecture == "qwen4_exp";
}

/// Build the backend matching `architecture` ("qwen2" / "gemma4"). Returns
/// nullptr for unsupported architectures — callers must check.
/// `moeGroupEnabled` maps to `features.moeGroup`; `moeFusedDownEnabled`
/// maps to `features.moeFusedDown != Disable`. Non-MoE architectures
/// ignore both.
std::unique_ptr<ArchBackend>
createArchBackend(const std::string&             architecture,
                  const model::LlmConfig&        config,
                  const core::gguf::WeightsMap&       weights,
                  const model::FusedQkvWeights*  fusedQkv,
                  compute::ComputeOps&               ops,
                  compute::ComputeMatmul&            gmm,
                  OpProfiler&                    opProfiler,
                  bool                           moeGroupEnabled     = true,
                  bool                           moeFusedDownEnabled = false);

} // namespace mimirmind::runtime::arch