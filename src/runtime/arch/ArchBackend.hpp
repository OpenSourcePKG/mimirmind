// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "compute/ComputeBuffer.hpp"
#include "core/gguf/GgufTypes.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
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

    /**
     * 7.2 Hugin: observe named intermediate stages of the forward (e.g.
     * post-RoPE "Qcur"/"Kcur", "Vcur", the pre-O-projection attention context
     * "attn_ctx"). `wants(stage, block)` gates the readback, so an installed
     * tap costs nothing for stages nobody asked for; `onStage` receives a
     * host copy [rows x cols] f32. Which stages exist is per backend; an
     * empty tap (the default) disables everything. Diagnostic path only —
     * the tap syncs the queue before each readback.
     */
    struct StageTap {
        std::function<bool(std::string_view stage, std::size_t block)> wants;
        std::function<void(std::string_view stage, std::size_t block,
                           std::vector<float>&& data, std::size_t rows,
                           std::size_t cols)>                           onStage;
        /// 7.3 Hugin (optional): edit a stage in place. Called only at stages
        /// a backend marks editable (tapStageMutable); returns true when it
        /// changed `data`, which is then written back to the device buffer
        /// before the forward continues. Diagnostic/preview path only — the
        /// serving engine never installs a tap.
        std::function<bool(std::string_view stage, std::size_t block,
                           std::vector<float>& data, std::size_t rows,
                           std::size_t cols)>                           edit;
    };
    void setStageTap(StageTap tap) { _stageTap = std::move(tap); }
    void clearStageTap() noexcept { _stageTap = StageTap{}; }

    /// 7.3 Rune Steer overlay (Futhark): a device vector [dModel] per block,
    /// added to every row of the residual stream right after that block on
    /// every forward path (prefill, decode, batched serving, verify). Set once
    /// at load, before any graph capture; the caller owns the device memory
    /// for the backend's lifetime. Several steers on one block must be summed
    /// by the caller (one vector per block).
    void setResidualSteers(compute::ComputeOps& ops, std::size_t dModel,
                           const std::vector<std::pair<std::size_t, const float*>>& steers);
    [[nodiscard]] bool hasResidualSteers() const noexcept { return _steerOps != nullptr; }
    /// Call after runBlock / runBlockBatched with the rows that block wrote.
    /// No work (and no device op) when no steer is registered for `block`.
    void applyResidualSteer(std::size_t block, float* x, std::size_t rows) const {
        if (_steerOps != nullptr && block < _steerByBlock.size() &&
            _steerByBlock[block] != nullptr) {
            addResidualSteer(_steerByBlock[block], x, rows);
        }
        if (block < _condAtBlock.size() && _condAtBlock[block]) {
            applyConditionalSteers(block, x, rows);
        }
    }

    /// 7.11 Rune projection / conditional steer overlays (Futhark
    /// steer_project / steer_if). A condition scores every row of the residual
    /// after its block as cos(x_r, direction); a steer gated on it changes only
    /// the rows whose score is > threshold. Rows are independent, so the gate is
    /// per token on every forward path (prefill, decode, batched, ragged,
    /// graph) without a row -> sequence map. Order at one block: the summed
    /// plain steers, then the conditions anchored there are scored, then the
    /// projection / conditional steers in registration order. Device vectors
    /// are owned by the caller for the backend's lifetime; set once at load,
    /// before any graph capture.
    struct SteerCondition {
        std::size_t  block{0};
        const float* direction{nullptr};   ///< device, unit [dModel]
        float        threshold{0.0F};
    };
    struct ConditionalSteer {
        std::size_t  block{0};
        bool         project{false};       ///< false: x += v; true: x -= alpha (x.v) v
        const float* values{nullptr};      ///< device [dModel] (unit when project)
        float        alpha{1.0F};          ///< project only
        int          condition{-1};        ///< index into the conditions, -1 = every row
    };
    void setConditionalSteers(compute::ComputeOps& ops, std::size_t dModel,
                              std::vector<SteerCondition> conditions,
                              std::vector<ConditionalSteer> steers);
    void clearConditionalSteers() noexcept;
    [[nodiscard]] bool hasConditionalSteers() const noexcept { return _condOps != nullptr; }
    /// Host copy of the last condition scores written for `condition` (rows of
    /// the most recent forward that reached its block). Diagnostic (Hugin
    /// preview / calibration); syncs the stream.
    [[nodiscard]] std::vector<float> readConditionScores(std::size_t condition,
                                                         std::size_t rows) const;

    /// 7.3 Rune expert disable (Futhark Mask op on a router): a device bias
    /// [nExperts] per MoE block, 0 for kept experts and kExpertMaskedLogit for
    /// disabled ones, added to the router logits before any top-K (host or
    /// device, single-session or batched). Set once at load, before any graph
    /// capture; the caller owns the device memory. One mask per block (the
    /// caller ANDs stacked masks).
    static constexpr float kExpertMaskedLogit = -1.0e30F;
    void setExpertMasks(compute::ComputeOps& ops, std::size_t nExperts,
                        const std::vector<std::pair<std::size_t, const float*>>& masks);
    [[nodiscard]] bool hasExpertMasks() const noexcept { return _maskOps != nullptr; }
    /// Call right after the router matmul wrote `logits` [rows, nExperts].
    /// Returns true when a mask was enqueued: a caller whose top-K reads the
    /// logits on the host must then sync the stream first. No work (and no
    /// device op) when `block` has no mask.
    [[nodiscard]] bool applyExpertMask(std::size_t block, float* logits, std::size_t rows) const {
        if (_maskOps != nullptr && block < _maskByBlock.size() &&
            _maskByBlock[block] != nullptr) {
            addExpertMask(_maskByBlock[block], logits, rows);
            return true;
        }
        return false;
    }

    /// 7.4.3 Rune LowRank overlay (LoRA): on projection `tensor` (GGUF suffix,
    /// e.g. "attn_q.weight") of `block`, out [rows, out] += B (A in), with A
    /// [rank, in] and B [out, rank] F32 device buffers owned by the caller
    /// (alpha already folded into B). Set once at load, before any capture.
    struct LowRankOverlay {
        std::size_t  block{0};
        std::string  tensor;
        std::size_t  rank{0};
        std::size_t  in{0};
        std::size_t  out{0};
        const float* a{nullptr};
        const float* b{nullptr};
    };
    /// True for backends whose block code calls applyLowRank; the engine
    /// refuses lowrank runes otherwise.
    [[nodiscard]] virtual bool supportsLowRank() const noexcept { return false; }
    /// True iff the block code calls applyLowRank for `tensor` (e.g.
    /// "attn_q.weight"); setLowRanks refuses every other target, so a rune
    /// can never be a silent no-op.
    [[nodiscard]] virtual bool supportsLowRankOn(std::string_view tensor) const noexcept {
        (void)tensor;
        return false;
    }
    void setLowRanks(compute::ComputeOps& ops, compute::ComputeMatmul& gmm,
                     std::vector<LowRankOverlay> overlays);
    [[nodiscard]] bool hasLowRanks() const noexcept { return _lrOps != nullptr; }
    [[nodiscard]] bool hasLowRank(std::size_t block, std::string_view tensor) const noexcept;
    /// Enqueue out += B (A in) when (block, tensor) has an overlay; returns
    /// whether it did. `matmulScratch` is the caller's matmul scratch.
    bool applyLowRank(std::size_t block, std::string_view tensor, const float* in,
                      std::size_t rows, float* out, float* matmulScratch);

protected:
    ArchBackend() = default;

    /// True iff a tap is installed and wants (stage, block).
    [[nodiscard]] bool stageTapWants(std::string_view stage, std::size_t block) const {
        return _stageTap.wants && _stageTap.onStage && _stageTap.wants(stage, block);
    }
    /// Device buffer -> host copy -> tap (no-op unless stageTapWants).
    void tapStage(compute::ComputeMatmul& gmm, compute::ComputeOps& ops,
                  std::string_view stage, std::size_t block,
                  const float* device, std::size_t rows, std::size_t cols) const;
    /// Like tapStage, but the stage may be edited in place (StageTap::edit)
    /// and is written back to `device` when changed. onStage then observes
    /// the edited values.
    void tapStageMutable(compute::ComputeMatmul& gmm, compute::ComputeOps& ops,
                         std::string_view stage, std::size_t block,
                         float* device, std::size_t rows, std::size_t cols) const;
    /// Host values -> tap (no-op unless stageTapWants).
    void tapStageHost(std::string_view stage, std::size_t block,
                      const float* host, std::size_t rows, std::size_t cols) const;
    /// Integer stages (e.g. MoE expert ids): delivered to the tap as exact
    /// floats (ids < 2^24). Device and host variants.
    void tapStageI32(compute::ComputeMatmul& gmm, compute::ComputeOps& ops,
                     std::string_view stage, std::size_t block,
                     const std::int32_t* device, std::size_t rows, std::size_t cols) const;
    void tapStageHostI32(std::string_view stage, std::size_t block,
                         const std::int32_t* host, std::size_t rows, std::size_t cols) const;

    /// Build this backend's capability flags (8.30.8). Replaces the former
    /// eight individual capability virtuals. Called once, lazily, by
    /// `capabilities()` after full construction — so a Qwen subclass reports
    /// its most-derived values (a constructor cannot). Must be a pure
    /// function of load-time-stable state (config / fused-QKV / delegated
    /// impl); it runs at most once per backend instance.
    [[nodiscard]] virtual ArchCapabilities computeCapabilities() const = 0;

private:
    void addResidualSteer(const float* steer, float* x, std::size_t rows) const;
    void addExpertMask(const float* bias, float* logits, std::size_t rows) const;
    void applyConditionalSteers(std::size_t block, float* x, std::size_t rows) const;
    /// Grows the per-condition score buffers + the projection scratch to
    /// `rows`. Old buffers are kept alive (a captured graph may still point at
    /// them); the initial capacity covers every decode batch, so growth only
    /// happens on large (uncaptured) prefill forwards.
    void ensureConditionalCapacity(std::size_t rows) const;

    compute::ComputeOps*     _steerOps{nullptr};
    std::size_t              _steerDModel{0};
    std::vector<const float*> _steerByBlock;
    compute::ComputeOps*     _maskOps{nullptr};
    std::size_t              _maskNExperts{0};
    std::vector<const float*> _maskByBlock;
    compute::ComputeOps*      _condOps{nullptr};
    std::size_t               _condDModel{0};
    std::vector<SteerCondition>   _conditions;
    std::vector<ConditionalSteer> _condSteers;
    std::vector<bool>             _condAtBlock;   ///< block has a condition or a cond. steer
    mutable std::size_t                          _condCapacity{0};
    mutable std::vector<compute::ComputeBuffer>  _condScores;    ///< one per condition
    mutable compute::ComputeBuffer               _condProj;      ///< projection coefficients
    mutable std::vector<compute::ComputeBuffer>  _condRetired;   ///< outgrown, kept alive
    compute::ComputeOps*                     _lrOps{nullptr};
    compute::ComputeMatmul*                  _lrGmm{nullptr};
    std::vector<std::vector<LowRankOverlay>> _lrByBlock;
    compute::ComputeBuffer                   _lrTmp;    ///< [kLowRankChunk, max rank]
    compute::ComputeBuffer                   _lrOut;    ///< [kLowRankChunk, max out]
    static constexpr std::size_t             kLowRankChunk = 256;
    StageTap                 _stageTap{};
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
           architecture == "qwen4_exp" || architecture == "kolibri1";
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