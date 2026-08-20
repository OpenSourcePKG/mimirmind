// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/gguf/GgufTypes.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
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

    /// True when `runBlockBatched` is implemented for this backend (L0
    /// Gemma 4 MoE in Phase 1). InferenceEngine's batched harness checks
    /// this before allocating per-sequence state. Default false.
    [[nodiscard]] virtual bool supportsBatchedDecode() const noexcept {
        return false;
    }

    /// True if the arch needs the token embedding to be scaled by
    /// sqrt(d_model) before the first block (Gemma family). InferenceEngine
    /// reads this to centralise the scale on prefill + decode.
    [[nodiscard]] virtual bool scalesEmbedding() const noexcept = 0;

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

    /// True if this backend routes its FP16-KV writes through an fp32 staging
    /// redirect (project → fp32 scratch → rmsnorm/rope in fp32 → kv_commit_fp16
    /// cast into the cache), so a raw fp32 K/V matmul never lands in an fp16
    /// slot. Backends that DON'T (the plain path writes fp32 straight into the
    /// cache slot) must return false — the engine then still requires fused-QKV
    /// for FP16 to avoid corrupting the fp16 cache. Default false.
    [[nodiscard]] virtual bool supportsFp16KvStaging() const noexcept {
        return false;
    }

    /// Short identifier for logs ("qwen2", "gemma4").
    [[nodiscard]] virtual const char* name() const noexcept = 0;

    /// True if the arch needs the per-head fused [Q|gate] scratch buffers
    /// (`BlockBuffers::qGateFused` / `gateScratch`). Qwen3-Next full-
    /// attention fuses the query projection with a per-head output gate;
    /// every other arch leaves this false. InferenceEngine reads it when
    /// sizing block scratch.
    [[nodiscard]] virtual bool needsQGateScratch() const noexcept {
        return false;
    }

    /// True if the arch needs the GatedDeltaNet linear-layer scratch
    /// (`BlockBuffers::ssm*`). Qwen3-Next hybrid-recurrent models set this;
    /// every other arch leaves it false. Read by InferenceEngine when
    /// sizing block scratch.
    [[nodiscard]] virtual bool needsSsmScratch() const noexcept {
        return false;
    }

    /// True when this MoE backend runs its decode block with fully
    /// device-side expert dispatch — no host read of the routing between
    /// the router matmul and the accumulator (M-CLR.MoE Increment 2). Such
    /// a block is Command-List-Replay-capturable: InferenceEngine may then
    /// enable CLR for the decode loop even though expertCount > 0. Dense
    /// (expertCount == 0) backends never need this; the MoE default is
    /// false (host routing bakes stale expert picks into the recording).
    /// Gemma4MoeBackend overrides it to reflect the device-dispatch gate.
    [[nodiscard]] virtual bool moeDecodeClrSafe() const noexcept {
        return false;
    }

    /// True when this backend's dense (non-MoE) decode block writes K/V
    /// through a Command-List-Replay-safe destination — a stable cache
    /// base plus the device-side curLen slot, as the fused-QKV split does.
    /// The UNFUSED QKV path instead projects K/V straight into
    /// `cache.writeSlotK/V()`, a host-computed per-token pointer baked into
    /// the recording at capture time; a replayed decode step then re-writes
    /// that same stale slot instead of the current one, so the KV cache
    /// never advances and generation degenerates after the first (recorded)
    /// step. A backend that can fall onto the unfused path for the loaded
    /// weights (e.g. mixed-quant QKV that FusedQkvWeights refuses to fuse)
    /// must override this to report false so InferenceEngine keeps decode in
    /// immediate mode. Default true: backends whose QKV is always fused (or
    /// which never use per-token slot writes) are replay-safe.
    [[nodiscard]] virtual bool decodeQkvClrSafe() const noexcept {
        return true;
    }

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

protected:
    ArchBackend() = default;
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
           architecture == "gemma4" || architecture == "qwen35moe";
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