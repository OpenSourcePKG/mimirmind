// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/gguf/GgufTypes.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
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
    return architecture == "qwen2" || architecture == "gemma4" ||
           architecture == "qwen35moe";
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