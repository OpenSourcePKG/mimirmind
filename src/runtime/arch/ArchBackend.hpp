#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace mimirmind::compute {
class GpuMatmul;
class GpuOps;
} // namespace mimirmind::compute

namespace mimirmind::runtime {
class OpProfiler;
} // namespace mimirmind::runtime

namespace mimirmind::model {
class FusedQkvWeights;
class WeightsMap;
struct LlmConfig;
} // namespace mimirmind::model

namespace mimirmind::runtime {
class KvCache;
struct BlockBuffers;
} // namespace mimirmind::runtime

namespace mimirmind::runtime::arch {

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

    /// Maximum hidden-state dim across layers for any of: Q output, KV
    /// output. BlockBuffers is sized for this so scratch survives the
    /// largest layer. Returns a pair {qDimMax, kvDimMax}.
    [[nodiscard]] virtual std::pair<std::size_t, std::size_t>
        maxQKVDims() const = 0;

    /// Short identifier for logs ("qwen2", "gemma4").
    [[nodiscard]] virtual const char* name() const noexcept = 0;

    /// Enable per-stage parity dumps. PREFIX is the same string carried by
    /// the MIMIRMIND_PARITY_DUMP env: each stage writes a file at
    ///   <prefix>-blk{N}-<stage>.bin
    /// matching the layout llama-parity-dump produces. Empty string =
    /// disabled (default). Default impl is no-op; backends that wire
    /// intermediate dumps override.
    virtual void setParityDumpPrefix(const std::string& /*prefix*/) noexcept {}

    /// Give the backend a heads-up about the token ids that are about to
    /// run through the block chain in the next `runBlock` sequence. Called
    /// once per forward pass — before prefill, before every decode step,
    /// and before `forwardVerify`.
    ///
    /// Non-E-series backends have no per-token per-layer state, so the
    /// default is a no-op. `Gemma4E4BBackend` overrides this to
    /// pre-fetch the per-layer-embedding (PLE) slices for the tokens
    /// into a USM scratch that `runBlock` slices per layer.
    ///
    /// The span refers to caller-owned memory that stays valid for the
    /// duration of the block-chain call. The backend copies whatever it
    /// needs synchronously here.
    virtual void prepareForward(std::span<const std::int32_t> /*tokIds*/) {}

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
    return architecture == "qwen2" || architecture == "gemma4";
}

/// Build the backend matching `architecture` ("qwen2" / "gemma4"). Returns
/// nullptr for unsupported architectures — callers must check.
std::unique_ptr<ArchBackend>
createArchBackend(const std::string&             architecture,
                  const model::LlmConfig&        config,
                  const model::WeightsMap&       weights,
                  const model::FusedQkvWeights*  fusedQkv,
                  compute::GpuOps&               ops,
                  compute::GpuMatmul&            gmm,
                  OpProfiler&                    opProfiler);

} // namespace mimirmind::runtime::arch