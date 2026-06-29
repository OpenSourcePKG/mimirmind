#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace mimirmind::compute {
class GpuMatmul;
class GpuOps;
} // namespace mimirmind::compute

namespace mimirmind::model {
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

protected:
    ArchBackend() = default;
};

/// Build the backend matching `architecture` ("qwen2" / "gemma4"). Returns
/// nullptr for unsupported architectures — callers must check.
std::unique_ptr<ArchBackend>
createArchBackend(const std::string&        architecture,
                  const model::LlmConfig&   config,
                  const model::WeightsMap&  weights,
                  compute::GpuOps&          ops,
                  compute::GpuMatmul&       gmm);

} // namespace mimirmind::runtime::arch