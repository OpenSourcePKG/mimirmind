#pragma once

#include "runtime/arch/ArchBackend.hpp"

#include <memory>
#include <string>

namespace mimirmind::compute {
class GpuMatmul;
class GpuOps;
} // namespace mimirmind::compute

namespace mimirmind::model {
class FusedQkvWeights;
class WeightsMap;
struct LlmConfig;
} // namespace mimirmind::model

namespace mimirmind::runtime::arch {

class GemmaBaseBackend;

/**
 * Gemma 4 architecture backend — public entry point registered with
 * `createArchBackend("gemma4", ...)`. Owns one of two concrete
 * implementations picked at construction from `config.expertCount`:
 *   - expertCount > 0 → `Gemma4MoeBackend`   (26B-A4B-it target)
 *   - expertCount == 0 → `Gemma4DenseBackend` (4B / 12B / E4B)
 *
 * Both concrete backends inherit from `GemmaBaseBackend`, which owns
 * the shared attention block + per-layer info + parity dump machinery.
 * The facade forwards every `ArchBackend` method to the chosen impl.
 *
 * The dispatch is decided once at construction; the impl pointer never
 * swaps at runtime. A model-load flip requires a fresh process.
 */
class Gemma4Backend final : public ArchBackend {
public:
    Gemma4Backend(const model::LlmConfig&        config,
                  const model::WeightsMap&       weights,
                  const model::FusedQkvWeights*  fusedQkv,
                  compute::GpuOps&               ops,
                  compute::GpuMatmul&            gmm,
                  runtime::OpProfiler&           opProfiler);

    ~Gemma4Backend() override;

    void runBlock(std::size_t   blockIdx,
                  float*        x,
                  std::size_t   T,
                  KvCache&      cache,
                  BlockBuffers& buffers,
                  bool          traceBlock0) override;

    [[nodiscard]] bool        scalesEmbedding() const noexcept override { return true; }
    [[nodiscard]] const char* name()            const noexcept override { return "gemma4"; }

    [[nodiscard]] std::vector<std::size_t>
        kvDimPerLayer() const override;
    [[nodiscard]] std::pair<std::size_t, std::size_t>
        maxQKVDims() const override;

    void setParityDumpPrefix(const std::string& prefix) noexcept override;

private:
    std::unique_ptr<GemmaBaseBackend> _impl;
};

} // namespace mimirmind::runtime::arch