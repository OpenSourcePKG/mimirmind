#pragma once

#include "runtime/arch/ArchBackend.hpp"

namespace mimirmind::compute {
class GpuMatmul;
class GpuOps;
} // namespace mimirmind::compute

namespace mimirmind::model {
class WeightsMap;
struct LlmConfig;
} // namespace mimirmind::model

namespace mimirmind::runtime::arch {

/**
 * Qwen2 / Qwen2.5 / Llama-family decoder block.
 *
 *   x -> rmsNorm(attn_norm) -> Q/K/V proj (+bias) -> RoPE -> GQA attention
 *     -> O proj -> residual -> rmsNorm(ffn_norm) -> SwiGLU (silu(gate)*up)
 *     -> down -> residual
 */
class Qwen2Backend final : public ArchBackend {
public:
    Qwen2Backend(const model::LlmConfig&   config,
                 const model::WeightsMap&  weights,
                 compute::GpuOps&          ops,
                 compute::GpuMatmul&       gmm);

    void runBlock(std::size_t   blockIdx,
                  float*        x,
                  std::size_t   T,
                  KvCache&      cache,
                  BlockBuffers& buffers,
                  bool          traceBlock0) override;

    [[nodiscard]] bool        scalesEmbedding() const noexcept override { return false; }
    [[nodiscard]] const char* name()            const noexcept override { return "qwen2"; }

private:
    const model::LlmConfig&   _config;
    const model::WeightsMap&  _weights;
    compute::GpuOps&          _ops;
    compute::GpuMatmul&       _gmm;
};

} // namespace mimirmind::runtime::arch