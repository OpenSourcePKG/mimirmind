#include "runtime/arch/Gemma4DenseBackend.hpp"

#include "compute/GpuMatmul.hpp"
#include "compute/GpuOps.hpp"
#include "core/gguf/GgufReader.hpp"
#include "core/gguf/GgufTypes.hpp"
#include "model/LlmConfig.hpp"
#include "runtime/BlockBuffers.hpp"
#include "runtime/KvCache.hpp"
#include "core/log/Log.hpp"
#include "runtime/OpProfiler.hpp"

#include <cstddef>

namespace mimirmind::runtime::arch {

Gemma4DenseBackend::Gemma4DenseBackend(const model::LlmConfig&        config,
                                       const core::gguf::WeightsMap&       weights,
                                       const model::FusedQkvWeights*  fusedQkv,
                                       compute::GpuOps&               ops,
                                       compute::GpuMatmul&            gmm,
                                       runtime::OpProfiler&           opProfiler)
    : GemmaBaseBackend{config, weights, fusedQkv, ops, gmm, opProfiler} {
    MM_LOG_INFO("gemma4-dense",
                "Gemma4DenseBackend ready — blocks={} d_model={} ff={} "
                "heads={} kv={} head_dim={}",
                _config.blockCount, _config.embeddingLength,
                _config.feedForwardLength,
                _config.headCount, _config.headCountKvFor(0),
                _layers.empty() ? 0 : _layers.front().headDim);
}

void Gemma4DenseBackend::runBlock(std::size_t   blockIdx,
                                  float*        x,
                                  std::size_t   T,
                                  KvCache&      cache,
                                  BlockBuffers& s,
                                  bool          traceBlock0) {
    const bool diag = (blockIdx == 0 && cache.length() == 0 && traceBlock0);
    auto trace = [&](const char* tag) {
        if (diag) MM_LOG_INFO("blkdiag-g4d", "blk0 {}", tag);
    };
    trace("enter (dense)");

    // Shared attention section. On return `x` holds
    // sa_out = inpL + post_attention_norm(W_o @ attn(...)).
    runAttentionSection(blockIdx, x, T, cache, s, diag);

    const auto* ffnNorm  = requireTensor(blockIdx, "ffn_norm.weight",           "Gemma4DenseBackend");
    const auto* ffnGate  = requireTensor(blockIdx, "ffn_gate.weight",           "Gemma4DenseBackend");
    const auto* ffnUp    = requireTensor(blockIdx, "ffn_up.weight",             "Gemma4DenseBackend");
    const auto* ffnDown  = requireTensor(blockIdx, "ffn_down.weight",           "Gemma4DenseBackend");
    const auto* ffwPost  = requireTensor(blockIdx, "post_ffw_norm.weight",      "Gemma4DenseBackend");
    const auto* outScale = requireTensor(blockIdx, "layer_output_scale.weight", "Gemma4DenseBackend");

    const std::size_t d_model  = s.d_model;
    const std::size_t ff_dim   = s.ff_dim;

    float* const normBuf       = s.normBuf.as<float>();
    float* const projOutBuf    = s.projOut.as<float>();
    float* const gateOutBuf    = s.gateOut.as<float>();
    float* const upOutBuf      = s.upOut.as<float>();
    float* const matmulScratch = s.matmulScratch.as<float>();

    // --- FFN — dense SwiGLU with GELU ---------------------------------
    // Fused attn-residual + ffn_norm: runAttentionSection left
    // `projOutBuf = attn_post_norm(attn_out)` for us to fold in here.

    _op.mark(runtime::OpProfiler::Cat::NORM);
    trace("attn residual + ffn_norm (fused)");
    _ops.addRmsNormAsync(x, projOutBuf, T, d_model,
                         static_cast<const float*>(ffnNorm->usmPtr),
                         _config.rmsNormEps,
                         normBuf);
    dumpStage("attn_out", blockIdx, x, T, d_model);

    // gate + up read normBuf, write disjoint outputs → can pipeline.
    _op.mark(runtime::OpProfiler::Cat::MATMUL);
    trace("FFN gate+up proj (unordered)");
    {
        runtime::UnorderedScope u{_ops.queue()};
        _gmm.matmulAsync(ffnGate->type, ffnGate->usmPtr, ff_dim, d_model,
                         normBuf, T, gateOutBuf, matmulScratch);
        _gmm.matmulAsync(ffnUp->type, ffnUp->usmPtr, ff_dim, d_model,
                         normBuf, T, upOutBuf, matmulScratch);
    }

    _op.mark(runtime::OpProfiler::Cat::ACTIVATION);
    trace("GELU + mul (fused)");
    _ops.geluMulAsync(gateOutBuf, upOutBuf, T * ff_dim);

    _op.mark(runtime::OpProfiler::Cat::MATMUL);
    trace("FFN down proj");
    _gmm.matmul(ffnDown->type, ffnDown->usmPtr, d_model, ff_dim,
                gateOutBuf, T,
                projOutBuf, matmulScratch);

    // Dense variants only have one post-FFN norm — `post_ffw_norm`. The
    // hybrid MoE runs a `post_ffw_norm_1` on Path A first and then
    // `post_ffw_norm` on the combined A+B output; here we apply it
    // directly to the single FFN output.
    _op.mark(runtime::OpProfiler::Cat::NORM);
    trace("post_ffw_norm (dense)");
    _ops.rmsNormAsync(projOutBuf, T, d_model,
                      static_cast<const float*>(ffwPost->usmPtr),
                      _config.rmsNormEps,
                      projOutBuf);            // in-place
    dumpStage("ffn_out", blockIdx, projOutBuf, T, d_model);

    _op.mark(runtime::OpProfiler::Cat::RESIDUAL);
    trace("ffn residual");
    _ops.addResidualAsync(x, projOutBuf, T * d_model);

    const float scaleVal = *static_cast<const float*>(outScale->usmPtr);
    _op.mark(runtime::OpProfiler::Cat::ACTIVATION);
    trace("layer_output_scale");
    _ops.mulScalarAsync(x, scaleVal, T * d_model);
    dumpStage("out_scaled", blockIdx, x, T, d_model);
    dumpStage("l_out",      blockIdx, x, T, d_model);

    // Close the last phase before returning so its time lands in the
    // accumulator. Cheap no-op when profiling is disabled.
    _op.finish();
}

} // namespace mimirmind::runtime::arch