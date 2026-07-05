#include "runtime/arch/Gemma4E4BBackend.hpp"

#include "compute/Dequant.hpp"
#include "compute/GpuMatmul.hpp"
#include "compute/GpuOps.hpp"
#include "compute/QuantType.hpp"
#include "compute/QuantTypeRegistry.hpp"
#include "model/GgufReader.hpp"
#include "model/GgufTypes.hpp"
#include "model/LlmConfig.hpp"
#include "model/WeightsMap.hpp"
#include "runtime/BlockBuffers.hpp"
#include "runtime/KvCache.hpp"
#include "runtime/Log.hpp"
#include "runtime/OpProfiler.hpp"
#include "runtime/UsmAllocator.hpp"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace mimirmind::runtime::arch {

Gemma4E4BBackend::Gemma4E4BBackend(const model::LlmConfig&        config,
                                   const model::WeightsMap&       weights,
                                   const model::FusedQkvWeights*  fusedQkv,
                                   compute::GpuOps&               ops,
                                   compute::GpuMatmul&            gmm,
                                   runtime::OpProfiler&           opProfiler)
    : GemmaBaseBackend{config, weights, fusedQkv, ops, gmm, opProfiler} {
    // Per-layer-dim comes from the block-level inp_gate.weight — GGUF
    // dims are [K, N] = [d_model, per_layer_dim].
    const auto* inp0 = weights.findBlock(0, "inp_gate.weight");
    if (inp0 == nullptr || inp0->dimensions.size() < 2) {
        throw std::runtime_error(
            "Gemma4E4BBackend: blk.0.inp_gate.weight missing or malformed");
    }
    _perLayerDim = inp0->dimensions[1];

    // per_layer_token_embd is a top-level tensor storing a
    // [num_layers * per_layer_dim, vocab_size] block-quantized weight.
    // Missing → we still construct (so a malformed model does not crash
    // the process) but runBlock skips the PLE branch and logs once.
    const auto* pleT = weights.find("per_layer_token_embd.weight");
    if (pleT != nullptr) {
        _pleTablePtr  = pleT->usmPtr;
        _pleTableType = pleT->type;
        _vocabSize    = pleT->dimensions.size() >= 2 ? pleT->dimensions[1] : 0;
        const auto* qt = compute::quantType(_pleTableType);
        if (qt != nullptr) {
            _pleBytesPerBlock = qt->blockBytes();
        } else {
            MM_LOG_WARN("gemma4-e4b",
                        "per_layer_token_embd.weight uses unsupported quant "
                        "type {} — PLE injection disabled, output will be "
                        "structurally degraded",
                        model::typeInfo(_pleTableType).name);
            _pleTablePtr = nullptr;
        }
    } else {
        MM_LOG_WARN("gemma4-e4b",
                    "per_layer_token_embd.weight missing — PLE injection "
                    "disabled, output will be structurally degraded");
    }

    MM_LOG_INFO("gemma4-e4b",
                "Gemma4E4BBackend ready — blocks={} d_model={} "
                "per_layer_dim={} ff={} heads={} kv={} head_dim={} "
                "ple_table={} vocab={}",
                _config.blockCount, _config.embeddingLength,
                _perLayerDim, _config.feedForwardLength,
                _config.headCount, _config.headCountKvFor(0),
                _layers.empty() ? 0 : _layers.front().headDim,
                _pleTablePtr != nullptr
                    ? model::typeInfo(_pleTableType).name
                    : std::string_view{"none"},
                _vocabSize);
}

void Gemma4E4BBackend::ensurePleCapacity(std::size_t T) {
    if (_pleTablePtr == nullptr) {
        return;
    }
    if (T > _pleBufCapT) {
        // Layout: [num_layers, T, per_layer_dim]
        const std::size_t bytes =
            _config.blockCount * T * _perLayerDim * sizeof(float);
        _pleBuf = UsmHandle{_ops.allocator(), bytes};
        _pleBufCapT = T;
    }
    if (T > _pleGateBufCapT) {
        const std::size_t bytes = T * _perLayerDim * sizeof(float);
        _pleGateBuf = UsmHandle{_ops.allocator(), bytes};
        _pleGateBufCapT = T;
    }
}

void Gemma4E4BBackend::prepareForward(std::span<const std::int32_t> tokIds) {
    if (_pleTablePtr == nullptr) {
        _pleActiveT = tokIds.size();
        return;
    }

    const std::size_t T = tokIds.size();
    if (T == 0) {
        _pleActiveT = 0;
        return;
    }
    ensurePleCapacity(T);

    float* const pleBuf = _pleBuf.as<float>();

    // Per-layer-dim (256) matches the K-quant super-block size, so for
    // every (token, layer) pair we dequant exactly one contiguous block
    // from the table. Layout on disk: [num_layers * per_layer_dim, vocab]
    // stored column-per-token, giving `num_layers * blockBytes` bytes
    // per token. Layer L's block sits at offset L * blockBytes within
    // the token column.
    const std::size_t bytesPerToken = _config.blockCount * _pleBytesPerBlock;
    const auto* base = static_cast<const std::uint8_t*>(_pleTablePtr);
    const std::size_t stride = T * _perLayerDim;   // [num_layers, T, per_layer_dim]

    for (std::size_t t = 0; t < T; ++t) {
        const std::int32_t tok = tokIds[t];
        if (tok < 0 || static_cast<std::size_t>(tok) >= _vocabSize) {
            // Out-of-range token: fill zeros for every layer of this slot.
            for (std::size_t L = 0; L < _config.blockCount; ++L) {
                std::memset(pleBuf + L * stride + t * _perLayerDim, 0,
                            _perLayerDim * sizeof(float));
            }
            continue;
        }

        const std::uint8_t* tokBase =
            base + static_cast<std::size_t>(tok) * bytesPerToken;
        for (std::size_t L = 0; L < _config.blockCount; ++L) {
            const std::uint8_t* blockPtr = tokBase + L * _pleBytesPerBlock;
            float* dstSlice = pleBuf + L * stride + t * _perLayerDim;
            compute::dequantToF32(_pleTableType, blockPtr,
                                  _perLayerDim, dstSlice);
        }
    }

    _pleActiveT = T;
}

void Gemma4E4BBackend::runBlock(std::size_t   blockIdx,
                                float*        x,
                                std::size_t   T,
                                KvCache&      cache,
                                BlockBuffers& s,
                                bool          traceBlock0) {
    const bool diag = (blockIdx == 0 && cache.length() == 0 && traceBlock0);
    auto trace = [&](const char* tag) {
        if (diag) MM_LOG_INFO("blkdiag-g4e", "blk0 {}", tag);
    };
    trace("enter (e4b)");

    // Attention section unchanged — matches Dense variant.
    runAttentionSection(blockIdx, x, T, cache, s, diag);

    const auto* ffnNorm  = requireTensor(blockIdx, "ffn_norm.weight",           "Gemma4E4BBackend");
    const auto* ffnGate  = requireTensor(blockIdx, "ffn_gate.weight",           "Gemma4E4BBackend");
    const auto* ffnUp    = requireTensor(blockIdx, "ffn_up.weight",             "Gemma4E4BBackend");
    const auto* ffnDown  = requireTensor(blockIdx, "ffn_down.weight",           "Gemma4E4BBackend");
    const auto* ffwPost  = requireTensor(blockIdx, "post_ffw_norm.weight",      "Gemma4E4BBackend");
    const auto* outScale = requireTensor(blockIdx, "layer_output_scale.weight", "Gemma4E4BBackend");

    const std::size_t d_model = s.d_model;
    const std::size_t ff_dim  = s.ff_dim;

    float* const normBuf       = s.normBuf.as<float>();
    float* const projOutBuf    = s.projOut.as<float>();
    float* const gateOutBuf    = s.gateOut.as<float>();
    float* const upOutBuf      = s.upOut.as<float>();
    float* const matmulScratch = s.matmulScratch.as<float>();

    // --- Dense FFN (GELU-SwiGLU) --------------------------------------

    _op.mark(runtime::OpProfiler::Cat::NORM);
    trace("ffn_norm (e4b pre)");
    _ops.rmsNormAsync(x, T, d_model,
                      static_cast<const float*>(ffnNorm->usmPtr),
                      _config.rmsNormEps,
                      normBuf);

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

    _op.mark(runtime::OpProfiler::Cat::NORM);
    trace("post_ffw_norm");
    _ops.rmsNormAsync(projOutBuf, T, d_model,
                      static_cast<const float*>(ffwPost->usmPtr),
                      _config.rmsNormEps,
                      projOutBuf);

    _op.mark(runtime::OpProfiler::Cat::RESIDUAL);
    trace("ffn residual");
    _ops.addResidualAsync(x, projOutBuf, T * d_model);

    // --- PLE injection --------------------------------------------------
    //
    // Skipped when the model didn't ship a per_layer_token_embd table —
    // the constructor already logged a warning. Skipped when the current
    // request has zero drafted tokens (defensive; can't happen from a
    // real generate() call but keeps a safe no-op path if a future
    // caller flushes prepareForward with empty tokIds).
    if (_pleTablePtr != nullptr && _pleActiveT >= T) {
        const auto* inpGate  = requireTensor(blockIdx, "inp_gate.weight",  "Gemma4E4BBackend");
        const auto* proj     = requireTensor(blockIdx, "proj.weight",      "Gemma4E4BBackend");
        const auto* postNorm = requireTensor(blockIdx, "post_norm.weight", "Gemma4E4BBackend");

        float* const pleGateBuf = _pleGateBuf.as<float>();
        const float* const pleSliceForLayer =
            _pleBuf.as<float>()
            + blockIdx * (_pleActiveT * _perLayerDim);

        _op.mark(runtime::OpProfiler::Cat::MATMUL);
        trace("PLE inp_gate proj");
        _gmm.matmul(inpGate->type, inpGate->usmPtr,
                    _perLayerDim, d_model,
                    x, T,
                    pleGateBuf, matmulScratch);

        _op.mark(runtime::OpProfiler::Cat::ACTIVATION);
        trace("PLE GELU * slice (fused)");
        // geluMulAsync: gate[i] = gelu(gate[i]) * up[i]
        // Reused here for "hidden = GELU(hidden) * PLE_slice".
        _ops.geluMulAsync(pleGateBuf, pleSliceForLayer, T * _perLayerDim);

        _op.mark(runtime::OpProfiler::Cat::MATMUL);
        trace("PLE proj");
        _gmm.matmul(proj->type, proj->usmPtr,
                    d_model, _perLayerDim,
                    pleGateBuf, T,
                    projOutBuf, matmulScratch);

        _op.mark(runtime::OpProfiler::Cat::NORM);
        trace("PLE post_norm");
        _ops.rmsNormAsync(projOutBuf, T, d_model,
                          static_cast<const float*>(postNorm->usmPtr),
                          _config.rmsNormEps,
                          projOutBuf);

        _op.mark(runtime::OpProfiler::Cat::RESIDUAL);
        trace("PLE residual");
        _ops.addResidualAsync(x, projOutBuf, T * d_model);
    }

    // --- Block-output scale --------------------------------------------

    const float scaleVal = *static_cast<const float*>(outScale->usmPtr);
    _op.mark(runtime::OpProfiler::Cat::ACTIVATION);
    trace("layer_output_scale");
    _ops.mulScalarAsync(x, scaleVal, T * d_model);

    _op.finish();
}

} // namespace mimirmind::runtime::arch