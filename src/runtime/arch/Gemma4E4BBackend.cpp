#include "runtime/arch/Gemma4E4BBackend.hpp"

#include "compute/Dequant.hpp"
#include "compute/GpuMatmul.hpp"
#include "compute/GpuOps.hpp"
#include "compute/QuantType.hpp"
#include "compute/QuantTypeRegistry.hpp"
#include "compute/quant/Q8_0.hpp"
#include "core/gguf/GgufReader.hpp"
#include "core/gguf/GgufTypes.hpp"
#include "model/LlmConfig.hpp"
#include "core/gguf/WeightsMap.hpp"
#include "runtime/BlockBuffers.hpp"
#include "runtime/KvCache.hpp"
#include "core/log/Log.hpp"
#include "runtime/OpProfiler.hpp"
#include "core/l0/UsmAllocator.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace mimirmind::runtime::arch {

namespace {

/// IEEE-754 binary32 → binary16 (round-to-nearest-even). Only used for
/// packing per-block scales into Q8_0 blocks at load time, so a scalar
/// implementation is fine.
std::uint16_t floatToHalf(float f) noexcept {
    std::uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    const std::uint32_t sign = (bits >> 16) & 0x8000U;
    std::int32_t        exp  = static_cast<std::int32_t>((bits >> 23) & 0xFFU);
    const std::uint32_t mant = bits & 0x7FFFFFU;

    if (exp == 0xFF) {
        // Inf / NaN
        return static_cast<std::uint16_t>(
            sign | 0x7C00U | (mant != 0 ? 0x0200U : 0U));
    }
    exp -= 127 - 15;
    if (exp <= 0) {
        if (exp < -10) return static_cast<std::uint16_t>(sign);
        const std::uint32_t m = (mant | 0x800000U) >> (14 - exp);
        return static_cast<std::uint16_t>(sign | m);
    }
    if (exp >= 0x1F) {
        return static_cast<std::uint16_t>(sign | 0x7C00U);
    }
    return static_cast<std::uint16_t>(sign
        | (static_cast<std::uint32_t>(exp) << 10)
        | (mant >> 13));
}

} // namespace

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

    // per_layer_token_embd — the big Q6_K embedding table.
    if (const auto* pleT = weights.find("per_layer_token_embd.weight")) {
        _pleTablePtr  = pleT->usmPtr;
        _pleTableType = pleT->type;
        _vocabSize    = pleT->dimensions.size() >= 2 ? pleT->dimensions[1] : 0;
        if (const auto* qt = compute::quantType(_pleTableType)) {
            _pleBytesPerBlock = qt->blockBytes();
        } else {
            MM_LOG_WARN("gemma4-e4b",
                        "per_layer_token_embd.weight uses unsupported quant "
                        "type {} — PLE disabled",
                        model::typeInfo(_pleTableType).name);
            _pleTablePtr = nullptr;
        }
    } else {
        MM_LOG_WARN("gemma4-e4b",
                    "per_layer_token_embd.weight missing — PLE disabled");
    }

    // per_layer_model_proj — the new BF16 [d_model, num_layers*per_layer_dim]
    // tensor that produces the per-layer inputs from hidden_states. GPU
    // matmul has no BF16 path, so we requantize to Q8_0 at load time.
    if (const auto* proj = weights.find("per_layer_model_proj.weight")) {
        try {
            requantizeModelProjToQ8_0(*proj);
        } catch (const std::exception& e) {
            MM_LOG_WARN("gemma4-e4b",
                        "per_layer_model_proj Q8_0 requantize failed ({}) — "
                        "PLE-input projection disabled, output degraded",
                        e.what());
            _projQ8 = UsmHandle{};
            _projQ8Bytes = 0;
        }
    } else {
        MM_LOG_WARN("gemma4-e4b",
                    "per_layer_model_proj.weight missing — PLE-input "
                    "projection disabled, output degraded");
    }

    // per_layer_proj_norm — RMSNorm weight [per_layer_dim].
    if (const auto* n = weights.find("per_layer_proj_norm.weight");
        n != nullptr && n->type == model::GgmlType::F32) {
        _projNorm = static_cast<const float*>(n->usmPtr);
    }

    MM_LOG_INFO("gemma4-e4b",
                "Gemma4E4BBackend ready — blocks={} d_model={} "
                "per_layer_dim={} ff={} heads={} kv={} head_dim={} "
                "ple_table={} proj_q8={} proj_norm={} vocab={}",
                _config.blockCount, _config.embeddingLength,
                _perLayerDim, _config.feedForwardLength,
                _config.headCount, _config.headCountKvFor(0),
                _layers.empty() ? 0 : _layers.front().headDim,
                _pleTablePtr != nullptr
                    ? model::typeInfo(_pleTableType).name
                    : std::string_view{"none"},
                _projQ8Bytes > 0 ? "on" : "off",
                _projNorm != nullptr ? "on" : "off",
                _vocabSize);
}

void Gemma4E4BBackend::requantizeModelProjToQ8_0(const model::GgufTensor& src) {
    if (src.dimensions.size() < 2) {
        throw std::runtime_error("per_layer_model_proj: expected 2D tensor");
    }
    const std::size_t K = src.dimensions[0];   // d_model
    const std::size_t N = src.dimensions[1];   // num_layers * per_layer_dim
    if (K % 32 != 0) {
        throw std::runtime_error(
            "per_layer_model_proj: K=" + std::to_string(K) +
            " not divisible by Q8_0 block size 32");
    }
    if (N != _config.blockCount * _perLayerDim) {
        throw std::runtime_error(
            "per_layer_model_proj: N=" + std::to_string(N) +
            " does not match num_layers*per_layer_dim=" +
            std::to_string(_config.blockCount * _perLayerDim));
    }

    const auto* srcQt = compute::quantType(src.type);
    if (srcQt == nullptr) {
        throw std::runtime_error(
            "per_layer_model_proj: source type " +
            std::string{model::typeInfo(src.type).name} +
            " not supported for dequant");
    }

    // GGUF stores weights as [rows × cols] with each ROW contiguous. For
    // our matmul convention rows are the OUTPUT dim (N), cols are the
    // INPUT dim (K). So one row = K floats = K/32 Q8_0 blocks. Total
    // output size = N * K/32 * 34 bytes.
    const std::size_t blocksPerRow = K / 32;
    _projQ8Bytes = N * blocksPerRow * 34;
    _projQ8 = UsmHandle{_ops.allocator(), _projQ8Bytes};
    auto* dst = _projQ8.as<std::uint8_t>();

    // Bytes per source ROW. For BF16 that's K*2; for F32 K*4; for
    // block-quantized types K/blockSize * blockBytes.
    const std::size_t srcRowElemBytes =
        srcQt->blockElements() > 1
            ? (K / srcQt->blockElements()) * srcQt->blockBytes()
            : K * srcQt->blockBytes();
    const auto* srcBase = static_cast<const std::uint8_t*>(src.usmPtr);

    std::vector<float> rowF32(K);
    for (std::size_t n = 0; n < N; ++n) {
        srcQt->dequantToF32(srcBase + n * srcRowElemBytes, K, rowF32.data());

        // Per-row: partition into K/32 blocks. Each block gets an
        // fp16 scale + 32 int8 quants.
        for (std::size_t b = 0; b < blocksPerRow; ++b) {
            const float* srcBlock = rowF32.data() + b * 32;
            std::uint8_t* dstBlock =
                dst + (n * blocksPerRow + b) * 34;

            float absMax = 0.0F;
            for (int i = 0; i < 32; ++i) {
                absMax = std::max(absMax, std::fabs(srcBlock[i]));
            }
            const float scale = (absMax > 0.0F) ? (absMax / 127.0F) : 0.0F;
            const std::uint16_t dHalf = floatToHalf(scale);
            std::memcpy(dstBlock, &dHalf, 2);

            const float invScale = (scale > 0.0F) ? (1.0F / scale) : 0.0F;
            for (int i = 0; i < 32; ++i) {
                const int q = static_cast<int>(std::lround(srcBlock[i] * invScale));
                const int qc = std::clamp(q, -127, 127);
                static_cast<std::int8_t*>(
                    static_cast<void*>(dstBlock + 2))[i] =
                    static_cast<std::int8_t>(qc);
            }
        }
    }

    MM_LOG_INFO("gemma4-e4b",
                "per_layer_model_proj: requantized {} → Q8_0, "
                "{} bytes ({} MiB), src bytes was {} ({} MiB)",
                model::typeInfo(src.type).name,
                _projQ8Bytes, _projQ8Bytes / (1024 * 1024),
                src.nbytes, src.nbytes / (1024 * 1024));

    // Round-trip sanity: dequant the first row of the requantized Q8_0
    // back to F32 and compare to the source's first row. Catches the
    // most obvious symmetric-quantize mistakes (scale-inverted values,
    // wrong block stride, etc.).
    {
        std::vector<float> srcFirstRow(K);
        std::vector<float> reFirstRow(K);
        srcQt->dequantToF32(srcBase, K, srcFirstRow.data());
        compute::dequantToF32(model::GgmlType::Q8_0, dst,
                              K, reFirstRow.data());
        float maxAbs = 0.0F, maxDiff = 0.0F;
        for (std::size_t k = 0; k < K; ++k) {
            maxAbs  = std::max(maxAbs,  std::fabs(srcFirstRow[k]));
            maxDiff = std::max(maxDiff, std::fabs(srcFirstRow[k] - reFirstRow[k]));
        }
        MM_LOG_INFO("gemma4-e4b",
                    "per_layer_model_proj Q8_0 roundtrip row0 — "
                    "maxAbs={:.6g} maxDiff={:.6g} first-src=[{:.4f} {:.4f} "
                    "{:.4f} {:.4f}] first-req=[{:.4f} {:.4f} {:.4f} {:.4f}]",
                    maxAbs, maxDiff,
                    srcFirstRow[0], srcFirstRow[1], srcFirstRow[2], srcFirstRow[3],
                    reFirstRow[0],  reFirstRow[1],  reFirstRow[2],  reFirstRow[3]);
    }

    // M8.K.Q8_0-Reorder — Phase 5 wiring for the E4B per_layer_model_proj
    // path. When the operator has enabled features.q8_0Reorder we
    // allocate a second USM buffer of the same size, copy the native
    // requantized weights into it, and reorder them in place to the
    // scales-then-quants layout that matmul_q8_0_vec_reorder consumes.
    // The native buffer stays untouched so prefill (M>1) still hits
    // the GEMM path through GpuMatmul; only decode (M=1) at
    // injectPerLayerInputs dispatches through the reorder kernel.
    // Reorder allocation is lazy: any mode == Disable, or any earlier
    // requant failure, and this block is skipped entirely.
    if (_ops.q8_0ReorderMode() != runtime::TriState::Disable) {
        try {
            _projQ8ReorderBytes = _projQ8Bytes;
            _projQ8Reorder = UsmHandle{_ops.allocator(),
                                       _projQ8ReorderBytes};
            std::memcpy(_projQ8Reorder.get(), _projQ8.get(),
                        _projQ8Bytes);
            std::vector<std::uint8_t> scratch(blocksPerRow * 34);
            compute::quant::Q8_0::reorderMatrixInPlace(
                _projQ8Reorder.get(), N, K, scratch.data());
            _ops.noteQ8_0ReorderApplied(_projQ8ReorderBytes,
                                        "per_layer_model_proj");
            MM_LOG_INFO("gemma4-e4b",
                        "per_layer_model_proj reorder copy: {} bytes "
                        "({} MiB) [mode={}] — decode (M=1) will dispatch "
                        "matmul_q8_0_vec_reorder, prefill (M>1) stays "
                        "on native GEMM",
                        _projQ8ReorderBytes,
                        _projQ8ReorderBytes / (1024 * 1024),
                        _ops.q8_0ReorderModeName());
        } catch (const std::exception& e) {
            MM_LOG_WARN("gemma4-e4b",
                        "per_layer_model_proj reorder copy failed ({}) "
                        "— decode falls back to native Q8_0 vec kernel",
                        e.what());
            _projQ8Reorder = UsmHandle{};
            _projQ8ReorderBytes = 0;
        }
    }
}

void Gemma4E4BBackend::ensurePleCapacity(std::size_t T) {
    if (T == 0) return;

    if (_pleTablePtr != nullptr && T > _pleBufCapT) {
        const std::size_t bytes =
            _config.blockCount * T * _perLayerDim * sizeof(float);
        _pleBuf = UsmHandle{_ops.allocator(), bytes};
        _pleBufCapT = T;
    }
    if (_projQ8Bytes > 0 && T > _pleProjBufCapT) {
        // Layout [T, num_layers, per_layer_dim] = [T, N] where N = 10752.
        const std::size_t bytes =
            T * _config.blockCount * _perLayerDim * sizeof(float);
        _pleProjBuf = UsmHandle{_ops.allocator(), bytes};
        _pleProjBufCapT = T;
    }
    if (T > _pleGateBufCapT) {
        const std::size_t bytes = T * _perLayerDim * sizeof(float);
        _pleGateBuf = UsmHandle{_ops.allocator(), bytes};
        _pleGateBufCapT = T;
    }
}

void Gemma4E4BBackend::prepareForward(std::span<const std::int32_t> tokIds,
                                      const float*                  hiddenStates,
                                      std::size_t                   T) {
    if (T == 0) {
        _pleActiveT = 0;
        return;
    }

    ensurePleCapacity(T);

    // ------------------------------------------------------------------
    // 1. Dequant per_layer_token_embd → _pleBuf, layout [L, T, D].
    // ------------------------------------------------------------------

    // Per llama.cpp/src/models/gemma4.cpp build_inp_per_layer():
    //   tok_embd_scale = sqrt(n_embd_per_layer)
    // Applied here directly during dequant so the combine step below
    // just sees the already-scaled embedding.
    const float tokEmbdScale =
        std::sqrt(static_cast<float>(_perLayerDim));

    float* const pleBuf = _pleBuf.as<float>();
    if (_pleTablePtr != nullptr) {
        const std::size_t bytesPerToken =
            _config.blockCount * _pleBytesPerBlock;
        const auto* base = static_cast<const std::uint8_t*>(_pleTablePtr);
        const std::size_t stride = T * _perLayerDim;

        for (std::size_t t = 0; t < T; ++t) {
            const std::int32_t tok = tokIds[t];
            if (tok < 0 || static_cast<std::size_t>(tok) >= _vocabSize) {
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
                for (std::size_t i = 0; i < _perLayerDim; ++i) {
                    dstSlice[i] *= tokEmbdScale;
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // 2. per_layer_model_proj chain — matmul + RMSNorm on hidden_states.
    // ------------------------------------------------------------------
    // Skip when the quantized weight or the norm couldn't load — the
    // model then runs with embedding-only per-layer inputs (degraded
    // but coherent enough for smoke).

    const bool haveProj = (_projQ8Bytes > 0)
                          && (_projNorm != nullptr)
                          && (hiddenStates != nullptr);
    if (haveProj) {
        const std::size_t d_model = _config.embeddingLength;
        const std::size_t projN   = _config.blockCount * _perLayerDim;
        float* const projBuf      = _pleProjBuf.as<float>();

        // M8.K.Q8_0-Reorder — for T==1 (decode) and when the reorder
        // copy was loaded, dispatch through matmul_q8_0_vec_reorder
        // directly. The reorder kernel is matvec-only, so prefill
        // (T>1) keeps hitting GpuMatmul's Q8_0 GEMM path against the
        // untouched `_projQ8` buffer.
        if (T == 1 && _projQ8ReorderBytes > 0) {
            _ops.matmulQ8_0VecReorderAsync(_projQ8Reorder.get(),
                                           projN, d_model,
                                           hiddenStates, projBuf);
        } else {
            _gmm.matmul(model::GgmlType::Q8_0, _projQ8.get(),
                        projN, d_model,
                        hiddenStates, T,
                        projBuf, /*scratch=*/nullptr);
        }

        // Per llama.cpp/src/models/gemma4.cpp project_per_layer_inputs():
        //   per_layer_projection_scale = 1/sqrt(n_embd)
        // Applied BEFORE the RMSNorm so the norm sees the correctly
        // scaled projection.
        const float projScale =
            1.0F / std::sqrt(static_cast<float>(d_model));
        _ops.mulScalarAsync(projBuf, projScale, T * projN);

        // RMSNorm per (t, L) slice of 256 elements — flatten to a
        // [T*num_layers, per_layer_dim] view with weight [per_layer_dim].
        _ops.rmsNormAsync(projBuf, T * _config.blockCount, _perLayerDim,
                          _projNorm, _config.rmsNormEps, projBuf);
    }

    // ------------------------------------------------------------------
    // 3. Combine into _pleBuf as (proj + embd) * 1/sqrt(2).
    // ------------------------------------------------------------------
    // proj layout [T, L, D] vs embd (_pleBuf) [L, T, D] — the combine
    // has to transpose across (T, L). Small enough loop that we do it
    // on the CPU against USM (unified memory on Meteor Lake iGPU).

    if (haveProj) {
        // Sync the queue so the GPU rmsnorm has landed before CPU reads.
        _ops.queue().flush();

        const float invSqrt2 = 0.70710678118654752440F;
        const float* proj    = _pleProjBuf.as<float>();
        for (std::size_t L = 0; L < _config.blockCount; ++L) {
            for (std::size_t t = 0; t < T; ++t) {
                float* dst =
                    pleBuf + L * T * _perLayerDim + t * _perLayerDim;
                const float* src =
                    proj + t * _config.blockCount * _perLayerDim
                         + L * _perLayerDim;
                for (std::size_t i = 0; i < _perLayerDim; ++i) {
                    dst[i] = (dst[i] + src[i]) * invSqrt2;
                }
            }
        }
    }

    _pleActiveT = T;

    if (!_pleDumpDone && T > 0) {
        _pleDumpDone = true;
        const float* p0  = pleBuf;
        const float* p20 = pleBuf + 20 * (T * _perLayerDim);
        MM_LOG_INFO("gemma4-e4b",
                    "per-layer input sanity: L=0 t=0 → [{:.4f} {:.4f} {:.4f} "
                    "{:.4f} {:.4f} {:.4f} {:.4f} {:.4f}] tok0={}",
                    p0[0], p0[1], p0[2], p0[3], p0[4], p0[5], p0[6], p0[7],
                    tokIds.front());
        MM_LOG_INFO("gemma4-e4b",
                    "per-layer input sanity: L=20 t=0 → [{:.4f} {:.4f} {:.4f} "
                    "{:.4f} {:.4f} {:.4f} {:.4f} {:.4f}]",
                    p20[0], p20[1], p20[2], p20[3], p20[4], p20[5], p20[6], p20[7]);

        // Diagnostics for the projection chain — separate the embed and
        // proj contributions so we can tell which one is producing the
        // final magnitude. Also show a hidden-state sample as sanity.
        if (haveProj) {
            const float* proj0  = _pleProjBuf.as<float>();
            const float* proj20 = proj0 + 20 * _perLayerDim;
            MM_LOG_INFO("gemma4-e4b",
                        "proj sanity (post-rmsnorm): L=0 t=0 → [{:.4f} {:.4f} "
                        "{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}]",
                        proj0[0], proj0[1], proj0[2], proj0[3],
                        proj0[4], proj0[5], proj0[6], proj0[7]);
            MM_LOG_INFO("gemma4-e4b",
                        "proj sanity (post-rmsnorm): L=20 t=0 → [{:.4f} {:.4f} "
                        "{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}]",
                        proj20[0], proj20[1], proj20[2], proj20[3],
                        proj20[4], proj20[5], proj20[6], proj20[7]);
            MM_LOG_INFO("gemma4-e4b",
                        "hidden sanity t=0 → [{:.4f} {:.4f} {:.4f} {:.4f} "
                        "{:.4f} {:.4f} {:.4f} {:.4f}]",
                        hiddenStates[0], hiddenStates[1], hiddenStates[2],
                        hiddenStates[3], hiddenStates[4], hiddenStates[5],
                        hiddenStates[6], hiddenStates[7]);
        }
    }
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

    // --- Dense SwiGLU-GELU FFN ----------------------------------------
    //
    // runAttentionSection intentionally does NOT do the attn residual —
    // we do it fused with ffn_norm in a single kernel here. `projOutBuf`
    // holds `attn_post_norm(attn_out)` on entry, and after this call:
    //   x += projOutBuf
    //   normBuf = rmsnorm(x, ffnNorm)

    _op.mark(runtime::OpProfiler::Cat::NORM);
    _ops.addRmsNormAsync(x, projOutBuf, T, d_model,
                         static_cast<const float*>(ffnNorm->usmPtr),
                         _config.rmsNormEps,
                         normBuf);
    dumpStage("attn_out", blockIdx, x, T, d_model);

    _op.mark(runtime::OpProfiler::Cat::MATMUL);
    {
        runtime::UnorderedScope u{_ops.queue()};
        _gmm.matmulAsync(ffnGate->type, ffnGate->usmPtr, ff_dim, d_model,
                         normBuf, T, gateOutBuf, matmulScratch);
        _gmm.matmulAsync(ffnUp->type, ffnUp->usmPtr, ff_dim, d_model,
                         normBuf, T, upOutBuf, matmulScratch);
    }

    _op.mark(runtime::OpProfiler::Cat::ACTIVATION);
    _ops.geluMulAsync(gateOutBuf, upOutBuf, T * ff_dim);

    _op.mark(runtime::OpProfiler::Cat::MATMUL);
    _gmm.matmul(ffnDown->type, ffnDown->usmPtr, d_model, ff_dim,
                gateOutBuf, T,
                projOutBuf, matmulScratch);

    _op.mark(runtime::OpProfiler::Cat::NORM);
    _ops.rmsNormAsync(projOutBuf, T, d_model,
                      static_cast<const float*>(ffwPost->usmPtr),
                      _config.rmsNormEps,
                      projOutBuf);

    _op.mark(runtime::OpProfiler::Cat::RESIDUAL);
    _ops.addResidualAsync(x, projOutBuf, T * d_model);

    // --- PLE injection -------------------------------------------------
    //
    // Skipped when either the PLE table or the model-proj weight is
    // missing — logs at construction already flagged that.

    if (_pleTablePtr != nullptr && _pleActiveT >= T) {
        const auto* inpGate  = requireTensor(blockIdx, "inp_gate.weight",  "Gemma4E4BBackend");
        const auto* proj     = requireTensor(blockIdx, "proj.weight",      "Gemma4E4BBackend");
        const auto* postNorm = requireTensor(blockIdx, "post_norm.weight", "Gemma4E4BBackend");

        float* const pleGateBuf = _pleGateBuf.as<float>();
        const float* const pleSliceForLayer =
            _pleBuf.as<float>()
            + blockIdx * (_pleActiveT * _perLayerDim);

        _op.mark(runtime::OpProfiler::Cat::MATMUL);
        _gmm.matmul(inpGate->type, inpGate->usmPtr,
                    _perLayerDim, d_model,
                    x, T,
                    pleGateBuf, matmulScratch);

        _op.mark(runtime::OpProfiler::Cat::ACTIVATION);
        // Fused GELU + element-wise mul: gate = GELU(gate) * per_layer_input.
        // Per llama.cpp gemma4.cpp line 373 (`ggml_gelu`) — the simplified
        // Gemma 4 path uses GELU, not SiLU (unlike the full Gemma 4 E-Series
        // path which does use SiLU alongside AltUp/Laurel).
        _ops.geluMulAsync(pleGateBuf, pleSliceForLayer, T * _perLayerDim);

        _op.mark(runtime::OpProfiler::Cat::MATMUL);
        _gmm.matmul(proj->type, proj->usmPtr,
                    d_model, _perLayerDim,
                    pleGateBuf, T,
                    projOutBuf, matmulScratch);

        _op.mark(runtime::OpProfiler::Cat::NORM);
        // Plain w * rmsnorm(x) — Gemma 4 E-Series uses the same convention
        // as Gemma 4 base (no shift).
        _ops.rmsNormAsync(projOutBuf, T, d_model,
                          static_cast<const float*>(postNorm->usmPtr),
                          _config.rmsNormEps,
                          projOutBuf);

        _op.mark(runtime::OpProfiler::Cat::RESIDUAL);
        _ops.addResidualAsync(x, projOutBuf, T * d_model);
    }

    // --- Block output scale ------------------------------------------
    // Learned scalar `layer_output_scale.weight` multiplied over the
    // block output. Matches `ggml_mul(cur, scalar)` in
    // llama.cpp/src/models/gemma4.cpp:393.

    const float scaleVal = *static_cast<const float*>(outScale->usmPtr);
    _op.mark(runtime::OpProfiler::Cat::ACTIVATION);
    _ops.mulScalarAsync(x, scaleVal, T * d_model);

    _op.finish();
}

} // namespace mimirmind::runtime::arch