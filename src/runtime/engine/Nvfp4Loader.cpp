// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/engine/Nvfp4Loader.hpp"

#include "runtime/InferenceEngine.hpp"

#ifdef MIMIRMIND_HAVE_CUDA

#include "compute/cuda/CudaMaterializerOps.hpp"
#include "core/gpu/cuda/CudaComputeContext.hpp"
#include "core/log/Log.hpp"
#include "core/modelopt/HfQuantConfig.hpp"
#include "core/modelopt/Qwen35MoeMaterializer.hpp"
#include "core/safetensors/SafetensorsModel.hpp"
#include "runtime/nvfp4/ComputeOpsUploader.hpp"
#include "runtime/nvfp4/NvFp4WeightsMap.hpp"
#include "runtime/nvfp4/Qwen35MoeConfig.hpp"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#endif

namespace mimirmind::runtime::engine {

#ifndef MIMIRMIND_HAVE_CUDA

void Nvfp4Loader::load(InferenceEngine&, std::string_view, std::string_view) {
    throw std::runtime_error("Nvfp4Loader::load: NVFP4 requires the CUDA "
                             "backend (built without MIMIRMIND_ENABLE_CUDA)");
}

#else

void Nvfp4Loader::load(InferenceEngine& e,
                       std::string_view checkpointDir,
                       std::string_view tokenizerGguf) {
    if (e._computeCtx->kind() != core::backend::BackendKind::Cuda) {
        throw std::runtime_error("Nvfp4Loader::load: NVFP4 needs the CUDA "
                                 "backend, but this engine picked another");
    }
    if (tokenizerGguf.empty()) {
        throw std::runtime_error("Nvfp4Loader::load: models[].tokenizerGguf is "
                                 "required (the NVFP4 checkpoint ships no GGUF tokenizer)");
    }

    const std::string dir{checkpointDir};
    auto readText = [](const std::filesystem::path& p) -> std::string {
        std::ifstream f(p);
        if (!f) throw std::runtime_error("Nvfp4Loader::load: cannot read " + p.string());
        std::stringstream ss;
        ss << f.rdbuf();
        return ss.str();
    };

    MM_LOG_INFO("engine", "loadModelNvfp4: '{}' (tokenizer from '{}')",
                checkpointDir, tokenizerGguf);

    // 1. Arch params from config.json (GGUF-metadata parse is GGUF-only).
    e._config = runtime::nvfp4::parseQwen35MoeSafetensorsConfig(
        readText(std::filesystem::path{dir} / "config.json"));

    // 2. Tokenizer via the GGUF shortcut (HF tokenizer.json not parsed yet).
    e._reader.open(tokenizerGguf);
    e._tokenizer.loadFromGguf(e._reader);

    // 3. Upload the NVFP4/FP8 weights (+ BF16 passthroughs) to the device.
    runtime::nvfp4::ComputeOpsUploader uploader(*e._ops);
    e._nvfp4Model = std::make_unique<runtime::nvfp4::NvFp4Model>(
        runtime::nvfp4::loadNvfp4Model(dir, uploader));

    // 4. Build the materialization plan (tensor shapes + per-module schemes).
    core::safetensors::SafetensorsModel sm;
    sm.open(dir);
    const core::modelopt::HfQuantConfig hfCfg =
        core::modelopt::HfQuantConfig::parse(
            readText(std::filesystem::path{dir} / "hf_quant_config.json"));
    const core::modelopt::Qwen35MoeArch arch{
        static_cast<int>(e._config.blockCount),
        static_cast<int>(e._config.expertCount),
        4 /* full_attention_interval; layer_types agrees for this model */};
    const std::vector<core::modelopt::MaterializationStep> steps =
        core::modelopt::planQwen35MoeMaterialization(sm, hfCfg, arch);

    // 5. Dequantise every weight to BF16 on device (weight-only W4A16).
    auto& cudaCtx = static_cast<core::cuda::CudaComputeContext&>(*e._computeCtx);
    compute::cuda::CudaMaterializerOps devOps(cudaCtx, *e._ops);
    e._materializedBf16 =
        runtime::nvfp4::executeMaterialization(steps, *e._nvfp4Model, devOps);
    cudaCtx.stream().synchronize(); // all dequant kernels + D2D copies complete

    // 5b. GatedDeltaNet value-head regroup (HF -> GGUF layout).
    //
    // llama.cpp's GGUF conversion stores the linear-attention value stream
    // as (gqa, num_k_heads, head_dim), while the HF checkpoint stores it as
    // (num_k_heads, gqa, head_dim). The NVFP4 materializer emits raw HF
    // layout, but the backend that consumes these BF16 tensors was written
    // for the GGUF convention — so without this regroup the 32 value heads
    // are scrambled, the delta-rule recurrence reads the wrong channels, and
    // the whole model degenerates into incoherent tokens. Full-attention
    // layers are unaffected. Verified against the coherent GGUF Q4_K weights.
    //
    // The regroup is a pure element permutation of the vDim value channels:
    //   dst(j,k,d) <- src(k,j,d)   (k: k-head, j: gqa slot, d: head dim)
    // applied to the value ROWS of attn_qkv / ssm_conv1d, ALL rows of
    // attn_gate, and the value COLUMNS of ssm_out.
    {
        const std::size_t KH  = e._config.ssmNumKHeads();   // num k heads (16)
        const std::size_t VH  = e._config.ssmNumVHeads();   // num v heads (32)
        const std::size_t HD  = e._config.ssmHeadDim();     // head dim (128)
        const std::size_t GQA = (KH > 0) ? VH / KH : 1;     // v-per-k (2)
        const std::size_t vDim = VH * HD;                   // value dim (4096)
        if (GQA >= 2 && KH * GQA * HD == vDim) {
            std::vector<std::size_t> perm(vDim);
            for (std::size_t k = 0; k < KH; ++k) {
                for (std::size_t j = 0; j < GQA; ++j) {
                    for (std::size_t d = 0; d < HD; ++d) {
                        const std::size_t src = k * GQA * HD + j * HD + d;
                        const std::size_t dst = j * KH * HD + k * HD + d;
                        perm[dst] = src;
                    }
                }
            }
            auto regroupRows = [&](runtime::nvfp4::MaterializedTensor& t,
                                   std::size_t rowStart) {
                const std::size_t inCols    = t.ggufDims[0];
                const std::size_t elemBytes = t.isF32 ? 4 : 2;
                const std::size_t rowBytes  = inCols * elemBytes;
                const std::size_t nbytes    = vDim * rowBytes;
                auto* base = static_cast<std::uint8_t*>(t.buffer.get())
                             + rowStart * rowBytes;
                std::vector<std::uint8_t> in(nbytes), out(nbytes);
                e._ops->readbackToHost(in.data(), base, nbytes);
                for (std::size_t r = 0; r < vDim; ++r) {
                    std::memcpy(out.data() + r * rowBytes,
                                in.data() + perm[r] * rowBytes, rowBytes);
                }
                e._ops->uploadHostBytes(base, out.data(), nbytes);
            };
            auto regroupCols = [&](runtime::nvfp4::MaterializedTensor& t) {
                const std::size_t cols      = t.ggufDims[0];  // == vDim
                const std::size_t rows      = t.ggufDims[1];
                const std::size_t elemBytes = t.isF32 ? 4 : 2;
                const std::size_t nbytes    = rows * cols * elemBytes;
                auto* base = static_cast<std::uint8_t*>(t.buffer.get());
                std::vector<std::uint8_t> in(nbytes), out(nbytes);
                e._ops->readbackToHost(in.data(), base, nbytes);
                for (std::size_t r = 0; r < rows; ++r) {
                    const std::size_t ro = r * cols;
                    for (std::size_t c = 0; c < cols; ++c) {
                        std::memcpy(out.data() + (ro + c) * elemBytes,
                                    in.data() + (ro + perm[c]) * elemBytes,
                                    elemBytes);
                    }
                }
                e._ops->uploadHostBytes(base, out.data(), nbytes);
            };
            std::size_t regrouped = 0;
            for (auto& t : e._materializedBf16) {
                const std::string& n = t.ggufName;
                if (n.ends_with(".attn_qkv.weight")) {
                    regroupRows(t, /*rowStart=*/2 * KH * HD);
                    ++regrouped;
                } else if (n.ends_with(".ssm_conv1d.weight")) {
                    regroupRows(t, /*rowStart=*/2 * KH * HD);
                    ++regrouped;
                } else if (n.ends_with(".attn_gate.weight")) {
                    regroupRows(t, /*rowStart=*/0);
                    ++regrouped;
                } else if (n.ends_with(".ssm_out.weight")) {
                    regroupCols(t);
                    ++regrouped;
                }
            }
            cudaCtx.stream().synchronize();
            MM_LOG_INFO("engine",
                        "loadModelNvfp4: GatedDeltaNet value-head regroup "
                        "applied to {} tensors (KH={} GQA={} HD={})",
                        regrouped, KH, GQA, HD);
        }
    }

    // 5c. Re-quantise the dense attention projections BF16 -> Q8_0.
    //
    // DEFAULT OFF: empirically this BREAKS coherence on qwen35moe. Q8_0 is a
    // per-32-block *linear* quant, but these projections are FP8(e4m3)/NVFP4
    // origin — *logarithmic*, wide-dynamic-range. In a block with one large +
    // many small weights, Q8_0's absmax scale crushes the small weights to ~0;
    // e4m3/NVFP4 preserve them via log/grouped scales, and the model is
    // calibrated for that. Kept behind MIMIRMIND_NVFP4_Q8_PROJ (off) for A/B.
    // See lesson q8_0-linear-requant-crushes-fp8-weights.
    //   "all"/"1"/"fa"/"gdn" — quantise that group   "0"/unset — keep BF16
    {
        const char* q8env = std::getenv("MIMIRMIND_NVFP4_Q8_PROJ");
        const std::string_view q8mode = (q8env == nullptr) ? "0" : q8env;
        const bool q8proj = (q8mode != "0");
        if (q8proj) {
            const bool wantFa  = (q8mode == "all" || q8mode == "1" || q8mode == "fa");
            const bool wantGdn = (q8mode == "all" || q8mode == "1" || q8mode == "gdn");
            auto isFa = [](std::string_view n) {
                return n.ends_with(".attn_q.weight") || n.ends_with(".attn_k.weight")
                    || n.ends_with(".attn_v.weight") || n.ends_with(".attn_output.weight");
            };
            auto isGdn = [](std::string_view n) {
                return n.ends_with(".attn_qkv.weight") || n.ends_with(".attn_gate.weight")
                    || n.ends_with(".ssm_out.weight");
            };
            auto isProj = [&](std::string_view n) {
                return (wantFa && isFa(n)) || (wantGdn && isGdn(n));
            };
            std::size_t nQuant = 0;
            std::uint64_t bytesBefore = 0, bytesAfter = 0;
            for (auto& t : e._materializedBf16) {
                if (t.isF32 || t.ggufDims.size() < 2 || !isProj(t.ggufName)) continue;
                const std::uint64_t K    = t.ggufDims[0];  // input dim (contiguous)
                const std::uint64_t rows = t.ggufDims[1];  // output dim
                if (K == 0 || rows == 0 || (K % 32) != 0 || K * rows != t.elems) continue;
                const std::size_t q8Bytes =
                    (static_cast<std::size_t>(t.elems) / 32) * 34;
                compute::ComputeBuffer q8 = devOps.allocate(q8Bytes);
                devOps.quantizeBf16ToQ8_0(q8.get(), t.buffer.get(), rows, K);
                cudaCtx.stream().synchronize(); // finish before the BF16 buffer frees
                bytesBefore += static_cast<std::uint64_t>(t.elems) * 2;
                bytesAfter  += q8Bytes;
                t.buffer = std::move(q8); // frees the BF16 buffer (RAII)
                t.isQ8_0 = true;
                ++nQuant;
            }
            MM_LOG_INFO("engine",
                        "loadModelNvfp4: re-quantised {} attention projections "
                        "BF16 -> Q8_0 (mode='{}', {} MiB -> {} MiB)",
                        nQuant, q8mode, bytesBefore >> 20, bytesAfter >> 20);
        }
    }

    // 5d. Re-quantise the MoE expert banks BF16 -> K-quant (gate/up->Q4_K,
    // down->Q6_K). The experts dominate the decode-time weight traffic and the
    // qwen35moe decode is memory-bound. K-quants keep per-sub-block scale+min,
    // preserving the log-distributed weights (unlike a flat linear Q8_0), and
    // are the format llama.cpp uses for these experts (proven coherent). They
    // feed the existing fused-K MoE kernels unchanged (the backend dispatches
    // on tensor .type). MIMIRMIND_NVFP4_MOE_Q4K=0 keeps BF16 (A/B).
    {
        const char* mq = std::getenv("MIMIRMIND_NVFP4_MOE_Q4K");
        const bool moeQ4K = (mq == nullptr) || (std::string_view{mq} != "0");
        if (moeQ4K) {
            std::size_t nQuant = 0;
            std::uint64_t bytesBefore = 0, bytesAfter = 0;
            for (auto& t : e._materializedBf16) {
                if (t.isF32 || t.isQ8_0) continue;
                const std::string& n = t.ggufName;
                const bool isGateUp = n.ends_with(".ffn_gate_exps.weight")
                                   || n.ends_with(".ffn_up_exps.weight");
                const bool isDown   = n.ends_with(".ffn_down_exps.weight");
                if (!isGateUp && !isDown) continue;
                if ((t.elems % 256) != 0) continue;
                const std::size_t qBytes = isGateUp
                    ? (static_cast<std::size_t>(t.elems) / 256) * 144
                    : (static_cast<std::size_t>(t.elems) / 256) * 210;
                compute::ComputeBuffer q = devOps.allocate(qBytes);
                if (isGateUp) devOps.quantizeBf16ToQ4K(q.get(), t.buffer.get(), t.elems);
                else          devOps.quantizeBf16ToQ6K(q.get(), t.buffer.get(), t.elems);
                cudaCtx.stream().synchronize();
                bytesBefore += static_cast<std::uint64_t>(t.elems) * 2;
                bytesAfter  += qBytes;
                t.buffer = std::move(q); // frees the BF16 bank (RAII)
                if (isGateUp) t.isQ4K = true;
                else          t.isQ6K = true;
                ++nQuant;
            }
            MM_LOG_INFO("engine",
                        "loadModelNvfp4: re-quantised {} MoE expert banks "
                        "(gate/up->Q4_K, down->Q6_K) BF16 -> Kquant ({} MiB -> {} MiB)",
                        nQuant, bytesBefore >> 20, bytesAfter >> 20);
        }
    }

    // 6. Expose the BF16 tensors as a GGUF-convention WeightsMap.
    e._weights.emplace(runtime::nvfp4::buildBf16WeightsMap(e._materializedBf16));

    // 7. Release the packed NVFP4 uploads — only the BF16 result is needed.
    e._nvfp4Model.reset();

    MM_LOG_INFO("engine", "loadModelNvfp4: materialised {} BF16 tensors",
                e._materializedBf16.size());
}

#endif // MIMIRMIND_HAVE_CUDA

} // namespace mimirmind::runtime::engine
