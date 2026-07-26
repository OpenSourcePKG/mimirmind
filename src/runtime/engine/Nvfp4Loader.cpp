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

#include <algorithm>
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

    // 5e. Re-quantise the dense attention projections BF16 -> blocked-FP8 (E4M3).
    //
    // These log-distributed FP8/NVFP4-origin projections are exactly what a
    // linear Q8_0 crushes (q8_0-linear-requant lesson). Blocked-FP8 keeps the
    // E4M3 log format per 32-block (34 B / 32 ≈ 1.06 B/elem vs 2 B BF16) so the
    // small weights survive; the matmul_fp8_vec/_gemm kernels consume it (scale
    // embedded per block, so no scale plumbing / matmulAsync signature change).
    // Halves the always-read attention traffic (all layers, every token).
    // MIMIRMIND_NVFP4_ATTN_FP8=0 keeps BF16 (A/B).
    {
        // MIMIRMIND_NVFP4_ATTN_FP8 selects which projections to keep FP8:
        //   unset / "gdn" — only the GatedDeltaNet linear projections
        //                   (attn_qkv/attn_gate/ssm_out), which ARE natively FP8
        //                   in the checkpoint → E4M3 is lossless-ish + coherent
        //   "fa"          — full-attention q/k/v/output only (NVFP4/BF16 origin;
        //                   E4M3 is a downgrade there — degrades)
        //   "all"         — both     "0" — disabled (keep BF16)
        const char* fp8env = std::getenv("MIMIRMIND_NVFP4_ATTN_FP8");
        const std::string_view mode = (fp8env == nullptr) ? "gdn" : fp8env;
        if (mode != "0") {
            const bool wantFa  = (mode == "all" || mode == "fa");
            const bool wantGdn = (mode == "all" || mode == "gdn");
            auto isFa = [](std::string_view n) {
                return n.ends_with(".attn_q.weight") || n.ends_with(".attn_k.weight")
                    || n.ends_with(".attn_v.weight") || n.ends_with(".attn_output.weight");
            };
            auto isGdn = [](std::string_view n) {
                return n.ends_with(".attn_qkv.weight") || n.ends_with(".attn_gate.weight")
                    || n.ends_with(".ssm_out.weight");
            };
            auto isAttnProj = [&](std::string_view n) {
                return (wantFa && isFa(n)) || (wantGdn && isGdn(n));
            };
            std::size_t nQuant = 0;
            std::uint64_t bytesBefore = 0, bytesAfter = 0;
            for (auto& t : e._materializedBf16) {
                if (t.isF32 || t.isQ8_0 || t.isQ4K || t.isQ6K) continue;
                if (t.ggufDims.size() < 2 || !isAttnProj(t.ggufName)) continue;
                const std::uint64_t K    = t.ggufDims[0];  // input dim (contiguous)
                const std::uint64_t rows = t.ggufDims[1];  // output dim
                if (K == 0 || rows == 0 || (K % 32) != 0 || K * rows != t.elems) continue;
                const std::size_t fp8Bytes =
                    (static_cast<std::size_t>(t.elems) / 32) * 34;
                compute::ComputeBuffer fp8 = devOps.allocate(fp8Bytes);
                devOps.quantizeBf16ToFp8(fp8.get(), t.buffer.get(), rows, K);
                cudaCtx.stream().synchronize();
                bytesBefore += static_cast<std::uint64_t>(t.elems) * 2;
                bytesAfter  += fp8Bytes;
                t.buffer = std::move(fp8); // frees the BF16 buffer (RAII)
                t.isFp8  = true;
                ++nQuant;
            }
            MM_LOG_INFO("engine",
                        "loadModelNvfp4: re-quantised {} attention projections "
                        "BF16 -> blocked-FP8 E4M3 (mode='{}', {} MiB -> {} MiB)",
                        nQuant, mode, bytesBefore >> 20, bytesAfter >> 20);
        }
    }

    // 5f. Keep the dense NVFP4 projections native 4-bit (blocked-NVFP4).
    //
    // The MoE shared-expert projections (ffn_*_shexp) are W4A16_NVFP4 in the
    // checkpoint and go through the dense matmulAsync (always active, read every
    // token). Their BF16 materialisation holds exactly the NVFP4 values widened,
    // so keeping them 4-bit is LOSSLESS (no re-quant; re-quantising BF16 back to
    // the very coarse E2M1 would double-quant and degrade). Repackage the
    // original NVFP4 (packed E2M1 + per-16 E4M3 block-scale + global, still
    // resident in _nvfp4Model) into a single-pointer blocked format the
    // matmul_nvfp4blk kernels consume (embedded folded scale, no plumbing).
    //
    // NB: the full-attention self_attn.{q,k,v,o} are NOT quantised in this
    // checkpoint (no quantized_layers entry) — they stay full-precision BF16 by
    // design, so there is no native NVFP4 form to keep for them.
    // MIMIRMIND_NVFP4_SHEXP=0 keeps BF16 (A/B).
    if (e._nvfp4Model) {
        const char* faenv = std::getenv("MIMIRMIND_NVFP4_SHEXP");
        const bool faNvblk = (faenv == nullptr) || (std::string_view{faenv} != "0");
        if (faNvblk) {
            auto isFullAttn = [](std::string_view n) {
                return n.ends_with(".ffn_gate_shexp.weight")
                    || n.ends_with(".ffn_up_shexp.weight")
                    || n.ends_with(".ffn_down_shexp.weight");
            };
            std::size_t nRepack = 0;
            std::uint64_t bytesBefore = 0, bytesAfter = 0;
            for (const auto& step : steps) {
                if (!isFullAttn(step.ggufName) || step.sources.size() != 1) continue;
                const auto& src = step.sources[0];
                if (src.kind != core::modelopt::SourceKind::Nvfp4) continue;
                if ((src.in % 32) != 0) continue;
                auto it = std::find_if(
                    e._materializedBf16.begin(), e._materializedBf16.end(),
                    [&](const runtime::nvfp4::MaterializedTensor& t) {
                        return t.ggufName == step.ggufName;
                    });
                if (it == e._materializedBf16.end() || it->isF32) continue;
                const auto* pk = e._nvfp4Model->find(src.hfWeightName);
                const std::string base{src.hfWeightName};
                const std::string baseNoW =
                    base.size() > 7 ? base.substr(0, base.size() - 7) : base;
                const auto* bs = e._nvfp4Model->find(baseNoW + ".weight_scale");
                const auto* gs = e._nvfp4Model->find(baseNoW + ".weight_scale_2");
                if (pk == nullptr || bs == nullptr || gs == nullptr) continue;
                const float global = devOps.readF32(gs->devPtr);
                const std::size_t blkBytes =
                    (static_cast<std::size_t>(it->elems) / 32) * 20;
                compute::ComputeBuffer nb = devOps.allocate(blkBytes);
                devOps.repackageNvfp4ToBlk(nb.get(), pk->devPtr, bs->devPtr, global,
                                           src.rows, src.in);
                cudaCtx.stream().synchronize();
                bytesBefore += static_cast<std::uint64_t>(it->elems) * 2;
                bytesAfter  += blkBytes;
                it->buffer     = std::move(nb); // frees the BF16 buffer (RAII)
                it->isNvfp4Blk = true;
                ++nRepack;
            }
            MM_LOG_INFO("engine",
                        "loadModelNvfp4: kept {} dense NVFP4 (shared-expert) "
                        "projections native blocked-NVFP4 ({} MiB -> {} MiB)",
                        nRepack, bytesBefore >> 20, bytesAfter >> 20);
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
