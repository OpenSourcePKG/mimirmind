// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/engine/Nvfp4Loader.hpp"

#include "runtime/InferenceEngine.hpp"

#ifdef MIMIRMIND_HAVE_CUDA

#include "compute/cuda/CudaMaterializerOps.hpp"
#include "core/gpu/cuda/CudaComputeContext.hpp"
#include "core/log/Log.hpp"
#include "core/modelopt/BlockScaleSwizzle.hpp"
#include "core/modelopt/CompressedTensorsConfig.hpp"
#include "core/modelopt/Gemma4Materializer.hpp"
#include "core/modelopt/HfQuantConfig.hpp"
#include "core/modelopt/Qwen3_5DenseMaterializer.hpp"
#include "core/modelopt/Qwen3_5MoeMaterializer.hpp"
#include "core/safetensors/SafetensorsModel.hpp"
#include "runtime/nvfp4/ComputeOpsUploader.hpp"
#include "runtime/nvfp4/Gemma4Config.hpp"
#include "runtime/nvfp4/NvFp4WeightsMap.hpp"
#include "runtime/nvfp4/Qwen3_5MoeConfig.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
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

void Nvfp4Loader::load(InferenceEngine&, std::string_view, std::string_view,
                       core::safetensors::SafetensorsModel*) {
    throw std::runtime_error("Nvfp4Loader::load: NVFP4 requires the CUDA "
                             "backend (built without MIMIRMIND_ENABLE_CUDA)");
}

#else

namespace {

/// True if `config.json` names a Gemma-4 model (compressed-tensors
/// "nvfp4-pack-quantized"), vs a ModelOpt qwen35moe checkpoint. Keyed on the
/// top-level `model_type` / `architectures[0]` (a Gemma-4 unified multimodal
/// checkpoint reports e.g. "Gemma4UnifiedForConditionalGeneration").
bool isGemma4Config(const std::string& configText) {
    nlohmann::json j =
        nlohmann::json::parse(configText, nullptr, /*allow_exceptions=*/false);
    if (!j.is_object()) {
        return false;
    }
    auto lower = [](std::string s) {
        for (char& ch : s) ch = static_cast<char>(std::tolower(
                              static_cast<unsigned char>(ch)));
        return s;
    };
    auto contains = [&](const std::string& hay, const char* needle) {
        return lower(hay).find(needle) != std::string::npos;
    };
    if (j.contains("model_type") && j["model_type"].is_string()
        && contains(j["model_type"].get<std::string>(), "gemma")) {
        return true;
    }
    if (j.contains("architectures") && j["architectures"].is_array()
        && !j["architectures"].empty() && j["architectures"][0].is_string()) {
        return contains(j["architectures"][0].get<std::string>(), "gemma");
    }
    return false;
}

} // namespace

void Nvfp4Loader::load(InferenceEngine&                     e,
                       std::string_view                     checkpointDir,
                       std::string_view                     tokenizerGguf,
                       core::safetensors::SafetensorsModel* attachedSm) {
    if (e._computeCtx->kind() != core::backend::BackendKind::Cuda) {
        throw std::runtime_error("Nvfp4Loader::load: NVFP4 needs the CUDA "
                                 "backend, but this engine picked another");
    }
    const std::string dir{checkpointDir};
    auto readText = [](const std::filesystem::path& p) -> std::string {
        std::ifstream f(p);
        if (!f) throw std::runtime_error("Nvfp4Loader::load: cannot read " + p.string());
        std::stringstream ss;
        ss << f.rdbuf();
        return ss.str();
    };

    const std::string configText =
        readText(std::filesystem::path{dir} / "config.json");

    // --- Arch dispatch --------------------------------------------------
    // compressed-tensors Gemma-4 (dense text tower) takes a dedicated,
    // self-contained path: a different config schema (quant config lives IN
    // config.json, not hf_quant_config.json), a different NVFP4 name-triple
    // (.weight_packed / .weight_scale / .weight_global_scale, reciprocal
    // global), and none of the qwen35moe post-passes (GDN regroup / MTP /
    // MoE banks / shared-expert repack). It reuses the generic upload +
    // dequant + WeightsMap tail. The shared finalize (createArchBackend off
    // _config.architecture == "gemma4") then selects Gemma4DenseBackend.
    if (isGemma4Config(configText)) {
        MM_LOG_INFO("engine",
                    "loadModelNvfp4: '{}' — Gemma-4 (compressed-tensors, dense "
                    "text tower)", checkpointDir);
        e._config = runtime::nvfp4::parseGemma4SafetensorsConfig(configText);

        if (tokenizerGguf.empty()) {
            e._tokenizer.loadFromHfJson(dir);
        } else {
            e._reader.open(tokenizerGguf);
            e._tokenizer.loadFromGguf(e._reader);
        }

        runtime::nvfp4::ComputeOpsUploader uploader(*e._ops);
        e._nvfp4Model = std::make_unique<runtime::nvfp4::NvFp4Model>(
            attachedSm ? runtime::nvfp4::loadNvfp4Model(*attachedSm, dir, uploader)
                       : runtime::nvfp4::loadNvfp4Model(dir, uploader));

        core::safetensors::SafetensorsModel localSm;
        if (attachedSm == nullptr) localSm.open(dir);
        core::safetensors::SafetensorsModel& sm = attachedSm ? *attachedSm : localSm;
        core::modelopt::Gemma4Arch garch;
        garch.numLayers = static_cast<int>(e._config.blockCount);
        garch.isFullAttn.assign(e._config.blockCount, false);
        for (std::size_t L = 0; L < e._config.blockCount; ++L) {
            // Full-attention where the per-layer SWA mask says NOT sliding.
            const bool sliding = (L < e._config.slidingWindowPattern.size())
                                     ? static_cast<bool>(
                                           e._config.slidingWindowPattern[L])
                                     : true;
            garch.isFullAttn[L] = !sliding;
        }
        const std::vector<core::modelopt::MaterializationStep> gsteps =
            core::modelopt::planGemma4Materialization(sm, garch);

        auto& gCudaCtx =
            static_cast<core::cuda::CudaComputeContext&>(*e._computeCtx);
        compute::cuda::CudaMaterializerOps gDevOps(gCudaCtx, *e._ops);
        e._materializedBf16 =
            runtime::nvfp4::executeMaterialization(gsteps, *e._nvfp4Model,
                                                   gDevOps);
        gCudaCtx.stream().synchronize();

        // 5.13 (M-Cuda.Gemma-NVFP4-TC) — keep the dense projections in native
        // blocked-NVFP4 instead of the just-materialised BF16, so decode reads
        // 4-bit weights (¼ the bytes) on the weight-bandwidth-bound path. Same
        // repackage as the qwen shared-expert keep-path below (repackageNvfp4ToBlk
        // + isNvfp4Blk → NVFP4_BLK in buildBf16WeightsMap; the Gemma backend
        // dispatch is type-driven so it routes to matmul_nvfp4blk automatically).
        // Gemma4 has no experts/GDN → EVERY NVFP4-sourced projection is dense;
        // norms/embeds are BF16 passthroughs (src.kind != Nvfp4) and stay BF16.
        // Opt-in, default OFF (prod stays on the BF16 path until A/B'd).
        if (const char* ge = std::getenv("MIMIRMIND_GEMMA_NVFP4_DECODE");
            ge != nullptr && std::string_view{ge} != "0") {
            std::size_t nRepack = 0;
            std::uint64_t bytesBefore = 0, bytesAfter = 0;
            for (const auto& step : gsteps) {
                if (step.sources.size() != 1) continue;
                const auto& src = step.sources[0];
                if (src.kind != core::modelopt::SourceKind::Nvfp4) continue;
                if ((src.in % 32) != 0) continue;
                auto it = std::find_if(
                    e._materializedBf16.begin(), e._materializedBf16.end(),
                    [&](const runtime::nvfp4::MaterializedTensor& t) {
                        return t.ggufName == step.ggufName;
                    });
                if (it == e._materializedBf16.end() || it->isF32) continue;
                // compressed-tensors gemma4 sets the scale sidecar names
                // explicitly on the source (packed=.weight_packed,
                // block=.weight_scale, global=.weight_global_scale stored as a
                // RECIPROCAL). Use those directly rather than reconstructing the
                // qwen35moe .weight/.weight_scale_2 convention.
                const auto* pk = e._nvfp4Model->find(src.hfWeightName);
                const auto* bs = src.blockScaleName.empty()
                                     ? nullptr
                                     : e._nvfp4Model->find(src.blockScaleName);
                const auto* gs = src.globalScaleName.empty()
                                     ? nullptr
                                     : e._nvfp4Model->find(src.globalScaleName);
                if (pk == nullptr || bs == nullptr || gs == nullptr) continue;
                float global = gDevOps.readF32(gs->devPtr);
                if (src.globalIsReciprocal) {
                    global = 1.0F / global;   // dequant MULTIPLIES by the global
                }
                const std::size_t blkBytes =
                    (static_cast<std::size_t>(it->elems) / 32) * 20;
                compute::ComputeBuffer nb = gDevOps.allocateWeight(blkBytes);
                gDevOps.repackageNvfp4ToBlk(nb.get(), pk->devPtr, bs->devPtr,
                                            global, src.rows, src.in);
                gCudaCtx.stream().synchronize();
                bytesBefore += static_cast<std::uint64_t>(it->elems) * 2;
                bytesAfter  += blkBytes;
                it->buffer     = std::move(nb);   // frees the BF16 buffer (RAII)
                it->isNvfp4Blk = true;
                ++nRepack;
            }
            MM_LOG_INFO("engine",
                        "loadModelNvfp4: gemma4 kept {} dense projections native "
                        "blocked-NVFP4 (MIMIRMIND_GEMMA_NVFP4_DECODE) "
                        "({} MiB -> {} MiB)",
                        nRepack, bytesBefore >> 20, bytesAfter >> 20);
        }

        e._weights.emplace(
            runtime::nvfp4::buildBf16WeightsMap(e._materializedBf16));
        e._nvfp4Model.reset();
        MM_LOG_INFO("engine",
                    "loadModelNvfp4: gemma4 materialised {} BF16 tensors",
                    e._materializedBf16.size());
        return;
    }

    // 1. Arch params from config.json (GGUF-metadata parse is GGUF-only).
    e._config = runtime::nvfp4::parseQwen3_5MoeSafetensorsConfig(configText);

    // 2. Tokenizer. NVFP4 checkpoints ship no GGUF tokenizer, so by default
    //    parse the checkpoint's HF tokenizer.json directly (byte-level BPE,
    //    Qwen family). An explicit models[].tokenizerGguf still wins — e.g.
    //    to reuse a llama.cpp GGUF tokenizer for parity testing.
    if (tokenizerGguf.empty()) {
        MM_LOG_INFO("engine",
                    "loadModelNvfp4: '{}' (tokenizer from tokenizer.json)",
                    checkpointDir);
        e._tokenizer.loadFromHfJson(dir);
    } else {
        MM_LOG_INFO("engine",
                    "loadModelNvfp4: '{}' (tokenizer from GGUF '{}')",
                    checkpointDir, tokenizerGguf);
        e._reader.open(tokenizerGguf);
        e._tokenizer.loadFromGguf(e._reader);
    }

    // 3. Upload the NVFP4/FP8 weights (+ BF16 passthroughs) to the device.
    runtime::nvfp4::ComputeOpsUploader uploader(*e._ops);
    e._nvfp4Model = std::make_unique<runtime::nvfp4::NvFp4Model>(
        attachedSm ? runtime::nvfp4::loadNvfp4Model(*attachedSm, dir, uploader)
                   : runtime::nvfp4::loadNvfp4Model(dir, uploader));

    // 4. Build the materialization plan (tensor shapes + per-module schemes).
    core::safetensors::SafetensorsModel localSm;
    if (attachedSm == nullptr) localSm.open(dir);
    core::safetensors::SafetensorsModel& sm = attachedSm ? *attachedSm : localSm;
    const core::modelopt::Qwen3_5MoeArch arch{
        static_cast<int>(e._config.blockCount),
        static_cast<int>(e._config.expertCount),
        4 /* full_attention_interval; layer_types agrees for this model */,
        static_cast<int>(e._config.nextnPredictLayers) /* MTP head blocks */};

    // Two quant-config formats share the qwen3_5 arch walk: the ModelOpt MoE
    // checkpoint (hf_quant_config.json sidecar, per-tensor NVFP4/FP8) and the
    // compressed-tensors DENSE checkpoint (Qwen3.8-27B: quant config lives IN
    // config.json, mixed per-channel FP8 / NVFP4 / BF16, dense SwiGLU MLP, no
    // experts / shared expert / MTP). Detect the latter by a valid
    // compressed-tensors `quantization_config` + a dense config (expertCount 0),
    // and swap only the plan — the shared upload, dequant, GatedDeltaNet
    // value-head regroup, and WeightsMap tail below are format-agnostic. The
    // MoE-only post-passes (5c-5g) key on expert / shared-expert / MTP tensor
    // names, so they no-op on the dense plan.
    const core::modelopt::CompressedTensorsConfig ctCfg =
        core::modelopt::CompressedTensorsConfig::parse(configText);
    std::vector<core::modelopt::MaterializationStep> steps;
    if (ctCfg.valid() && e._config.expertCount == 0) {
        MM_LOG_INFO("engine",
                    "loadModelNvfp4: '{}' — qwen3_5 DENSE (compressed-tensors, "
                    "mixed FP8/NVFP4)", checkpointDir);
        steps = core::modelopt::planQwen3_5DenseMaterialization(sm, ctCfg, arch);
    } else {
        const core::modelopt::HfQuantConfig hfCfg =
            core::modelopt::HfQuantConfig::parse(
                readText(std::filesystem::path{dir} / "hf_quant_config.json"));
        steps = core::modelopt::planQwen3_5MoeMaterialization(sm, hfCfg, arch);
    }

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
    // attn_gate, and the value COLUMNS of ssm_out. The SAME regroup at head
    // granularity (dst(j,k) <- src(k,j)) applies to the per-value-head decay /
    // beta tensors — ssm_alpha.weight / ssm_beta.weight (VH output rows) and
    // ssm_a / ssm_dt.bias (VH scalars) — so each head's decay and beta stay
    // paired with the regrouped value head. Missing these was the Qwen3.6
    // long-generation degeneration bug (vLLM-oracle-localised): the recurrence
    // applied every value head's decay/beta to the wrong head, invisible per
    // token but compounding over the sequence.
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
            // Per-HEAD permutation (same (k,j)->(j,k) regroup, but at head
            // granularity) for the per-value-head decay/beta tensors. These
            // (ssm_alpha/ssm_beta projections + ssm_a/ssm_dt biases) select one
            // scalar per value head, so they must be regrouped to [rep,k] to
            // stay paired with the [rep,k] value stream — otherwise the
            // delta-rule recurrence applies each head's decay/beta to the wrong
            // value head, which is norm-invisible per token but compounds over
            // the sequence into long-generation degeneration.
            std::vector<std::size_t> permHead(VH);
            for (std::size_t k = 0; k < KH; ++k) {
                for (std::size_t j = 0; j < GQA; ++j) {
                    permHead[j * KH + k] = k * GQA + j;
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
            // Permute the VH output rows (one per value head) of a per-head
            // projection weight ([VH, inCols] row-major), or the VH elements of
            // a 1-D per-head bias, by permHead.
            auto regroupHeadRows = [&](runtime::nvfp4::MaterializedTensor& t) {
                const std::size_t inCols    = t.ggufDims[0];
                const std::size_t elemBytes = t.isF32 ? 4 : 2;
                const std::size_t rowBytes  = inCols * elemBytes;
                const std::size_t nbytes    = VH * rowBytes;
                auto* base = static_cast<std::uint8_t*>(t.buffer.get());
                std::vector<std::uint8_t> in(nbytes), out(nbytes);
                e._ops->readbackToHost(in.data(), base, nbytes);
                for (std::size_t r = 0; r < VH; ++r) {
                    std::memcpy(out.data() + r * rowBytes,
                                in.data() + permHead[r] * rowBytes, rowBytes);
                }
                e._ops->uploadHostBytes(base, out.data(), nbytes);
            };
            auto regroupHeadVec = [&](runtime::nvfp4::MaterializedTensor& t) {
                const std::size_t elemBytes = t.isF32 ? 4 : 2;
                const std::size_t nbytes    = VH * elemBytes;
                auto* base = static_cast<std::uint8_t*>(t.buffer.get());
                std::vector<std::uint8_t> in(nbytes), out(nbytes);
                e._ops->readbackToHost(in.data(), base, nbytes);
                for (std::size_t r = 0; r < VH; ++r) {
                    std::memcpy(out.data() + r * elemBytes,
                                in.data() + permHead[r] * elemBytes, elemBytes);
                }
                e._ops->uploadHostBytes(base, out.data(), nbytes);
            };
            auto numElems = [](const runtime::nvfp4::MaterializedTensor& t) {
                std::size_t n = 1;
                for (std::size_t dd : t.ggufDims) n *= dd;
                return n;
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
                } else if (n.ends_with(".ssm_alpha.weight") ||
                           n.ends_with(".ssm_beta.weight")) {
                    // Per-head decay/beta projections: VH output rows.
                    regroupHeadRows(t);
                    ++regrouped;
                } else if (n.ends_with(".ssm_a") || n.ends_with(".ssm_dt.bias")) {
                    // Per-head scalar biases ([VH]); regroup the VH elements.
                    if (numElems(t) == VH) {
                        regroupHeadVec(t);
                        ++regrouped;
                    } else {
                        MM_LOG_WARN("engine",
                                    "GDN regroup: {} has {} elems (expected VH={}) "
                                    "— skipped",
                                    n, numElems(t), VH);
                    }
                }
            }
            cudaCtx.stream().synchronize();
            MM_LOG_INFO("engine",
                        "loadModelNvfp4: GatedDeltaNet value-head regroup "
                        "applied to {} tensors (KH={} GQA={} HD={})",
                        regrouped, KH, GQA, HD);
        }
    }

    // 5b'. MTP eh_proj concat-half swap. The HF checkpoint stores the fused
    // pre-fc projection as fc(cat(hnorm, enorm)) — hidden-norm half first. The
    // backend's runMtpBlock feeds cat(enorm, hnorm) (embed-norm first, matching
    // the llama.cpp GGUF convert which swaps the halves). So swap the two
    // input-halves of blk.<blockCount>.nextn.eh_proj.weight. Layout is
    // [out(ne1) rows][in(ne0) cols] with in = 2*d_model contiguous per row.
    // eh_proj concat-half swap is OPT-IN: MTP accept-rate testing showed this
    // Qwen3.6-VL checkpoint's mtp.fc is already cat(enorm, hnorm) (no swap:
    // accept 0.30 vs 0.04 with swap). The llama.cpp swap-on-convert note
    // applies to Qwen3-Next, not this VL head. Kept behind MIMIRMIND_MTP_EHSWAP
    // for other checkpoints.
    if (std::getenv("MIMIRMIND_MTP_EHSWAP") != nullptr) {
        for (auto& t : e._materializedBf16) {
            if (!t.ggufName.ends_with(".nextn.eh_proj.weight")) {
                continue;
            }
            if (t.ggufDims.size() < 2 || (t.ggufDims[0] % 2) != 0) {
                MM_LOG_WARN("engine", "MTP eh_proj: unexpected dims for {}",
                            t.ggufName);
                break;
            }
            const std::size_t inCols    = t.ggufDims[0];       // 2 * d_model
            const std::size_t rows      = t.ggufDims[1];       // d_model (out)
            const std::size_t half      = inCols / 2;
            const std::size_t elemBytes = t.isF32 ? 4 : 2;
            const std::size_t rowBytes  = inCols * elemBytes;
            const std::size_t nbytes    = rows * rowBytes;
            auto* base = static_cast<std::uint8_t*>(t.buffer.get());
            std::vector<std::uint8_t> in(nbytes), out(nbytes);
            e._ops->readbackToHost(in.data(), base, nbytes);
            for (std::size_t r = 0; r < rows; ++r) {
                const std::uint8_t* ri = in.data() + r * rowBytes;
                std::uint8_t*       ro = out.data() + r * rowBytes;
                std::memcpy(ro,                       ri + half * elemBytes,
                            half * elemBytes);          // enorm half -> front
                std::memcpy(ro + half * elemBytes,    ri,
                            half * elemBytes);          // hnorm half -> back
            }
            e._ops->uploadHostBytes(base, out.data(), nbytes);
            cudaCtx.stream().synchronize();
            MM_LOG_INFO("engine",
                        "loadModelNvfp4: MTP eh_proj concat-half swap applied "
                        "({} rows x {} in)", rows, inCols);
            break;
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
                compute::ComputeBuffer q8 = devOps.allocateWeight(q8Bytes);
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

    // 5d. MoE routed-expert banks. The experts dominate the decode-time weight
    // traffic and the qwen35moe decode is memory-bound. MIMIRMIND_NVFP4_MOE:
    //   "nvfp4" (default) — keep the checkpoint's native NVFP4 by repackaging
    //       each expert into the blocked format (32-elem supers, exact E2M1).
    //       LOSSLESS (fused-K NVFP4 kernels), ~0.625 B/elem.
    //   "kquant"          — gate/up -> Q4_K, down -> Q6_K (a lossy but coherent
    //       re-quant, make_qkx2/make_qx_quants). Slightly smaller (~0.56 B).
    //   "0"               — keep BF16.
    {
        const char* mq = std::getenv("MIMIRMIND_NVFP4_MOE");
        const std::string_view moeMode = (mq == nullptr) ? "nvfp4" : mq;
        auto isExpert = [](std::string_view n) {
            return n.ends_with(".ffn_gate_exps.weight")
                || n.ends_with(".ffn_up_exps.weight")
                || n.ends_with(".ffn_down_exps.weight");
        };
        if (moeMode == "nvfp4" && e._nvfp4Model) {
            // Routed experts, three modes (MIMIRMIND_GROUPED_MOE):
            //   default = "additive": blocked-NVFP4 bank (decode) + FP4-TC side
            //             banks (prefill grouped GEMM). Both wins, ~2x routed
            //             memory. The default when CUTLASS is linked.
            //   "3" = "tc-only": one plain-nibble bank (NVFP4_TC) used by BOTH
            //             prefill AND decode; no blocked bank (saves memory, but
            //             the TC decode kernel is ~18% slower than blocked).
            //   "0" = "blocked": legacy blocked-NVFP4 only (no prefill win).
            const bool tcAvail = e._ops->moeGroupedGemmNvfp4TcAvailable();
            const char* gEnv = std::getenv("MIMIRMIND_GROUPED_MOE");
            const std::string_view gv = gEnv ? std::string_view(gEnv) : std::string_view();
            const bool forceBlocked = (gv == "0") || !tcAvail;
            const bool tcOnly   = !forceBlocked && (gv == "3");
            const bool additive = !forceBlocked && !tcOnly;   // default
            std::size_t nBanks = 0, tcBanks = 0;
            std::uint64_t bytesBefore = 0, bytesAfter = 0;

            auto findSrc = [&](const core::modelopt::MaterializationSource& src,
                               const runtime::nvfp4::NvFp4DeviceTensor*& pk,
                               const runtime::nvfp4::NvFp4DeviceTensor*& bs,
                               const runtime::nvfp4::NvFp4DeviceTensor*& gs) {
                pk = e._nvfp4Model->find(src.hfWeightName);
                const std::string base{src.hfWeightName};
                const std::string baseNoW =
                    base.size() > 7 ? base.substr(0, base.size() - 7) : base;
                bs = e._nvfp4Model->find(baseNoW + ".weight_scale");
                gs = e._nvfp4Model->find(baseNoW + ".weight_scale_2");
                return pk != nullptr && bs != nullptr && gs != nullptr;
            };

            for (const auto& step : steps) {
                if (!isExpert(step.ggufName)) continue;
                auto it = std::find_if(
                    e._materializedBf16.begin(), e._materializedBf16.end(),
                    [&](const runtime::nvfp4::MaterializedTensor& t) {
                        return t.ggufName == step.ggufName;
                    });
                if (it == e._materializedBf16.end() || it->isF32) continue;
                if ((it->elems % 32) != 0) continue;
                const int nExp = static_cast<int>(step.sources.size());
                const std::uint64_t tcN = step.sources.empty() ? 0 : step.sources.front().rows;
                const std::uint64_t tcK = step.sources.empty() ? 0 : step.sources.front().in;

                if (tcOnly && nExp > 0 && tcK % 32 == 0) {
                    // --- FP4-TC-only banks (nibbles = the tensor buffer) -------
                    const std::size_t nibBytes = static_cast<std::size_t>(it->elems) / 2;
                    const std::size_t sfbBytes =
                        core::modelopt::moeSwizzledScaleBankBytes(
                            static_cast<std::uint64_t>(nExp), tcN, tcK / 16);
                    compute::ComputeBuffer nibBank  = devOps.allocateWeight(nibBytes);
                    compute::ComputeBuffer sfbBank  = devOps.allocateWeight(sfbBytes);
                    compute::ComputeBuffer globBank = devOps.allocateWeight(
                        static_cast<std::size_t>(nExp) * sizeof(float));
                    auto* nibB = static_cast<std::uint8_t*>(nibBank.get());
                    auto* sfbB = static_cast<std::uint8_t*>(sfbBank.get());
                    std::vector<float> globHost(static_cast<std::size_t>(nExp), 0.0f);
                    bool ok = true;
                    for (const auto& src : step.sources) {
                        const runtime::nvfp4::NvFp4DeviceTensor *pk, *bs, *gs;
                        if (src.kind != core::modelopt::SourceKind::Nvfp4
                            || (src.in % 32) != 0 || !findSrc(src, pk, bs, gs)) {
                            ok = false; break;
                        }
                        const int eIdx = static_cast<int>(
                            src.dstElemOffset
                            / (static_cast<std::uint64_t>(src.rows) * src.in));
                        devOps.copyBytes(
                            nibB + static_cast<std::size_t>(eIdx) * src.rows * (src.in / 2),
                            pk->devPtr,
                            static_cast<std::size_t>(src.rows) * (src.in / 2));
                        devOps.swizzleWeightSf(
                            sfbB + static_cast<std::size_t>(eIdx)
                                * core::modelopt::moeSwizzledScaleStride(src.rows, src.in / 16),
                            bs->devPtr, src.rows, src.in);
                        if (eIdx >= 0 && eIdx < nExp) {
                            globHost[static_cast<std::size_t>(eIdx)] = devOps.readF32(gs->devPtr);
                        }
                    }
                    if (!ok) continue;
                    e._ops->uploadHostBytes(globBank.get(), globHost.data(),
                                            globHost.size() * sizeof(float));
                    cudaCtx.stream().synchronize();
                    bytesBefore += static_cast<std::uint64_t>(it->elems) * 2;
                    bytesAfter  += nibBytes + sfbBytes;
                    it->buffer        = std::move(nibBank);  // frees the BF16 bank
                    it->tcSfbBank     = std::move(sfbBank);
                    it->tcGlobalsBank = std::move(globBank);
                    it->isNvfp4Tc     = true;
                    ++tcBanks;
                } else {
                    // --- blocked-NVFP4 bank (decode) [+ additive TC sidecars
                    //     for the prefill grouped GEMM] --------------------------
                    const std::size_t blkBytes =
                        (static_cast<std::size_t>(it->elems) / 32) * 20;
                    compute::ComputeBuffer bank = devOps.allocateWeight(blkBytes);
                    auto* bankBytes = static_cast<std::uint8_t*>(bank.get());
                    const bool tcAdd = additive && nExp > 0 && tcK % 32 == 0;
                    compute::ComputeBuffer tcNib, tcSfb, tcGlob;
                    std::uint8_t *tcNibB = nullptr, *tcSfbB = nullptr;
                    std::size_t sfbBytes = 0;
                    std::vector<float> tcGlobHost;
                    if (tcAdd) {
                        sfbBytes = core::modelopt::moeSwizzledScaleBankBytes(
                            static_cast<std::uint64_t>(nExp), tcN, tcK / 16);
                        tcNib  = devOps.allocateWeight(static_cast<std::size_t>(it->elems) / 2);
                        tcSfb  = devOps.allocateWeight(sfbBytes);
                        tcGlob = devOps.allocateWeight(static_cast<std::size_t>(nExp) * sizeof(float));
                        tcNibB = static_cast<std::uint8_t*>(tcNib.get());
                        tcSfbB = static_cast<std::uint8_t*>(tcSfb.get());
                        tcGlobHost.assign(static_cast<std::size_t>(nExp), 0.0f);
                    }
                    bool ok = true;
                    for (const auto& src : step.sources) {
                        const runtime::nvfp4::NvFp4DeviceTensor *pk, *bs, *gs;
                        if (src.kind != core::modelopt::SourceKind::Nvfp4
                            || (src.in % 32) != 0 || !findSrc(src, pk, bs, gs)) {
                            ok = false; break;
                        }
                        const float global = devOps.readF32(gs->devPtr);
                        const std::size_t byteOff =
                            (static_cast<std::size_t>(src.dstElemOffset) / 32) * 20;
                        devOps.repackageNvfp4ToBlk(bankBytes + byteOff, pk->devPtr,
                                                   bs->devPtr, global, src.rows, src.in);
                        if (tcAdd) {
                            const int eIdx = static_cast<int>(
                                src.dstElemOffset
                                / (static_cast<std::uint64_t>(src.rows) * src.in));
                            devOps.copyBytes(
                                tcNibB + static_cast<std::size_t>(eIdx) * src.rows * (src.in / 2),
                                pk->devPtr,
                                static_cast<std::size_t>(src.rows) * (src.in / 2));
                            devOps.swizzleWeightSf(
                                tcSfbB + static_cast<std::size_t>(eIdx)
                                    * core::modelopt::moeSwizzledScaleStride(src.rows, src.in / 16),
                                bs->devPtr, src.rows, src.in);
                            if (eIdx >= 0 && eIdx < nExp) {
                                tcGlobHost[static_cast<std::size_t>(eIdx)] = global;
                            }
                        }
                    }
                    if (!ok) continue;
                    if (tcAdd) {
                        e._ops->uploadHostBytes(tcGlob.get(), tcGlobHost.data(),
                                                tcGlobHost.size() * sizeof(float));
                    }
                    cudaCtx.stream().synchronize();
                    bytesBefore += static_cast<std::uint64_t>(it->elems) * 2;
                    bytesAfter  += blkBytes;
                    it->buffer     = std::move(bank);
                    it->isNvfp4Blk = true;
                    ++nBanks;
                    if (tcAdd) {
                        bytesAfter += static_cast<std::uint64_t>(it->elems) / 2 + sfbBytes;
                        it->tcNibbleBank  = std::move(tcNib);
                        it->tcSfbBank     = std::move(tcSfb);
                        it->tcGlobalsBank = std::move(tcGlob);
                        ++tcBanks;
                    }
                }
            }
            MM_LOG_INFO("engine",
                        "loadModelNvfp4: routed experts -> {} blocked-NVFP4 + {} "
                        "FP4-TC banks (BF16 {} MiB -> {} MiB)",
                        nBanks, tcBanks, bytesBefore >> 20, bytesAfter >> 20);
        } else if (moeMode == "kquant") {
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
                compute::ComputeBuffer q = devOps.allocateWeight(qBytes);
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
        // Default flipped to "0" (keep BF16) 2026-08-03: the FP8 requant halves
        // the attention weight bytes but matmul_fp8_gemm is slower than the
        // BF16 tf32-TC path at decode-M, and it dominates the decode step (it is
        // on all 30 GDN layers × 3 proj). Back-to-back HTTP A/B on GB10
        // (qwen3.6-35B-NVFP4): BF16 84.6 vs FP8 70.5 gen-tok/s @conc32 (+20%),
        // for +900 MiB (trivial on 128 GB). Precision-neutral (these projections
        // are natively FP8 in the checkpoint, so BF16 holds them exactly).
        // `MIMIRMIND_NVFP4_ATTN_FP8=gdn` restores the memory-saving FP8 path.
        const char* fp8env = std::getenv("MIMIRMIND_NVFP4_ATTN_FP8");
        const std::string_view mode = (fp8env == nullptr) ? "0" : fp8env;
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
                compute::ComputeBuffer fp8 = devOps.allocateWeight(fp8Bytes);
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

    // 5e-dense. Qwen3_5 DENSE decode bandwidth lever — keep the dense
    // compressed-tensors NVFP4 projections native 4-bit (blocked-NVFP4) instead
    // of the just-materialised BF16. The dense 27B is decode weight-bandwidth
    // bound (~53.7 GiB BF16 / token @ 273 GB/s ≈ 197 ms floor; measured ~346 ms),
    // and the MLP (ffn_gate/up/down, ff=17408) dominates. Every NVFP4-sourced
    // dense projection is read every token (no routing), so repackaging E2M1 +
    // per-16 E4M3 block-scale + global into the single-pointer blocked format the
    // matmul_nvfp4blk kernels consume (¼ the bytes) is LOSSLESS (the BF16 held
    // the widened NVFP4 values; re-quantising BF16 would double-quant instead).
    // compressed-tensors names the sidecars explicitly on the source
    // (.weight_packed / .weight_scale / .weight_global_scale, reciprocal global)
    // — use those directly, NOT the ModelOpt .weight/.weight_scale_2 convention.
    // Opt-in (default OFF until A/B'd on the HTTP anchor); FP8-sourced
    // projections (late-layer MLP + lm_head) stay BF16 here.
    if (ctCfg.valid() && e._config.expertCount == 0 && e._nvfp4Model) {
        const char* de = std::getenv("MIMIRMIND_QWEN_DENSE_NVFP4_DECODE");
        if (de != nullptr && std::string_view{de} != "0") {
            std::size_t nRepack = 0;
            std::uint64_t bytesBefore = 0, bytesAfter = 0;
            for (const auto& step : steps) {
                if (step.sources.size() != 1) continue;
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
                const auto* bs = src.blockScaleName.empty()
                                     ? nullptr
                                     : e._nvfp4Model->find(src.blockScaleName);
                const auto* gs = src.globalScaleName.empty()
                                     ? nullptr
                                     : e._nvfp4Model->find(src.globalScaleName);
                if (pk == nullptr || bs == nullptr || gs == nullptr) continue;
                float global = devOps.readF32(gs->devPtr);
                if (src.globalIsReciprocal) global = 1.0F / global;
                const std::size_t blkBytes =
                    (static_cast<std::size_t>(it->elems) / 32) * 20;
                compute::ComputeBuffer nb = devOps.allocateWeight(blkBytes);
                devOps.repackageNvfp4ToBlk(nb.get(), pk->devPtr, bs->devPtr,
                                           global, src.rows, src.in);
                cudaCtx.stream().synchronize();
                bytesBefore += static_cast<std::uint64_t>(it->elems) * 2;
                bytesAfter  += blkBytes;
                it->buffer     = std::move(nb);   // frees the BF16 buffer (RAII)
                it->isNvfp4Blk = true;
                ++nRepack;
            }
            MM_LOG_INFO("engine",
                        "loadModelNvfp4: qwen3_5 dense kept {} NVFP4 projections "
                        "native blocked-NVFP4 (MIMIRMIND_QWEN_DENSE_NVFP4_DECODE) "
                        "({} MiB -> {} MiB)",
                        nRepack, bytesBefore >> 20, bytesAfter >> 20);
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
        // DEFAULT ON: the shared experts stay native blocked-NVFP4 alongside the
        // routed experts. An earlier "routed-NVFP4 + shared-NVFP4 degenerates"
        // report was a MISDIAGNOSIS (it compared different prompts): a 40-block
        // probe shows shared-NVFP4 matches shared-BF16 to <=0.18% with no NaN,
        // and an A/B with repetition_penalty has the both-NVFP4 output as the
        // MOST coherent of the set. The residual short-prompt greedy repetition
        // collapse is identical with shared BF16, so it is a decode/chat-template
        // artefact, not a quant bug. MIMIRMIND_NVFP4_SHEXP=0 keeps BF16 for A/B.
        const char* faenv = std::getenv("MIMIRMIND_NVFP4_SHEXP");
        const bool faNvblk = (faenv == nullptr) || (std::string_view{faenv} != "0");
        if (faNvblk) {
            auto isFullAttn = [](std::string_view n) {
                return n.ends_with(".ffn_gate_shexp.weight")
                    || n.ends_with(".ffn_up_shexp.weight")
                    || n.ends_with(".ffn_down_shexp.weight");
            };
            // Track B (stage R prefill fix): in addition to the blocked-NVFP4
            // bank (kept for decode), build FP4-tensor-core sidecars (plain
            // E2M1 nibbles + swizzled UE4M3 SFB + F32 global) for each shared
            // expert so the prefill path can run it through the CUTLASS grouped
            // GEMM as a single group (nExp=1). The dense blocked kernel's
            // kGemmMaxM=16 loop re-streams the whole weight ~M/16 times at large
            // M — the measured 48 s@7.6k prefill bottleneck. A shared expert is
            // a single matrix, so its sidecar is the nExp=1, eIdx=0 degenerate
            // case of the routed tcAdd path above. Additive in every TC mode;
            // skipped only when CUTLASS is absent or GROUPED_MOE=0 forces
            // blocked-only. The generic NvFp4WeightsMap bridge exposes these as
            // tcNibblePtr/tcSfbPtr/tcGlobalsPtr once tcSfbBank is set.
            const bool tcAvail = e._ops->moeGroupedGemmNvfp4TcAvailable();
            const char* gEnv = std::getenv("MIMIRMIND_GROUPED_MOE");
            const std::string_view gv =
                gEnv ? std::string_view(gEnv) : std::string_view();
            const bool shexpTcAdd = tcAvail && (gv != "0");
            std::size_t nRepack = 0, tcSidecars = 0;
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
                compute::ComputeBuffer nb = devOps.allocateWeight(blkBytes);
                devOps.repackageNvfp4ToBlk(nb.get(), pk->devPtr, bs->devPtr, global,
                                           src.rows, src.in);
                cudaCtx.stream().synchronize();
                bytesBefore += static_cast<std::uint64_t>(it->elems) * 2;
                bytesAfter  += blkBytes;
                it->buffer     = std::move(nb); // frees the BF16 buffer (RAII)
                it->isNvfp4Blk = true;
                ++nRepack;

                // Additive FP4-TC sidecar (single matrix -> nExp=1, eIdx=0).
                if (shexpTcAdd) {
                    const std::size_t nibBytes =
                        static_cast<std::size_t>(it->elems) / 2;
                    const std::size_t sfbBytes =
                        core::modelopt::moeSwizzledScaleBankBytes(
                            1, src.rows, src.in / 16);
                    compute::ComputeBuffer tcNib  = devOps.allocateWeight(nibBytes);
                    compute::ComputeBuffer tcSfb  = devOps.allocateWeight(sfbBytes);
                    compute::ComputeBuffer tcGlob = devOps.allocateWeight(sizeof(float));
                    devOps.copyBytes(tcNib.get(), pk->devPtr, nibBytes);
                    devOps.swizzleWeightSf(tcSfb.get(), bs->devPtr,
                                           src.rows, src.in);
                    e._ops->uploadHostBytes(tcGlob.get(), &global, sizeof(float));
                    cudaCtx.stream().synchronize();
                    bytesAfter        += nibBytes + sfbBytes;
                    it->tcNibbleBank   = std::move(tcNib);
                    it->tcSfbBank      = std::move(tcSfb);
                    it->tcGlobalsBank  = std::move(tcGlob);
                    ++tcSidecars;
                }
            }
            MM_LOG_INFO("engine",
                        "loadModelNvfp4: kept {} dense NVFP4 (shared-expert) "
                        "projections native blocked-NVFP4 (+{} FP4-TC sidecars) "
                        "({} MiB -> {} MiB)",
                        nRepack, tcSidecars, bytesBefore >> 20, bytesAfter >> 20);
        }
    }

    // 5g. M-dependent dense FP8 (dual-copy). KEEP the BF16 dense projections
    // AND add a blocked-FP8 E4M3 ".fp8" variant of each, exposed as a separate
    // WeightsMap tensor. The backend then reads FP8 at low batch (decode is
    // memory-bound there: FP8 halves the always-read dense traffic, +~15%
    // single-request, bit-coherent since these projections are natively FP8 in
    // the checkpoint) and the BF16 copy at high batch (compute-bound, where the
    // TF32 tensor-core path wins). Opt-in via MIMIRMIND_DENSE_FP8_LOWM=<maxSeq>
    // (the FP8 copy costs ~1.3 GiB — trivial on 128 GB unified). Independent of
    // the static 5e MIMIRMIND_NVFP4_ATTN_FP8 replace-in-place path.
    {
        const char* lowmEnv = std::getenv("MIMIRMIND_DENSE_FP8_LOWM");
        if (lowmEnv != nullptr && std::atoi(lowmEnv) > 0) {
            auto isDense = [](std::string_view n) {
                return n.ends_with(".attn_qkv.weight")
                    || n.ends_with(".attn_gate.weight")
                    || n.ends_with(".ssm_out.weight")
                    || n.ends_with(".attn_q.weight")
                    || n.ends_with(".attn_k.weight")
                    || n.ends_with(".attn_v.weight")
                    || n.ends_with(".attn_output.weight");
            };
            std::vector<runtime::nvfp4::MaterializedTensor> variants;
            std::size_t   nDual = 0;
            std::uint64_t addBytes = 0;
            for (auto& t : e._materializedBf16) {
                if (t.isF32 || t.isQ8_0 || t.isQ4K || t.isQ6K || t.isFp8 ||
                    t.isNvfp4Blk) {
                    continue;
                }
                if (t.ggufDims.size() < 2 || !isDense(t.ggufName)) {
                    continue;
                }
                const std::uint64_t K    = t.ggufDims[0];
                const std::uint64_t rows = t.ggufDims[1];
                if (K == 0 || rows == 0 || (K % 32) != 0 || K * rows != t.elems) {
                    continue;
                }
                const std::size_t fp8Bytes =
                    (static_cast<std::size_t>(t.elems) / 32) * 34;
                compute::ComputeBuffer fp8 = devOps.allocateWeight(fp8Bytes);
                devOps.quantizeBf16ToFp8(fp8.get(), t.buffer.get(), rows, K);
                cudaCtx.stream().synchronize();
                runtime::nvfp4::MaterializedTensor v;
                v.ggufName = t.ggufName + ".fp8";
                v.buffer   = std::move(fp8);
                v.ggufDims = t.ggufDims;
                v.elems    = t.elems;
                v.isFp8    = true;
                variants.push_back(std::move(v));
                addBytes += fp8Bytes;
                ++nDual;
            }
            for (auto& v : variants) {
                e._materializedBf16.push_back(std::move(v));
            }
            MM_LOG_INFO("engine",
                        "loadModelNvfp4: M-dependent dense FP8 — added {} FP8 "
                        "'.fp8' variants (+{} MiB), BF16 kept (FP8 used for "
                        "nSeq<={})",
                        nDual, addBytes >> 20, std::atoi(lowmEnv));
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
