// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/modelopt/Qwen3_5DenseMaterializer.hpp"

#include "core/modelopt/Qwen3_5MoeGgufMap.hpp"
#include "core/safetensors/SafetensorsModel.hpp"

#include <stdexcept>
#include <string>
#include <string_view>

namespace mimirmind::core::modelopt {

namespace {

namespace st = safetensors;

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("qwen35dense materialize: " + msg);
}

bool endsWith(const std::string& s, std::string_view suf) {
    return s.size() >= suf.size()
        && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

const st::SafetensorsTensor& require(const st::SafetensorsModel& m,
                                     const std::string& name) {
    const auto* t = m.find(name);
    if (t == nullptr) {
        fail("missing HF tensor '" + name + "'");
    }
    return *t;
}

/// GGUF ne-order dims = reverse of the HF shape with size-1 dims dropped.
std::vector<std::uint64_t> ggufDimsFromHf(const std::vector<std::uint64_t>& hf) {
    std::vector<std::uint64_t> out;
    for (auto it = hf.rbegin(); it != hf.rend(); ++it) {
        if (*it != 1) out.push_back(*it);
    }
    if (out.empty()) out.push_back(1);
    return out;
}

/// Strip a trailing ".weight" to get the compressed-tensors module base.
std::string moduleBase(const std::string& weightName) {
    return endsWith(weightName, ".weight")
               ? weightName.substr(0, weightName.size() - 7)
               : weightName;
}

/// Append a Direct step (one HF `.weight` -> one GGUF tensor), resolving the
/// compressed-tensors scheme (NVFP4 / per-channel FP8 / BF16 passthrough).
void addDirect(std::vector<MaterializationStep>& steps,
               const st::SafetensorsModel& model, const CompressedTensorsConfig& cfg,
               const std::string& ggufName, const std::string& hfWeightName) {
    const std::string base = moduleBase(hfWeightName);
    const auto scheme = cfg.schemeForTensor(hfWeightName);

    MaterializationStep step;
    step.ggufName = ggufName;
    MaterializationSource src;

    if (scheme == CompressedTensorsConfig::Scheme::Nvfp4) {
        // `<base>.weight_packed` (U8, packed cols) + `<base>.weight_scale`
        // (F8_E4M3 per-16) + `<base>.weight_global_scale` (F32, reciprocal).
        const st::SafetensorsTensor& w = require(model, base + ".weight_packed");
        if (w.shape.size() != 2) {
            fail("NVFP4 tensor '" + base + ".weight_packed' is not 2-D");
        }
        src.hfWeightName       = base + ".weight_packed";
        src.kind               = SourceKind::Nvfp4;
        src.rows               = w.shape[0];        // out
        src.in                 = w.shape[1] * 2;    // packed cols * 2 = in
        src.blockScaleName     = base + ".weight_scale";
        src.globalScaleName    = base + ".weight_global_scale";
        src.globalIsReciprocal = true;
        step.ggufDims   = {src.in, src.rows};
        step.totalElems = src.rows * src.in;
        step.outF32     = false;
    } else if (scheme == CompressedTensorsConfig::Scheme::Fp8) {
        // Per-channel FP8: `<base>.weight` (E4M3, [out,in]) + `<base>.weight_scale`
        // ([out,1] BF16, one scale per output row). Dequantised to BF16.
        const st::SafetensorsTensor& w = require(model, hfWeightName);
        if (w.shape.size() != 2) {
            fail("FP8 tensor '" + hfWeightName + "' is not 2-D");
        }
        src.hfWeightName = hfWeightName;
        src.kind         = SourceKind::Fp8PerChannel;
        src.rows         = w.shape[0];              // out
        src.in           = w.shape[1];              // in (unpacked)
        step.ggufDims   = {src.in, src.rows};
        step.totalElems = src.rows * src.in;
        step.outF32     = false;
    } else {
        // Unquantised (config `ignore`, or unmatched): widen BF16 -> F32 so the
        // runtime reads it as `const float*` (norms, conv1d, ssm scalars, the
        // small GatedDeltaNet in_proj_a/b, embed). rows*in = element count.
        const st::SafetensorsTensor& w = require(model, hfWeightName);
        src.hfWeightName = hfWeightName;
        src.kind         = SourceKind::Bf16Passthrough;
        src.rows         = w.nelements;
        src.in           = 1;
        step.ggufDims   = ggufDimsFromHf(w.shape);
        step.totalElems = w.nelements;
        step.outF32     = true;
    }

    // Element-wise fix-ups (only valid on F32 passthrough outputs, which is
    // where they land): ssm_a = -exp(A_log); transformer RMSNorm (1 + w). The
    // GatedDeltaNet ssm_norm keeps the plain convention (excluded). Same as the
    // MoE plan (planQwen3_5MoeMaterialization).
    if (endsWith(ggufName, ".ssm_a")) {
        step.postTransform = PostTransform::NegExp;
    } else if (endsWith(ggufName, "norm.weight") && !endsWith(ggufName, "ssm_norm.weight")) {
        step.postTransform = PostTransform::AddOne;
    }

    step.sources.push_back(std::move(src));
    steps.push_back(std::move(step));
}

} // namespace

std::vector<MaterializationStep>
planQwen3_5DenseMaterialization(const st::SafetensorsModel& model,
                                const CompressedTensorsConfig& cfg,
                                const Qwen3_5MoeArch&          arch) {
    std::vector<MaterializationStep> steps;

    // --- model-level ------------------------------------------------------
    for (const auto& t : qwen35moeTopLevelTensors()) {
        addDirect(steps, model, cfg, std::string(t.ggufSuffix), std::string(t.hfSuffix));
    }

    // --- per layer --------------------------------------------------------
    for (int L = 0; L < arch.numLayers; ++L) {
        const std::string blk = "blk." + std::to_string(L) + ".";
        const auto attnTable = qwen35moeIsFullAttnLayer(L, arch.fullAttnInterval)
                                   ? qwen35moeFullAttnTensors()
                                   : qwen35moeDeltaNetTensors();
        for (const auto& t : attnTable) {
            addDirect(steps, model, cfg, blk + std::string(t.ggufSuffix),
                      qwen35moeHfName(t.hfSuffix, L));
        }
        // Dense SwiGLU MLP (no router / experts / shared expert).
        for (const auto& t : qwen35DenseMlpTensors()) {
            addDirect(steps, model, cfg, blk + std::string(t.ggufSuffix),
                      qwen35moeHfName(t.hfSuffix, L));
        }
    }

    return steps;
}

} // namespace mimirmind::core::modelopt
