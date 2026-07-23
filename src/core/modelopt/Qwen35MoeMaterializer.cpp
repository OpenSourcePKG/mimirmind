// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/modelopt/Qwen35MoeMaterializer.hpp"

#include "core/modelopt/ModelOptQuant.hpp"
#include "core/modelopt/Qwen35MoeGgufMap.hpp"
#include "core/safetensors/SafetensorsModel.hpp"

#include <stdexcept>

namespace mimirmind::core::modelopt {

namespace {

namespace st = safetensors;

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("qwen35moe materialize: " + msg);
}

const st::SafetensorsTensor& require(const st::SafetensorsModel& m, const std::string& name) {
    const auto* t = m.find(name);
    if (t == nullptr) {
        fail("missing HF tensor '" + name + "'");
    }
    return *t;
}

/// GGUF ne-order dims = reverse of the HF shape with size-1 dims dropped
/// (so conv1d [8192,1,4] -> [4,8192], weight [out,in] -> [in,out], [d] -> [d]).
std::vector<std::uint64_t> ggufDimsFromHf(const std::vector<std::uint64_t>& hf) {
    std::vector<std::uint64_t> out;
    for (auto it = hf.rbegin(); it != hf.rend(); ++it) {
        if (*it != 1) {
            out.push_back(*it);
        }
    }
    if (out.empty()) {
        out.push_back(1);
    }
    return out;
}

/// Resolve a single HF `.weight` to a source descriptor (kind + rows/in).
MaterializationSource sourceFor(const st::SafetensorsModel& model,
                                const HfQuantConfig&        cfg,
                                const std::string&          hfWeightName,
                                std::uint64_t               dstElemOffset) {
    const st::SafetensorsTensor& w = require(model, hfWeightName);
    MaterializationSource s;
    s.hfWeightName  = hfWeightName;
    s.dstElemOffset = dstElemOffset;

    const auto scheme = cfg.schemeForTensor(hfWeightName);
    if (!scheme.has_value()) {
        // Unquantised -> BF16 passthrough. rows*in = element count.
        s.kind = SourceKind::Bf16Passthrough;
        s.rows = w.nelements;
        s.in   = 1;
        return s;
    }
    if (w.shape.size() != 2) {
        fail("quantised tensor '" + hfWeightName + "' is not 2-D");
    }
    if (*scheme == ModelOptQuantScheme::NVFP4_E2M1_BLK16) {
        s.kind = SourceKind::Nvfp4;
        s.rows = w.shape[0];               // out
        s.in   = w.shape[1] * 2;           // packed cols * 2 = in
    } else {
        s.kind = SourceKind::Fp8;
        s.rows = w.shape[0];               // out
        s.in   = w.shape[1];               // in (unpacked)
    }
    return s;
}

/// Append a Direct step (one HF source -> one GGUF tensor).
void addDirect(std::vector<MaterializationStep>& steps,
               const st::SafetensorsModel& model, const HfQuantConfig& cfg,
               const std::string& ggufName, const std::string& hfWeightName) {
    const st::SafetensorsTensor& w = require(model, hfWeightName);
    MaterializationStep step;
    step.ggufName            = ggufName;
    MaterializationSource src = sourceFor(model, cfg, hfWeightName, 0);
    // GGUF dims come from the LOGICAL (dequantised) shape: for a quantised
    // 2-D weight that is [in(ne0), out(ne1)] (NVFP4's packed cols are half of
    // `in`, so the raw shape is wrong); passthrough tensors keep their real
    // (reverse+squeezed) HF shape (norms, conv1d, embed, ...).
    step.ggufDims   = (src.kind == SourceKind::Bf16Passthrough)
                          ? ggufDimsFromHf(w.shape)
                          : std::vector<std::uint64_t>{src.in, src.rows};
    step.totalElems = src.rows * src.in;
    step.sources.push_back(std::move(src));
    steps.push_back(std::move(step));
}

/// Append a StackExperts step (N per-expert HF sources -> one GGUF 3-D tensor
/// [in, out, n_expert] in ne-order).
void addStacked(std::vector<MaterializationStep>& steps,
                const st::SafetensorsModel& model, const HfQuantConfig& cfg,
                const std::string& ggufName, std::string_view hfTemplate,
                int layer, int numExperts) {
    MaterializationStep step;
    step.ggufName = ggufName;
    std::uint64_t perExpert = 0;
    for (int e = 0; e < numExperts; ++e) {
        const std::string hf = qwen35moeHfName(hfTemplate, layer, e);
        MaterializationSource src = sourceFor(model, cfg, hf, 0);
        if (e == 0) {
            perExpert = src.rows * src.in;
            // Logical per-expert dims [in(ne0), out(ne1)], stacked over ne2.
            step.ggufDims = {src.in, src.rows, static_cast<std::uint64_t>(numExperts)};
        }
        src.dstElemOffset = static_cast<std::uint64_t>(e) * perExpert;
        step.sources.push_back(std::move(src));
    }
    step.totalElems = perExpert * static_cast<std::uint64_t>(numExperts);
    steps.push_back(std::move(step));
}

} // namespace

std::vector<MaterializationStep>
planQwen35MoeMaterialization(const st::SafetensorsModel& model,
                             const HfQuantConfig&        cfg,
                             const Qwen35MoeArch&        arch) {
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
        for (const auto& t : qwen35moeMoeTensors()) {
            if (t.xform == WeightXform::StackExperts) {
                addStacked(steps, model, cfg, blk + std::string(t.ggufSuffix),
                           t.hfSuffix, L, arch.numExperts);
            } else {
                addDirect(steps, model, cfg, blk + std::string(t.ggufSuffix),
                          qwen35moeHfName(t.hfSuffix, L));
            }
        }
    }

    return steps;
}

} // namespace mimirmind::core::modelopt