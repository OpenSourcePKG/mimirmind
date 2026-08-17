// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/modelopt/Qwen3_5MoeMaterializer.hpp"

#include "core/modelopt/ModelOptQuant.hpp"
#include "core/modelopt/Qwen3_5MoeGgufMap.hpp"
#include "core/safetensors/SafetensorsModel.hpp"

#include <stdexcept>
#include <string_view>

namespace mimirmind::core::modelopt {

namespace {

namespace st = safetensors;

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("qwen35moe materialize: " + msg);
}

bool endsWith(const std::string& s, std::string_view suf) {
    return s.size() >= suf.size()
        && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
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
    // Unquantised passthrough tensors (norms, ssm scalars, conv1d, biases,
    // router, embed) are read as F32 device pointers by the runtime; widen
    // them here. Dequantised NVFP4/FP8 weights stay BF16.
    step.outF32     = (src.kind == SourceKind::Bf16Passthrough);
    // The GGUF `ssm_a` (SSM_A_NOSCAN) is the pre-computed decay coefficient
    // A = -exp(A_log); llama.cpp bakes the -exp() into the checkpoint at
    // conversion time. The NVFP4 checkpoint stores the raw HF `A_log`, so we
    // apply it here. The DeltaNet gate then multiplies softplus by ssm_a
    // directly (GatedDeltaNet.cpp / deltanet_gate.cu) — without this the gate
    // sees +A_log, the decay blows up and the state diverges to garbage.
    if (endsWith(ggufName, ".ssm_a")) {
        step.postTransform = PostTransform::NegExp;
    } else if (endsWith(ggufName, "norm.weight") && !endsWith(ggufName, "ssm_norm.weight")) {
        // Transformer RMSNorm weights (attn/q/k/post/output norms) are stored
        // centred at 0; llama.cpp bakes the (1 + w) into the GGUF tensor, and
        // the runtime multiplies by it directly. The NVFP4 checkpoint keeps
        // the raw HF weight, so add 1 here. The GatedDeltaNet ssm_norm uses
        // the plain convention (weights ~1) and is excluded — verified by
        // per-tensor NVFP4-vs-GGUF weight parity on GB10.
        step.postTransform = PostTransform::AddOne;
    }
    step.sources.push_back(std::move(src));
    steps.push_back(std::move(step));
}

/// Append a StackExperts step (N per-expert HF sources -> one GGUF 3-D tensor
/// [in, out, n_expert] in ne-order).
void addStacked(std::vector<MaterializationStep>& steps,
                const st::SafetensorsModel& model, const HfQuantConfig& cfg,
                const std::string& ggufName, std::string_view hfTemplate,
                int layer, int numExperts, bool mtp = false) {
    MaterializationStep step;
    step.ggufName = ggufName;
    std::uint64_t perExpert = 0;
    for (int e = 0; e < numExperts; ++e) {
        const std::string hf = mtp ? qwen35moeMtpHfName(hfTemplate, e)
                                   : qwen35moeHfName(hfTemplate, layer, e);
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

/// Append a de-stacked BF16 expert step sliced out of ONE fused HF source
/// tensor (the MTP MoE format: `experts.gate_up_proj [n_exp, 2*ff, d]` and
/// `experts.down_proj [n_exp, d, ff]`, expert-major, unquantised BF16). Per
/// expert e, copy `outRows*inCols` BF16 elements from
/// `src[e*srcStride + srcOffsetInExpert]` into the stacked GGUF tensor at
/// `e*outRows*inCols`. Byte layout is compatible (HF per-expert [out,in]
/// row-major == GGUF [in,out] ne-order), so this is a pure sliced copy.
void addFusedStack(std::vector<MaterializationStep>& steps,
                   const st::SafetensorsModel& model,
                   const std::string& ggufName, const std::string& hfName,
                   std::uint64_t nExp, std::uint64_t outRows, std::uint64_t inCols,
                   std::uint64_t srcStride, std::uint64_t srcOffsetInExpert) {
    (void)require(model, hfName);  // validate presence up front
    MaterializationStep step;
    step.ggufName   = ggufName;
    const std::uint64_t perExpert = outRows * inCols;
    step.ggufDims   = {inCols, outRows, nExp};  // GGUF ne-order [in, out, n_expert]
    step.totalElems = perExpert * nExp;
    step.outF32     = false;                    // BF16 matmul weight, kept BF16
    for (std::uint64_t e = 0; e < nExp; ++e) {
        MaterializationSource src;
        src.hfWeightName  = hfName;
        src.kind          = SourceKind::Bf16Copy;
        src.rows          = outRows;
        src.in            = inCols;
        src.dstElemOffset = e * perExpert;
        src.srcElemOffset = e * srcStride + srcOffsetInExpert;
        step.sources.push_back(std::move(src));
    }
    steps.push_back(std::move(step));
}

} // namespace

std::vector<MaterializationStep>
planQwen3_5MoeMaterialization(const st::SafetensorsModel& model,
                             const HfQuantConfig&        cfg,
                             const Qwen3_5MoeArch&        arch) {
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

    // --- MTP (nextn) head, as GGUF block index numLayers ------------------
    // A full-attention + MoE transformer block (mtp.layers.0.*) plus the four
    // nextn.* projections (mtp.fc / mtp.pre_fc_norm_* / mtp.norm). The backend
    // (Qwen3_5MoeBackend::runMtpBlock) addresses it at block index blockCount.
    if (arch.mtpLayers > 0) {
        const std::string blk = "blk." + std::to_string(arch.numLayers) + ".";
        for (const auto& t : qwen35moeNextnTensors()) {
            addDirect(steps, model, cfg, blk + std::string(t.ggufSuffix),
                      "mtp." + std::string(t.hfSuffix));
        }
        for (const auto& t : qwen35moeFullAttnTensors()) {
            addDirect(steps, model, cfg, blk + std::string(t.ggufSuffix),
                      qwen35moeMtpHfName(t.hfSuffix));
        }
        for (const auto& t : qwen35moeMoeTensors()) {
            // Routed experts use the fused, expert-stacked BF16 MTP layout
            // (below), NOT the main stack's per-expert `experts.{E}.*`; skip
            // them here and emit via addFusedStack. Router + shared-expert
            // (Direct) map normally.
            if (t.xform == WeightXform::StackExperts) {
                continue;
            }
            addDirect(steps, model, cfg, blk + std::string(t.ggufSuffix),
                      qwen35moeMtpHfName(t.hfSuffix));
        }

        // Routed experts, de-stacked from the two fused BF16 tensors:
        //   gate_up_proj [n_exp, 2*ff, d]  ->  ffn_gate_exps (rows 0:ff)
        //                                       ffn_up_exps   (rows ff:2*ff)
        //   down_proj    [n_exp, d, ff]    ->  ffn_down_exps
        const std::string guName = qwen35moeMtpHfName("mlp.experts.gate_up_proj");
        const std::string dnName = qwen35moeMtpHfName("mlp.experts.down_proj");
        const st::SafetensorsTensor& gu = require(model, guName);
        const st::SafetensorsTensor& dn = require(model, dnName);
        if (gu.shape.size() != 3 || dn.shape.size() != 3) {
            fail("MTP fused expert tensors are not 3-D");
        }
        const std::uint64_t nExp  = gu.shape[0];
        const std::uint64_t twoFf = gu.shape[1];
        const std::uint64_t d     = gu.shape[2];
        const std::uint64_t ff    = twoFf / 2;
        // gate_up_proj is gate-first: ffn_gate_exps <- rows [0:ff], ffn_up_exps
        // <- rows [ff:2*ff]. Pinned by MTP accept-rate on GB10: gate-first gives
        // 0.89 vs up-first 0.30 (with the eh_proj swap off — see Nvfp4Loader).
        addFusedStack(steps, model, blk + "ffn_gate_exps.weight", guName,
                      nExp, /*out=*/ff, /*in=*/d, /*stride=*/twoFf * d, /*off=*/0);
        addFusedStack(steps, model, blk + "ffn_up_exps.weight", guName,
                      nExp, /*out=*/ff, /*in=*/d, /*stride=*/twoFf * d, /*off=*/ff * d);
        addFusedStack(steps, model, blk + "ffn_down_exps.weight", dnName,
                      nExp, /*out=*/d, /*in=*/ff, /*stride=*/d * ff, /*off=*/0);
    }

    return steps;
}

} // namespace mimirmind::core::modelopt