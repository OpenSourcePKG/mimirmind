// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/modelopt/Kolibri1Materializer.hpp"

#include "core/safetensors/SafetensorsModel.hpp"

#include <stdexcept>
#include <string>
#include <string_view>

namespace mimirmind::core::modelopt {

namespace {

namespace st = safetensors;

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("kolibri1 materialize: " + msg);
}

bool endsWith(const std::string& s, std::string_view suf) {
    return s.size() >= suf.size()
        && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

const st::SafetensorsTensor& require(const st::SafetensorsModel& m,
                                     const std::string&          name) {
    const auto* t = m.find(name);
    if (t == nullptr) {
        fail("missing HF tensor '" + name + "'");
    }
    return *t;
}

/// GGUF ne-order dims = reverse of the HF shape with size-1 dims dropped
/// (so a 2-D weight [out,in] -> [in,out], a 1-D vector [d] -> [d]).
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

/// A quantised Kolibri weight is a 2-D `.weight` with a sibling F32 128x128
/// block-scale grid `<base>.weight_scale_inv`. The checkpoint truth (the
/// sidecar's presence) is authoritative — the per-layer `mlp.gate` and
/// `lm_head` carry no sidecar and stay BF16, exactly as config.json's
/// `modules_to_not_convert` lists them.
bool hasBlockScaleInv(const st::SafetensorsModel& m, const std::string& weightName) {
    if (!endsWith(weightName, ".weight")) {
        return false;
    }
    const std::string base = weightName.substr(0, weightName.size() - 7);
    return m.find(base + ".weight_scale_inv") != nullptr;
}

/// Resolve one HF `.weight` to a source descriptor (kind + rows/in). Fp8Block128
/// when a `.weight_scale_inv` sibling is present (rows=shape[0], in=shape[1],
/// unpacked — F8 is one byte/elem), else an unquantised BF16 passthrough.
MaterializationSource sourceFor(const st::SafetensorsModel& model,
                                const std::string&          hfWeightName,
                                std::uint64_t               dstElemOffset) {
    const st::SafetensorsTensor& w = require(model, hfWeightName);
    MaterializationSource s;
    s.hfWeightName  = hfWeightName;
    s.dstElemOffset = dstElemOffset;
    if (hasBlockScaleInv(model, hfWeightName)) {
        if (w.shape.size() != 2) {
            fail("block-128 FP8 tensor '" + hfWeightName + "' is not 2-D");
        }
        s.kind = SourceKind::Fp8Block128;
        s.rows = w.shape[0];   // out features
        s.in   = w.shape[1];   // in features (unpacked)
    } else {
        s.kind = SourceKind::Bf16Passthrough;
        s.rows = w.nelements;  // widened to F32, element count
        s.in   = 1;
    }
    return s;
}

/// Append a Direct step (one HF source -> one GGUF tensor).
void addDirect(std::vector<MaterializationStep>& steps,
               const st::SafetensorsModel& model, const std::string& ggufName,
               const std::string& hfWeightName) {
    const st::SafetensorsTensor& w = require(model, hfWeightName);
    MaterializationStep step;
    step.ggufName = ggufName;
    MaterializationSource src = sourceFor(model, hfWeightName, 0);

    // GGUF dims come from the LOGICAL shape: a quantised 2-D weight is
    // [in(ne0), out(ne1)]; a passthrough keeps its (reverse+squeezed) HF shape.
    step.ggufDims   = (src.kind == SourceKind::Bf16Passthrough)
                          ? ggufDimsFromHf(w.shape)
                          : std::vector<std::uint64_t>{src.in, src.rows};
    step.totalElems = src.rows * src.in;
    // Passthrough tensors (norms, router, bias, embed, lm_head) widen to F32 so
    // the runtime reads them as `const float*`; dequantised FP8 stays BF16.
    step.outF32     = (src.kind == SourceKind::Bf16Passthrough);
    // NO PostTransform::AddOne for Kolibri norms: the reference (aleph-alpha-
    // inference vLLM plugin) uses vLLM's PLAIN RMSNorm (weight * x_normed), NOT
    // the Gemma/Qwen (1 + weight) GGUF convention. The weights are stored as-is
    // so mimirmind's `weight * x_normed` norm kernel matches the oracle. See
    // decisions/2026-10-12-kolibri1-i4-i5-forward-spec.
    step.sources.push_back(std::move(src));
    steps.push_back(std::move(step));
}

/// Append a StackExperts step (N per-expert HF sources -> one GGUF 3-D tensor
/// [in, out, n_expert] in ne-order). All experts share one dtype/shape.
void addStacked(std::vector<MaterializationStep>& steps,
                const st::SafetensorsModel& model, const std::string& ggufName,
                const std::string& hfPrefix, const std::string& projSuffix,
                int numExperts) {
    MaterializationStep step;
    step.ggufName = ggufName;
    std::uint64_t perExpert = 0;
    for (int e = 0; e < numExperts; ++e) {
        const std::string hf =
            hfPrefix + std::to_string(e) + "." + projSuffix + ".weight";
        MaterializationSource src = sourceFor(model, hf, 0);
        if (src.kind != SourceKind::Fp8Block128) {
            fail("routed expert '" + hf + "' is not block-128 FP8 (no "
                 "weight_scale_inv sibling)");
        }
        if (e == 0) {
            perExpert = src.rows * src.in;
            step.ggufDims = {src.in, src.rows,
                             static_cast<std::uint64_t>(numExperts)};
        } else if (src.rows * src.in != perExpert) {
            fail("routed expert '" + hf + "' has a different element count "
                 "than expert 0 (ragged expert stack)");
        }
        src.dstElemOffset = static_cast<std::uint64_t>(e) * perExpert;
        step.sources.push_back(std::move(src));
    }
    step.totalElems = perExpert * static_cast<std::uint64_t>(numExperts);
    steps.push_back(std::move(step));
}

} // namespace

std::vector<MaterializationStep>
planKolibri1Materialization(const st::SafetensorsModel& model,
                            const Kolibri1Arch&         arch) {
    std::vector<MaterializationStep> steps;

    // --- model-level ------------------------------------------------------
    addDirect(steps, model, "token_embd.weight",  "model.embed_tokens.weight");
    addDirect(steps, model, "output_norm.weight", "model.norm.weight");
    addDirect(steps, model, "output.weight",      "lm_head.weight");

    // --- per layer --------------------------------------------------------
    for (int L = 0; L < arch.numLayers; ++L) {
        const std::string blk = "blk." + std::to_string(L) + ".";
        const std::string hf  = "model.layers." + std::to_string(L) + ".";

        // Attention: q/k/v/o projections (FP8 block-128) + the 4 per-layer
        // norms and the per-head q/k RMSNorms (BF16 -> F32).
        addDirect(steps, model, blk + "attn_norm.weight",
                  hf + "input_layernorm.weight");
        addDirect(steps, model, blk + "attn_q.weight",
                  hf + "self_attn.q_proj.weight");
        addDirect(steps, model, blk + "attn_k.weight",
                  hf + "self_attn.k_proj.weight");
        addDirect(steps, model, blk + "attn_v.weight",
                  hf + "self_attn.v_proj.weight");
        addDirect(steps, model, blk + "attn_output.weight",
                  hf + "self_attn.o_proj.weight");
        addDirect(steps, model, blk + "attn_q_norm.weight",
                  hf + "self_attn.q_norm.weight");
        addDirect(steps, model, blk + "attn_k_norm.weight",
                  hf + "self_attn.k_norm.weight");
        addDirect(steps, model, blk + "post_attn_norm.weight",
                  hf + "post_attn_norm.weight");
        addDirect(steps, model, blk + "post_attention_norm.weight",
                  hf + "post_attention_layernorm.weight");
        addDirect(steps, model, blk + "post_ffn_norm.weight",
                  hf + "post_ffn_norm.weight");

        // Router: the gate logits projection (unquantised BF16 -> F32) and the
        // DeepSeek-V3-style per-expert selection bias (sigmoid gate, I-5).
        addDirect(steps, model, blk + "ffn_gate_inp.weight",
                  hf + "mlp.gate.weight");
        addDirect(steps, model, blk + "exp_probs_b.weight",
                  hf + "moe.router.expert_bias");

        // Routed experts (FP8 block-128), stacked over n_expert. These are the
        // tensors the loader defers + streams into requant banks.
        const std::string expPrefix = hf + "mlp.experts.";
        addStacked(steps, model, blk + "ffn_gate_exps.weight",
                   expPrefix, "gate_proj", arch.numExperts);
        addStacked(steps, model, blk + "ffn_up_exps.weight",
                   expPrefix, "up_proj", arch.numExperts);
        addStacked(steps, model, blk + "ffn_down_exps.weight",
                   expPrefix, "down_proj", arch.numExperts);

        // Always-on shared expert (FP8 block-128 -> BF16; small, not deferred).
        addDirect(steps, model, blk + "ffn_gate_shexp.weight",
                  hf + "mlp.shared_experts.gate_proj.weight");
        addDirect(steps, model, blk + "ffn_up_shexp.weight",
                  hf + "mlp.shared_experts.up_proj.weight");
        addDirect(steps, model, blk + "ffn_down_shexp.weight",
                  hf + "mlp.shared_experts.down_proj.weight");
    }

    return steps;
}

} // namespace mimirmind::core::modelopt
