// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/modelopt/Gemma4Materializer.hpp"

#include "core/safetensors/SafetensorsModel.hpp"

#include <stdexcept>
#include <string>
#include <string_view>

namespace mimirmind::core::modelopt {

namespace {

namespace st = safetensors;

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("gemma4 materialize: " + msg);
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

std::string layerPrefix(int layer) {
    return "model.language_model.layers." + std::to_string(layer) + ".";
}

/// A compressed-tensors NVFP4 Linear: `<base>.weight_packed` (U8, packed cols)
/// + `<base>.weight_scale` (F8_E4M3 per-16) + `<base>.weight_global_scale`
/// (F32, stored reciprocal). Dequantised to BF16.
void addQuant(std::vector<MaterializationStep>& steps,
              const st::SafetensorsModel& model,
              const std::string& ggufName, const std::string& hfBase) {
    const st::SafetensorsTensor& w = require(model, hfBase + ".weight_packed");
    if (w.shape.size() != 2) {
        fail("quantised tensor '" + hfBase + ".weight_packed' is not 2-D");
    }
    MaterializationSource src;
    src.hfWeightName       = hfBase + ".weight_packed";
    src.kind               = SourceKind::Nvfp4;
    src.rows               = w.shape[0];        // out
    src.in                 = w.shape[1] * 2;    // packed cols * 2 = in
    src.blockScaleName     = hfBase + ".weight_scale";
    src.globalScaleName    = hfBase + ".weight_global_scale";
    src.globalIsReciprocal = true;              // compressed-tensors: 1/global

    MaterializationStep step;
    step.ggufName   = ggufName;
    step.ggufDims   = {src.in, src.rows};       // logical [in, out] ne-order
    step.totalElems = src.rows * src.in;
    step.outF32     = false;                    // BF16 matmul weight
    step.sources.push_back(std::move(src));
    steps.push_back(std::move(step));
}

/// An unquantised 1-D/embedding tensor kept as an F32 device pointer (norms,
/// the per-layer scalar, embeddings). `addOne` bakes the Gemma (1 + w) RMSNorm
/// convention (the backend multiplies by the stored weight directly).
void addPass(std::vector<MaterializationStep>& steps,
             const st::SafetensorsModel& model,
             const std::string& ggufName, const std::string& hfName,
             bool addOne) {
    const st::SafetensorsTensor& w = require(model, hfName);
    MaterializationSource src;
    src.hfWeightName = hfName;
    src.kind         = SourceKind::Bf16Passthrough;
    src.rows         = w.nelements;
    src.in           = 1;

    MaterializationStep step;
    step.ggufName      = ggufName;
    step.ggufDims      = ggufDimsFromHf(w.shape);
    step.totalElems    = w.nelements;
    step.outF32        = true;
    step.postTransform = addOne ? PostTransform::AddOne : PostTransform::None;
    step.sources.push_back(std::move(src));
    steps.push_back(std::move(step));
}

/// An already-BF16 2-D matmul weight kept BF16 verbatim (lm_head — it is in the
/// checkpoint's `ignore` list, so unquantised). Byte layout HF [out,in]
/// row-major == GGUF [in,out] ne-order.
void addBf16Matmul(std::vector<MaterializationStep>& steps,
                   const st::SafetensorsModel& model,
                   const std::string& ggufName, const std::string& hfName) {
    const st::SafetensorsTensor& w = require(model, hfName);
    if (w.shape.size() != 2) {
        fail("BF16 matmul tensor '" + hfName + "' is not 2-D");
    }
    MaterializationSource src;
    src.hfWeightName  = hfName;
    src.kind          = SourceKind::Bf16Copy;
    src.rows          = w.shape[0];   // out
    src.in            = w.shape[1];   // in
    src.dstElemOffset = 0;
    src.srcElemOffset = 0;

    MaterializationStep step;
    step.ggufName   = ggufName;
    step.ggufDims   = {w.shape[1], w.shape[0]};   // [in, out] ne-order
    step.totalElems = w.shape[0] * w.shape[1];
    step.outF32     = false;                      // keep BF16
    step.sources.push_back(std::move(src));
    steps.push_back(std::move(step));
}

} // namespace

std::vector<MaterializationStep>
planGemma4Materialization(const st::SafetensorsModel& model,
                          const Gemma4Arch&           arch) {
    std::vector<MaterializationStep> steps;

    // --- model-level ------------------------------------------------------
    // token_embd kept BF16 (F32 would be 4 GiB); the CPU embeddingLookup
    // dequantises per row via dequantToF32 (same path as the GGUF Q6_K embed).
    addBf16Matmul(steps, model, "token_embd.weight",
                  "model.language_model.embed_tokens.weight");
    // NB: this compressed-tensors checkpoint stores the RMSNorm weights already
    // in final (1 + w)-baked form — verified byte-identical to the coherent
    // llama.cpp GGUF (e.g. q_norm = 1.023, i.e. 1 + 0.023), which the Gemma4
    // backend consumes directly. So NO AddOne here (adding another 1 was the
    // norm-scale bug that produced token soup while the projections were fine).
    addPass(steps, model, "output_norm.weight",
            "model.language_model.norm.weight", /*addOne=*/false);
    // lm_head is unquantised (config `ignore`), BF16 in the checkpoint.
    addBf16Matmul(steps, model, "output.weight", "lm_head.weight");

    // --- per layer --------------------------------------------------------
    for (int L = 0; L < arch.numLayers; ++L) {
        const std::string blk = "blk." + std::to_string(L) + ".";
        const std::string pre = layerPrefix(L);
        const bool fullAttn =
            (L < static_cast<int>(arch.isFullAttn.size())) && arch.isFullAttn[L];

        addPass(steps, model, blk + "attn_norm.weight",
                pre + "input_layernorm.weight", false);
        addQuant(steps, model, blk + "attn_q.weight", pre + "self_attn.q_proj");
        addPass(steps, model, blk + "attn_q_norm.weight",
                pre + "self_attn.q_norm.weight", false);
        addQuant(steps, model, blk + "attn_k.weight", pre + "self_attn.k_proj");
        addPass(steps, model, blk + "attn_k_norm.weight",
                pre + "self_attn.k_norm.weight", false);
        // Full-attention layers are attention_k_eq_v (V = raw K) — no v_proj.
        if (!fullAttn) {
            addQuant(steps, model, blk + "attn_v.weight",
                     pre + "self_attn.v_proj");
        }
        addQuant(steps, model, blk + "attn_output.weight",
                 pre + "self_attn.o_proj");
        addPass(steps, model, blk + "post_attention_norm.weight",
                pre + "post_attention_layernorm.weight", false);

        addPass(steps, model, blk + "ffn_norm.weight",
                pre + "pre_feedforward_layernorm.weight", false);
        addQuant(steps, model, blk + "ffn_gate.weight", pre + "mlp.gate_proj");
        addQuant(steps, model, blk + "ffn_up.weight", pre + "mlp.up_proj");
        addQuant(steps, model, blk + "ffn_down.weight", pre + "mlp.down_proj");
        addPass(steps, model, blk + "post_ffw_norm.weight",
                pre + "post_feedforward_layernorm.weight", false);

        // Per-layer output scalar (Gemma `layer_scalar` -> layer_output_scale).
        // Plain passthrough, NOT a norm — no (1 + w).
        addPass(steps, model, blk + "layer_output_scale.weight",
                pre + "layer_scalar", false);
    }

    // Defensive: the AddOne convention keys on the GGUF suffix elsewhere; make
    // sure no non-norm passthrough accidentally carries it.
    for (const auto& s : steps) {
        if (s.postTransform == PostTransform::AddOne
            && !endsWith(s.ggufName, "norm.weight")) {
            fail("AddOne on a non-norm tensor '" + s.ggufName + "'");
        }
    }

    return steps;
}

} // namespace mimirmind::core::modelopt
