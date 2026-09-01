// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/encoder/EncoderModel.hpp"

#include "compute/ComputeOps.hpp"
#include "core/gpu/AllocCategory.hpp"
#include "core/safetensors/SafetensorsDtype.hpp"
#include "core/safetensors/SafetensorsModel.hpp"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace mimirmind::runtime::encoder {

namespace {

std::string readTextFile(const std::filesystem::path& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) {
        throw std::runtime_error("EncoderModel: cannot read " + p.string());
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// Round a binary32 to bfloat16 (round-to-nearest-even), returned as the raw
// 16-bit pattern. Weight-only; NaN/Inf are not special-cased (weights finite).
std::uint16_t f32ToBf16(float f) {
    std::uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const std::uint32_t rounding = 0x7FFFu + ((x >> 16) & 1u);
    return static_cast<std::uint16_t>((x + rounding) >> 16);
}

} // namespace

EncoderConfig parseEncoderConfig(std::string_view configJson) {
    const nlohmann::json j =
        nlohmann::json::parse(configJson.begin(), configJson.end(), nullptr,
                              /*allow_exceptions=*/true, /*ignore_comments=*/true);

    EncoderConfig c{};
    c.hidden     = j.at("hidden_size").get<std::size_t>();
    c.numLayers  = j.at("num_hidden_layers").get<std::size_t>();
    c.heads      = j.at("num_attention_heads").get<std::size_t>();
    c.headDim    = c.hidden / c.heads;
    c.ffn        = j.at("intermediate_size").get<std::size_t>();
    c.vocab      = j.at("vocab_size").get<std::size_t>();
    c.maxPos     = j.at("max_position_embeddings").get<std::size_t>();
    c.typeVocab  = j.value("type_vocab_size", std::size_t{1});
    c.padTokenId = j.value("pad_token_id", std::int32_t{1});
    c.posOffset  = static_cast<std::size_t>(c.padTokenId) + 1;
    c.lnEps      = j.value("layer_norm_eps", 1e-5F);
    if (j.contains("id2label")) {
        c.numLabels = j.at("id2label").size();
    }
    c.numLabels = j.value("num_labels", c.numLabels);
    if (c.numLabels == 0) {
        c.numLabels = 1;
    }
    return c;
}

void EncoderModel::load(std::string_view dir, compute::ComputeOps& ops,
                        core::gguf::GgmlType matmulType) {
    // 8.16 Stage B: bge-m3 embed/rerank weights load outside the guarded
    // InferenceEngine path -> tag them Weights (else they land in `unknown`).
    core::gpu::ScopedAllocCategory _wc{core::gpu::AllocCategory::Weights};
    _matmulType = matmulType;
    const std::filesystem::path root{dir};
    _config = parseEncoderConfig(readTextFile(root / "config.json"));

    core::safetensors::SafetensorsModel sm;
    sm.open(dir);

    auto findF32 = [&](const std::string& name)
        -> std::pair<const float*, std::size_t> {
        const auto* t = sm.find(name);
        if (t == nullptr) {
            throw std::runtime_error("EncoderModel: missing tensor '" + name + "'");
        }
        if (t->dtype != core::safetensors::SafetensorsDtype::F32) {
            throw std::runtime_error(
                "EncoderModel: tensor '" + name +
                "' is not F32 (only dense float32 checkpoints are supported)");
        }
        const std::span<const std::uint8_t> bytes = sm.tensorBytes(name);
        return {reinterpret_cast<const float*>(bytes.data()), bytes.size() / sizeof(float)};
    };

    // Upload one F32 tensor verbatim to USM (biases, LayerNorm, embeddings).
    auto upload = [&](const std::string& name) -> const float* {
        const auto [src, n] = findF32(name);
        compute::ComputeBuffer buf = ops.allocate(n * sizeof(float));
        ops.uploadHostBytes(buf.get(), src, n * sizeof(float));
        const float* p = static_cast<const float*>(buf.get());
        _owned.push_back(std::move(buf));
        return p;
    };

    // Upload a dense LINEAR weight as _matmulType: BF16 (round F32->bf16, half
    // the bytes, tensor-core GEMM at M>1) or F32 verbatim. Returned as an
    // opaque device pointer (ComputeMatmul reads it per GgmlType). Ranking is
    // robust to the bf16 rounding; activations stay F32.
    auto uploadW = [&](const std::string& name) -> const float* {
        const auto [src, n] = findF32(name);
        if (_matmulType == core::gguf::GgmlType::BF16) {
            std::vector<std::uint16_t> bf(n);
            for (std::size_t i = 0; i < n; ++i) {
                bf[i] = f32ToBf16(src[i]);
            }
            compute::ComputeBuffer buf = ops.allocate(n * sizeof(std::uint16_t));
            ops.uploadHostBytes(buf.get(), bf.data(), n * sizeof(std::uint16_t));
            const auto* p = reinterpret_cast<const float*>(buf.get());
            _owned.push_back(std::move(buf));
            return p;
        }
        compute::ComputeBuffer buf = ops.allocate(n * sizeof(float));
        ops.uploadHostBytes(buf.get(), src, n * sizeof(float));
        const float* p = static_cast<const float*>(buf.get());
        _owned.push_back(std::move(buf));
        return p;
    };

    // Embeddings block.
    _wordEmb  = upload("roberta.embeddings.word_embeddings.weight");
    _posTable = upload("roberta.embeddings.position_embeddings.weight");
    _typeVec  = upload("roberta.embeddings.token_type_embeddings.weight");
    _embLnW   = upload("roberta.embeddings.LayerNorm.weight");
    _embLnB   = upload("roberta.embeddings.LayerNorm.bias");

    // Encoder layers.
    _layers.resize(_config.numLayers);
    for (std::size_t i = 0; i < _config.numLayers; ++i) {
        const std::string pre =
            "roberta.encoder.layer." + std::to_string(i) + ".";
        EncoderLayerWeights& L = _layers[i];
        L.qW   = uploadW(pre + "attention.self.query.weight");
        L.qB   = upload(pre + "attention.self.query.bias");
        L.kW   = uploadW(pre + "attention.self.key.weight");
        L.kB   = upload(pre + "attention.self.key.bias");
        L.vW   = uploadW(pre + "attention.self.value.weight");
        L.vB   = upload(pre + "attention.self.value.bias");
        L.aoW  = uploadW(pre + "attention.output.dense.weight");
        L.aoB  = upload(pre + "attention.output.dense.bias");
        L.aoLnW = upload(pre + "attention.output.LayerNorm.weight");
        L.aoLnB = upload(pre + "attention.output.LayerNorm.bias");
        L.fiW  = uploadW(pre + "intermediate.dense.weight");
        L.fiB  = upload(pre + "intermediate.dense.bias");
        L.foW  = uploadW(pre + "output.dense.weight");
        L.foB  = upload(pre + "output.dense.bias");
        L.outLnW = upload(pre + "output.LayerNorm.weight");
        L.outLnB = upload(pre + "output.LayerNorm.bias");
    }

    // Classifier head (dense -> tanh -> out_proj on the <s>/CLS token) —
    // present only on a sequence-classification checkpoint (reranker). A pure
    // embedding checkpoint (e.g. bge-m3 dense) ships encoder + embeddings only;
    // EncoderRunner::embed() pools the CLS hidden directly, so skip the head
    // when it is absent instead of hard-failing on the missing tensors.
    if (sm.find("classifier.out_proj.weight") != nullptr) {
        _clsDenseW = uploadW("classifier.dense.weight");
        _clsDenseB = upload("classifier.dense.bias");
        _clsOutW   = uploadW("classifier.out_proj.weight");
        _clsOutB   = upload("classifier.out_proj.bias");
        _hasClassifier = true;
        // num_labels from the head if config didn't pin it.
        if (const auto* op = sm.find("classifier.out_proj.weight");
            op != nullptr && !op->shape.empty()) {
            _config.numLabels = static_cast<std::size_t>(op->shape[0]);
        }
    } else {
        _hasClassifier   = false;
        _config.numLabels = 0;   // embedding-only checkpoint: no logits
    }
}

} // namespace mimirmind::runtime::encoder
