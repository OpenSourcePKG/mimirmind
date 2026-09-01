// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/audio/WhisperModel.hpp"

#include "compute/ComputeOps.hpp"
#include "core/gpu/AllocCategory.hpp"
#include "compute/quant/Float16.hpp"
#include "core/safetensors/SafetensorsDtype.hpp"
#include "core/safetensors/SafetensorsModel.hpp"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mimirmind::runtime::audio {

namespace {

std::string readTextFile(const std::filesystem::path& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) {
        throw std::runtime_error("WhisperModel: cannot read " + p.string());
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// binary32 -> bfloat16 (round-to-nearest-even), raw 16-bit pattern. Weight-only.
std::uint16_t f32ToBf16(float f) {
    std::uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const std::uint32_t rounding = 0x7FFFu + ((x >> 16) & 1u);
    return static_cast<std::uint16_t>((x + rounding) >> 16);
}

} // namespace

bool isWhisperConfig(std::string_view configJson) {
    const nlohmann::json j = nlohmann::json::parse(
        configJson.begin(), configJson.end(), nullptr,
        /*allow_exceptions=*/false, /*ignore_comments=*/true);
    if (j.is_discarded()) {
        return false;
    }
    if (j.contains("model_type") && j["model_type"].is_string() &&
        j["model_type"].get<std::string>() == "whisper") {
        return true;
    }
    if (j.contains("architectures") && j["architectures"].is_array()) {
        for (const auto& a : j["architectures"]) {
            if (a.is_string() && a.get<std::string>().rfind("Whisper", 0) == 0) {
                return true;
            }
        }
    }
    return false;
}

WhisperConfig parseWhisperConfig(std::string_view configJson) {
    const nlohmann::json j =
        nlohmann::json::parse(configJson.begin(), configJson.end(), nullptr,
                              /*allow_exceptions=*/true, /*ignore_comments=*/true);

    WhisperConfig c{};
    c.numMelBins         = j.value("num_mel_bins", std::size_t{80});
    c.dModel             = j.at("d_model").get<std::size_t>();
    c.encoderLayers      = j.at("encoder_layers").get<std::size_t>();
    c.decoderLayers      = j.at("decoder_layers").get<std::size_t>();
    c.encoderHeads       = j.at("encoder_attention_heads").get<std::size_t>();
    c.decoderHeads       = j.at("decoder_attention_heads").get<std::size_t>();
    c.encoderFfn         = j.at("encoder_ffn_dim").get<std::size_t>();
    c.decoderFfn         = j.at("decoder_ffn_dim").get<std::size_t>();
    c.vocab              = j.at("vocab_size").get<std::size_t>();
    c.maxSourcePositions = j.value("max_source_positions", std::size_t{1500});
    c.maxTargetPositions = j.value("max_target_positions", std::size_t{448});
    c.bosTokenId         = j.value("bos_token_id", std::int32_t{50257});
    c.eosTokenId         = j.value("eos_token_id", std::int32_t{50257});
    c.decoderStartTokenId = j.value("decoder_start_token_id", std::int32_t{50258});
    c.padTokenId         = j.value("pad_token_id", std::int32_t{50257});
    return c;
}

void WhisperModel::load(std::string_view dir, compute::ComputeOps& ops,
                        core::gguf::GgmlType matmulType) {
    // 8.16 Stage B: Whisper weights load outside the guarded InferenceEngine
    // path -> tag them Weights (else `unknown`).
    core::gpu::ScopedAllocCategory _wc{core::gpu::AllocCategory::Weights};
    _matmulType = matmulType;
    const std::filesystem::path root{dir};
    const std::string cfgText = readTextFile(root / "config.json");
    if (!isWhisperConfig(cfgText)) {
        throw std::runtime_error(
            "WhisperModel: config.json is not a Whisper checkpoint");
    }
    _config = parseWhisperConfig(cfgText);

    core::safetensors::SafetensorsModel sm;
    sm.open(dir);

    // Read one tensor into a host F32 vector, converting F16 source in place.
    // (Whisper checkpoints ship as F32 or F16.)
    auto hostF32 = [&](const std::string& name) -> std::vector<float> {
        const auto* t = sm.find(name);
        if (t == nullptr) {
            throw std::runtime_error("WhisperModel: missing tensor '" + name + "'");
        }
        const std::span<const std::uint8_t> bytes = sm.tensorBytes(name);
        std::vector<float> out(static_cast<std::size_t>(t->nelements));
        switch (t->dtype) {
        case core::safetensors::SafetensorsDtype::F32:
            std::memcpy(out.data(), bytes.data(), out.size() * sizeof(float));
            break;
        case core::safetensors::SafetensorsDtype::F16:
            compute::quant::Float16::instance().dequantToF32(
                bytes.data(), out.size(), out.data());
            break;
        default:
            throw std::runtime_error(
                "WhisperModel: tensor '" + name +
                "' has unsupported dtype (expected F32 or F16)");
        }
        return out;
    };

    // Assert a tensor's shape; a dim of 0 means "don't care".
    auto expectShape = [&](const std::string& name,
                           const std::vector<std::size_t>& dims) {
        const auto* t = sm.find(name);
        if (t == nullptr) {
            throw std::runtime_error("WhisperModel: missing tensor '" + name + "'");
        }
        if (t->shape.size() != dims.size()) {
            throw std::runtime_error("WhisperModel: tensor '" + name +
                                     "' rank mismatch");
        }
        for (std::size_t i = 0; i < dims.size(); ++i) {
            if (dims[i] != 0 &&
                static_cast<std::size_t>(t->shape[i]) != dims[i]) {
                throw std::runtime_error("WhisperModel: tensor '" + name +
                                         "' dim " + std::to_string(i) +
                                         " = " + std::to_string(t->shape[i]) +
                                         ", expected " + std::to_string(dims[i]));
            }
        }
    };

    // Upload an F32 host buffer to USM, return the device pointer.
    auto uploadHost = [&](const std::vector<float>& h) -> const float* {
        compute::ComputeBuffer buf = ops.allocate(h.size() * sizeof(float));
        ops.uploadHostBytes(buf.get(), h.data(), h.size() * sizeof(float));
        const float* p = static_cast<const float*>(buf.get());
        _owned.push_back(std::move(buf));
        return p;
    };

    // Verbatim F32 upload (conv, embeddings, positional tables, biases, norms).
    auto upload = [&](const std::string& name) -> const float* {
        return uploadHost(hostF32(name));
    };

    // Dense LINEAR weight upload as _matmulType: BF16 (half the bytes,
    // tensor-core GEMM at M>1) or F32 verbatim. Opaque device pointer that
    // ComputeMatmul reads per GgmlType.
    auto uploadW = [&](const std::string& name) -> const float* {
        const std::vector<float> h = hostF32(name);
        if (_matmulType == core::gguf::GgmlType::BF16) {
            std::vector<std::uint16_t> bf(h.size());
            for (std::size_t i = 0; i < h.size(); ++i) {
                bf[i] = f32ToBf16(h[i]);
            }
            compute::ComputeBuffer buf = ops.allocate(bf.size() * sizeof(std::uint16_t));
            ops.uploadHostBytes(buf.get(), bf.data(), bf.size() * sizeof(std::uint16_t));
            const auto* p = reinterpret_cast<const float*>(buf.get());
            _owned.push_back(std::move(buf));
            return p;
        }
        return uploadHost(h);
    };

    // Detect the state-dict prefix: WhisperForConditionalGeneration wraps the
    // encoder/decoder under "model."; a bare WhisperModel checkpoint does not.
    const std::string pfx =
        (sm.find("model.encoder.conv1.weight") != nullptr) ? "model." : "";

    const std::size_t d    = _config.dModel;
    const std::size_t mel  = _config.numMelBins;

    // ---- Convolutional stem -------------------------------------------------
    const std::string enc = pfx + "encoder.";
    expectShape(enc + "conv1.weight", {d, mel, 3});
    expectShape(enc + "conv2.weight", {d, d, 3});
    _conv1W = upload(enc + "conv1.weight");
    _conv1B = upload(enc + "conv1.bias");
    _conv2W = upload(enc + "conv2.weight");
    _conv2B = upload(enc + "conv2.bias");

    // ---- Encoder ------------------------------------------------------------
    expectShape(enc + "embed_positions.weight",
                {_config.maxSourcePositions, d});
    _encPosEmb = upload(enc + "embed_positions.weight");

    _encLayers.resize(_config.encoderLayers);
    for (std::size_t i = 0; i < _config.encoderLayers; ++i) {
        const std::string pre = enc + "layers." + std::to_string(i) + ".";
        WhisperEncoderLayer& L = _encLayers[i];
        L.qW = uploadW(pre + "self_attn.q_proj.weight");
        L.qB = upload(pre + "self_attn.q_proj.bias");
        L.kW = uploadW(pre + "self_attn.k_proj.weight");   // no bias
        L.vW = uploadW(pre + "self_attn.v_proj.weight");
        L.vB = upload(pre + "self_attn.v_proj.bias");
        L.oW = uploadW(pre + "self_attn.out_proj.weight");
        L.oB = upload(pre + "self_attn.out_proj.bias");
        L.attnLnW = upload(pre + "self_attn_layer_norm.weight");
        L.attnLnB = upload(pre + "self_attn_layer_norm.bias");
        L.fc1W = uploadW(pre + "fc1.weight");
        L.fc1B = upload(pre + "fc1.bias");
        L.fc2W = uploadW(pre + "fc2.weight");
        L.fc2B = upload(pre + "fc2.bias");
        L.finalLnW = upload(pre + "final_layer_norm.weight");
        L.finalLnB = upload(pre + "final_layer_norm.bias");
    }
    _encLnW = upload(enc + "layer_norm.weight");
    _encLnB = upload(enc + "layer_norm.bias");

    // ---- Decoder ------------------------------------------------------------
    const std::string dec = pfx + "decoder.";
    expectShape(dec + "embed_tokens.weight", {_config.vocab, d});
    expectShape(dec + "embed_positions.weight",
                {_config.maxTargetPositions, d});
    _decTokEmb = upload(dec + "embed_tokens.weight");
    _decPosEmb = upload(dec + "embed_positions.weight");

    _decLayers.resize(_config.decoderLayers);
    for (std::size_t i = 0; i < _config.decoderLayers; ++i) {
        const std::string pre = dec + "layers." + std::to_string(i) + ".";
        WhisperDecoderLayer& L = _decLayers[i];
        // Masked self-attention.
        L.qW = uploadW(pre + "self_attn.q_proj.weight");
        L.qB = upload(pre + "self_attn.q_proj.bias");
        L.kW = uploadW(pre + "self_attn.k_proj.weight");
        L.vW = uploadW(pre + "self_attn.v_proj.weight");
        L.vB = upload(pre + "self_attn.v_proj.bias");
        L.oW = uploadW(pre + "self_attn.out_proj.weight");
        L.oB = upload(pre + "self_attn.out_proj.bias");
        L.selfLnW = upload(pre + "self_attn_layer_norm.weight");
        L.selfLnB = upload(pre + "self_attn_layer_norm.bias");
        // Cross-attention over encoder states.
        L.cqW = uploadW(pre + "encoder_attn.q_proj.weight");
        L.cqB = upload(pre + "encoder_attn.q_proj.bias");
        L.ckW = uploadW(pre + "encoder_attn.k_proj.weight");
        L.cvW = uploadW(pre + "encoder_attn.v_proj.weight");
        L.cvB = upload(pre + "encoder_attn.v_proj.bias");
        L.coW = uploadW(pre + "encoder_attn.out_proj.weight");
        L.coB = upload(pre + "encoder_attn.out_proj.bias");
        L.crossLnW = upload(pre + "encoder_attn_layer_norm.weight");
        L.crossLnB = upload(pre + "encoder_attn_layer_norm.bias");
        // Feed-forward.
        L.fc1W = uploadW(pre + "fc1.weight");
        L.fc1B = upload(pre + "fc1.bias");
        L.fc2W = uploadW(pre + "fc2.weight");
        L.fc2B = upload(pre + "fc2.bias");
        L.finalLnW = upload(pre + "final_layer_norm.weight");
        L.finalLnB = upload(pre + "final_layer_norm.bias");
    }
    _decLnW = upload(dec + "layer_norm.weight");
    _decLnB = upload(dec + "layer_norm.bias");

    // Output projection: explicit if present, otherwise tied to embed_tokens
    // (Whisper's default) — leave _projOut null so callers GEMM against
    // decTokEmb().
    if (sm.find(pfx + "proj_out.weight") != nullptr) {
        expectShape(pfx + "proj_out.weight", {_config.vocab, d});
        _projOut = uploadW(pfx + "proj_out.weight");
    } else if (sm.find("proj_out.weight") != nullptr) {
        expectShape("proj_out.weight", {_config.vocab, d});
        _projOut = uploadW("proj_out.weight");
    } else {
        _projOut = nullptr;
    }
}

} // namespace mimirmind::runtime::audio
