// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/encoder/EmbedEngine.hpp"

#include <cstdint>
#include <filesystem>
#include <vector>

namespace mimirmind::runtime::encoder {

EmbedEngine::EmbedEngine(std::string_view modelDir,
                         compute::ComputeOps& ops,
                         compute::ComputeMatmul& matmul)
    : _runner{_model, ops, matmul} {
    // BF16 linear weights -> tensor-core GEMM at M>1; activations stay F32.
    // Embedding cosine similarity is robust to the bf16 rounding, same as the
    // reranker path.
    _model.load(modelDir, ops, core::gguf::GgmlType::BF16);
    const std::filesystem::path spm =
        std::filesystem::path{modelDir} / "sentencepiece.bpe.model";
    _tokenizer.load(spm.string());
}

std::vector<float> EmbedEngine::embed(std::string_view text) const {
    // [<s>] text [</s>] — the <s>/CLS hidden (row 0) is what EncoderRunner::embed
    // pools and L2-normalizes.
    const std::vector<std::int32_t> ids = _tokenizer.encodeSingle(text);
    return _runner.embed(ids);
}

std::vector<std::vector<float>>
EmbedEngine::embedBatch(std::span<const std::string> texts) const {
    std::vector<std::vector<float>> out;
    out.reserve(texts.size());
    for (const auto& t : texts) {
        out.push_back(embed(t));
    }
    return out;
}

} // namespace mimirmind::runtime::encoder
