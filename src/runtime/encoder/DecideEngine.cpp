// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/encoder/DecideEngine.hpp"

#include "core/log/Log.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <stdexcept>

namespace mimirmind::runtime::encoder {

DecideEngine::DecideEngine(std::string_view encoderDir,
                           std::string_view headsDir,
                           compute::ComputeOps& ops,
                           compute::ComputeMatmul& matmul)
    : _runner{_model, ops, matmul} {
    // Same frozen bge-m3 substrate as EmbedEngine: BF16 linear weights (TC GEMM
    // at M>1), F32 activations, CLS-pool + L2-normalize. The decision heads sit
    // on top of that unit embedding.
    _model.load(encoderDir, ops, core::gguf::GgmlType::BF16);
    const std::filesystem::path spm =
        std::filesystem::path{encoderDir} / "sentencepiece.bpe.model";
    _tokenizer.load(spm.string());

    // Discover heads: every immediate subdirectory of headsDir holding a
    // head.json. A missing/empty dir is fine — the engine then serves
    // fallback-only until trained heads are dropped in.
    namespace fs = std::filesystem;
    const fs::path root{headsDir};
    if (!headsDir.empty() && fs::is_directory(root)) {
        std::vector<fs::path> dirs;
        for (const auto& e : fs::directory_iterator{root}) {
            if (e.is_directory() && fs::exists(e.path() / "head.json")) {
                dirs.push_back(e.path());
            }
        }
        // Deterministic load order (directory iteration order is unspecified).
        std::sort(dirs.begin(), dirs.end());
        for (const auto& d : dirs) {
            DecisionHead h = DecisionHead::loadFromDir(d);
            if (h.hidden() != _model.config().hidden) {
                throw std::runtime_error(
                    "DecideEngine: head '" + h.name() + "' hidden " +
                    std::to_string(h.hidden()) + " != encoder hidden " +
                    std::to_string(_model.config().hidden));
            }
            if (findHead(h.name()) != nullptr) {
                throw std::runtime_error(
                    "DecideEngine: duplicate head name '" + h.name() + "'");
            }
            MM_LOG_INFO("decide", "loaded decision head '{}' ({} labels)",
                        h.name(), h.labels().size());
            _heads.push_back(std::move(h));
        }
    }
}

const DecisionHead* DecideEngine::findHead(std::string_view name) const {
    for (const auto& h : _heads) {
        if (h.name() == name) {
            return &h;
        }
    }
    return nullptr;
}

std::vector<std::string> DecideEngine::headNames() const {
    std::vector<std::string> out;
    out.reserve(_heads.size());
    for (const auto& h : _heads) {
        out.push_back(h.name());
    }
    return out;
}

std::vector<DecideEngine::Answer>
DecideEngine::decide(std::string_view text,
                     std::span<const std::string> questions) const {
    // One encoder forward, shared across every head. [<s>] text [</s>] → CLS
    // hidden (row 0), L2-normalized — identical to /v1/embeddings.
    const std::vector<std::int32_t> ids = _tokenizer.encodeSingle(text);
    const std::vector<float> emb = _runner.embed(ids);

    std::vector<Answer> out;
    if (questions.empty()) {
        // No explicit questions → answer every loaded head.
        out.reserve(_heads.size());
        for (const auto& h : _heads) {
            Answer a{};
            a.question = h.name();
            a.hasHead  = true;
            a.result   = h.decide(emb);
            out.push_back(std::move(a));
        }
        return out;
    }

    out.reserve(questions.size());
    for (const auto& q : questions) {
        Answer a{};
        a.question = q;
        if (const DecisionHead* h = findHead(q); h != nullptr) {
            a.hasHead = true;
            a.result  = h->decide(emb);
        }
        out.push_back(std::move(a));
    }
    return out;
}

} // namespace mimirmind::runtime::encoder
