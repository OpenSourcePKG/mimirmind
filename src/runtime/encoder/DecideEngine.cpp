// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/encoder/DecideEngine.hpp"

#include "core/log/Log.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <set>
#include <stdexcept>
#include <unordered_map>

namespace mimirmind::runtime::encoder {

DecideEngine::DecideEngine(std::string_view encoderDir,
                           std::string_view headsDir,
                           compute::ComputeOps& ops,
                           compute::ComputeMatmul& matmul)
    : _runner{_model, ops, matmul}, _headsDir{headsDir} {
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

DecideEngine::TrainReport DecideEngine::trainHead(
        const std::string& name, const std::vector<std::string>& labelsIn,
        const std::vector<TrainExample>& examples,
        const LinearProbeTrainer::Config& cfg, float temperature, float threshold) {
    if (examples.empty()) {
        throw std::runtime_error("DecideEngine::trainHead '" + name + "': no examples");
    }

    // Label vocabulary: explicit order when given, else sorted-unique from data.
    std::vector<std::string> labels = labelsIn;
    if (labels.empty()) {
        std::set<std::string> uniq;
        for (const auto& e : examples) {
            uniq.insert(e.label);
        }
        labels.assign(uniq.begin(), uniq.end());
    }
    if (labels.size() < 2) {
        throw std::runtime_error("DecideEngine::trainHead '" + name + "': need >= 2 labels");
    }
    std::unordered_map<std::string, std::int32_t> labelIdx;
    for (std::size_t i = 0; i < labels.size(); ++i) {
        labelIdx[labels[i]] = static_cast<std::int32_t>(i);
    }

    // Embed every example through the SAME path decide() uses → train features
    // are identical to serve features.
    const std::size_t D = _model.config().hidden;
    std::vector<float>        features;
    std::vector<std::int32_t> y;
    features.reserve(examples.size() * D);
    y.reserve(examples.size());

    for (const auto& e : examples) {
        const auto it = labelIdx.find(e.label);
        if (it == labelIdx.end()) {
            throw std::runtime_error(
                "DecideEngine::trainHead '" + name + "': example label '" +
                e.label + "' is not in the label set");
        }
        const std::vector<std::int32_t> ids = _tokenizer.encodeSingle(e.text);
        const std::vector<float> emb = _runner.embed(ids);
        if (emb.size() != D) {
            throw std::runtime_error("DecideEngine::trainHead: embedding dim mismatch");
        }
        features.insert(features.end(), emb.begin(), emb.end());
        y.push_back(it->second);
    }

    const LinearProbeTrainer::Result res =
        LinearProbeTrainer::train(features, D, y, labels.size(), cfg);

    DecisionHead::Spec spec{};
    spec.name        = name;
    spec.labels      = labels;
    spec.hidden      = D;
    spec.temperature = temperature > 0.0F ? temperature : 1.0F;
    spec.threshold   = threshold;
    spec.encoder     = "bge-m3";
    spec.weight      = res.weight;
    spec.bias        = res.bias;
    upsertHead(spec);   // persist + hot-reload (re-validates on disk)

    MM_LOG_INFO("decide",
                "trained head '{}' on {} examples ({} train / {} val): "
                "train_acc={:.3f} val_acc={:.3f} loss={:.4f}",
                name, examples.size(), res.nTrain, res.nVal,
                res.trainAccuracy, res.valAccuracy, res.finalLoss);

    TrainReport rep{};
    rep.name          = name;
    rep.labels        = labels;
    rep.nExamples     = examples.size();
    rep.nTrain        = res.nTrain;
    rep.nVal          = res.nVal;
    rep.trainAccuracy = res.trainAccuracy;
    rep.valAccuracy   = res.valAccuracy;
    rep.finalLoss     = res.finalLoss;
    rep.epochs        = res.epochsRun;
    return rep;
}

void DecideEngine::upsertHead(const DecisionHead::Spec& spec) {
    if (_headsDir.empty()) {
        throw std::runtime_error(
            "DecideEngine: no headsDir configured — cannot persist a pushed head");
    }
    // Path safety: the name becomes a directory under headsDir.
    if (spec.name.find('/') != std::string::npos ||
        spec.name.find('\\') != std::string::npos ||
        spec.name.find("..") != std::string::npos ||
        spec.name == "." || spec.name == "..") {
        throw std::runtime_error("DecideEngine: unsafe head name '" + spec.name + "'");
    }
    if (spec.hidden != _model.config().hidden) {
        throw std::runtime_error(
            "DecideEngine: head '" + spec.name + "' hidden " +
            std::to_string(spec.hidden) + " != encoder hidden " +
            std::to_string(_model.config().hidden));
    }

    namespace fs = std::filesystem;
    const fs::path root{_headsDir};
    const fs::path finalDir = root / spec.name;
    const fs::path tmpDir   = root / ("." + spec.name + ".tmp");

    // Write + validate-by-load in a temp dir, then atomically swap into place so
    // a bad push can never corrupt a live head's on-disk copy.
    fs::remove_all(tmpDir);
    DecisionHead::writeToDir(tmpDir, spec);
    DecisionHead loaded = DecisionHead::loadFromDir(tmpDir);   // re-validates on disk
    fs::remove_all(finalDir);
    fs::rename(tmpDir, finalDir);

    for (auto& h : _heads) {
        if (h.name() == spec.name) {
            h = std::move(loaded);
            MM_LOG_INFO("decide", "reloaded decision head '{}' ({} labels)",
                        spec.name, spec.labels.size());
            return;
        }
    }
    _heads.push_back(std::move(loaded));
    MM_LOG_INFO("decide", "installed decision head '{}' ({} labels)",
                spec.name, spec.labels.size());
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
