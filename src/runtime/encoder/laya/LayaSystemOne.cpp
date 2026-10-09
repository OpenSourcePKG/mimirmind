// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/encoder/laya/LayaSystemOne.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace mimirmind::runtime::encoder::laya {

namespace {

std::string replaceAll(std::string s, const std::string& from, const std::string& to) {
    if (from.empty()) {
        return s;
    }
    std::size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
    return s;
}

const char* qtypeName(LayaSystemOne::QType t) {
    switch (t) {
    case LayaSystemOne::QType::Choice: return "choice";
    case LayaSystemOne::QType::Score:  return "score";
    case LayaSystemOne::QType::Noul:   return "noul";
    }
    return "choice";
}

} // namespace

LayaSystemOne::LayaSystemOne(compute::ComputeOps& ops, compute::ComputeMatmul& matmul,
                             std::size_t maxLen, std::size_t headMaxLen)
    : _ops{ops}, _mm{matmul}, _maxLen{maxLen}, _headMaxLen{headMaxLen} {}

void LayaSystemOne::load(std::string_view modelDir, core::gguf::GgmlType matmulType,
                         std::size_t nHeadLayers) {
    _model.load(modelDir, _ops, matmulType, "encoder.");
    _head.load(modelDir, _ops, _model.config().hidden, nHeadLayers, matmulType);
    _tok.load(modelDir);
    _enc = std::make_unique<LayaEncoderRunner>(_model, _ops, _mm);
    _dec = std::make_unique<LayaDecisionRunner>(_model, *_enc, _head, _ops, _mm);
}

std::vector<std::string> LayaSystemOne::renderOptions(const Question& q) {
    std::vector<std::string> out;
    switch (q.type) {
    case QType::Choice:
        for (const auto& [key, desc] : q.choice) {
            out.push_back(desc.empty() ? key : (key + ": " + desc));
        }
        break;
    case QType::Score:
        for (std::size_t i = 0; i < q.levels.size(); ++i) {
            out.push_back("level " + std::to_string(i) + ": " + q.levels[i]);
        }
        break;
    case QType::Noul:
        out.push_back(q.noulFalseLabel + ": " +
                      (q.noulFalse.empty() ? "no, the statement does not hold" : q.noulFalse));
        out.push_back(q.noulTrueLabel + ": " +
                      (q.noulTrue.empty() ? "yes, the statement holds" : q.noulTrue));
        break;
    }
    return out;
}

LayaSystemOne::Answer LayaSystemOne::decide(std::string_view state, const Question& q) {
    if (_enc == nullptr || _dec == nullptr) {
        throw std::runtime_error("LayaSystemOne::decide: not loaded");
    }
    const std::string maskText = "<mask>";

    // ---- head: "<t> question: <instructions>" ----
    const std::string headText =
        std::string(qtypeName(q.type)) + " question: " + replaceAll(q.instructions, maskText, " ");
    std::vector<std::int32_t> headIds = _tok.encode(headText);

    // ---- options: [mask] + first-48 tokens of (" " + optText) ----
    const std::vector<std::string> opts = renderOptions(q);
    const std::size_t nOpts = opts.size();
    if (nOpts == 0) {
        throw std::runtime_error("LayaSystemOne::decide: question has no options");
    }
    std::vector<std::vector<std::int32_t>> optIds(nOpts);
    for (std::size_t o = 0; o < nOpts; ++o) {
        std::vector<std::int32_t> t = _tok.encode(" " + replaceAll(opts[o], maskText, " "));
        if (t.size() > 48) {
            t.resize(48);
        }
        optIds[o].push_back(_tok.maskId());
        optIds[o].insert(optIds[o].end(), t.begin(), t.end());
    }

    // ---- head budget ----
    auto optSum = [&]() {
        std::size_t s = 0;
        for (const auto& o : optIds) s += o.size();
        return s;
    };
    long optBudget = static_cast<long>(_headMaxLen) - static_cast<long>(optSum());
    if (optBudget < 16) {
        const std::size_t per =
            std::max<std::size_t>(4, (_headMaxLen - 16) / std::max<std::size_t>(1, nOpts));
        for (auto& o : optIds) {
            if (o.size() > per) o.resize(per);
        }
        optBudget = static_cast<long>(_headMaxLen) - static_cast<long>(optSum());
    }
    const std::size_t headKeep = std::max<long>(8, optBudget);
    if (headIds.size() > headKeep) {
        headIds.resize(headKeep);
    }

    // ---- assemble head: [cls] head [sep] ([mask] opt)* [sep] ----
    std::vector<std::int32_t> ids;
    ids.push_back(_tok.clsId());
    ids.insert(ids.end(), headIds.begin(), headIds.end());
    ids.push_back(_tok.sepId());
    std::vector<std::int32_t> markers;
    markers.reserve(nOpts);
    for (const auto& o : optIds) {
        markers.push_back(static_cast<std::int32_t>(ids.size()));
        ids.insert(ids.end(), o.begin(), o.end());
    }
    ids.push_back(_tok.sepId());

    // ---- state (mask->space), clamped to room, then closing [sep] ----
    const long room = std::max<long>(0, static_cast<long>(_maxLen) - static_cast<long>(ids.size()) - 1);
    std::vector<std::int32_t> stateIds = _tok.encode(replaceAll(std::string(state), maskText, " "));
    if (stateIds.size() > static_cast<std::size_t>(room)) {
        stateIds.resize(static_cast<std::size_t>(room));   // truncate_left=false
    }
    ids.insert(ids.end(), stateIds.begin(), stateIds.end());
    ids.push_back(_tok.sepId());
    if (ids.size() > _maxLen) {
        ids.resize(_maxLen);
    }
    // drop markers past max_len (every marker is in the head, well within max_len here).
    std::vector<std::int32_t> keptMarkers;
    for (std::int32_t m : markers) {
        if (static_cast<std::size_t>(m) < _maxLen) keptMarkers.push_back(m);
    }

    // ---- decide (backbone + head) ----
    const auto res = _dec->decide(ids, keptMarkers, static_cast<std::int32_t>(q.type));
    const std::size_t k = res.logits.size();

    // ---- decode: softmax over option logits (temperature == 1 for the shipped checkpoints) ----
    std::vector<float> p(k);
    {
        const float mx = *std::max_element(res.logits.begin(), res.logits.end());
        double sum = 0.0;
        for (std::size_t o = 0; o < k; ++o) { p[o] = std::exp(res.logits[o] - mx); sum += p[o]; }
        for (std::size_t o = 0; o < k; ++o) { p[o] = static_cast<float>(p[o] / sum); }
    }
    const std::size_t argmax =
        static_cast<std::size_t>(std::max_element(p.begin(), p.end()) - p.begin());

    Answer a;
    a.type = q.type;
    a.actProbability = res.actProbability;
    a.answerConfidence = *std::max_element(p.begin(), p.end());   // max(p)

    // confidence_from_probs = 1 - H(p)/log(k) for k>=2 else 1.
    float confEntropy = 1.0F;
    if (k >= 2) {
        double ent = 0.0;
        for (std::size_t o = 0; o < k; ++o) {
            ent -= static_cast<double>(p[o]) * std::log(std::max(p[o], 1e-12F));
        }
        confEntropy = static_cast<float>(std::clamp(1.0 - ent / std::log(static_cast<double>(k)), 0.0, 1.0));
    }

    if (q.type == QType::Choice) {
        a.choice = q.choice[argmax].first;
        for (std::size_t o = 0; o < k && o < q.choice.size(); ++o) {
            a.probabilities.emplace_back(q.choice[o].first, p[o]);
        }
        a.confidence = confEntropy;
    } else if (q.type == QType::Score) {
        double exp = 0.0;
        for (std::size_t o = 0; o < k; ++o) exp += static_cast<double>(o) * p[o];
        a.score = static_cast<float>(exp);
        for (std::size_t o = 0; o < k; ++o) a.probabilities.emplace_back(std::to_string(o), p[o]);
        a.confidence = confEntropy;
    } else {   // Noul: p = [false, true]; noul = P(true)
        a.noul = (k >= 2) ? p[1] : p[0];
        a.confidence = std::max(a.noul, 1.0F - a.noul);
        a.probabilities.emplace_back("false", (k >= 1) ? p[0] : 0.0F);
        a.probabilities.emplace_back("true", (k >= 2) ? p[1] : 0.0F);
    }
    return a;
}

} // namespace mimirmind::runtime::encoder::laya
