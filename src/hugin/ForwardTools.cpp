// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "hugin/ForwardTools.hpp"

#include "core/config/Config.hpp"
#include "hugin/LogitMath.hpp"
#include "runtime/InferenceEngine.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace mimirmind::hugin {

namespace {

using nlohmann::json;
using Engine = runtime::InferenceEngine;

json schema(json properties, json required = json::array()) {
    return {{"type", "object"}, {"properties", std::move(properties)}, {"required", std::move(required)}};
}

json textProps(json extra = json::object()) {
    json p{{"text", {{"type", "string"}, {"description", "raw text (no chat template is applied)"}}},
           {"tokens", {{"type", "array"}, {"items", {{"type", "integer"}}},
                       {"description", "token ids; used instead of text when given"}}},
           {"add_bos", {{"type", "boolean"}, {"description", "prepend BOS when tokenizing text (default true)"}}}};
    for (auto it = extra.begin(); it != extra.end(); ++it) {
        p[it.key()] = it.value();
    }
    return p;
}

std::vector<std::int32_t> inputTokens(const Engine& e, const json& a) {
    std::vector<std::int32_t> ids;
    if (a.contains("tokens")) {
        ids = a["tokens"].get<std::vector<std::int32_t>>();
    } else if (a.contains("text")) {
        ids = e.tokenizer().encode(a["text"].get<std::string>(), a.value("add_bos", true));
    } else {
        throw std::runtime_error("pass 'text' or 'tokens'");
    }
    if (ids.empty()) {
        throw std::runtime_error("input tokenizes to zero tokens");
    }
    return ids;
}

std::string piece(const Engine& e, std::int32_t id) {
    return std::string{e.tokenizer().tokenText(id)};
}

/// One stateless forward: empty cache in, empty cache out.
std::vector<std::vector<float>> forward(Engine& e, std::span<const std::int32_t> ids,
                                        Engine::ForwardProbe* probe = nullptr) {
    e.resetCache();
    auto logits = e.forwardVerify(ids, probe);
    e.resetCache();
    return logits;
}

std::vector<double> logProbs(const Engine& e, std::vector<float> logits) {
    LogitMath::softcap(logits, e.finalLogitSoftcap());
    return LogitMath::logSoftmax(logits);
}

json topJson(const Engine& e, const std::vector<double>& logp, std::size_t k) {
    json out = json::array();
    for (const std::int32_t id : LogitMath::topK(logp, k)) {
        out.push_back({{"id", id}, {"piece", piece(e, id)},
                       {"prob", std::exp(logp[static_cast<std::size_t>(id)])}});
    }
    return out;
}

/// Requested blocks (default: all), validated against the model.
std::vector<std::uint32_t> blockList(const Engine& e, const json& a) {
    const std::uint32_t n = e.config().blockCount;
    std::vector<std::uint32_t> out;
    if (a.contains("blocks")) {
        for (const auto& v : a["blocks"]) {
            const auto b = v.get<std::int64_t>();
            if (b < 0 || b >= static_cast<std::int64_t>(n)) {
                throw std::runtime_error("block " + std::to_string(b) + " out of range [0," +
                                         std::to_string(n) + ")");
            }
            out.push_back(static_cast<std::uint32_t>(b));
        }
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
    } else {
        for (std::uint32_t b = 0; b < n; ++b) {
            out.push_back(b);
        }
    }
    return out;
}

std::size_t positionArg(const json& a, std::size_t n) {
    if (!a.contains("position")) {
        return n - 1;
    }
    const auto p = a["position"].get<std::int64_t>();
    const auto q = p < 0 ? static_cast<std::int64_t>(n) + p : p;   // -1 = last
    if (q < 0 || q >= static_cast<std::int64_t>(n)) {
        throw std::runtime_error("position " + std::to_string(p) + " out of range for " +
                                 std::to_string(n) + " tokens");
    }
    return static_cast<std::size_t>(q);
}

/// Per-token log-probs of ids[1..] under `logits` (position i predicts i+1).
std::vector<double> tokenLogProbs(const Engine& e, const std::vector<std::vector<float>>& logits,
                                  std::span<const std::int32_t> ids) {
    std::vector<double> out;
    for (std::size_t i = 0; i + 1 < ids.size(); ++i) {
        const auto lp = logProbs(e, logits[i]);
        out.push_back(lp[static_cast<std::size_t>(ids[i + 1])]);
    }
    return out;
}

double perplexityOf(const std::vector<double>& lps) {
    if (lps.empty()) {
        return 0.0;
    }
    double nll = 0.0;
    for (double lp : lps) {
        nll -= lp;
    }
    return std::exp(nll / static_cast<double>(lps.size()));
}

} // namespace

struct ForwardTools::Loaded {
    std::string                 path;
    std::vector<std::string>    runes;
    core::config::Config        config;
    std::unique_ptr<Engine>     engine;
};

ForwardTools::ForwardTools(std::function<std::string()> modelPath)
    : _modelPath(std::move(modelPath)) {}

ForwardTools::~ForwardTools() = default;

std::unique_ptr<ForwardTools::Loaded>
ForwardTools::load(const std::string& path, const std::vector<std::string>& runes) const {
    auto l    = std::make_unique<Loaded>();
    l->path   = path;
    l->runes  = runes;
    l->engine = std::make_unique<Engine>(l->config);
    l->engine->setRunes(runes);
    std::cerr << "hugin: loading engine for '" << path << "'"
              << (runes.empty() ? std::string{} : " with " + std::to_string(runes.size()) + " rune(s)")
              << "\n";
    l->engine->loadModel(path);
    return l;
}

Engine& ForwardTools::base() {
    const std::string path = _modelPath();
    if (path.empty()) {
        throw std::runtime_error("no model loaded — call open_model first");
    }
    if (!_base || _base->path != path) {
        _base.reset();
        _variant.reset();
        _base = load(path, {});
    }
    return *_base->engine;
}

Engine& ForwardTools::withRunes(const std::vector<std::string>& runes) {
    (void)base();   // validates the path and keeps base/variant on the same model
    if (!_variant || _variant->path != _base->path || _variant->runes != runes) {
        _variant.reset();
        _variant = load(_base->path, runes);
    }
    return *_variant->engine;
}

void ForwardTools::registerOn(McpServer& server) {
    server.addTool(
        "tokenize", "Tokenize text with the model's own tokenizer: ids and pieces.",
        schema(textProps(), {}),
        [this](const json& a) {
            const Engine& e = base();
            const auto ids  = inputTokens(e, a);
            json pieces = json::array();
            for (const auto id : ids) {
                pieces.push_back(piece(e, id));
            }
            return json{{"count", ids.size()}, {"ids", ids}, {"pieces", pieces}};
        });

    server.addTool(
        "next_token",
        "Next-token distribution after the input (engine forward, final soft-cap applied): "
        "top-k tokens with probabilities and the entropy.",
        schema(textProps({{"top_k", {{"type", "integer"}}}}), {}),
        [this](const json& a) {
            Engine&    e   = base();
            const auto ids = inputTokens(e, a);
            const auto lg  = forward(e, ids);
            const auto lp  = logProbs(e, lg.back());
            return json{{"tokens", ids.size()},
                        {"entropy", LogitMath::entropy(lp)},
                        {"top", topJson(e, lp, a.value("top_k", std::size_t{10}))}};
        });

    server.addTool(
        "perplexity",
        "Teacher-forced log-prob of every token of the input given its prefix; mean NLL and "
        "perplexity. 'detail' adds the per-token table (logprob, rank, model's argmax).",
        schema(textProps({{"detail", {{"type", "boolean"}}}}), {}),
        [this](const json& a) {
            Engine&    e   = base();
            const auto ids = inputTokens(e, a);
            if (ids.size() < 2) {
                throw std::runtime_error("perplexity needs at least 2 tokens");
            }
            const auto lg  = forward(e, ids);
            const auto lps = tokenLogProbs(e, lg, ids);
            json out{{"tokens", ids.size()}, {"scored", lps.size()},
                     {"mean_nll", std::log(perplexityOf(lps))}, {"perplexity", perplexityOf(lps)}};
            if (a.value("detail", false)) {
                json rows = json::array();
                for (std::size_t i = 0; i + 1 < ids.size(); ++i) {
                    const auto lp  = logProbs(e, lg[i]);
                    const auto top = LogitMath::topK(lp, 1).front();
                    rows.push_back({{"pos", i + 1}, {"token", piece(e, ids[i + 1])},
                                    {"logprob", lps[i]},
                                    {"rank", LogitMath::rankOf(lp, static_cast<std::size_t>(ids[i + 1]))},
                                    {"argmax", piece(e, top)}});
                }
                out["per_token"] = rows;
            }
            return out;
        });

    server.addTool(
        "logit_lens",
        "Logit lens: after each block, project the residual stream at 'position' (default last) "
        "through the model's own final norm + lm_head. Per block: top-k tokens, entropy, and the "
        "probability/rank of the model's final prediction. 'blocks' limits the blocks (default all).",
        schema(textProps({{"blocks", {{"type", "array"}, {"items", {{"type", "integer"}}}}},
                          {"position", {{"type", "integer"}, {"description", "token index; negative counts from the end"}}},
                          {"top_k", {{"type", "integer"}}}}),
               {}),
        [this](const json& a) {
            Engine&    e   = base();
            const auto ids = inputTokens(e, a);
            Engine::ForwardProbe probe;
            probe.lensBlocks   = blockList(e, a);
            probe.lensPosition = positionArg(a, ids.size());
            const auto lg      = forward(e, ids, &probe);
            const auto finalLp = logProbs(e, lg[probe.lensPosition]);
            const auto finalId = LogitMath::topK(finalLp, 1).front();
            const std::size_t k = a.value("top_k", std::size_t{5});
            json blocks = json::array();
            for (const auto& [b, logits] : probe.lens) {
                const auto lp = logProbs(e, logits);
                blocks.push_back({{"block", b}, {"entropy", LogitMath::entropy(lp)},
                                  {"final_token_prob", std::exp(lp[static_cast<std::size_t>(finalId)])},
                                  {"final_token_rank", LogitMath::rankOf(lp, static_cast<std::size_t>(finalId))},
                                  {"top", topJson(e, lp, k)}});
            }
            return json{{"position", probe.lensPosition},
                        {"token_at_position", piece(e, ids[probe.lensPosition])},
                        {"final_prediction", {{"id", finalId}, {"piece", piece(e, finalId)},
                                              {"prob", std::exp(finalLp[static_cast<std::size_t>(finalId)])}}},
                        {"blocks", blocks}};
        });

    server.addTool(
        "activations",
        "Residual-stream statistics after each block at 'position' (default last): L2 norm, mean, "
        "std, abs-max, the update size and cosine vs the previous block, and the strongest "
        "dimensions. 'values': true also returns the raw vector (large).",
        schema(textProps({{"blocks", {{"type", "array"}, {"items", {{"type", "integer"}}}}},
                          {"position", {{"type", "integer"}}},
                          {"top_dims", {{"type", "integer"}}},
                          {"values", {{"type", "boolean"}}}}),
               {}),
        [this](const json& a) {
            Engine&    e   = base();
            const auto ids = inputTokens(e, a);
            const std::size_t pos = positionArg(a, ids.size());
            const std::size_t d   = e.config().embeddingLength;
            const auto wanted     = blockList(e, a);
            // Each requested block also needs its predecessor for the update size.
            Engine::ForwardProbe probe;
            for (const auto b : wanted) {
                probe.hiddenBlocks.push_back(b);
                if (b > 0) {
                    probe.hiddenBlocks.push_back(b - 1);
                }
            }
            (void)forward(e, ids, &probe);
            const std::size_t k = a.value("top_dims", std::size_t{8});
            const bool values   = a.value("values", false);
            const auto row = [&](std::uint32_t b) {
                const auto& h = probe.hidden.at(b);
                return std::span<const float>(h.data() + pos * d, d);
            };
            json blocks = json::array();
            for (const auto b : wanted) {
                const auto x = row(b);
                double sum = 0, sq = 0, amax = 0;
                std::vector<double> mag(d);
                for (std::size_t i = 0; i < d; ++i) {
                    const double v = x[i];
                    sum += v;
                    sq += v * v;
                    mag[i] = std::fabs(v);
                    amax   = std::max(amax, mag[i]);
                }
                const double mean = sum / static_cast<double>(d);
                json dims = json::array();
                for (const auto i : LogitMath::topK(mag, k)) {
                    dims.push_back({{"dim", i}, {"value", x[static_cast<std::size_t>(i)]}});
                }
                json o{{"block", b}, {"l2", std::sqrt(sq)}, {"mean", mean},
                       {"std", std::sqrt(std::max(0.0, sq / static_cast<double>(d) - mean * mean))},
                       {"abs_max", amax}, {"top_dims", dims}};
                if (b > 0) {
                    const auto p = row(b - 1);
                    double dsq = 0, dot = 0, psq = 0;
                    for (std::size_t i = 0; i < d; ++i) {
                        const double xv = x[i];
                        const double pv = p[i];
                        dsq += (xv - pv) * (xv - pv);
                        dot += xv * pv;
                        psq += pv * pv;
                    }
                    o["update_l2"]      = std::sqrt(dsq);
                    o["cos_prev_block"] = (sq > 0 && psq > 0) ? dot / std::sqrt(sq * psq) : 0.0;
                }
                if (values) {
                    o["values"] = std::vector<float>(x.begin(), x.end());
                }
                blocks.push_back(std::move(o));
            }
            return json{{"position", pos}, {"token_at_position", piece(e, ids[pos])},
                        {"d_model", d}, {"blocks", blocks}};
        });

    server.addTool(
        "rune_compare",
        "Run the same input through the base model and through the model with the given runes "
        "applied by Futhark at load (exactly the serving path). Reports per-position "
        "KL(base||rune), top-1 agreement, perplexity of both, and the last-position top-k of both.",
        schema(textProps({{"runes", {{"type", "array"}, {"items", {{"type", "string"}}}}},
                          {"top_k", {{"type", "integer"}}}}),
               {"runes"}),
        [this](const json& a) {
            const auto runes = a.at("runes").get<std::vector<std::string>>();
            if (runes.empty()) {
                throw std::runtime_error("'runes' must not be empty");
            }
            Engine&    b   = base();
            const auto ids = inputTokens(b, a);
            const auto lb  = forward(b, ids);
            Engine&    r   = withRunes(runes);
            const auto lr  = forward(r, ids);
            json   positions = json::array();
            double klSum = 0, klMax = 0;
            std::size_t agree = 0;
            for (std::size_t i = 0; i < ids.size(); ++i) {
                const auto pb = logProbs(b, lb[i]);
                const auto pr = logProbs(r, lr[i]);
                const double kl = LogitMath::kl(pb, pr);
                const auto tb = LogitMath::topK(pb, 1).front();
                const auto tr = LogitMath::topK(pr, 1).front();
                klSum += kl;
                klMax = std::max(klMax, kl);
                agree += tb == tr ? 1 : 0;
                positions.push_back({{"pos", i}, {"token", piece(b, ids[i])}, {"kl", kl},
                                     {"top1_base", piece(b, tb)}, {"top1_rune", piece(r, tr)}});
            }
            const std::size_t k = a.value("top_k", std::size_t{5});
            json out{{"tokens", ids.size()},
                     {"mean_kl", klSum / static_cast<double>(ids.size())},
                     {"max_kl", klMax},
                     {"top1_agreement", static_cast<double>(agree) / static_cast<double>(ids.size())},
                     {"last_top_base", topJson(b, logProbs(b, lb.back()), k)},
                     {"last_top_rune", topJson(r, logProbs(r, lr.back()), k)},
                     {"positions", positions}};
            if (ids.size() >= 2) {
                out["perplexity_base"] = perplexityOf(tokenLogProbs(b, lb, ids));
                out["perplexity_rune"] = perplexityOf(tokenLogProbs(r, lr, ids));
            }
            return out;
        });
}

} // namespace mimirmind::hugin
