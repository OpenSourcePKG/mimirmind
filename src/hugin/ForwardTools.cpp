// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "hugin/ForwardTools.hpp"

#include "core/config/Config.hpp"
#include "hugin/LogitMath.hpp"
#include "runtime/InferenceEngine.hpp"
#include "runtime/nvfp4/ModelFormatResolver.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
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

std::size_t nonFiniteCount(const std::vector<double>& v) {
    return static_cast<std::size_t>(
        std::count_if(v.begin(), v.end(), [](double x) { return !std::isfinite(x); }));
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

/// Teacher-forced score of `cont` after `prompt` in one forward.
struct ContinuationScore {
    double      logprob{0};
    bool        greedyMatch{true};   // every continuation token is the argmax
    std::size_t firstRank{0};
    std::size_t tokens{0};
};

ContinuationScore scoreContinuation(Engine& e, const std::vector<std::int32_t>& prompt,
                                    const std::vector<std::int32_t>& cont) {
    if (prompt.empty() || cont.empty()) {
        throw std::runtime_error("eval item needs a non-empty prompt and continuation");
    }
    std::vector<std::int32_t> ids = prompt;
    ids.insert(ids.end(), cont.begin(), cont.end());
    const auto lg = forward(e, ids);
    ContinuationScore sc;
    sc.tokens = cont.size();
    for (std::size_t j = 0; j < cont.size(); ++j) {
        const auto lp  = logProbs(e, lg[prompt.size() - 1 + j]);
        const auto tok = static_cast<std::size_t>(cont[j]);
        sc.logprob += lp[tok];
        const std::size_t rank = LogitMath::rankOf(lp, tok);
        if (j == 0) {
            sc.firstRank = rank;
        }
        sc.greedyMatch = sc.greedyMatch && rank == 0;
    }
    return sc;
}

/// Goldset items: inline array or a .json (array) / .jsonl file.
json loadGoldset(const json& a) {
    if (a.contains("items")) {
        return a["items"];
    }
    if (!a.contains("path")) {
        throw std::runtime_error("pass 'items' or 'path'");
    }
    const std::string path = a["path"].get<std::string>();
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("cannot open goldset '" + path + "'");
    }
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string text = ss.str();
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first != std::string::npos && text[first] == '[') {
        return json::parse(text);
    }
    json items = json::array();
    std::istringstream lines(text);
    std::string line;
    while (std::getline(lines, line)) {
        if (line.find_first_not_of(" \t\r") != std::string::npos) {
            items.push_back(json::parse(line));
        }
    }
    return items;
}

/// One eval item on one engine: {"prompt", "expect"} (exact continuation) or
/// {"prompt", "choices":[...], "answer": idx} (multiple choice by logprob).
json evalItem(Engine& e, const json& item, bool addBos) {
    const auto prompt = e.tokenizer().encode(item.at("prompt").get<std::string>(), addBos);
    const auto encodeCont = [&](const std::string& t) { return e.tokenizer().encode(t, false); };
    if (item.contains("choices")) {
        const auto choices = item["choices"].get<std::vector<std::string>>();
        const auto answer  = item.at("answer").get<std::size_t>();
        if (answer >= choices.size()) {
            throw std::runtime_error("eval item 'answer' out of range");
        }
        std::vector<double> lps;
        for (const auto& c : choices) {
            lps.push_back(scoreContinuation(e, prompt, encodeCont(c)).logprob);
        }
        const auto best = static_cast<std::size_t>(
            std::max_element(lps.begin(), lps.end()) - lps.begin());
        return json{{"kind", "choice"}, {"correct", best == answer}, {"picked", best},
                    {"logprob", lps[answer]}, {"choice_logprobs", lps}};
    }
    const auto sc = scoreContinuation(e, prompt, encodeCont(item.at("expect").get<std::string>()));
    return json{{"kind", "expect"}, {"correct", sc.greedyMatch}, {"logprob", sc.logprob},
                {"tokens", sc.tokens}, {"first_rank", sc.firstRank}};
}

json evalRun(Engine& e, const json& items, bool addBos, json& rows) {
    std::size_t correct = 0, nonFinite = 0;
    double      lpSum   = 0;
    for (std::size_t i = 0; i < items.size(); ++i) {
        json r = evalItem(e, items[i], addBos);
        correct += r["correct"].get<bool>() ? 1U : 0U;
        const double lp = r["logprob"].get<double>();
        if (std::isfinite(lp)) {
            lpSum += lp;
        } else {
            ++nonFinite;   // NaN/Inf logits from the forward: report, don't poison the mean
        }
        if (rows.size() <= i) {
            rows.push_back(json{{"index", i}, {"id", items[i].contains("id") ? items[i]["id"] : json(i)}});
        }
        rows[i]["base_or_run"] = std::move(r);
    }
    const double n = static_cast<double>(std::max<std::size_t>(1, items.size()));
    const double finiteN = static_cast<double>(std::max<std::size_t>(1, items.size() - nonFinite));
    return json{{"items", items.size()}, {"accuracy", static_cast<double>(correct) / n},
                {"mean_logprob", lpSum / finiteN}, {"non_finite_items", nonFinite}};
}

} // namespace

struct ForwardTools::Loaded {
    Target                      target;
    std::vector<std::string>    runes;
    core::config::Config        config;
    std::unique_ptr<Engine>     engine;
};

ForwardTools::ForwardTools(std::function<Target()> target) : _target(std::move(target)) {}

ForwardTools::~ForwardTools() = default;

std::unique_ptr<ForwardTools::Loaded>
ForwardTools::load(const Target& t, const std::vector<std::string>& runes) const {
    auto l    = std::make_unique<Loaded>();
    l->target = t;
    l->runes  = runes;
    l->engine = std::make_unique<Engine>(l->config);
    l->engine->setRunes(runes);
    std::cerr << "hugin: loading engine for '" << t.path << "'"
              << (runes.empty() ? std::string{} : " with " + std::to_string(runes.size()) + " rune(s)")
              << "\n";
    if (runtime::nvfp4::resolveModelFormat(core::config::ModelFormat::Auto, t.path) ==
        core::config::ModelFormat::Nvfp4) {
        l->engine->loadModelNvfp4(t.path, t.tokenizerGguf);
    } else {
        l->engine->loadModel(t.path);
    }
    return l;
}

namespace {
bool sameTarget(const ForwardTools::Target& a, const ForwardTools::Target& b) {
    return a.path == b.path && a.tokenizerGguf == b.tokenizerGguf;
}
} // namespace

Engine& ForwardTools::base() {
    const Target t = _target();
    if (t.path.empty()) {
        throw std::runtime_error("no model loaded — call open_model first");
    }
    if (!_base || !sameTarget(_base->target, t)) {
        _base.reset();
        _variant.reset();
        _base = load(t, {});
    }
    return *_base->engine;
}

Engine& ForwardTools::withRunes(const std::vector<std::string>& runes) {
    (void)base();   // validates the target and keeps base/variant on the same model
    if (!_variant || !sameTarget(_variant->target, _base->target) || _variant->runes != runes) {
        // One engine at a time besides the base: drop the old variant first so
        // two large models are never resident at once beyond base + variant.
        _variant.reset();
        _variant = load(_base->target, runes);
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
            // non_finite > 0 means the forward produced NaN/Inf logits — an
            // engine defect worth reporting, not a property of the text.
            json out{{"tokens", ids.size()}, {"scored", lps.size()},
                     {"mean_nll", std::log(perplexityOf(lps))}, {"perplexity", perplexityOf(lps)},
                     {"non_finite", nonFiniteCount(lps)}};
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
                // Statistics over finite values; non-finite ones are counted
                // (a NaN/Inf residual is itself a finding worth reporting).
                double sum = 0, sq = 0, amax = 0;
                std::size_t nonFinite = 0;
                std::vector<double> mag(d, 0.0);
                for (std::size_t i = 0; i < d; ++i) {
                    const double v = x[i];
                    if (!std::isfinite(v)) {
                        ++nonFinite;
                        continue;
                    }
                    sum += v;
                    sq += v * v;
                    mag[i] = std::fabs(v);
                    amax   = std::max(amax, mag[i]);
                }
                const double finiteN = static_cast<double>(std::max<std::size_t>(1, d - nonFinite));
                const double mean = sum / finiteN;
                json dims = json::array();
                for (const auto i : LogitMath::topK(mag, k)) {
                    dims.push_back({{"dim", i}, {"value", x[static_cast<std::size_t>(i)]}});
                }
                json o{{"block", b}, {"l2", std::sqrt(sq)}, {"mean", mean},
                       {"std", std::sqrt(std::max(0.0, sq / finiteN - mean * mean))},
                       {"abs_max", amax}, {"non_finite", nonFinite}, {"top_dims", dims}};
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
    server.addTool(
        "attention_pattern",
        "Attention weights of one block for the query at 'position' (default last), rebuilt from "
        "the engine's own post-RoPE Q/K (stage tap) with the kernel's exact scale. Per head: "
        "entropy, weight on BOS and on itself, top-k attended tokens. 'consistency' compares "
        "weights x V against the attention output the kernel produced (proves the pattern). "
        "Supported where the backend exposes the stages (llama / qwen2 today).",
        schema(textProps({{"block", {{"type", "integer"}}},
                          {"heads", {{"type", "array"}, {"items", {{"type", "integer"}}}}},
                          {"position", {{"type", "integer"}}},
                          {"top_k", {{"type", "integer"}}},
                          {"full", {{"type", "boolean"}, {"description", "include the full weight row per head"}}}}),
               {"block"}),
        [this](const json& a) {
            Engine&    e   = base();
            const auto ids = inputTokens(e, a);
            const auto blk = blockList(e, json{{"blocks", json::array({a.at("block")})}}).front();
            const std::size_t pos = positionArg(a, ids.size());
            Engine::ForwardProbe probe;
            probe.stages      = {"Qcur", "Kcur", "Vcur", "attn_ctx", "attn_scale"};
            probe.stageBlocks = {blk};
            (void)forward(e, ids, &probe);
            const auto get = [&](const char* name) -> const Engine::ForwardProbe::Stage& {
                const auto it = probe.stageData.find({name, blk});
                if (it == probe.stageData.end()) {
                    throw std::runtime_error("architecture '" + e.config().architecture +
                                             "' does not expose stage '" + name +
                                             "' (attention_pattern supports llama/qwen2 today)");
                }
                return it->second;
            };
            const auto& Q = get("Qcur");
            const auto& K = get("Kcur");
            const auto& V = get("Vcur");
            const auto& C = get("attn_ctx");
            const double scale = get("attn_scale").data.at(0);
            const std::size_t H   = e.config().headCount;
            const std::size_t Hkv = e.config().headCountKv;
            const std::size_t hd  = e.config().headDim();
            if (Q.cols != H * hd || K.cols != Hkv * hd || V.cols != Hkv * hd || C.cols != H * hd ||
                Q.rows != ids.size() || K.rows < ids.size()) {
                throw std::runtime_error("unexpected stage shapes for attention_pattern");
            }
            std::vector<std::size_t> heads;
            if (a.contains("heads")) {
                for (const auto& v : a["heads"]) {
                    const auto h = v.get<std::size_t>();
                    if (h >= H) throw std::runtime_error("head " + std::to_string(h) + " out of range");
                    heads.push_back(h);
                }
            } else {
                for (std::size_t h = 0; h < H; ++h) heads.push_back(h);
            }
            const std::size_t k = a.value("top_k", std::size_t{5});
            const bool full     = a.value("full", false);
            const std::size_t n = pos + 1;   // causal: keys 0..pos
            double maxErr = 0, maxRef = 0;
            std::vector<double> avg(n, 0.0);
            json out = json::array();
            for (const auto h : heads) {
                const std::size_t kvh = h / (H / Hkv);
                const float* q = Q.data.data() + pos * Q.cols + h * hd;
                std::vector<double> w(n);
                double mx = -1e300;
                for (std::size_t j = 0; j < n; ++j) {
                    const float* kr = K.data.data() + j * K.cols + kvh * hd;
                    double dot = 0;
                    for (std::size_t d = 0; d < hd; ++d) dot += static_cast<double>(q[d]) * static_cast<double>(kr[d]);
                    w[j] = dot * scale;
                    mx = std::max(mx, w[j]);
                }
                double z = 0;
                for (auto& x : w) { x = std::exp(x - mx); z += x; }
                double ent = 0;
                for (std::size_t j = 0; j < n; ++j) {
                    w[j] /= z;
                    avg[j] += w[j] / static_cast<double>(heads.size());
                    if (w[j] > 0) ent -= w[j] * std::log(w[j]);
                }
                for (std::size_t d = 0; d < hd; ++d) {
                    double ctx = 0;
                    for (std::size_t j = 0; j < n; ++j) ctx += w[j] * static_cast<double>(V.data[j * V.cols + kvh * hd + d]);
                    const double ref = C.data[pos * C.cols + h * hd + d];
                    maxErr = std::max(maxErr, std::fabs(ctx - ref));
                    maxRef = std::max(maxRef, std::fabs(ref));
                }
                json top = json::array();
                for (const auto j : LogitMath::topK(w, k)) {
                    top.push_back({{"pos", j}, {"token", piece(e, ids[static_cast<std::size_t>(j)])},
                                   {"weight", w[static_cast<std::size_t>(j)]}});
                }
                json o{{"head", h}, {"kv_head", kvh}, {"entropy", ent}, {"bos_weight", w[0]},
                       {"self_weight", w[pos]}, {"top", top}};
                if (full) o["weights"] = w;
                out.push_back(std::move(o));
            }
            json avgTop = json::array();
            for (const auto j : LogitMath::topK(avg, k)) {
                avgTop.push_back({{"pos", j}, {"token", piece(e, ids[static_cast<std::size_t>(j)])},
                                  {"weight", avg[static_cast<std::size_t>(j)]}});
            }
            return json{{"block", blk}, {"position", pos}, {"query_token", piece(e, ids[pos])},
                        {"heads", H}, {"kv_heads", Hkv}, {"head_dim", hd}, {"scale", scale},
                        {"consistency", {{"max_abs_err", maxErr}, {"max_abs_ref", maxRef}}},
                        {"head_average_top", avgTop}, {"per_head", out}};
        });

    server.addTool(
        "eval",
        "Forward-only goldset eval (teacher-forced, no sampling). Items: {prompt, expect} -> "
        "correct iff every expected token is the argmax; or {prompt, choices[], answer} -> "
        "correct iff the answer has the highest total logprob. Goldset inline ('items') or a "
        ".json/.jsonl file ('path'). With 'runes' the same set also runs on the rune-applied "
        "model and the deltas are reported.",
        schema({{"items", {{"type", "array"}, {"items", {{"type", "object"}}}}},
                {"path", {{"type", "string"}}},
                {"runes", {{"type", "array"}, {"items", {{"type", "string"}}}}},
                {"add_bos", {{"type", "boolean"}}},
                {"detail", {{"type", "boolean"}}}}),
        [this](const json& a) {
            const json items = loadGoldset(a);
            if (!items.is_array() || items.empty()) {
                throw std::runtime_error("goldset is empty");
            }
            const bool addBos = a.value("add_bos", true);
            json rows = json::array();
            json baseRows = json::array();
            const json baseSum = evalRun(base(), items, addBos, baseRows);
            json out{{"base", baseSum}};
            for (auto& r : baseRows) {
                rows.push_back(json{{"index", r["index"]}, {"id", r["id"]}, {"base", r["base_or_run"]}});
            }
            if (a.contains("runes")) {
                const auto runes = a["runes"].get<std::vector<std::string>>();
                json runeRows = json::array();
                const json runeSum = evalRun(withRunes(runes), items, addBos, runeRows);
                out["rune"] = runeSum;
                out["delta_accuracy"] = runeSum["accuracy"].get<double>() - baseSum["accuracy"].get<double>();
                out["delta_mean_logprob"] = runeSum["mean_logprob"].get<double>() - baseSum["mean_logprob"].get<double>();
                json flips = json::array();
                for (std::size_t i = 0; i < rows.size(); ++i) {
                    rows[i]["rune"] = runeRows[i]["base_or_run"];
                    if (rows[i]["base"]["correct"] != rows[i]["rune"]["correct"]) {
                        flips.push_back({{"index", i}, {"id", rows[i]["id"]},
                                         {"now_correct", rows[i]["rune"]["correct"]}});
                    }
                }
                out["flips"] = flips;
            }
            if (a.value("detail", false)) {
                out["per_item"] = rows;
            }
            return out;
        });
    server.addTool(
        "expert_usage",
        "MoE routing actually taken by the engine (stage tap at the router's selection, so it "
        "reflects bias / device top-K exactly). Per block: distinct experts used, load entropy, "
        "top experts by token count and by routing weight; 'position' adds that token's picks. "
        "Supported: qwen35moe-family and kolibri1 backends.",
        schema(textProps({{"blocks", {{"type", "array"}, {"items", {{"type", "integer"}}}}},
                          {"position", {{"type", "integer"}}},
                          {"top_k", {{"type", "integer"}}}}),
               {}),
        [this](const json& a) {
            Engine&    e   = base();
            const auto ids = inputTokens(e, a);
            Engine::ForwardProbe probe;
            probe.stages = {"moe_topk_idx", "moe_topk_w"};
            if (a.contains("blocks")) {
                probe.stageBlocks = blockList(e, a);
            }
            (void)forward(e, ids, &probe);
            const std::size_t topN = a.value("top_k", std::size_t{8});
            const std::size_t nExperts = e.config().expertCount;
            std::optional<std::size_t> pos;
            if (a.contains("position")) {
                pos = positionArg(a, ids.size());
            }
            json blocks = json::array();
            std::vector<double> totalCount(nExperts, 0.0);
            for (const auto& [key, st] : probe.stageData) {
                if (key.first != "moe_topk_idx") {
                    continue;
                }
                const auto wIt = probe.stageData.find({"moe_topk_w", key.second});
                const std::size_t T = st.rows, K = st.cols;
                std::vector<double> count(nExperts, 0.0), weight(nExperts, 0.0);
                for (std::size_t t = 0; t < T; ++t) {
                    for (std::size_t k = 0; k < K; ++k) {
                        const auto ex = static_cast<std::size_t>(st.data[t * K + k]);
                        if (ex >= nExperts) {
                            throw std::runtime_error("expert id out of range in block " +
                                                     std::to_string(key.second));
                        }
                        count[ex] += 1.0;
                        totalCount[ex] += 1.0;
                        if (wIt != probe.stageData.end()) {
                            weight[ex] += static_cast<double>(wIt->second.data[t * K + k]);
                        }
                    }
                }
                std::size_t distinct = 0;
                double ent = 0;
                const double picks = static_cast<double>(T * K);
                for (const double c : count) {
                    if (c > 0) {
                        ++distinct;
                        ent -= (c / picks) * std::log(c / picks);
                    }
                }
                json byCount = json::array(), byWeight = json::array();
                for (const auto ex : LogitMath::topK(count, topN)) {
                    if (count[static_cast<std::size_t>(ex)] > 0)
                        byCount.push_back({{"expert", ex}, {"tokens", count[static_cast<std::size_t>(ex)]}});
                }
                for (const auto ex : LogitMath::topK(weight, topN)) {
                    if (weight[static_cast<std::size_t>(ex)] > 0)
                        byWeight.push_back({{"expert", ex}, {"weight", weight[static_cast<std::size_t>(ex)]}});
                }
                json b{{"block", key.second}, {"tokens", T}, {"k", K}, {"distinct_experts", distinct},
                       {"load_entropy", ent},
                       {"max_entropy", std::log(static_cast<double>(std::min<std::size_t>(nExperts, T * K)))},
                       {"top_by_count", byCount}, {"top_by_weight", byWeight}};
                if (pos) {
                    json picksAt = json::array();
                    for (std::size_t k = 0; k < K; ++k) {
                        json p{{"expert", static_cast<std::int64_t>(st.data[*pos * K + k])}};
                        if (wIt != probe.stageData.end()) p["weight"] = wIt->second.data[*pos * K + k];
                        picksAt.push_back(std::move(p));
                    }
                    b["picks_at_position"] = picksAt;
                }
                blocks.push_back(std::move(b));
            }
            if (blocks.empty()) {
                throw std::runtime_error("no MoE routing captured — '" + e.config().architecture +
                                         "' is dense or its backend does not expose the routing "
                                         "tap yet (supported: qwen35moe-family, kolibri1)");
            }
            json overall = json::array();
            for (const auto ex : LogitMath::topK(totalCount, topN)) {
                overall.push_back({{"expert", ex}, {"tokens", totalCount[static_cast<std::size_t>(ex)]}});
            }
            std::size_t neverUsed = 0;
            for (const double c : totalCount) neverUsed += c == 0 ? 1U : 0U;
            json out{{"tokens", ids.size()}, {"experts", nExperts}, {"moe_blocks", blocks.size()},
                     {"never_used_experts", neverUsed}, {"top_overall", overall}, {"blocks", blocks}};
            if (pos) {
                out["position"] = *pos;
                out["token_at_position"] = piece(e, ids[*pos]);
            }
            return out;
        });
}

} // namespace mimirmind::hugin
