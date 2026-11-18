// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "hugin/ForwardSupport.hpp"

#include "hugin/GenerationCheck.hpp"
#include "hugin/LogitMath.hpp"
#include "model/ChatTemplate.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace mimirmind::hugin::forward_support {

using nlohmann::json;

json schema(json properties, json required) {
    return {{"type", "object"}, {"properties", std::move(properties)}, {"required", std::move(required)}};
}

json textProps(json extra) {
    json p{{"text", {{"type", "string"}, {"description", "raw text (no chat template is applied; special "
                                                        "markup such as <|im_start|> maps to its token id)"}}},
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
        ids = encodeText(e, a["text"].get<std::string>(), a.value("add_bos", true));
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
                                        Engine::ForwardProbe* probe) {
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

std::vector<std::int32_t> encodeText(const Engine& e, std::string_view text, bool addBos) {
    return e.tokenizer().encode(text, addBos, /*parseSpecial=*/true);
}

std::vector<std::int32_t> textTokens(const Engine& e, const std::string& text, bool addBos) {
    auto ids = encodeText(e, text, addBos);
    if (ids.empty()) {
        throw std::runtime_error("input tokenizes to zero tokens");
    }
    return ids;
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
ContinuationScore scoreContinuation(Engine& e, const std::vector<std::int32_t>& prompt,
                                    const std::vector<std::int32_t>& cont,
                                    const ProbeSetup* setup) {
    if (prompt.empty() || cont.empty()) {
        throw std::runtime_error("eval item needs a non-empty prompt and continuation");
    }
    std::vector<std::int32_t> ids = prompt;
    ids.insert(ids.end(), cont.begin(), cont.end());
    Engine::ForwardProbe probe;
    if (setup != nullptr && *setup) {
        (*setup)(probe, ids.size());
    }
    const auto lg = forward(e, ids, (setup != nullptr && *setup) ? &probe : nullptr);
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

std::vector<std::int32_t> generateGreedy(Engine& e, const std::vector<std::int32_t>& prompt,
                                         std::size_t maxTokens, const ProbeSetup* setup) {
    if (prompt.empty()) {
        throw std::runtime_error("generate item needs a non-empty prompt");
    }
    std::vector<std::int32_t> stops = model::ChatTemplate::stopIds(
        model::ChatTemplate::detectFromArch(e.config().architecture), e.tokenizer());
    stops.push_back(e.tokenizer().eosId());
    const auto isStop = [&](std::int32_t t) { return std::find(stops.begin(), stops.end(), t) != stops.end(); };

    std::vector<std::int32_t> out;
    if (setup == nullptr || !*setup) {
        runtime::GenerateParams p;
        p.maxNewTokens = maxTokens;   // sampling default = temperature 0 = argmax
        p.stopIds      = stops;
        e.resetCache();               // no prefix reuse: every item starts cold, like forward()
        out = e.generate(prompt, p);
        e.resetCache();
        while (!out.empty() && isStop(out.back())) {
            out.pop_back();
        }
        return out;
    }
    std::vector<std::int32_t> ids = prompt;
    for (std::size_t i = 0; i < maxTokens; ++i) {
        Engine::ForwardProbe probe;
        (*setup)(probe, ids.size());
        const auto lg  = forward(e, ids, &probe);
        const auto& last = lg.back();
        const auto tok = static_cast<std::int32_t>(std::max_element(last.begin(), last.end()) - last.begin());
        if (isStop(tok)) {
            break;
        }
        out.push_back(tok);
        ids.push_back(tok);
    }
    return out;
}

/// Goldset items: inline array or a .json (array) / .jsonl file.
namespace {

/// Free-generation specs are validated up front, so a bad regex or language
/// fails the call before any engine is built or any forward runs.
json validated(json items) {
    if (items.is_array()) {
        for (std::size_t i = 0; i < items.size(); ++i) {
            if (items[i].is_object() && items[i].contains("generate")) {
                try {
                    (void)GenerationCheck{items[i]["generate"]};
                } catch (const std::exception& ex) {
                    throw std::runtime_error("goldset item " + std::to_string(i) + ": " + ex.what());
                }
            }
        }
    }
    return items;
}

} // namespace

json loadGoldset(const json& a) {
    if (a.contains("items")) {
        return validated(a["items"]);
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
        return validated(json::parse(text));
    }
    json items = json::array();
    std::istringstream lines(text);
    std::string line;
    while (std::getline(lines, line)) {
        if (line.find_first_not_of(" \t\r") != std::string::npos) {
            items.push_back(json::parse(line));
        }
    }
    return validated(std::move(items));
}

/// One eval item on one engine: {"prompt", "expect"} (exact continuation) or
/// {"prompt", "choices":[...], "answer": idx} (multiple choice by logprob).
json evalItem(Engine& e, const json& item, bool addBos, const ProbeSetup* setup) {
    if (item.contains("generate")) {
        const GenerationCheck check{item["generate"]};   // validate before the forward
        const auto prompt = encodeText(e, item.at("prompt").get<std::string>(), addBos);
        const auto out    = generateGreedy(e, prompt, check.maxTokens(), setup);
        const std::string text = e.tokenizer().decode(out);
        json r = check.evaluate(text);
        r["kind"]   = "generate";
        r["text"]   = text;
        r["tokens"] = out.size();
        return r;
    }
    const auto prompt = encodeText(e, item.at("prompt").get<std::string>(), addBos);
    const auto encodeCont = [&](const std::string& t) { return e.tokenizer().encode(t, false); };
    if (item.contains("choices")) {
        const auto choices = item["choices"].get<std::vector<std::string>>();
        const auto answer  = item.at("answer").get<std::size_t>();
        if (answer >= choices.size()) {
            throw std::runtime_error("eval item 'answer' out of range");
        }
        std::vector<double> lps;
        for (const auto& c : choices) {
            lps.push_back(scoreContinuation(e, prompt, encodeCont(c), setup).logprob);
        }
        const auto best = static_cast<std::size_t>(
            std::max_element(lps.begin(), lps.end()) - lps.begin());
        return json{{"kind", "choice"}, {"correct", best == answer}, {"picked", best},
                    {"logprob", lps[answer]}, {"choice_logprobs", lps}};
    }
    const auto sc = scoreContinuation(e, prompt, encodeCont(item.at("expect").get<std::string>()), setup);
    return json{{"kind", "expect"}, {"correct", sc.greedyMatch}, {"logprob", sc.logprob},
                {"tokens", sc.tokens}, {"first_rank", sc.firstRank}};
}

json evalRun(Engine& e, const json& items, bool addBos, json& rows, const ProbeSetup* setup) {
    std::size_t correct = 0, nonFinite = 0, generated = 0;
    double      lpSum   = 0;
    for (std::size_t i = 0; i < items.size(); ++i) {
        json r = evalItem(e, items[i], addBos, setup);
        correct += r["correct"].get<bool>() ? 1U : 0U;
        if (!r.contains("logprob")) {
            ++generated;   // free generation: checked, not scored
        } else if (const double lp = r["logprob"].get<double>(); std::isfinite(lp)) {
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
    const double finiteN =
        static_cast<double>(std::max<std::size_t>(1, items.size() - nonFinite - generated));
    return json{{"items", items.size()}, {"accuracy", static_cast<double>(correct) / n},
                {"mean_logprob", lpSum / finiteN}, {"non_finite_items", nonFinite},
                {"generate_items", generated}};
}

json checkRegressions(const json& baseRows, const json& runRows) {
    json out = json::array();
    for (std::size_t i = 0; i < baseRows.size() && i < runRows.size(); ++i) {
        const json& b = baseRows[i]["base_or_run"];
        const json& r = runRows[i]["base_or_run"];
        if (!b.contains("checks") || !r.contains("checks")) {
            continue;
        }
        for (const auto& [name, passed] : b["checks"].items()) {
            if (passed.get<bool>() && !r["checks"].value(name, false)) {
                out.push_back({{"index", i}, {"id", baseRows[i]["id"]}, {"check", name},
                               {"base_text", b.value("text", std::string{})},
                               {"text", r.value("text", std::string{})}});
            }
        }
    }
    return out;
}

} // namespace mimirmind::hugin::forward_support
