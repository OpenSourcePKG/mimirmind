// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "hugin/ForwardTools.hpp"
#include "hugin/ForwardSupport.hpp"

#include "core/config/Config.hpp"
#include "hugin/LogitMath.hpp"
#include "runtime/InferenceEngine.hpp"
#include "runtime/nvfp4/ModelFormatResolver.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>

namespace mimirmind::hugin {

namespace {

using nlohmann::json;
using Engine = runtime::InferenceEngine;
using namespace forward_support;

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

void ForwardTools::dropVariant() noexcept {
    _variant.reset();
}

std::string ForwardTools::targetKey() const {
    const Target t = _target();
    return t.path + '\n' + t.tokenizerGguf;
}

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
    const Target t = _target();
    if (t.path.empty()) {
        throw std::runtime_error("no model loaded — call open_model first");
    }
    if (_variant && sameTarget(_variant->target, t) && _variant->runes == runes) {
        return *_variant->engine;
    }
    // One engine at a time besides the base: drop the old variant first so
    // two large models are never resident at once beyond base + variant.
    _variant.reset();
    if (_singleEngine) {
        _base.reset();   // base is reloaded lazily by the next base()
    } else {
        (void)base();    // keeps base/variant on the same model
    }
    _variant = load(t, runes);
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
            const Engine* bp = &base();
            const auto    ids = inputTokens(*bp, a);
            const auto    lb  = forward(base(), ids);
            Engine&       r   = withRunes(runes);
            const auto    lr  = forward(r, ids);
            // --single-engine released the base when the variant loaded; the
            // variant has the same tokenizer and logit softcap (runes change
            // weights only), so it stands in for the base's metadata below.
            const Engine& b = _singleEngine ? r : *bp;
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
        "'span' [start, end) adds each head's weight mass on that token range and ranks the "
        "heads by it (e.g. which heads read the system instruction). Sliding-window layers are "
        "masked like the kernel. Supported where the backend exposes the stages (llama / qwen2 / "
        "kolibri1 today).",
        schema(textProps({{"block", {{"type", "integer"}}},
                          {"heads", {{"type", "array"}, {"items", {{"type", "integer"}}}}},
                          {"position", {{"type", "integer"}}},
                          {"top_k", {{"type", "integer"}}},
                          {"span", {{"type", "array"}, {"items", {{"type", "integer"}}},
                                    {"description", "[start, end) token positions"}}},
                          {"full", {{"type", "boolean"}, {"description", "include the full weight row per head"}}}}),
               {"block"}),
        [this](const json& a) {
            Engine&    e   = base();
            const auto ids = inputTokens(e, a);
            const auto blk = blockList(e, json{{"blocks", json::array({a.at("block")})}}).front();
            const std::size_t pos = positionArg(a, ids.size());
            Engine::ForwardProbe probe;
            probe.stages      = {"Qcur", "Kcur", "Vcur", "attn_ctx", "attn_scale", "attn_window"};
            probe.stageBlocks = {blk};
            (void)forward(e, ids, &probe);
            const auto get = [&](const char* name) -> const Engine::ForwardProbe::Stage& {
                const auto it = probe.stageData.find({name, blk});
                if (it == probe.stageData.end()) {
                    throw std::runtime_error("architecture '" + e.config().architecture +
                                             "' does not expose stage '" + name +
                                             "' (attention_pattern supports llama/qwen2/kolibri1 today)");
                }
                return it->second;
            };
            const auto& Q = get("Qcur");
            const auto& K = get("Kcur");
            const auto& V = get("Vcur");
            const auto& C = get("attn_ctx");
            const double scale = get("attn_scale").data.at(0);
            // Per-layer geometry from the captured shapes (GQA width and head
            // dim may differ per layer, e.g. kolibri1 SWA vs full layers).
            const std::size_t H = e.config().headCount;
            if (H == 0 || C.cols % H != 0) {
                throw std::runtime_error("attn_ctx width is not a multiple of the head count");
            }
            const std::size_t hd  = C.cols / H;
            const std::size_t Hkv = hd > 0 ? K.cols / hd : 0;
            if (Q.cols != H * hd || Hkv == 0 || K.cols != Hkv * hd || V.cols != Hkv * hd ||
                H % Hkv != 0 || Q.rows != ids.size() || K.rows < ids.size() || V.rows < ids.size()) {
                throw std::runtime_error("unexpected stage shapes for attention_pattern");
            }
            std::size_t window = 0;
            if (const auto it = probe.stageData.find({"attn_window", blk}); it != probe.stageData.end()) {
                window = static_cast<std::size_t>(it->second.data.at(0));
            }
            std::size_t spanLo = 0, spanHi = 0;
            const bool hasSpan = a.contains("span");
            if (hasSpan) {
                const auto sp = a["span"].get<std::vector<std::int64_t>>();
                if (sp.size() != 2 || sp[0] < 0 || sp[1] <= sp[0] ||
                    sp[1] > static_cast<std::int64_t>(ids.size())) {
                    throw std::runtime_error("'span' must be [start, end) with 0 <= start < end <= tokens");
                }
                spanLo = static_cast<std::size_t>(sp[0]);
                spanHi = static_cast<std::size_t>(sp[1]);
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
            // Sliding window as the kernel applies it: keys [n - window, n).
            const std::size_t kMin = (window > 0 && n > window) ? n - window : 0;
            double maxErr = 0, maxRef = 0;
            std::vector<double> avg(n, 0.0);
            json out = json::array();
            std::vector<std::pair<double, std::size_t>> bySpan;
            for (const auto h : heads) {
                const std::size_t kvh = h / (H / Hkv);
                const float* q = Q.data.data() + pos * Q.cols + h * hd;
                std::vector<double> w(n, 0.0);
                double mx = -1e300;
                for (std::size_t j = kMin; j < n; ++j) {
                    const float* kr = K.data.data() + j * K.cols + kvh * hd;
                    double dot = 0;
                    for (std::size_t d = 0; d < hd; ++d) dot += static_cast<double>(q[d]) * static_cast<double>(kr[d]);
                    w[j] = dot * scale;
                    mx = std::max(mx, w[j]);
                }
                double z = 0;
                for (std::size_t j = kMin; j < n; ++j) { w[j] = std::exp(w[j] - mx); z += w[j]; }
                double ent = 0, spanW = 0;
                for (std::size_t j = kMin; j < n; ++j) {
                    w[j] /= z;
                    avg[j] += w[j] / static_cast<double>(heads.size());
                    if (w[j] > 0) ent -= w[j] * std::log(w[j]);
                    if (hasSpan && j >= spanLo && j < spanHi) spanW += w[j];
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
                if (hasSpan) {
                    o["span_weight"] = spanW;
                    bySpan.emplace_back(spanW, h);
                }
                if (full) o["weights"] = w;
                out.push_back(std::move(o));
            }
            json avgTop = json::array();
            for (const auto j : LogitMath::topK(avg, k)) {
                avgTop.push_back({{"pos", j}, {"token", piece(e, ids[static_cast<std::size_t>(j)])},
                                  {"weight", avg[static_cast<std::size_t>(j)]}});
            }
            json res{{"block", blk}, {"position", pos}, {"query_token", piece(e, ids[pos])},
                     {"heads", H}, {"kv_heads", Hkv}, {"head_dim", hd}, {"scale", scale},
                     {"window", window},
                     {"consistency", {{"max_abs_err", maxErr}, {"max_abs_ref", maxRef}}},
                     {"head_average_top", avgTop}, {"per_head", out}};
            if (hasSpan) {
                std::sort(bySpan.begin(), bySpan.end(), std::greater<>{});
                json ranked = json::array();
                for (const auto& [wgt, h] : bySpan) ranked.push_back({{"head", h}, {"span_weight", wgt}});
                res["span"] = {spanLo, spanHi};
                res["heads_by_span_weight"] = ranked;
            }
            return res;
        });

    server.addTool(
        "eval",
        "Forward-only goldset eval (teacher-forced, no sampling). Items: {prompt, expect} -> "
        "correct iff every expected token is the argmax; or {prompt, choices[], answer} -> "
        "correct iff the answer has the highest total logprob; or {prompt, generate:{max_tokens, "
        "lang:'de'|'en', max_other_markers, match:[regex], no_match:[regex]}} -> greedy free "
        "generation (KV-cached), correct iff every check passes (answer language, no code-mixing, "
        "facts). Goldset inline ('items') or a "
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
                out["check_regressions"] = checkRegressions(baseRows, runeRows);
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
        "With 'runes' the routing of the rune-applied model is reported (e.g. to prove an expert "
        "disable). Supported: qwen35moe-family and kolibri1 backends.",
        schema(textProps({{"blocks", {{"type", "array"}, {"items", {{"type", "integer"}}}}},
                          {"position", {{"type", "integer"}}},
                          {"top_k", {{"type", "integer"}}},
                          {"runes", {{"type", "array"}, {"items", {{"type", "string"}}}}}}),
               {}),
        [this](const json& a) {
            const auto runes = a.value("runes", std::vector<std::string>{});
            Engine&    e   = runes.empty() ? base() : withRunes(runes);
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
            if (!runes.empty()) {
                out["runes"] = runes;
            }
            if (pos) {
                out["position"] = *pos;
                out["token_at_position"] = piece(e, ids[*pos]);
            }
            return out;
        });
}

} // namespace mimirmind::hugin
