// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "hugin/ForwardTools.hpp"
#include "hugin/ForwardReport.hpp"
#include "hugin/ForwardSupport.hpp"

#include "core/config/Config.hpp"
#include "core/ipc/WeightArenaExporter.hpp"
#include "hugin/LogitMath.hpp"
#include "runtime/InferenceEngine.hpp"
#include "runtime/nvfp4/ModelFormatResolver.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
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

/// MemAvailable from /proc/meminfo in GiB (0 if unreadable).
double memAvailableGiB() {
    std::ifstream f("/proc/meminfo");
    std::string key;
    std::uint64_t kib = 0;
    std::string unit;
    while (f >> key >> kib >> unit) {
        if (key == "MemAvailable:") {
            return static_cast<double>(kib) / (1024.0 * 1024.0);
        }
    }
    return 0.0;
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
    if (!_attachSocket.empty()) {
        // 7.2.3: the live engine's weights, read-only, no second copy.
        const double availGiB = memAvailableGiB();
        if (availGiB < _attachMinFreeGiB) {
            throw std::runtime_error(
                "refusing to attach: " + std::to_string(availGiB) + " GiB available < " +
                std::to_string(_attachMinFreeGiB) + " GiB required beside the serving engine");
        }
        auto h = core::ipc::fetchWeightArena(_attachSocket, "hugin");
        if (!h) {
            throw std::runtime_error(h.error());
        }
        std::cerr << "hugin: attaching to '" << h->manifest.modelId << "' via '" << _attachSocket
                  << "' (" << h->manifest.tensors.size() << " tensors, "
                  << (h->manifest.mapBytes >> 20) << " MiB, " << availGiB
                  << " GiB available)\n";
        l->engine->loadModelSharedArena(t.path, t.tokenizerGguf, std::move(*h));
    } else if (runtime::nvfp4::resolveModelFormat(core::config::ModelFormat::Auto, t.path) ==
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

/// The model's tokenizer as a ForwardReport::Piece.
ForwardReport::Piece pieceOf(const Engine& e) {
    return [&e](std::int32_t id) { return piece(e, id); };
}

/// The attention stages one forward captured for `block`, or nullopt when the
/// backend does not expose them.
std::optional<ForwardReport::AttentionStages>
attentionStages(const Engine& e, const Engine::ForwardProbe& probe, std::uint32_t block) {
    const auto find = [&](const char* name) -> const Engine::ForwardProbe::Stage* {
        const auto it = probe.stageData.find({name, block});
        return it == probe.stageData.end() ? nullptr : &it->second;
    };
    ForwardReport::AttentionStages s;
    s.q   = find("Qcur");
    s.k   = find("Kcur");
    s.v   = find("Vcur");
    s.ctx = find("attn_ctx");
    const auto* scale = find("attn_scale");
    if (s.q == nullptr || s.k == nullptr || s.v == nullptr || s.ctx == nullptr || scale == nullptr ||
        scale->data.empty()) {
        return std::nullopt;
    }
    s.scale     = scale->data.front();
    s.headCount = e.config().headCount;
    if (const auto* w = find("attn_window"); w != nullptr && !w->data.empty()) {
        s.window = static_cast<std::size_t>(w->data.front());
    }
    return s;
}

/// expert_usage over the routing taps of one forward; null when none was captured.
json expertUsage(const Engine& e, std::span<const std::int32_t> ids, const Engine::ForwardProbe& probe,
                 std::optional<std::size_t> pos, std::size_t topN) {
    const std::size_t nExperts = e.config().expertCount;
    json blocks = json::array();
    std::vector<double> totalCount(nExperts, 0.0);
    for (const auto& [key, st] : probe.stageData) {
        if (key.first != "moe_topk_idx") {
            continue;
        }
        const auto wIt = probe.stageData.find({"moe_topk_w", key.second});
        blocks.push_back(ForwardReport::expertBlock(key.second, st,
                                                    wIt == probe.stageData.end() ? nullptr : &wIt->second,
                                                    nExperts, topN, pos, totalCount));
    }
    if (blocks.empty()) {
        return nullptr;
    }
    json out = ForwardReport::expertSummary(totalCount, topN);
    out["tokens"]     = ids.size();
    out["experts"]    = nExperts;
    out["moe_blocks"] = blocks.size();
    out["blocks"]     = std::move(blocks);
    return out;
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
                const auto prev = b > 0 ? std::optional<std::span<const float>>(row(b - 1)) : std::nullopt;
                blocks.push_back(ForwardReport::residual(b, row(b), prev, k, values));
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
        "kolibri1 / gemma4 / the full-attention blocks of qwen35-family hybrids) with an F32 KV "
        "cache. Under cuDNN SDPA (Q/K/V staged to bf16) 'consistency' shows that rounding "
        "(~1e-2 relative); MIMIRMIND_ATTN_CUDNN=0 gives the exact F32 kernel.",
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
            const auto stages = attentionStages(e, probe, blk);
            if (!stages) {
                throw std::runtime_error("architecture '" + e.config().architecture +
                                         "' does not expose the attention stages "
                                         "(attention_pattern supports llama/qwen2/kolibri1 today)");
            }
            ForwardReport::AttentionOptions o;
            o.position = pos;
            o.topK     = a.value("top_k", std::size_t{5});
            o.full     = a.value("full", false);
            if (a.contains("span")) {
                const auto sp = a["span"].get<std::vector<std::int64_t>>();
                if (sp.size() != 2 || sp[0] < 0 || sp[1] <= sp[0] ||
                    sp[1] > static_cast<std::int64_t>(ids.size())) {
                    throw std::runtime_error("'span' must be [start, end) with 0 <= start < end <= tokens");
                }
                o.span = std::pair{static_cast<std::size_t>(sp[0]), static_cast<std::size_t>(sp[1])};
            }
            if (a.contains("heads")) {
                for (const auto& v : a["heads"]) {
                    o.heads.push_back(v.get<std::size_t>());
                }
            }
            return ForwardReport::attention(*stages, ids, pieceOf(e), blk, o);
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
        "disable). Supported: qwen35moe-family, gemma4 MoE and kolibri1 backends.",
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
            std::optional<std::size_t> pos;
            if (a.contains("position")) {
                pos = positionArg(a, ids.size());
            }
            json out = expertUsage(e, ids, probe, pos, topN);
            if (out.is_null()) {
                throw std::runtime_error("no MoE routing captured — '" + e.config().architecture +
                                         "' is dense or its backend does not expose the routing "
                                         "tap yet (supported: qwen35moe-family, kolibri1)");
            }
            if (!runes.empty()) {
                out["runes"] = runes;
            }
            if (pos) {
                out["position"] = *pos;
                out["token_at_position"] = piece(e, ids[*pos]);
            }
            return out;
        });

    server.addTool(
        "trace",
        "One prompt through the whole model in ONE forward, with everything a picture of the "
        "run needs: tokens, the next-token top-k, per block the logit lens and the residual "
        "statistics (raw vector with 'values', default true), the attention pattern of every "
        "block where the backend exposes the stages ('attention': false skips it), and the MoE "
        "routing. Sections carry the single tools' names and shapes (tokenize, next_token, "
        "logit_lens, activations, attention_pattern[], expert_usage) plus 'model' (sizes and "
        "per-block layer kinds) and 'notes' for what could not be captured. Memory grows with "
        "tokens x blocks; meant for prompts up to a few hundred tokens.",
        schema(textProps({{"position", {{"type", "integer"}, {"description", "token index; negative counts from the end"}}},
                          {"top_k", {{"type", "integer"}, {"description", "lens / next-token top-k (default 5)"}}},
                          {"top_dims", {{"type", "integer"}}},
                          {"values", {{"type", "boolean"}, {"description", "raw residual per block (default true)"}}},
                          {"attention", {{"type", "boolean"}}},
                          {"attention_top_k", {{"type", "integer"}}},
                          {"expert_top_k", {{"type", "integer"}}},
                          {"runes", {{"type", "array"}, {"items", {{"type", "string"}}}}}}),
               {}),
        [this](const json& a) {
            const auto runes = a.value("runes", std::vector<std::string>{});
            Engine&    e   = runes.empty() ? base() : withRunes(runes);
            const auto ids = inputTokens(e, a);
            const std::size_t pos = positionArg(a, ids.size());
            const auto&       cfg = e.config();
            const std::uint32_t B = cfg.blockCount;
            const std::size_t   d = cfg.embeddingLength;
            const bool withAttention = a.value("attention", true);

            Engine::ForwardProbe probe;
            for (std::uint32_t b = 0; b < B; ++b) {
                probe.hiddenBlocks.push_back(b);
                probe.lensBlocks.push_back(b);
            }
            probe.lensPosition = pos;
            probe.stages = {"moe_topk_idx", "moe_topk_w"};
            if (withAttention) {
                for (const char* s : {"Qcur", "Kcur", "Vcur", "attn_ctx", "attn_scale", "attn_window"}) {
                    probe.stages.emplace_back(s);
                }
            }
            const auto lg = forward(e, ids, &probe);
            const std::size_t k = a.value("top_k", std::size_t{5});
            const auto piece_ = pieceOf(e);
            json notes = json::array();

            json pieces = json::array();
            for (const auto id : ids) {
                pieces.push_back(piece(e, id));
            }

            const auto finalLp = logProbs(e, lg[pos]);
            const auto finalId = LogitMath::topK(finalLp, 1).front();
            json lens = json::array();
            for (const auto& [b, logits] : probe.lens) {
                const auto lp = logProbs(e, logits);
                lens.push_back({{"block", b}, {"entropy", LogitMath::entropy(lp)},
                                {"final_token_prob", std::exp(lp[static_cast<std::size_t>(finalId)])},
                                {"final_token_rank", LogitMath::rankOf(lp, static_cast<std::size_t>(finalId))},
                                {"top", topJson(e, lp, k)}});
            }

            const std::size_t topDims = a.value("top_dims", std::size_t{8});
            const bool values = a.value("values", true);
            json acts = json::array();
            std::optional<std::span<const float>> prev;
            for (std::uint32_t b = 0; b < B; ++b) {
                const auto it = probe.hidden.find(b);
                if (it == probe.hidden.end() || it->second.size() < (pos + 1) * d) {
                    prev.reset();
                    continue;
                }
                const std::span<const float> x(it->second.data() + pos * d, d);
                acts.push_back(ForwardReport::residual(b, x, prev, topDims, values));
                prev = x;
            }

            json attention = json::array();
            if (withAttention) {
                ForwardReport::AttentionOptions o;
                o.position = pos;
                o.topK     = a.value("attention_top_k", std::size_t{3});
                for (std::uint32_t b = 0; b < B; ++b) {
                    const auto st = attentionStages(e, probe, b);
                    if (!st) {
                        continue;   // e.g. a linear-attention block of a hybrid model
                    }
                    try {
                        attention.push_back(ForwardReport::attention(*st, ids, piece_, b, o));
                    } catch (const std::exception& ex) {
                        notes.push_back("attention block " + std::to_string(b) + ": " + ex.what());
                    }
                }
                if (attention.empty()) {
                    notes.push_back("'" + cfg.architecture + "' does not expose the attention stages");
                }
            }

            const json moe = expertUsage(e, ids, probe, pos, a.value("expert_top_k", std::size_t{8}));

            json model{{"name", std::filesystem::path(_target().path).filename().string()},
                       {"architecture", cfg.architecture}, {"d_model", d}, {"blocks", B},
                       {"heads", cfg.headCount}, {"kv_heads", cfg.headCountKv},
                       {"experts", cfg.expertCount}, {"experts_per_token", cfg.expertUsedCount}};
            if (cfg.isHybridRecurrent()) {
                json kinds = json::array();
                for (std::uint32_t b = 0; b < B; ++b) kinds.push_back(cfg.isRecurrentLayer(b) ? "linear" : "full");
                model["block_kinds"] = kinds;
            } else if (cfg.slidingWindowPattern.size() == B) {
                json kinds = json::array();
                for (std::uint32_t b = 0; b < B; ++b) kinds.push_back(cfg.slidingWindowPattern[b] ? "sliding" : "global");
                model["block_kinds"] = kinds;
            }

            json out{{"model", model},
                     {"tokenize", {{"count", ids.size()}, {"ids", ids}, {"pieces", pieces}}},
                     {"next_token", {{"entropy", LogitMath::entropy(finalLp)}, {"top", topJson(e, finalLp, k)}}},
                     {"logit_lens", {{"position", pos}, {"token_at_position", piece(e, ids[pos])},
                                     {"final_prediction", {{"id", finalId}, {"piece", piece(e, finalId)},
                                                           {"prob", std::exp(finalLp[static_cast<std::size_t>(finalId)])}}},
                                     {"blocks", lens}}},
                     {"activations", {{"position", pos}, {"token_at_position", piece(e, ids[pos])},
                                      {"d_model", d}, {"blocks", acts}}},
                     {"attention_pattern", attention},
                     {"notes", notes}};
            if (!moe.is_null()) {
                out["expert_usage"] = moe;
                out["expert_usage"]["position"] = pos;
            }
            if (a.contains("text")) {
                out["prompt"] = a["text"];
            }
            if (!runes.empty()) {
                out["runes"] = runes;
            }
            return out;
        });
}

} // namespace mimirmind::hugin
