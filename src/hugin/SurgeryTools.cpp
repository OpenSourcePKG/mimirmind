// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "hugin/SurgeryTools.hpp"

#include "hugin/ForwardSupport.hpp"
#include "hugin/ForwardTools.hpp"
#include "hugin/LogitMath.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>

namespace mimirmind::hugin {

namespace {

using nlohmann::json;
using namespace forward_support;
using Probe = Engine::ForwardProbe;

constexpr std::size_t kAllPositions   = static_cast<std::size_t>(-1);
constexpr std::size_t kMaxGenerate    = 256;
constexpr std::size_t kDefaultForward = 400;

/// One forward's interventions (steer and/or head ablation).
struct Interventions {
    const SurgeryTools::SteerVector* steer{nullptr};
    double                           alpha{0};
    bool                             steerLastOnly{false};
    bool                             ablate{false};
    std::uint32_t                    ablateBlock{0};
    std::vector<std::size_t>         ablateHeads;
};

std::uint32_t blockArg(const Engine& e, const json& v) {
    return blockList(e, json{{"blocks", json::array({v})}}).front();
}

std::vector<std::size_t> headList(const Engine& e, const json& v) {
    const std::size_t H = e.config().headCount;
    std::vector<std::size_t> out;
    for (const auto& x : v) {
        const auto h = x.get<std::int64_t>();
        if (h < 0 || h >= static_cast<std::int64_t>(H)) {
            throw std::runtime_error("head " + std::to_string(h) + " out of range [0," +
                                     std::to_string(H) + ")");
        }
        out.push_back(static_cast<std::size_t>(h));
    }
    if (out.empty()) {
        throw std::runtime_error("'heads' must not be empty");
    }
    return out;
}

/// Positions mode: "all" (default) or "last".
bool lastOnlyArg(const json& a) {
    const std::string m = a.value("positions", std::string{"all"});
    if (m != "all" && m != "last") {
        throw std::runtime_error("'positions' must be \"all\" or \"last\"");
    }
    return m == "last";
}

/// Installs `iv` into `probe` for an n-token forward. The returned flag turns
/// true once the backend actually reached the ablation stage — an
/// architecture without an editable attn_ctx would otherwise silently no-op.
std::shared_ptr<bool> install(const Engine& e, const Interventions& iv, std::size_t n, Probe& probe) {
    auto reached = std::make_shared<bool>(!iv.ablate);
    if (iv.steer != nullptr && iv.alpha != 0.0) {
        Probe::ResidualEdit re;
        re.block    = iv.steer->block;
        re.mode     = Probe::ResidualEdit::Mode::Add;
        re.position = iv.steerLastOnly ? n - 1 : kAllPositions;
        re.values.resize(iv.steer->values.size());
        for (std::size_t i = 0; i < re.values.size(); ++i) {
            re.values[i] = static_cast<float>(iv.alpha * static_cast<double>(iv.steer->values[i]));
        }
        probe.residualEdits.push_back(std::move(re));
    }
    if (iv.ablate) {
        const std::size_t H     = e.config().headCount;
        const auto        heads = iv.ablateHeads;
        probe.stageEdits.push_back(Probe::StageEdit{
            "attn_ctx", iv.ablateBlock,
            [H, heads, reached](std::vector<float>& data, std::size_t rows, std::size_t cols) {
                if (H == 0 || cols % H != 0) {
                    throw std::runtime_error("attn_ctx width is not a multiple of the head count");
                }
                const std::size_t hd = cols / H;
                for (std::size_t r = 0; r < rows; ++r) {
                    for (const auto h : heads) {
                        std::fill_n(data.begin() + static_cast<std::ptrdiff_t>(r * cols + h * hd),
                                    hd, 0.0F);
                    }
                }
                *reached = true;
                return true;
            }});
    }
    return reached;
}

/// Forward with interventions; returns all logits rows.
std::vector<std::vector<float>> runWith(Engine& e, std::span<const std::int32_t> ids,
                                        const Interventions& iv) {
    Probe probe;
    const auto reached = install(e, iv, ids.size(), probe);
    auto lg = forward(e, ids, &probe);
    if (!*reached) {
        throw std::runtime_error("architecture '" + e.config().architecture +
                                 "' does not expose an editable attn_ctx stage "
                                 "(head ablation supports llama / qwen2 today)");
    }
    return lg;
}

std::vector<double> lastLogProbs(Engine& e, std::span<const std::int32_t> ids, const Interventions& iv) {
    return logProbs(e, runWith(e, ids, iv).back());
}

/// Greedy continuation by repeated stateless forwards (interventions apply to
/// every forward, so a steered/ablated model generates consistently).
std::vector<std::int32_t> greedy(Engine& e, std::vector<std::int32_t> ids, std::size_t maxTokens,
                                 const Interventions& iv) {
    std::vector<std::int32_t> out;
    const std::int32_t eos = e.tokenizer().eosId();
    for (std::size_t i = 0; i < maxTokens; ++i) {
        const auto lp  = lastLogProbs(e, ids, iv);
        const auto tok = LogitMath::topK(lp, 1).front();
        if (tok == eos) {
            break;
        }
        out.push_back(tok);
        ids.push_back(tok);
    }
    return out;
}

std::size_t maxTokensArg(const json& a, std::size_t dflt) {
    const auto n = a.value("max_tokens", dflt);
    if (n > kMaxGenerate) {
        throw std::runtime_error("max_tokens is capped at " + std::to_string(kMaxGenerate));
    }
    return n;
}

/// Optional single target token: first token of `target` (no BOS).
std::optional<std::int32_t> targetArg(const Engine& e, const json& a) {
    if (!a.contains("target")) {
        return std::nullopt;
    }
    const auto ids = e.tokenizer().encode(a["target"].get<std::string>(), false);
    if (ids.empty()) {
        throw std::runtime_error("'target' tokenizes to zero tokens");
    }
    return ids.front();
}

json shiftJson(const Engine& e, const std::vector<double>& base, const std::vector<double>& edited,
               std::size_t k, std::optional<std::int32_t> target) {
    const auto b1 = LogitMath::topK(base, 1).front();
    const auto e1 = LogitMath::topK(edited, 1).front();
    json out{{"kl", LogitMath::kl(base, edited)},
             {"top1_changed", b1 != e1},
             {"base_top", topJson(e, base, k)},
             {"edited_top", topJson(e, edited, k)}};
    if (target) {
        const auto t = static_cast<std::size_t>(*target);
        out["target"] = {{"id", *target}, {"piece", piece(e, *target)},
                         {"base_prob", std::exp(base[t])}, {"edited_prob", std::exp(edited[t])},
                         {"base_rank", LogitMath::rankOf(base, t)},
                         {"edited_rank", LogitMath::rankOf(edited, t)}};
    }
    return out;
}

double l2(std::span<const float> v) {
    double s = 0;
    for (const float x : v) {
        s += static_cast<double>(x) * static_cast<double>(x);
    }
    return std::sqrt(s);
}

} // namespace

SurgeryTools::SurgeryTools(ForwardTools& forward) : _forward(forward) {}

void SurgeryTools::dropStaleVectors() {
    const std::string key = _forward.targetKey();
    if (key != _vectorsKey) {
        _vectors.clear();
        _vectorsKey = key;
    }
}

const SurgeryTools::SteerVector& SurgeryTools::vector(const std::string& name) {
    dropStaleVectors();
    const auto it = _vectors.find(name);
    if (it == _vectors.end()) {
        throw std::runtime_error("no steering vector '" + name + "' — build it with steer_vector");
    }
    return it->second;
}

void SurgeryTools::registerOn(McpServer& server) {
    server.addTool(
        "causal_trace",
        "Activation patching (forward only). Runs the 'clean' and the 'corrupt' prompt (same token "
        "count), then for each (block, position) re-runs the corrupt prompt with the residual "
        "stream at that block/position replaced by the clean one, and reports the probability of "
        "the target token (default: the clean argmax). effect = (p_patched - p_corrupt) / "
        "(p_clean - p_corrupt): 1 = that site alone carries the clean answer. Cost: one forward "
        "per cell; 'max_forwards' (default 400) guards it.",
        schema({{"clean", {{"type", "string"}}},
                {"corrupt", {{"type", "string"}}},
                {"target", {{"type", "string"}, {"description", "first token is scored; default clean argmax"}}},
                {"blocks", {{"type", "array"}, {"items", {{"type", "integer"}}}}},
                {"positions", {{"type", "array"}, {"items", {{"type", "integer"}}}}},
                {"add_bos", {{"type", "boolean"}}},
                {"max_forwards", {{"type", "integer"}}},
                {"top_cells", {{"type", "integer"}}}},
               {"clean", "corrupt"}),
        [this](const json& a) {
            Engine&    e      = _forward.engine();
            const bool addBos = a.value("add_bos", true);
            const auto clean   = textTokens(e, a.at("clean").get<std::string>(), addBos);
            const auto corrupt = textTokens(e, a.at("corrupt").get<std::string>(), addBos);
            if (clean.size() != corrupt.size()) {
                throw std::runtime_error("clean and corrupt must tokenize to the same length (" +
                                         std::to_string(clean.size()) + " vs " +
                                         std::to_string(corrupt.size()) + ")");
            }
            const std::size_t n = clean.size();
            const auto blocks   = blockList(e, a);
            std::vector<std::size_t> positions;
            if (a.contains("positions")) {
                for (const auto& v : a["positions"]) {
                    positions.push_back(positionArg(json{{"position", v}}, n));
                }
            } else {
                for (std::size_t p = 0; p < n; ++p) positions.push_back(p);
            }
            const std::size_t cells = blocks.size() * positions.size();
            if (cells > a.value("max_forwards", kDefaultForward)) {
                throw std::runtime_error(std::to_string(cells) + " forwards requested (" +
                                         std::to_string(blocks.size()) + " blocks x " +
                                         std::to_string(positions.size()) +
                                         " positions) — narrow 'blocks'/'positions' or raise 'max_forwards'");
            }
            const std::size_t d = e.config().embeddingLength;

            Probe cleanProbe;
            cleanProbe.hiddenBlocks = blocks;
            const auto cleanLp   = logProbs(e, forward(e, clean, &cleanProbe).back());
            const auto corruptLp = logProbs(e, forward(e, corrupt).back());
            const auto target    = targetArg(e, a).value_or(LogitMath::topK(cleanLp, 1).front());
            const auto t         = static_cast<std::size_t>(target);
            const double pClean   = std::exp(cleanLp[t]);
            const double pCorrupt = std::exp(corruptLp[t]);
            const double denom    = pClean - pCorrupt;

            json grid = json::array();
            struct Cell { std::uint32_t block; std::size_t pos; double effect, prob; };
            std::vector<Cell> all;
            for (const auto b : blocks) {
                const auto& h = cleanProbe.hidden.at(b);
                json probs = json::array(), effects = json::array();
                for (const auto p : positions) {
                    Probe probe;
                    probe.residualEdits.push_back(Probe::ResidualEdit{
                        b, Probe::ResidualEdit::Mode::Set, p,
                        std::vector<float>(h.begin() + static_cast<std::ptrdiff_t>(p * d),
                                           h.begin() + static_cast<std::ptrdiff_t>((p + 1) * d))});
                    const auto lp   = logProbs(e, forward(e, corrupt, &probe).back());
                    const double pr = std::exp(lp[t]);
                    const double ef = std::fabs(denom) > 1e-9 ? (pr - pCorrupt) / denom : 0.0;
                    probs.push_back(pr);
                    effects.push_back(ef);
                    all.push_back({b, p, ef, pr});
                }
                grid.push_back({{"block", b}, {"prob", probs}, {"effect", effects}});
            }
            std::sort(all.begin(), all.end(), [](const Cell& x, const Cell& y) { return x.effect > y.effect; });
            json top = json::array();
            for (std::size_t i = 0; i < std::min(all.size(), a.value("top_cells", std::size_t{10})); ++i) {
                top.push_back({{"block", all[i].block}, {"position", all[i].pos},
                               {"clean_token", piece(e, clean[all[i].pos])},
                               {"corrupt_token", piece(e, corrupt[all[i].pos])},
                               {"effect", all[i].effect}, {"prob", all[i].prob}});
            }
            json tokens = json::array();
            for (const auto p : positions) {
                tokens.push_back({{"position", p}, {"clean", piece(e, clean[p])},
                                  {"corrupt", piece(e, corrupt[p])}, {"differs", clean[p] != corrupt[p]}});
            }
            return json{{"target", {{"id", target}, {"piece", piece(e, target)}}},
                        {"p_clean", pClean}, {"p_corrupt", pCorrupt},
                        {"weak_contrast", std::fabs(denom) < 0.05},
                        {"forwards", cells + 2}, {"positions", tokens},
                        {"top_cells", top}, {"grid", grid}};
        });

    server.addTool(
        "steer_vector",
        "Build a steering direction at one block: mean residual (last position) over the "
        "'positive' prompts minus the mean over the 'negative' prompts. Stored in this session "
        "under 'name' (dropped when the model changes). Reports its norm next to the typical "
        "residual norm at that block, which is the scale 'alpha' is relative to.",
        schema({{"name", {{"type", "string"}}},
                {"positive", {{"type", "array"}, {"items", {{"type", "string"}}}}},
                {"negative", {{"type", "array"}, {"items", {{"type", "string"}}}}},
                {"block", {{"type", "integer"}}},
                {"add_bos", {{"type", "boolean"}}}},
               {"name", "positive", "negative", "block"}),
        [this](const json& a) {
            Engine&    e      = _forward.engine();
            dropStaleVectors();
            const auto blk    = blockArg(e, a.at("block"));
            const bool addBos = a.value("add_bos", true);
            const std::size_t d = e.config().embeddingLength;
            double residualSum = 0;
            std::size_t residualN = 0;
            const auto meanOf = [&](const json& texts) {
                if (!texts.is_array() || texts.empty()) {
                    throw std::runtime_error("'positive' and 'negative' must be non-empty arrays");
                }
                std::vector<double> m(d, 0.0);
                for (const auto& t : texts) {
                    const auto ids = textTokens(e, t.get<std::string>(), addBos);
                    Probe probe;
                    probe.hiddenBlocks = {blk};
                    (void)forward(e, ids, &probe);
                    const auto& h = probe.hidden.at(blk);
                    const std::span<const float> last{h.data() + (ids.size() - 1) * d, d};
                    residualSum += l2(last);
                    ++residualN;
                    for (std::size_t i = 0; i < d; ++i) m[i] += static_cast<double>(last[i]);
                }
                for (auto& x : m) x /= static_cast<double>(texts.size());
                return m;
            };
            const auto pos = meanOf(a.at("positive"));
            const auto neg = meanOf(a.at("negative"));
            SteerVector v;
            v.block = blk;
            v.values.resize(d);
            double dot = 0, np = 0, nn = 0;
            for (std::size_t i = 0; i < d; ++i) {
                v.values[i] = static_cast<float>(pos[i] - neg[i]);
                dot += pos[i] * neg[i];
                np  += pos[i] * pos[i];
                nn  += neg[i] * neg[i];
            }
            v.norm         = l2(v.values);
            v.residualNorm = residualSum / static_cast<double>(residualN);
            v.positives    = a["positive"].size();
            v.negatives    = a["negative"].size();
            const std::string name = a.at("name").get<std::string>();
            _vectors[name] = v;
            return json{{"name", name}, {"block", blk}, {"norm", v.norm},
                        {"mean_residual_norm", v.residualNorm},
                        {"relative_norm", v.residualNorm > 0 ? v.norm / v.residualNorm : 0.0},
                        {"cosine_pos_neg", (np > 0 && nn > 0) ? dot / std::sqrt(np * nn) : 0.0},
                        {"positives", v.positives}, {"negatives", v.negatives}};
        });

    server.addTool(
        "steer_list", "Steering vectors held in this session.",
        schema(json::object()),
        [this](const json&) {
            dropStaleVectors();
            json out = json::array();
            for (const auto& [name, v] : _vectors) {
                out.push_back({{"name", name}, {"block", v.block}, {"norm", v.norm},
                               {"mean_residual_norm", v.residualNorm},
                               {"positives", v.positives}, {"negatives", v.negatives}});
            }
            return json{{"vectors", out}};
        });

    server.addTool(
        "steer_preview",
        "Preview a steering vector: add alpha x vector to the residual stream after its block "
        "(at every position, or only the last), compare the next-token distribution with the "
        "unsteered model (KL, top-k, optional target token), and with 'max_tokens' > 0 the greedy "
        "continuations of both. 'alphas' sweeps several strengths in one call. Preview only — "
        "nothing is written.",
        schema(textProps({{"vector", {{"type", "string"}}},
                          {"alpha", {{"type", "number"}}},
                          {"alphas", {{"type", "array"}, {"items", {{"type", "number"}}}}},
                          {"positions", {{"type", "string"}, {"enum", {"all", "last"}}}},
                          {"target", {{"type", "string"}}},
                          {"top_k", {{"type", "integer"}}},
                          {"max_tokens", {{"type", "integer"}}}}),
               {"vector"}),
        [this](const json& a) {
            Engine&    e   = _forward.engine();
            const auto ids = inputTokens(e, a);
            Interventions iv;
            iv.steer         = &vector(a.at("vector").get<std::string>());
            iv.steerLastOnly = lastOnlyArg(a);
            const std::size_t k   = a.value("top_k", std::size_t{5});
            const std::size_t gen = maxTokensArg(a, 0);
            const auto target     = targetArg(e, a);
            const auto baseLp     = lastLogProbs(e, ids, Interventions{});
            const auto decode     = [&](const std::vector<std::int32_t>& t) { return e.tokenizer().decode(t); };
            json out{{"vector", a["vector"]}, {"block", iv.steer->block},
                     {"positions", iv.steerLastOnly ? "last" : "all"}};
            if (gen > 0) {
                out["base_text"] = decode(greedy(e, ids, gen, Interventions{}));
            }
            std::vector<double> alphas;
            if (a.contains("alphas")) {
                alphas = a["alphas"].get<std::vector<double>>();
            } else {
                alphas = {a.value("alpha", 1.0)};
            }
            json runs = json::array();
            for (const double alpha : alphas) {
                iv.alpha = alpha;
                json r   = shiftJson(e, baseLp, lastLogProbs(e, ids, iv), k, target);
                r["alpha"] = alpha;
                if (gen > 0) {
                    r["steered_text"] = decode(greedy(e, ids, gen, iv));
                }
                runs.push_back(std::move(r));
            }
            out["runs"] = runs;
            return out;
        });

    server.addTool(
        "ablate_preview",
        "Preview zeroing attention heads of one block (their slice of the attention output, "
        "before the O-projection). With 'heads': next-token shift vs the intact model (KL, top-k, "
        "optional target) and optional greedy text. With 'scan': each head ablated alone, ranked "
        "by KL (and target-probability drop). Preview only — nothing is written. Supported where "
        "the backend exposes an editable attn_ctx (llama / qwen2 today).",
        schema(textProps({{"block", {{"type", "integer"}}},
                          {"heads", {{"type", "array"}, {"items", {{"type", "integer"}}}}},
                          {"scan", {{"type", "boolean"}}},
                          {"target", {{"type", "string"}}},
                          {"top_k", {{"type", "integer"}}},
                          {"max_tokens", {{"type", "integer"}}}}),
               {"block"}),
        [this](const json& a) {
            Engine&    e   = _forward.engine();
            const auto ids = inputTokens(e, a);
            Interventions iv;
            iv.ablate      = true;
            iv.ablateBlock = blockArg(e, a.at("block"));
            const std::size_t k = a.value("top_k", std::size_t{5});
            const auto target   = targetArg(e, a);
            const auto baseLp   = lastLogProbs(e, ids, Interventions{});
            if (a.value("scan", false)) {
                const std::size_t H = e.config().headCount;
                struct Row { std::size_t head; double kl; bool top1; double dTarget; };
                std::vector<Row> rows;
                const auto b1 = LogitMath::topK(baseLp, 1).front();
                for (std::size_t h = 0; h < H; ++h) {
                    iv.ablateHeads = {h};
                    const auto lp = lastLogProbs(e, ids, iv);
                    double dT = 0;
                    if (target) {
                        const auto t = static_cast<std::size_t>(*target);
                        dT = std::exp(lp[t]) - std::exp(baseLp[t]);
                    }
                    rows.push_back({h, LogitMath::kl(baseLp, lp), LogitMath::topK(lp, 1).front() != b1, dT});
                }
                std::sort(rows.begin(), rows.end(), [](const Row& x, const Row& y) { return x.kl > y.kl; });
                json out = json::array();
                for (std::size_t i = 0; i < std::min(rows.size(), a.value("top_k", std::size_t{10})); ++i) {
                    json r{{"head", rows[i].head}, {"kl", rows[i].kl}, {"top1_changed", rows[i].top1}};
                    if (target) r["target_prob_delta"] = rows[i].dTarget;
                    out.push_back(std::move(r));
                }
                return json{{"block", iv.ablateBlock}, {"heads", H}, {"ranked", out}};
            }
            if (!a.contains("heads")) {
                throw std::runtime_error("pass 'heads' or 'scan': true");
            }
            iv.ablateHeads = headList(e, a["heads"]);
            json out = shiftJson(e, baseLp, lastLogProbs(e, ids, iv), k, target);
            out["block"] = iv.ablateBlock;
            out["heads"] = iv.ablateHeads;
            const std::size_t gen = maxTokensArg(a, 0);
            if (gen > 0) {
                out["base_text"]    = e.tokenizer().decode(greedy(e, ids, gen, Interventions{}));
                out["ablated_text"] = e.tokenizer().decode(greedy(e, ids, gen, iv));
            }
            return out;
        });

    server.addTool(
        "generate",
        "Greedy continuation (repeated stateless forwards of the engine — slow, for inspection, "
        "not serving), optionally with a steering vector {vector, alpha, positions} and/or a head "
        "ablation {block, heads} applied to every forward.",
        schema(textProps({{"max_tokens", {{"type", "integer"}}},
                          {"steer", {{"type", "object"}}},
                          {"ablate", {{"type", "object"}}}}),
               {}),
        [this](const json& a) {
            Engine&    e   = _forward.engine();
            const auto ids = inputTokens(e, a);
            Interventions iv;
            if (a.contains("steer")) {
                const json& s    = a["steer"];
                iv.steer         = &vector(s.at("vector").get<std::string>());
                iv.alpha         = s.value("alpha", 1.0);
                iv.steerLastOnly = lastOnlyArg(s);
            }
            if (a.contains("ablate")) {
                iv.ablate      = true;
                iv.ablateBlock = blockArg(e, a["ablate"].at("block"));
                iv.ablateHeads = headList(e, a["ablate"].at("heads"));
            }
            const auto out = greedy(e, ids, maxTokensArg(a, 32), iv);
            return json{{"prompt_tokens", ids.size()}, {"tokens", out},
                        {"text", e.tokenizer().decode(out)}};
        });
}

} // namespace mimirmind::hugin
