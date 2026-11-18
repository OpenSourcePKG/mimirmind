// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "hugin/SurgeryTools.hpp"

#include "hugin/ForwardSupport.hpp"
#include "hugin/ForwardTools.hpp"
#include "hugin/LogitMath.hpp"
#include "core/rune/RuneApplier.hpp"
#include "runtime/encoder/LinearProbeTrainer.hpp"

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

/// Scale attention heads of one block (their attn_ctx slice, before the
/// O-projection): 0 = ablation, > 1 = amplification.
struct HeadEdit {
    std::uint32_t            block{0};
    std::vector<std::size_t> heads;
    double                   scale{0};
};

/// One forward's interventions (steer and/or head edits).
struct Interventions {
    const SurgeryTools::SteerVector* steer{nullptr};
    double                           alpha{0};
    bool                             steerLastOnly{false};
    std::vector<HeadEdit>            headEdits;
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

/// {block, heads, scale (default 0)} -> HeadEdit.
HeadEdit headEditArg(const Engine& e, const json& v) {
    HeadEdit he;
    he.block = blockArg(e, v.at("block"));
    he.heads = headList(e, v.at("heads"));
    he.scale = v.value("scale", 0.0);
    if (!std::isfinite(he.scale)) {
        throw std::runtime_error("head 'scale' must be finite");
    }
    return he;
}

/// Per-head-edit "reached" flags: an architecture without an editable attn_ctx
/// would otherwise silently no-op.
using Reached = std::shared_ptr<std::vector<char>>;

void requireReached(const Engine& e, const Reached& reached) {
    if (std::find(reached->begin(), reached->end(), char{0}) != reached->end()) {
        throw std::runtime_error("architecture '" + e.config().architecture +
                                 "' does not expose an editable attn_ctx stage "
                                 "(head edits support llama / qwen2 / kolibri1 today)");
    }
}

/// Installs `iv` into `probe` for an n-token forward, marking `reached[i]`
/// once head edit i hit its stage.
void installInto(const Engine& e, const Interventions& iv, std::size_t n, Probe& probe,
                 const Reached& reached) {
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
    const std::size_t H = e.config().headCount;
    for (std::size_t i = 0; i < iv.headEdits.size(); ++i) {
        const HeadEdit& he = iv.headEdits[i];
        probe.stageEdits.push_back(Probe::StageEdit{
            "attn_ctx", he.block,
            [H, he, reached, i](std::vector<float>& data, std::size_t rows, std::size_t cols) {
                if (H == 0 || cols % H != 0) {
                    throw std::runtime_error("attn_ctx width is not a multiple of the head count");
                }
                const std::size_t hd = cols / H;
                const auto        f  = static_cast<float>(he.scale);
                for (std::size_t r = 0; r < rows; ++r) {
                    for (const auto h : he.heads) {
                        float* p = data.data() + r * cols + h * hd;
                        for (std::size_t d = 0; d < hd; ++d) {
                            p[d] = f == 0.0F ? 0.0F : p[d] * f;
                        }
                    }
                }
                (*reached)[i] = 1;
                return true;
            }});
    }
}

/// Forward with interventions; returns all logits rows.
std::vector<std::vector<float>> runWith(Engine& e, std::span<const std::int32_t> ids,
                                        const Interventions& iv) {
    Probe probe;
    const auto reached = std::make_shared<std::vector<char>>(iv.headEdits.size(), char{0});
    installInto(e, iv, ids.size(), probe, reached);
    auto lg = forward(e, ids, &probe);
    requireReached(e, reached);
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
        _conditions.clear();
        _baseEvals.clear();
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

const SurgeryTools::ConditionVector& SurgeryTools::conditionVector(const std::string& name) {
    dropStaleVectors();
    const auto it = _conditions.find(name);
    if (it == _conditions.end()) {
        throw std::runtime_error("no condition '" + name + "' — build it with condition_vector");
    }
    return it->second;
}

namespace {

/// Removes the preview overlay on scope exit (also on a throwing tool call),
/// so a failed preview never leaves the session engine steered.
class PreviewOverlay {
public:
    PreviewOverlay(Engine& e, std::vector<core::rune::RuneCondSteer> steers) : _e(e) {
        _e.setPreviewConditionalSteers(std::move(steers));
    }
    ~PreviewOverlay() {
        try {
            _e.setPreviewConditionalSteers({});
        } catch (...) {
            // Destructor: the engine reports its own error; nothing to unwind.
        }
    }
    PreviewOverlay(const PreviewOverlay&)            = delete;
    PreviewOverlay& operator=(const PreviewOverlay&) = delete;

private:
    Engine& _e;
};

/// Fraction of `scores[from, to)` above `threshold` (0 for an empty range).
double fireRate(const std::vector<float>& scores, std::size_t from, std::size_t to,
                double threshold) {
    if (to <= from) {
        return 0.0;
    }
    std::size_t n = 0;
    for (std::size_t i = from; i < to; ++i) {
        n += static_cast<double>(scores[i]) > threshold ? std::size_t{1} : std::size_t{0};
    }
    return static_cast<double>(n) / static_cast<double>(to - from);
}

} // namespace

/// 7.11: the conditional-steer tools (condition_vector, steer_if_preview).
void SurgeryTools::registerConditionalOn(McpServer& server) {
    server.addTool(
        "condition_vector",
        "Fit the condition of a conditional steer (CAST-style, roadmap 7.11): a binary linear "
        "probe (softmax regression, the 8.23 trainer) on the L2-normalised residual after "
        "'block', positive vs negative prompts. 'positions': 'generated' (default — the rows "
        "that produce the first 'max_tokens' greedy tokens of the base model, i.e. where a "
        "per-token gate has to fire during the answer), 'last' or 'all'. Result: a unit "
        "direction + cosine threshold (rows with cos(h, direction) > threshold are 'on'), "
        "train/val accuracy, per-class score ranges and per-prompt fire rates. Stored in this "
        "session under 'name'. 'threshold' overrides the fitted one.",
        schema({{"name", {{"type", "string"}}},
                {"positive", {{"type", "array"}, {"items", {{"type", "string"}}}}},
                {"negative", {{"type", "array"}, {"items", {{"type", "string"}}}}},
                {"block", {{"type", "integer"}}},
                {"positions", {{"type", "string"}, {"enum", {"generated", "last", "all"}}}},
                {"max_tokens", {{"type", "integer"}}},
                {"l2", {{"type", "number"}}},
                {"threshold", {{"type", "number"}}},
                {"add_bos", {{"type", "boolean"}}}},
               {"name", "positive", "negative", "block"}),
        [this](const json& a) {
            Engine& e = _forward.engine();
            dropStaleVectors();
            const auto        blk    = blockArg(e, a.at("block"));
            const bool        addBos = a.value("add_bos", true);
            const std::string mode   = a.value("positions", std::string{"generated"});
            if (mode != "generated" && mode != "last" && mode != "all") {
                throw std::runtime_error("'positions' must be generated, last or all");
            }
            const std::size_t gen = maxTokensArg(a, 16);
            const std::size_t d   = e.config().embeddingLength;

            std::vector<float>        feats;
            std::vector<std::int32_t> labels;
            struct Span { std::size_t begin; std::size_t end; int label; std::string text; };
            std::vector<Span> spans;
            const auto collect = [&](const json& texts, int label) {
                if (!texts.is_array() || texts.empty()) {
                    throw std::runtime_error("'positive' and 'negative' must be non-empty arrays");
                }
                for (const auto& t : texts) {
                    const std::string text = t.get<std::string>();
                    auto              seq  = textTokens(e, text, addBos);
                    const std::size_t n    = seq.size();
                    std::size_t from = 0, to = n;
                    if (mode == "last") {
                        from = n - 1;
                    } else if (mode == "generated") {
                        const auto out = generateGreedy(e, seq, gen);
                        seq.insert(seq.end(), out.begin(), out.end());
                        from = n - 1;
                        to   = n - 1 + std::max<std::size_t>(out.size(), 1);
                    }
                    Probe probe;
                    probe.hiddenBlocks = {blk};
                    (void)forward(e, seq, &probe);
                    const auto& h = probe.hidden.at(blk);
                    const std::size_t begin = labels.size();
                    for (std::size_t r = from; r < to; ++r) {
                        const std::span<const float> row{h.data() + r * d, d};
                        const double nrm = l2(row);
                        for (std::size_t i = 0; i < d; ++i) {
                            feats.push_back(nrm > 0 ? static_cast<float>(static_cast<double>(row[i]) / nrm) : 0.0F);
                        }
                        labels.push_back(label);
                    }
                    spans.push_back({begin, labels.size(), label, text});
                }
            };
            collect(a.at("positive"), 1);
            collect(a.at("negative"), 0);

            runtime::encoder::LinearProbeTrainer::Config cfg;
            cfg.l2 = a.value("l2", cfg.l2);
            const auto fit = runtime::encoder::LinearProbeTrainer::train(feats, d, labels, 2, cfg);
            std::vector<float> w(d);
            for (std::size_t i = 0; i < d; ++i) {
                w[i] = fit.weight[d + i] - fit.weight[i];
            }
            const double nw = l2(w);
            if (!(nw > 0)) {
                throw std::runtime_error("probe weight is zero — the classes are not separable here");
            }
            ConditionVector c;
            c.block = blk;
            c.direction.resize(d);
            for (std::size_t i = 0; i < d; ++i) {
                c.direction[i] = static_cast<float>(static_cast<double>(w[i]) / nw);
            }
            const double b = static_cast<double>(fit.bias[1]) - static_cast<double>(fit.bias[0]);
            c.threshold = a.contains("threshold") ? a["threshold"].get<double>() : -b / nw;
            if (!(c.threshold > -1.0 && c.threshold < 1.0)) {
                throw std::runtime_error("threshold " + std::to_string(c.threshold) +
                                         " is not a cosine in (-1, 1); pass 'threshold'");
            }
            c.positions     = mode;
            c.trainAccuracy = fit.trainAccuracy;
            c.valAccuracy   = fit.valAccuracy;

            // Cosine of every row (features are unit rows) + stats at the threshold.
            std::vector<float> score(labels.size());
            for (std::size_t r = 0; r < labels.size(); ++r) {
                double s = 0;
                for (std::size_t i = 0; i < d; ++i) {
                    s += static_cast<double>(feats[r * d + i]) * static_cast<double>(c.direction[i]);
                }
                score[r] = static_cast<float>(s);
            }
            std::size_t correct = 0;
            double lo[2] = {1, 1}, hi[2] = {-1, -1}, sum[2] = {0, 0};
            std::size_t cnt[2] = {0, 0};
            for (std::size_t r = 0; r < labels.size(); ++r) {
                const int  l  = labels[r];
                const bool on = static_cast<double>(score[r]) > c.threshold;
                correct += (on == (l == 1)) ? 1 : 0;
                lo[l] = std::min(lo[l], static_cast<double>(score[r]));
                hi[l] = std::max(hi[l], static_cast<double>(score[r]));
                sum[l] += static_cast<double>(score[r]);
                ++cnt[l];
            }
            c.positiveRows = cnt[1];
            c.negativeRows = cnt[0];
            json prompts = json::array();
            for (const auto& sp : spans) {
                prompts.push_back({{"label", sp.label == 1 ? "positive" : "negative"},
                                   {"rows", sp.end - sp.begin},
                                   {"fire_rate", fireRate(score, sp.begin, sp.end, c.threshold)},
                                   {"text", sp.text.substr(0, 80)}});
            }
            const auto cls = [&](int l) {
                return json{{"rows", cnt[l]}, {"min", lo[l]}, {"max", hi[l]},
                            {"mean", cnt[l] ? sum[l] / static_cast<double>(cnt[l]) : 0.0}};
            };
            const std::string name = a.at("name").get<std::string>();
            _conditions[name] = c;
            return json{{"name", name}, {"block", blk}, {"positions", mode},
                        {"threshold", c.threshold}, {"fitted_threshold", -b / nw},
                        {"train_accuracy", c.trainAccuracy}, {"val_accuracy", c.valAccuracy},
                        {"accuracy_at_threshold",
                         static_cast<double>(correct) / static_cast<double>(labels.size())},
                        {"positive", cls(1)}, {"negative", cls(0)}, {"prompts", prompts}};
        });

    server.addTool(
        "steer_if_preview",
        "Preview a projection or conditional steer (roadmap 7.11) through the SAME engine "
        "overlay a steer_if / steer_project Rune installs (preview == rune by construction). "
        "'vector' = a steer_vector; 'mode' 'add' (x += alpha v) or 'project' (x -= alpha (x.u) u, "
        "u = v/|v|, 0 < alpha <= 1); 'condition' (optional) = a condition_vector — then only "
        "rows with cos(h, c) > threshold are changed, per token. For each of 'prompts': base and "
        "edited greedy text (KV-cached serving decode) and the gate fire rate over prompt and "
        "answer rows. Optional goldset ('items' / 'path') evaluated base vs edited with "
        "per-check regressions. Preview only — the overlay is removed afterwards.",
        schema({{"vector", {{"type", "string"}}},
                {"alpha", {{"type", "number"}}},
                {"mode", {{"type", "string"}, {"enum", {"add", "project"}}}},
                {"condition", {{"type", "string"}}},
                {"threshold", {{"type", "number"}}},
                {"prompts", {{"type", "array"}, {"items", {{"type", "string"}}}}},
                {"max_tokens", {{"type", "integer"}}},
                {"items", {{"type", "array"}, {"items", {{"type", "object"}}}}},
                {"path", {{"type", "string"}}},
                {"add_bos", {{"type", "boolean"}}},
                {"detail", {{"type", "boolean"}}}},
               {"vector"}),
        [this](const json& a) {
            Engine& e = _forward.engine();
            dropStaleVectors();
            const bool addBos = a.value("add_bos", true);
            core::rune::RuneCondSteer cs = condSteerArg(a);
            const std::size_t gen = maxTokensArg(a, 48);
            const auto decode = [&](const std::vector<std::int32_t>& t) { return e.tokenizer().decode(t); };

            // Everything unedited first, then ONE overlay install for the edited runs.
            std::vector<std::vector<std::int32_t>> ids;
            std::vector<std::string>               baseText;
            if (a.contains("prompts")) {
                for (const auto& p : a["prompts"]) {
                    ids.push_back(textTokens(e, p.get<std::string>(), addBos));
                    baseText.push_back(decode(generateGreedy(e, ids.back(), gen)));
                }
            }
            json items;
            json baseSum, baseRows = json::array();
            if (a.contains("items") || a.contains("path")) {
                items = loadGoldset(a);
                baseSum = evalRun(e, items, addBos, baseRows);
            }

            json out{{"vector", a["vector"]}, {"block", cs.block},
                     {"mode", std::string{core::rune::steerModeName(cs.mode)}},
                     {"alpha", a.value("alpha", 1.0)}};
            if (cs.hasCondition) {
                out["condition"]       = a["condition"];
                out["condition_block"] = cs.conditionBlock;
                out["threshold"]       = cs.threshold;
            }
            {
                PreviewOverlay overlay{e, {cs}};
                json runs = json::array();
                for (std::size_t i = 0; i < ids.size(); ++i) {
                    const auto edited = generateGreedy(e, ids[i], gen);
                    json r{{"prompt", a["prompts"][i]}, {"base_text", baseText[i]},
                           {"edited_text", decode(edited)}};
                    r["changed"] = r["base_text"] != r["edited_text"];
                    if (cs.hasCondition) {
                        // One full forward over prompt + edited answer: the gate
                        // scores of exactly the rows serving would gate on.
                        auto seq = ids[i];
                        seq.insert(seq.end(), edited.begin(), edited.end());
                        (void)forward(e, seq);
                        const auto sc = e.previewConditionScores(0, seq.size());
                        r["fire_rate_prompt"] = fireRate(sc, 0, ids[i].size(), cs.threshold);
                        r["fire_rate_answer"] =
                            fireRate(sc, ids[i].size() - 1, seq.size() - 1, cs.threshold);
                    }
                    runs.push_back(std::move(r));
                }
                out["runs"] = runs;
                if (!items.is_null()) {
                    json editRows = json::array();
                    const json editSum = evalRun(e, items, addBos, editRows);
                    out["eval"] = {{"base", baseSum}, {"edited", editSum},
                                   {"check_regressions", checkRegressions(baseRows, editRows)}};
                    json flips = json::array();
                    for (std::size_t i = 0; i < editRows.size(); ++i) {
                        const json& b = baseRows[i]["base_or_run"];
                        const json& r = editRows[i]["base_or_run"];
                        if (b.value("correct", false) != r.value("correct", false)) {
                            flips.push_back({{"index", i}, {"id", editRows[i]["id"]},
                                             {"now_correct", r.value("correct", false)}});
                        }
                    }
                    out["eval"]["flips"] = flips;
                    if (a.value("detail", false)) {
                        out["eval"]["per_item"] = editRows;
                    }
                }
            }
            return out;
        });
}

SurgeryTools::CondSteerSpec SurgeryTools::condSteerSpec(const json& a) {
    const SteerVector& v = vector(a.at("vector").get<std::string>());
    const double alpha = a.value("alpha", 1.0);
    const std::string mode = a.value("mode", std::string{"add"});
    if (mode != "add" && mode != "project") {
        throw std::runtime_error("'mode' must be add or project");
    }
    if (!std::isfinite(alpha) || alpha == 0.0) {
        throw std::runtime_error("'alpha' must be finite and non-zero");
    }
    const bool project = mode == "project";
    if (project && !(alpha > 0.0 && alpha <= 1.0)) {
        throw std::runtime_error("projection alpha must be in (0, 1] (> 1 reflects the "
                                 "component and installs the opposite behaviour)");
    }
    if (project && !(l2(v.values) > 0)) {
        throw std::runtime_error("steering vector is zero");
    }
    CondSteerSpec spec;
    const std::string block = "blk." + std::to_string(v.block);
    spec.op.target  = block;
    spec.op.payload = "steer." + block;
    spec.op.alpha   = alpha;
    spec.payloads.emplace(spec.op.payload, v.values);
    spec.params = {{"vector", a["vector"]}, {"block", v.block}, {"alpha", alpha},
                   {"mode", mode}, {"vector_norm", v.norm},
                   {"mean_residual_norm", v.residualNorm},
                   {"relative_norm", v.residualNorm > 0
                                         ? (project ? alpha : std::fabs(alpha) * v.norm / v.residualNorm)
                                         : 0.0},
                   {"positives", v.positives}, {"negatives", v.negatives}};
    if (a.contains("condition")) {
        const ConditionVector& c = conditionVector(a["condition"].get<std::string>());
        if (c.block > v.block) {
            throw std::runtime_error("condition block " + std::to_string(c.block) +
                                     " is after the steered block " + std::to_string(v.block));
        }
        const double thr = a.value("threshold", c.threshold);
        if (!(thr > -1.0 && thr < 1.0)) {
            throw std::runtime_error("'threshold' must be a cosine in (-1, 1)");
        }
        spec.op.kind           = core::rune::RuneOpKind::SteerIf;
        spec.op.steerMode      = project ? core::rune::RuneSteerMode::Project
                                         : core::rune::RuneSteerMode::Add;
        spec.op.condition      = "cond." + block;
        spec.op.conditionBlock = c.block;
        spec.op.threshold      = thr;
        spec.payloads.emplace(spec.op.condition, c.direction);
        spec.params["condition"]          = a["condition"];
        spec.params["condition_block"]    = c.block;
        spec.params["threshold"]          = thr;
        spec.params["condition_positions"] = c.positions;
        spec.params["condition_val_accuracy"] = c.valAccuracy;
    } else if (project) {
        spec.op.kind = core::rune::RuneOpKind::SteerProject;
    } else {
        throw std::runtime_error("an ungated add steer is steer_preview / rune_stage_steer; pass "
                                 "'condition' or mode 'project'");
    }
    return spec;
}

core::rune::RuneCondSteer SurgeryTools::condSteerArg(const json& a) {
    // Resolve through the real Futhark applier, so the preview overlay is
    // bit-identical to what a staged Rune of the same spec installs.
    CondSteerSpec spec = condSteerSpec(a);
    core::rune::RuneManifest man;
    man.name            = "preview";
    man.baseFingerprint = "preview";
    man.ops.push_back(spec.op);
    const core::rune::RuneFile rune{std::move(man), std::move(spec.payloads)};
    core::rune::RuneApplier applier;
    auto rep = applier.apply(rune, "preview",
                             [](std::string_view) { return std::optional<core::rune::RuneTensorView>{}; });
    if (!rep) {
        throw std::runtime_error(rep.error());
    }
    if (rep->condSteers.size() != 1) {
        throw std::runtime_error("internal: expected one conditional steer");
    }
    return rep->condSteers.front();
}

void SurgeryTools::registerOn(McpServer& server) {
    registerConditionalOn(server);
    // steer {vector, alpha, positions} / ablate {block, heads, scale} /
    // head_edits [{block, heads, scale}] -> Interventions.
    const auto interventionsArg = [this](const json& a) {
        Engine&       e = _forward.engine();
        Interventions iv;
        if (a.contains("steer")) {
            const json& st   = a["steer"];
            iv.steer         = &vector(st.at("vector").get<std::string>());
            iv.alpha         = st.value("alpha", 1.0);
            iv.steerLastOnly = lastOnlyArg(st);
        }
        if (a.contains("ablate")) {
            iv.headEdits.push_back(headEditArg(e, a["ablate"]));
        }
        if (a.contains("head_edits")) {
            for (const auto& he : a["head_edits"]) {
                iv.headEdits.push_back(headEditArg(e, he));
            }
        }
        return iv;
    };

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
        "before the O-projection), or scaling them by 'scale' (default 0 = ablation, > 1 = "
        "amplification). With 'heads': next-token shift vs the intact model (KL, top-k, "
        "optional target) and optional greedy text. With 'scan': each head edited alone, ranked "
        "by KL (and target-probability change). Preview only — nothing is written. Supported "
        "where the backend exposes an editable attn_ctx (llama / qwen2 / kolibri1 today).",
        schema(textProps({{"block", {{"type", "integer"}}},
                          {"heads", {{"type", "array"}, {"items", {{"type", "integer"}}}}},
                          {"scale", {{"type", "number"}}},
                          {"scan", {{"type", "boolean"}}},
                          {"target", {{"type", "string"}}},
                          {"top_k", {{"type", "integer"}}},
                          {"max_tokens", {{"type", "integer"}}}}),
               {"block"}),
        [this](const json& a) {
            Engine&    e   = _forward.engine();
            const auto ids = inputTokens(e, a);
            Interventions iv;
            HeadEdit he;
            he.block = blockArg(e, a.at("block"));
            he.scale = a.value("scale", 0.0);
            iv.headEdits = {he};
            const std::size_t k = a.value("top_k", std::size_t{5});
            const auto target   = targetArg(e, a);
            const auto baseLp   = lastLogProbs(e, ids, Interventions{});
            if (a.value("scan", false)) {
                const std::size_t H = e.config().headCount;
                struct Row { std::size_t head; double kl; bool top1; double dTarget; };
                std::vector<Row> rows;
                const auto b1 = LogitMath::topK(baseLp, 1).front();
                for (std::size_t h = 0; h < H; ++h) {
                    iv.headEdits.front().heads = {h};
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
                return json{{"block", he.block}, {"scale", he.scale}, {"heads", H}, {"ranked", out}};
            }
            if (!a.contains("heads")) {
                throw std::runtime_error("pass 'heads' or 'scan': true");
            }
            iv.headEdits.front().heads = headList(e, a["heads"]);
            json out = shiftJson(e, baseLp, lastLogProbs(e, ids, iv), k, target);
            out["block"] = he.block;
            out["heads"] = iv.headEdits.front().heads;
            out["scale"] = he.scale;
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
        "not serving), optionally with a steering vector {vector, alpha, positions}, a head "
        "ablation {block, heads, scale} and/or 'head_edits' [{block, heads, scale}] applied to "
        "every forward.",
        schema(textProps({{"max_tokens", {{"type", "integer"}}},
                          {"steer", {{"type", "object"}}},
                          {"ablate", {{"type", "object"}}},
                          {"head_edits", {{"type", "array"}, {"items", {{"type", "object"}}}}}}),
               {}),
        [this, interventionsArg](const json& a) {
            Engine&    e   = _forward.engine();
            const auto ids = inputTokens(e, a);
            const Interventions iv = interventionsArg(a);
            const auto out = greedy(e, ids, maxTokensArg(a, 32), iv);
            return json{{"prompt_tokens", ids.size()}, {"tokens", out},
                        {"text", e.tokenizer().decode(out)}};
        });

    server.addTool(
        "eval_preview",
        "Goldset eval (same items and scoring as 'eval') with interventions applied to every "
        "forward: 'steer' {vector, alpha, positions} and/or 'head_edits' [{block, heads, scale}] "
        "(scale 0 = ablate, > 1 = amplify). Compares against the unedited model; the base result "
        "of a goldset 'path' is cached for this model, so sweeps only pay the edited runs. "
        "Preview only — nothing is written; a passing setting is then written as a Rune.",
        schema({{"items", {{"type", "array"}, {"items", {{"type", "object"}}}}},
                {"path", {{"type", "string"}}},
                {"steer", {{"type", "object"}}},
                {"head_edits", {{"type", "array"}, {"items", {{"type", "object"}}}}},
                {"add_bos", {{"type", "boolean"}}},
                {"detail", {{"type", "boolean"}}}}),
        [this, interventionsArg](const json& a) {
            Engine& e = _forward.engine();
            dropStaleVectors();   // also drops cached base evals of another model
            const json items = loadGoldset(a);
            if (!items.is_array() || items.empty()) {
                throw std::runtime_error("goldset is empty");
            }
            const bool addBos = a.value("add_bos", true);
            const Interventions iv = interventionsArg(a);
            if (iv.headEdits.empty() && (iv.steer == nullptr || iv.alpha == 0.0)) {
                throw std::runtime_error("pass 'steer' (alpha != 0) and/or 'head_edits'");
            }

            json baseSum, baseRows = json::array();
            const std::string cacheKey =
                a.contains("path") ? a["path"].get<std::string>() + (addBos ? "#bos" : "#nobos") : "";
            bool cached = false;
            if (const auto it = _baseEvals.find(cacheKey); !cacheKey.empty() && it != _baseEvals.end()) {
                baseSum  = it->second.first;
                baseRows = it->second.second;
                cached   = true;
            } else {
                baseSum = evalRun(e, items, addBos, baseRows);
                if (!cacheKey.empty()) {
                    _baseEvals[cacheKey] = {baseSum, baseRows};
                }
            }

            const auto reached = std::make_shared<std::vector<char>>(iv.headEdits.size(), char{0});
            const ProbeSetup setup = [&](Probe& probe, std::size_t n) {
                installInto(e, iv, n, probe, reached);
            };
            json editRows = json::array();
            const json editSum = evalRun(e, items, addBos, editRows, &setup);
            requireReached(e, reached);

            json out{{"base", baseSum}, {"edited", editSum}, {"base_cached", cached},
                     {"delta_accuracy",
                      editSum["accuracy"].get<double>() - baseSum["accuracy"].get<double>()},
                     {"delta_mean_logprob",
                      editSum["mean_logprob"].get<double>() - baseSum["mean_logprob"].get<double>()}};
            json flips = json::array(), rows = json::array();
            for (std::size_t i = 0; i < editRows.size(); ++i) {
                const json& b = baseRows[i]["base_or_run"];
                const json& r = editRows[i]["base_or_run"];
                if (b["correct"] != r["correct"]) {
                    flips.push_back({{"index", i}, {"id", editRows[i]["id"]}, {"now_correct", r["correct"]}});
                }
                rows.push_back({{"index", i}, {"id", editRows[i]["id"]}, {"base", b}, {"edited", r}});
            }
            out["flips"] = flips;
            out["check_regressions"] = checkRegressions(baseRows, editRows);
            if (a.value("detail", false)) {
                out["per_item"] = rows;
            }
            return out;
        });
}

} // namespace mimirmind::hugin
