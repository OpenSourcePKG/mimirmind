// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "hugin/TrainTools.hpp"

#include "compute/ComputeOps.hpp"
#include "compute/Dequant.hpp"
#include "hugin/ForwardSupport.hpp"
#include "hugin/ForwardTools.hpp"
#include "hugin/RuneStaging.hpp"
#include "hugin/train/AdamW.hpp"
#include "hugin/train/DenseBlock.hpp"
#include "hugin/train/DenseModel.hpp"
#include "hugin/train/Parallel.hpp"
#include "model/ChatTemplate.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace mimirmind::hugin {

using json = nlohmann::json;
using namespace forward_support;
using train::DenseBlock;

namespace {

/// Whole tensor as F32, read back from wherever the engine holds it.
std::vector<float> tensorF32(Engine& e, const core::gguf::GgufTensor& t) {
    std::vector<std::uint8_t> bytes(t.nbytes);
    e.ops()->readbackToHost(bytes.data(), t.usmPtr, bytes.size());
    std::vector<float> out(static_cast<std::size_t>(t.nelements));
    compute::dequantToF32(t.type, bytes.data(), out.size(), out.data());
    return out;
}

std::vector<float> blockTensor(Engine& e, std::size_t L, const char* name, bool required) {
    const auto* t = e.weights().findBlock(L, name);
    if (t == nullptr || t->usmPtr == nullptr) {
        if (required) {
            throw std::runtime_error("blk." + std::to_string(L) + "." + name + " not found");
        }
        return {};
    }
    return tensorF32(e, *t);
}

DenseBlock loadDenseBlock(Engine& e, std::size_t L) {
    const auto& c = e.config();
    const std::string& arch = c.architecture;
    if (arch != "llama" && arch != "qwen2") {
        throw std::runtime_error("train_block_check supports the dense llama / qwen2 block only (model is '" +
                                 arch + "'); other architectures follow in 7.4.4");
    }
    if (!c.ropeScalingType.empty() && c.ropeScalingFactor > 1.0F) {
        throw std::runtime_error("rope_scaling '" + c.ropeScalingType + "' is not supported by the train ops yet");
    }
    DenseBlock::Config cfg;
    cfg.dModel     = c.embeddingLength;
    cfg.heads      = c.headCount;
    cfg.kvHeads    = c.headCountKv;
    cfg.headDim    = c.headDim();
    cfg.ff         = c.feedForwardLength;
    cfg.eps        = c.rmsNormEps;
    cfg.ropeBase   = c.ropeFreqBase;
    cfg.ropeLayout = arch == "llama" ? train::RopeGrad::Layout::Interleaved : train::RopeGrad::Layout::Split;
    cfg.attnScale  = 1.0F / std::sqrt(static_cast<float>(cfg.headDim));
    if (const auto* rf = e.weights().find("rope_freqs.weight");
        rf != nullptr && rf->type == core::gguf::GgmlType::F32 && rf->nelements >= cfg.headDim / 2) {
        auto f = tensorF32(e, *rf);
        f.resize(cfg.headDim / 2);
        cfg.ropeFactors = std::move(f);
    }
    DenseBlock::Weights w;
    w.attnNorm = blockTensor(e, L, "attn_norm.weight", true);
    w.wq       = blockTensor(e, L, "attn_q.weight", true);
    w.bq       = blockTensor(e, L, "attn_q.bias", false);
    w.wk       = blockTensor(e, L, "attn_k.weight", true);
    w.bk       = blockTensor(e, L, "attn_k.bias", false);
    w.wv       = blockTensor(e, L, "attn_v.weight", true);
    w.bv       = blockTensor(e, L, "attn_v.bias", false);
    w.wo       = blockTensor(e, L, "attn_output.weight", true);
    w.ffnNorm  = blockTensor(e, L, "ffn_norm.weight", true);
    w.wGate    = blockTensor(e, L, "ffn_gate.weight", true);
    w.wUp      = blockTensor(e, L, "ffn_up.weight", true);
    w.wDown    = blockTensor(e, L, "ffn_down.weight", true);
    return DenseBlock(std::move(cfg), std::move(w));
}

std::vector<float> randn(std::size_t n, std::mt19937& rng, float sd) {
    std::normal_distribution<float> dist(0.0F, sd);
    std::vector<float> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

double dot(const std::vector<float>& a, const std::vector<float>& b) {
    double s = 0;
    for (std::size_t i = 0; i < a.size(); ++i) s += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    return s;
}

const char* projName(DenseBlock::Proj p) {
    switch (p) {
        case DenseBlock::Proj::Q:    return "q";
        case DenseBlock::Proj::K:    return "k";
        case DenseBlock::Proj::V:    return "v";
        case DenseBlock::Proj::O:    return "o";
        case DenseBlock::Proj::Gate: return "gate";
        case DenseBlock::Proj::Up:   return "up";
        case DenseBlock::Proj::Down: return "down";
    }
    return "?";
}

const char* projTensor(DenseBlock::Proj p) {
    switch (p) {
        case DenseBlock::Proj::Q:    return "attn_q.weight";
        case DenseBlock::Proj::K:    return "attn_k.weight";
        case DenseBlock::Proj::V:    return "attn_v.weight";
        case DenseBlock::Proj::O:    return "attn_output.weight";
        case DenseBlock::Proj::Gate: return "ffn_gate.weight";
        case DenseBlock::Proj::Up:   return "ffn_up.weight";
        case DenseBlock::Proj::Down: return "ffn_down.weight";
    }
    return "?";
}

constexpr DenseBlock::Proj kAllProjs[] = {DenseBlock::Proj::Q, DenseBlock::Proj::K, DenseBlock::Proj::V,
                                          DenseBlock::Proj::O, DenseBlock::Proj::Gate, DenseBlock::Proj::Up,
                                          DenseBlock::Proj::Down};

DenseBlock::Proj projArg(const std::string& s) {
    for (const auto p : {DenseBlock::Proj::Q, DenseBlock::Proj::K, DenseBlock::Proj::V, DenseBlock::Proj::O,
                         DenseBlock::Proj::Gate, DenseBlock::Proj::Up, DenseBlock::Proj::Down}) {
        if (s == projName(p)) return p;
    }
    throw std::runtime_error("unknown LoRA target '" + s + "' (q, k, v, o, gate, up, down)");
}

std::vector<float> globalTensor(Engine& e, const char* name, bool required) {
    const auto* t = e.weights().find(name);
    if (t == nullptr || t->usmPtr == nullptr) {
        if (required) throw std::runtime_error(std::string(name) + " not found");
        return {};
    }
    return tensorF32(e, *t);
}

/// One SFT example: ids = prompt + response, targets on the response only.
struct SftExample {
    std::vector<std::int32_t> ids;
    std::vector<std::int32_t> targets;
    std::size_t               promptTokens{0};
    std::size_t               responseTokens{0};
    bool                      truncated{false};
    std::string               response;
};

} // namespace

TrainTools::TrainTools(ForwardTools& forward, RuneStaging& staging)
    : _forward(forward), _staging(staging) {}

TrainTools::~TrainTools() = default;

train::DenseModel& TrainTools::model() {
    const auto t = _forward.target();
    if (_model && _modelKey == t.path) return *_model;
    _model.reset();
    Engine& e = _forward.engine();
    std::vector<DenseBlock> blocks;
    for (std::size_t L = 0; L < e.config().blockCount; ++L) blocks.push_back(loadDenseBlock(e, L));
    train::DenseModel::Head head;
    head.embed     = globalTensor(e, "token_embd.weight", true);
    head.finalNorm = globalTensor(e, "output_norm.weight", true);
    head.lmHead    = globalTensor(e, "output.weight", false);   // empty = tied
    head.eps       = e.config().rmsNormEps;
    head.vocab     = head.embed.size() / e.config().embeddingLength;
    _model    = std::make_unique<train::DenseModel>(std::move(blocks), std::move(head));
    _modelKey = t.path;
    return *_model;
}

void TrainTools::registerOn(McpServer& server) {
    server.addTool(
        "train_block_check",
        "7.4.2 training-core check on real weights. Builds dense block 'block' (llama / qwen2; >= 1) "
        "from Hugin's train ops with the engine's dequantised weights, feeds it the engine's residual "
        "after block-1 for 'text', and compares the output with the engine's residual after 'block' "
        "(max abs / relative error). Then installs random LoRA adapters (rank 'lora_rank', default 4) "
        "on 'targets' (default all: q k v o gate up down) and checks the recompute backward with "
        "directional central differences (loss = r . y): one check for the block input and one per "
        "adapter (A and B together). Each check runs a ladder of FD steps (input 1e-3..1e-2 of the "
        "residual RMS, adapters 1e-2..3e-1 of their own scale) and reports the best one: small steps "
        "drown in float rounding of the large residual, large steps in curvature, while a wrong "
        "gradient matches at no step. Best rel_err <= 1e-2 counts as ok.",
        schema(textProps({{"block", {{"type", "integer"}}},
                          {"lora_rank", {{"type", "integer"}}},
                          {"targets", {{"type", "array"}, {"items", {{"type", "string"}}}}},
                          {"seed", {{"type", "integer"}}}}),
               {"block"}),
        [this](const json& a) {
            Engine& e = _forward.engine();
            const auto L = a.at("block").get<std::size_t>();
            if (L < 1 || L >= e.config().blockCount) {
                throw std::runtime_error("'block' must be in [1, blockCount)");
            }
            const auto ids = inputTokens(e, a);
            const auto t0  = std::chrono::steady_clock::now();

            Engine::ForwardProbe probe;
            probe.hiddenBlocks = {static_cast<std::uint32_t>(L - 1), static_cast<std::uint32_t>(L)};
            (void)forward(e, ids, &probe);
            const std::vector<float>& xin = probe.hidden.at(static_cast<std::uint32_t>(L - 1));
            const std::vector<float>& ref = probe.hidden.at(static_cast<std::uint32_t>(L));

            DenseBlock blk = loadDenseBlock(e, L);
            const std::size_t T = ids.size(), d = blk.config().dModel;
            std::vector<float> y(T * d);
            blk.forward(xin.data(), T, y.data());
            double maxAbs = 0, refMax = 0, sqErr = 0, sqRef = 0;
            for (std::size_t i = 0; i < y.size(); ++i) {
                const double diff = std::fabs(static_cast<double>(y[i]) - static_cast<double>(ref[i]));
                maxAbs = std::max(maxAbs, diff);
                refMax = std::max(refMax, std::fabs(static_cast<double>(ref[i])));
                sqErr += diff * diff;
                sqRef += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
            }
            json parity{{"max_abs_err", maxAbs}, {"ref_abs_max", refMax},
                        {"rel_l2_err", std::sqrt(sqErr / std::max(sqRef, 1e-30))}};

            // Gradient checks (directional central differences).
            std::mt19937 rng(a.value("seed", 7402U));
            const auto  rank = a.value("lora_rank", std::size_t{4});
            std::vector<DenseBlock::Proj> targets;
            if (a.contains("targets")) {
                for (const auto& s : a["targets"]) targets.push_back(projArg(s.get<std::string>()));
            } else {
                targets = {DenseBlock::Proj::Q, DenseBlock::Proj::K, DenseBlock::Proj::V, DenseBlock::Proj::O,
                           DenseBlock::Proj::Gate, DenseBlock::Proj::Up, DenseBlock::Proj::Down};
            }
            for (const auto p : targets) {
                const auto [in, out] = blk.projShape(p);
                // Small adapters; B != 0 so dA is non-trivial.
                blk.setLora(p, rank, 1.0F, randn(rank * in, rng, 1.0F / std::sqrt(static_cast<float>(in))),
                            randn(out * rank, rng, 0.02F));
            }
            const std::vector<float> r = randn(T * d, rng, 1.0F / std::sqrt(static_cast<float>(T * d)));
            std::vector<float> x = xin;
            const auto loss = [&] {
                std::vector<float> yy(T * d);
                blk.forward(x.data(), T, yy.data());
                return dot(yy, r);
            };
            std::vector<float> dx(T * d);
            blk.backward(x.data(), T, r.data(), dx.data());

            json checks = json::array();
            bool allOk  = true;
            // Best relative error over a ladder of steps along one direction.
            const auto ladder = [&](const std::string& what, double analytic,
                                    const std::vector<float>& steps,
                                    const std::function<void(float)>& setOffset) {
                json tries = json::array();
                double bestRel = INFINITY, bestFd = 0, bestEps = 0;
                for (const float h : steps) {
                    setOffset(h);
                    const double lp = loss();
                    setOffset(-h);
                    const double lm = loss();
                    setOffset(0.0F);
                    const double fd  = (lp - lm) / (2.0 * static_cast<double>(h));
                    const double rel = std::fabs(analytic - fd) / std::max(std::fabs(fd), 1e-12);
                    tries.push_back({{"eps", h}, {"fd", fd}, {"rel_err", rel}});
                    if (rel < bestRel) {
                        bestRel = rel;
                        bestFd  = fd;
                        bestEps = h;
                    }
                }
                const bool ok = bestRel <= 1e-2;
                allOk = allOk && ok;
                checks.push_back({{"what", what}, {"analytic", analytic}, {"fd", bestFd}, {"eps", bestEps},
                                  {"rel_err", bestRel}, {"ok", ok}, {"ladder", tries}});
            };
            {
                // Block input: direction scaled to the residual's RMS.
                const double rms = std::sqrt(dot(x, x) / static_cast<double>(x.size()));
                const auto u = randn(x.size(), rng, static_cast<float>(rms));
                const std::vector<float> keep = x;
                ladder("input", dot(dx, u), {1e-3F, 3e-3F, 1e-2F}, [&](float h) {
                    for (std::size_t i = 0; i < x.size(); ++i) x[i] = keep[i] + h * u[i];
                });
            }
            for (const auto p : targets) {
                auto* l = blk.lora(p);
                const auto ua = randn(l->a.size(), rng, 1.0F / std::sqrt(static_cast<float>(l->a.size() / rank)));
                const auto ub = randn(l->b.size(), rng, 0.02F);
                const double analytic = dot(l->dA, ua) + dot(l->dB, ub);
                const auto ka = l->a, kb = l->b;
                ladder(std::string("lora_") + projName(p), analytic, {1e-2F, 3e-2F, 1e-1F, 3e-1F}, [&](float h) {
                    for (std::size_t i = 0; i < ka.size(); ++i) l->a[i] = ka[i] + h * ua[i];
                    for (std::size_t i = 0; i < kb.size(); ++i) l->b[i] = kb[i] + h * ub[i];
                });
            }
            const double secs =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            return json{{"block", L}, {"tokens", T}, {"arch", e.config().architecture},
                        {"rope", blk.config().ropeLayout == train::RopeGrad::Layout::Interleaved ? "interleaved" : "split"},
                        {"rope_factors", !blk.config().ropeFactors.empty()},
                        {"forward_parity", parity}, {"gradient_checks", checks},
                        {"all_gradients_ok", allOk}, {"seconds", secs}};
        });

    server.addTool(
        "train_lora_sft",
        "7.4.3 LoRA-SFT of the whole dense model (llama / qwen2) on CPU train ops; the base stays "
        "frozen, only adapters learn. Data: 'items' or 'path' (.json / .jsonl) of {prompt, response} "
        "or {prompt, teacher_prompt}: with teacher_prompt the response is the BASE model's greedy "
        "answer to teacher_prompt (self-distillation, e.g. the same question asked in German), ended "
        "by the template's stop token unless cut at 'max_teacher_tokens' (default 96). Loss = CE on "
        "the response tokens only. Adapters: 'targets' (default q k v o gate up down) on 'blocks' "
        "(default all), 'rank' (8), 'alpha' (16; scale alpha/rank), B = 0 at start. AdamW 'lr' "
        "(5e-4), 'epochs' (2), 'batch' (4, gradient accumulation), global grad-norm clip 'clip' "
        "(1.0), 'seed'. The result is staged as a LowRank Rune 'name' (then rune_gate / rune_commit "
        "as usual), and the trainer's logits are compared with the engine running that Rune "
        "(Futhark load path) on the first example.",
        schema({{"name", {{"type", "string"}}},
                {"items", {{"type", "array"}, {"items", {{"type", "object"}}}}},
                {"path", {{"type", "string"}}},
                {"add_bos", {{"type", "boolean"}}},
                {"targets", {{"type", "array"}, {"items", {{"type", "string"}}}}},
                {"blocks", {{"type", "array"}, {"items", {{"type", "integer"}}}}},
                {"rank", {{"type", "integer"}}},
                {"alpha", {{"type", "number"}}},
                {"lr", {{"type", "number"}}},
                {"epochs", {{"type", "integer"}}},
                {"batch", {{"type", "integer"}}},
                {"clip", {{"type", "number"}}},
                {"max_teacher_tokens", {{"type", "integer"}}},
                {"seed", {{"type", "integer"}}},
                {"threads", {{"type", "integer"}}},
                {"note", {{"type", "string"}}}},
               {"name"}),
        [this](const json& a) {
            const auto t0 = std::chrono::steady_clock::now();
            const auto elapsed = [&] {
                return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            };
            const std::string name = a.at("name").get<std::string>();
            if (a.contains("threads")) train::Parallel::setThreads(a["threads"].get<std::size_t>());
            const bool addBos = a.value("add_bos", false);
            const json items  = loadGoldset(a);
            if (items.empty()) throw std::runtime_error("no training items");
            Engine& e = _forward.engine();

            // 1. Examples (teacher responses from the base engine, before any adapter).
            std::vector<std::int32_t> stops = model::ChatTemplate::stopIds(
                model::ChatTemplate::detectFromArch(e.config().architecture), e.tokenizer());
            stops.push_back(e.tokenizer().eosId());
            const auto maxTeacher = a.value("max_teacher_tokens", std::size_t{96});
            std::vector<SftExample> data;
            std::size_t truncatedN = 0;
            for (const auto& it : items) {
                SftExample ex;
                const auto prompt = encodeText(e, it.at("prompt").get<std::string>(), addBos);
                std::vector<std::int32_t> resp;
                if (it.contains("response")) {
                    resp = encodeText(e, it["response"].get<std::string>(), false);
                } else if (it.contains("teacher_prompt")) {
                    const auto tp = encodeText(e, it["teacher_prompt"].get<std::string>(), addBos);
                    resp = generateGreedy(e, tp, maxTeacher);
                    ex.truncated = resp.size() >= maxTeacher;
                    if (!ex.truncated) resp.push_back(stops.front());
                } else {
                    throw std::runtime_error("every item needs 'response' or 'teacher_prompt'");
                }
                if (resp.empty()) throw std::runtime_error("empty response for an item");
                truncatedN += ex.truncated ? 1U : 0U;
                ex.response = e.tokenizer().decode(resp, false);
                ex.ids = prompt;
                ex.ids.insert(ex.ids.end(), resp.begin(), resp.end());
                ex.targets.assign(ex.ids.size(), -1);
                for (std::size_t i = prompt.size() - 1; i + 1 < ex.ids.size(); ++i) ex.targets[i] = ex.ids[i + 1];
                ex.promptTokens   = prompt.size();
                ex.responseTokens = resp.size();
                data.push_back(std::move(ex));
            }
            std::cerr << "hugin train_lora_sft: " << data.size() << " examples ready after " << elapsed() << " s\n";

            // 2. Model + fresh adapters.
            train::DenseModel& m = model();
            for (std::size_t b = 0; b < m.blockCount(); ++b) m.block(b).clearLora();
            std::vector<DenseBlock::Proj> projs;
            if (a.contains("targets")) {
                for (const auto& s : a["targets"]) projs.push_back(projArg(s.get<std::string>()));
            } else {
                projs.assign(std::begin(kAllProjs), std::end(kAllProjs));
            }
            std::vector<std::size_t> blocks;
            if (a.contains("blocks")) {
                for (const auto& b : a["blocks"]) {
                    const auto v = b.get<std::size_t>();
                    if (v >= m.blockCount()) throw std::runtime_error("block out of range");
                    blocks.push_back(v);
                }
            } else {
                blocks.resize(m.blockCount());
                std::iota(blocks.begin(), blocks.end(), std::size_t{0});
            }
            const auto  rank  = a.value("rank", std::size_t{8});
            const float alpha = a.value("alpha", 16.0F);
            const float scale = alpha / static_cast<float>(rank);
            std::mt19937 rng(a.value("seed", 7403U));
            struct Slot {
                std::size_t        block;
                DenseBlock::Proj   proj;
                train::AdamW       optA, optB;
            };
            const train::AdamW::Config oc{a.value("lr", 5e-4F), 0.9F, 0.999F, 1e-8F, 0.0F};
            std::vector<Slot> slots;
            for (const auto b : blocks) {
                for (const auto p : projs) {
                    const auto [in, out] = m.block(b).projShape(p);
                    // Kaiming-uniform A (as PEFT), B = 0: the model starts exactly at the base.
                    std::uniform_real_distribution<float> u(-1.0F / std::sqrt(static_cast<float>(in)),
                                                            1.0F / std::sqrt(static_cast<float>(in)));
                    std::vector<float> A(rank * in);
                    for (auto& v : A) v = u(rng);
                    m.block(b).setLora(p, rank, scale, std::move(A), std::vector<float>(out * rank, 0.0F));
                    slots.push_back(Slot{b, p, train::AdamW(rank * in, oc), train::AdamW(out * rank, oc)});
                }
            }

            // 3. Train.
            const auto epochs = a.value("epochs", std::size_t{2});
            const auto batch  = std::max<std::size_t>(1, a.value("batch", std::size_t{4}));
            const float clip  = a.value("clip", 1.0F);
            std::vector<std::size_t> order(data.size());
            std::iota(order.begin(), order.end(), std::size_t{0});
            json epochLoss = json::array(), stepLoss = json::array();
            std::size_t steps = 0;
            m.zeroGrad();
            for (std::size_t ep = 0; ep < epochs; ++ep) {
                std::shuffle(order.begin(), order.end(), rng);
                double sum = 0;
                double batchSum = 0;
                std::size_t inBatch = 0;
                for (std::size_t i = 0; i < order.size(); ++i) {
                    const SftExample& ex = data[order[i]];
                    const double l = m.lossAndGrad(ex.ids, ex.targets);
                    sum += l;
                    batchSum += l;
                    ++inBatch;
                    if (inBatch == batch || i + 1 == order.size()) {
                        // Mean over the batch, global-norm clip, one AdamW step per buffer.
                        const float inv = 1.0F / static_cast<float>(inBatch);
                        double sq = 0;
                        for (auto& s : slots) {
                            auto* l2 = m.block(s.block).lora(s.proj);
                            for (auto& g : l2->dA) { g *= inv; sq += static_cast<double>(g) * static_cast<double>(g); }
                            for (auto& g : l2->dB) { g *= inv; sq += static_cast<double>(g) * static_cast<double>(g); }
                        }
                        const double gn = std::sqrt(sq);
                        const float  cf = (clip > 0.0F && gn > static_cast<double>(clip))
                                              ? static_cast<float>(static_cast<double>(clip) / gn) : 1.0F;
                        for (auto& s : slots) {
                            auto* l2 = m.block(s.block).lora(s.proj);
                            if (cf != 1.0F) {
                                for (auto& g : l2->dA) g *= cf;
                                for (auto& g : l2->dB) g *= cf;
                            }
                            s.optA.step(l2->a, l2->dA);
                            s.optB.step(l2->b, l2->dB);
                        }
                        m.zeroGrad();
                        ++steps;
                        stepLoss.push_back({{"step", steps}, {"loss", batchSum / static_cast<double>(inBatch)},
                                            {"grad_norm", gn}});
                        std::cerr << "hugin train_lora_sft: epoch " << ep + 1 << " step " << steps << " loss "
                                  << batchSum / static_cast<double>(inBatch) << " |g| " << gn << " ("
                                  << elapsed() << " s)\n";
                        batchSum = 0;
                        inBatch  = 0;
                    }
                }
                epochLoss.push_back(sum / static_cast<double>(order.size()));
            }

            // 4. Stage as a LowRank rune.
            std::vector<RuneStaging::LowRankSpec> specs;
            for (const auto& s : slots) {
                const auto* l2 = m.block(s.block).lora(s.proj);
                specs.push_back({"blk." + std::to_string(s.block) + "." + projTensor(s.proj),
                                 static_cast<std::uint32_t>(rank), static_cast<double>(scale), l2->a, l2->b});
            }
            std::size_t respTok = 0;
            for (const auto& ex : data) respTok += ex.responseTokens;
            json tnames = json::array();
            for (const auto p : projs) tnames.push_back(projName(p));
            json params{{"method", "lora_sft"}, {"examples", data.size()}, {"response_tokens", respTok},
                        {"teacher_truncated", truncatedN}, {"targets", tnames}, {"blocks", blocks},
                        {"rank", rank}, {"alpha", alpha}, {"lr", oc.lr}, {"epochs", epochs},
                        {"batch", batch}, {"clip", clip}, {"seed", a.value("seed", 7403U)},
                        {"epoch_loss", epochLoss}, {"steps", steps},
                        {"note", a.value("note", std::string{})}};
            if (a.contains("path")) params["data_path"] = a["path"];
            json staged = _staging.stageLowRank(name, specs, params);

            // 5. Parity: trainer (with adapters) vs engine + rune on example 0, last prompt row.
            const SftExample& ex0 = data.front();
            const std::vector<std::int32_t> promptIds(ex0.ids.begin(),
                                                      ex0.ids.begin() + static_cast<std::ptrdiff_t>(ex0.promptTokens));
            const std::size_t row = ex0.promptTokens - 1;
            const std::vector<std::size_t> rows{row};
            const auto mine = m.logits(promptIds, rows);
            const auto base = forward(e, promptIds).at(row);
            Engine& v = _forward.engineWithRunes({staged.at("path").get<std::string>()});
            const auto ref = forward(v, promptIds).at(row);
            double maxAbs = 0, refMax = 0, shift = 0;
            for (std::size_t i = 0; i < ref.size(); ++i) {
                maxAbs = std::max(maxAbs, std::fabs(static_cast<double>(mine[i]) - static_cast<double>(ref[i])));
                refMax = std::max(refMax, std::fabs(static_cast<double>(ref[i])));
                shift  = std::max(shift, std::fabs(static_cast<double>(ref[i]) - static_cast<double>(base[i])));
            }
            const auto argmax = [](const std::vector<float>& x) {
                return static_cast<std::size_t>(std::max_element(x.begin(), x.end()) - x.begin());
            };
            // shift = how far the rune moves the engine's logits off the base;
            // the parity diff must be far below it to mean anything.
            const json parity{{"max_abs_logit_diff", maxAbs}, {"ref_abs_max", refMax},
                              {"rune_shift_vs_base", shift},
                              {"argmax_equal", argmax(mine) == argmax(ref)}};

            json samples = json::array();
            for (std::size_t i = 0; i < std::min<std::size_t>(3, data.size()); ++i) {
                samples.push_back({{"prompt_tokens", data[i].promptTokens}, {"response", data[i].response}});
            }
            return json{{"staged", staged}, {"epoch_loss", epochLoss},
                        {"first_step_loss", stepLoss.empty() ? json() : stepLoss.front()["loss"]},
                        {"last_step_loss", stepLoss.empty() ? json() : stepLoss.back()["loss"]},
                        {"steps", steps}, {"adapters", slots.size()},
                        {"trainable_params", [&] {
                             std::size_t n = 0;
                             for (const auto& s : slots) {
                                 const auto* l2 = m.block(s.block).lora(s.proj);
                                 n += l2->a.size() + l2->b.size();
                             }
                             return n;
                         }()},
                        {"engine_parity", parity}, {"teacher_samples", samples},
                        {"threads", train::Parallel::threads()}, {"seconds", elapsed()}};
        });
}

} // namespace mimirmind::hugin
