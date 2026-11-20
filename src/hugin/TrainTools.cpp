// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "hugin/TrainTools.hpp"

#include "compute/ComputeOps.hpp"
#include "core/modelopt/BlockScaleSwizzle.hpp"
#include "hugin/ForwardSupport.hpp"
#include "hugin/ForwardTools.hpp"
#include "hugin/GenerationCheck.hpp"
#include "hugin/HostDequant.hpp"
#include "hugin/RuneStaging.hpp"
#include "hugin/train/AdamW.hpp"
#include "hugin/train/DenseBlock.hpp"
#include "hugin/train/MoeBlock.hpp"
#include "hugin/train/TrainModel.hpp"
#include "hugin/train/Parallel.hpp"
#include "model/ChatTemplate.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
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
    HostDequant::toF32(t.type, bytes.data(), out.size(), out.data());
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

/// Loads one routed expert of a stacked bank ([experts, rows, in]) to F32.
/// NVFP4_TC reads the expert's nibbles, swizzled scale slot and global; every
/// other format is a per-expert byte slice of the main buffer. The engine and
/// the tensor are resolved on every load: under --single-engine the base can
/// be released and reloaded (rune parity / gate) while a TrainModel is cached.
class ExpertBank {
public:
    ExpertBank(ForwardTools& fw, std::size_t block, std::string name, std::size_t experts, std::size_t rows,
               std::size_t in)
        : _fw(&fw), _block(block), _name(std::move(name)), _experts(experts), _rows(rows), _in(in) {}

    void load(std::size_t ex, float* dst) const {
        if (ex >= _experts) throw std::runtime_error(_name + ": expert out of range");
        Engine& e = _fw->engine();
        const auto* t = e.weights().findBlock(_block, _name);
        if (t == nullptr || (t->usmPtr == nullptr && t->tcNibblePtr == nullptr)) {
            throw std::runtime_error("blk." + std::to_string(_block) + "." + _name + " not found");
        }
        if (t->type == core::gguf::GgmlType::NVFP4_TC) {
            if (t->tcSfbPtr == nullptr || t->tcGlobalsPtr == nullptr) {
                throw std::runtime_error(_name + ": NVFP4_TC bank without scale / global side banks");
            }
            const std::size_t nibBytes = _rows * _in / 2;
            const std::size_t sfbBytes = core::modelopt::moeSwizzledScaleStride(_rows, _in / 16);
            std::vector<std::uint8_t> nib(nibBytes), sfb(sfbBytes);
            float global = 0.0F;
            const void* nibSrc = t->tcNibblePtr != nullptr ? t->tcNibblePtr : t->usmPtr;
            e.ops()->readbackToHost(nib.data(), static_cast<const std::uint8_t*>(nibSrc) + ex * nibBytes, nibBytes);
            e.ops()->readbackToHost(sfb.data(), static_cast<const std::uint8_t*>(t->tcSfbPtr) + ex * sfbBytes,
                                    sfbBytes);
            e.ops()->readbackToHost(&global, static_cast<const float*>(t->tcGlobalsPtr) + ex, sizeof(float));
            HostDequant::nvfp4TcToF32(nib.data(), sfb.data(), global, _rows, _in, dst);
            return;
        }
        if (t->nbytes % _experts != 0) throw std::runtime_error(_name + ": bank size not divisible by experts");
        const std::size_t slice = t->nbytes / _experts;
        std::vector<std::uint8_t> bytes(slice);
        e.ops()->readbackToHost(bytes.data(), static_cast<const std::uint8_t*>(t->usmPtr) + ex * slice, slice);
        HostDequant::toF32(t->type, bytes.data(), _rows * _in, dst);
    }

private:
    ForwardTools* _fw;
    std::size_t   _block;
    std::string   _name;
    std::size_t   _experts, _rows, _in;
};

/// 7.4.4: one Kolibri-1 block (see Kolibri1Backend::runBlock) with its routed
/// experts streamed from the engine's banks one expert at a time.
train::MoeBlock loadKolibriBlock(ForwardTools& fw, std::size_t L) {
    Engine& e = fw.engine();
    const auto& c = e.config();
    train::MoeBlock::Config cfg;
    cfg.dModel     = c.embeddingLength;
    cfg.heads      = c.headCount;
    cfg.kvHeads    = c.headCountKvFor(L);
    cfg.headDim    = c.headDim(L);
    cfg.experts    = c.expertCount;
    cfg.topK       = c.expertUsedCount;
    cfg.expertFf   = c.expertFeedForwardLength;
    cfg.sharedFf   = c.expertSharedFeedForwardLength;
    cfg.eps        = c.rmsNormEps;
    cfg.sliding    = L < c.slidingWindowPattern.size() && static_cast<bool>(c.slidingWindowPattern[L]);
    cfg.window     = cfg.sliding ? c.slidingWindow : 0;
    cfg.ropeBase   = c.ropeFreqBaseSwa;
    cfg.ropeLayout = train::RopeGrad::Layout::Split;   // ropeInPlaceAsync = NEOX split
    cfg.attnScale  = c.attentionScaleFor(cfg.headDim);
    train::MoeBlock::Weights w;
    w.attnNorm     = blockTensor(e, L, "attn_norm.weight", true);
    w.wq           = blockTensor(e, L, "attn_q.weight", true);
    w.wk           = blockTensor(e, L, "attn_k.weight", true);
    w.wv           = blockTensor(e, L, "attn_v.weight", true);
    w.wo           = blockTensor(e, L, "attn_output.weight", true);
    w.qNorm        = blockTensor(e, L, "attn_q_norm.weight", true);
    w.kNorm        = blockTensor(e, L, "attn_k_norm.weight", true);
    w.postAttnNorm = blockTensor(e, L, "post_attn_norm.weight", true);
    w.ffnNorm      = blockTensor(e, L, "post_attention_norm.weight", true);
    w.postFfnNorm  = blockTensor(e, L, "post_ffn_norm.weight", true);
    w.router       = blockTensor(e, L, "ffn_gate_inp.weight", true);
    w.routerBias   = blockTensor(e, L, "exp_probs_b.weight", true);
    w.shGate       = blockTensor(e, L, "ffn_gate_shexp.weight", true);
    w.shUp         = blockTensor(e, L, "ffn_up_shexp.weight", true);
    w.shDown       = blockTensor(e, L, "ffn_down_shexp.weight", true);
    const std::size_t E = cfg.experts, ff = cfg.expertFf, d = cfg.dModel;
    auto gate = std::make_shared<ExpertBank>(fw, L, "ffn_gate_exps.weight", E, ff, d);
    auto up   = std::make_shared<ExpertBank>(fw, L, "ffn_up_exps.weight", E, ff, d);
    auto down = std::make_shared<ExpertBank>(fw, L, "ffn_down_exps.weight", E, d, ff);
    std::vector<float> probe(ff * d);
    gate->load(0, probe.data());   // fail at build time, not mid-training, on an unknown bank format
    auto loader = [gate, up, down](std::size_t ex, float* g, float* u, float* dn) {
        gate->load(ex, g);
        up->load(ex, u);
        down->load(ex, dn);
    };
    return train::MoeBlock(cfg, std::move(w), std::move(loader));
}

/// The train block of layer L for the engine's architecture.
std::unique_ptr<train::TrainBlock> loadBlock(ForwardTools& fw, std::size_t L) {
    Engine& e = fw.engine();
    const std::string& arch = e.config().architecture;
    if (arch == "llama" || arch == "qwen2") return std::make_unique<DenseBlock>(loadDenseBlock(e, L));
    if (arch == "kolibri1") return std::make_unique<train::MoeBlock>(loadKolibriBlock(fw, L));
    throw std::runtime_error("the train ops support llama / qwen2 / kolibri1 blocks (model is '" + arch + "')");
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

/// Engine tensor of a LoRA slot; on kolibri1 the MLP slots are the shared expert.
const char* projTensor(const std::string& arch, DenseBlock::Proj p) {
    if (arch == "kolibri1") {
        if (p == DenseBlock::Proj::Gate) return "ffn_gate_shexp.weight";
        if (p == DenseBlock::Proj::Up) return "ffn_up_shexp.weight";
        if (p == DenseBlock::Proj::Down) return "ffn_down_shexp.weight";
    }
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
    std::string               kind;
    std::vector<std::int32_t> teacherPrompt;   ///< 7.4.5 distill: the teacher's prompt tokens
};

} // namespace

TrainTools::TrainTools(ForwardTools& forward, RuneStaging& staging)
    : _forward(forward), _staging(staging) {}

TrainTools::~TrainTools() = default;

train::TrainModel& TrainTools::model() {
    const auto t = _forward.target();
    if (_model && _modelKey == t.path) return *_model;
    _model.reset();
    Engine& e = _forward.engine();
    std::vector<std::unique_ptr<train::TrainBlock>> blocks;
    for (std::size_t L = 0; L < e.config().blockCount; ++L) blocks.push_back(loadBlock(_forward, L));
    train::TrainModel::Head head;
    head.embed     = globalTensor(e, "token_embd.weight", true);
    head.finalNorm = globalTensor(e, "output_norm.weight", true);
    head.lmHead    = globalTensor(e, "output.weight", false);   // empty = tied
    head.eps       = e.config().rmsNormEps;
    head.vocab     = head.embed.size() / e.config().embeddingLength;
    _model    = std::make_unique<train::TrainModel>(std::move(blocks), std::move(head));
    _modelKey = t.path;
    return *_model;
}

void TrainTools::registerOn(McpServer& server) {
    server.addTool(
        "train_block_check",
        "7.4.2 training-core check on real weights. Builds block 'block' (>= 1; dense llama / qwen2 or the kolibri1 MoE block, routed experts streamed per expert) "
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

            const auto blkPtr = loadBlock(_forward, L);
            train::TrainBlock& blk = *blkPtr;
            const std::size_t T = ids.size(), d = blk.dModel();
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
            json out{{"block", L}, {"tokens", T}, {"arch", e.config().architecture},
                     {"forward_parity", parity}, {"gradient_checks", checks},
                     {"all_gradients_ok", allOk}, {"seconds", secs}};
            if (const auto* db = dynamic_cast<const DenseBlock*>(&blk)) {
                out["rope"] = db->config().ropeLayout == train::RopeGrad::Layout::Interleaved ? "interleaved" : "split";
                out["rope_factors"] = !db->config().ropeFactors.empty();
            }
            if (const auto* mb = dynamic_cast<const train::MoeBlock*>(&blk)) {
                const auto& sel = mb->lastSelection();
                std::vector<std::int32_t> used(sel.begin(), sel.end());
                std::sort(used.begin(), used.end());
                used.erase(std::unique(used.begin(), used.end()), used.end());
                out["sliding"] = mb->config().sliding;
                out["experts_used"] = used.size();
            }
            return out;
        });

    server.addTool(
        "train_lora_sft",
        "7.4.3 LoRA-SFT of the whole model (llama / qwen2; kolibri1 since 7.4.4 with the MLP slots on the shared expert) on CPU train ops; the base stays "
        "frozen, only adapters learn. Data: 'items' or 'path' (.json / .jsonl) of {prompt, response} "
        "or {prompt, teacher_prompt}: with teacher_prompt the response is the BASE model's greedy "
        "answer to teacher_prompt (self-distillation, e.g. the same question asked in German), ended "
        "by the template's stop token unless cut at 'max_teacher_tokens' (default 96); an item's "
        "'teacher_check' (eval generate spec: lang / match / ...) drops teacher answers that fail it. "
        "'data_out' writes the final {kind, prompt, response} pairs as .jsonl for re-use. Loss = CE on "
        "the response tokens only. Adapters: 'targets' (default q k v o gate up down) on 'blocks' "
        "(default all), 'rank' (8), 'alpha' (16; scale alpha/rank), B = 0 at start. AdamW 'lr' "
        "(5e-4), 'epochs' (2), 'batch' (4, gradient accumulation), global grad-norm clip 'clip' "
        "(1.0), 'seed'. 7.4.5 KL anchor: 'kl_lambda' > 0 adds lambda * KL(base || adapted) on the "
        "response rows of every item whose kind is in 'kl_kinds' (default: every kind except "
        "'target'); the base distribution is cached before training as its top 'kl_topk' (64) tokens "
        "plus one tail bucket. Those items then train on the KL only, plus their CE with 'kl_ce'. "
        "Context distillation: 'distill_lambda' > 0 makes items of 'distill_kinds' (default 'target') "
        "whose teacher_prompt differs from their prompt also learn the base's full top-K distribution "
        "under the teacher prompt on the same response tokens (KL(teacher || student)), next to "
        "their CE unless 'distill_ce' is false; teacher_prompt is kept when a response is given. "
        "The result is staged as a LowRank Rune 'name' (then rune_gate / rune_commit "
        "as usual), and the trainer's logits are compared with the engine running that Rune "
        "(Futhark load path) on the first example.",
        schema({{"name", {{"type", "string"}}},
                {"items", {{"type", "array"}, {"items", {{"type", "object"}}}}},
                {"data_out", {{"type", "string"}}},
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
                {"kl_lambda", {{"type", "number"}}},
                {"kl_topk", {{"type", "integer"}}},
                {"kl_kinds", {{"type", "array"}, {"items", {{"type", "string"}}}}},
                {"kl_ce", {{"type", "boolean"}}},
                {"distill_lambda", {{"type", "number"}}},
                {"distill_kinds", {{"type", "array"}, {"items", {{"type", "string"}}}}},
                {"distill_ce", {{"type", "boolean"}}},
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
            std::size_t truncatedN = 0, droppedN = 0;
            json dropped = json::array();
            for (const auto& it : items) {
                SftExample ex;
                ex.kind = it.value("kind", std::string{});
                const auto prompt = encodeText(e, it.at("prompt").get<std::string>(), addBos);
                if (it.contains("teacher_prompt")) {
                    ex.teacherPrompt = encodeText(e, it["teacher_prompt"].get<std::string>(), addBos);
                }
                std::vector<std::int32_t> resp;
                if (it.contains("response")) {
                    resp = encodeText(e, it["response"].get<std::string>(), false);
                } else if (it.contains("teacher_prompt")) {
                    const auto tp = encodeText(e, it["teacher_prompt"].get<std::string>(), addBos);
                    resp = generateGreedy(e, tp, maxTeacher);
                    ex.truncated = resp.size() >= maxTeacher;
                    if (it.contains("teacher_check")) {
                        // Keep only teacher answers that pass the item's own check
                        // (e.g. the right language): a wrong answer would be learned.
                        const GenerationCheck check(it["teacher_check"]);
                        const json r = check.evaluate(e.tokenizer().decode(resp, true));
                        if (!r.at("correct").get<bool>()) {
                            ++droppedN;
                            if (dropped.size() < 8) {
                                dropped.push_back({{"id", it.value("id", std::string{})},
                                                   {"checks", r.at("checks")},
                                                   {"text", e.tokenizer().decode(resp, true)}});
                            }
                            continue;
                        }
                    }
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
            if (data.empty()) throw std::runtime_error("no example passed its teacher_check");
            std::cerr << "hugin train_lora_sft: " << data.size() << " examples ready (" << droppedN
                      << " dropped by teacher_check) after " << elapsed() << " s\n";
            if (a.contains("data_out")) {
                // Re-usable data: the exact prompt/response pairs trained on (the
                // response text includes the stop token unless truncated).
                const std::filesystem::path out = a["data_out"].get<std::string>();
                std::ofstream f(out);
                if (!f) throw std::runtime_error("cannot write data_out '" + out.string() + "'");
                for (const auto& ex : data) {
                    const std::vector<std::int32_t> pr(ex.ids.begin(),
                                                       ex.ids.begin() + static_cast<std::ptrdiff_t>(ex.promptTokens));
                    f << json{{"kind", ex.kind}, {"prompt", e.tokenizer().decode(pr, false)},
                              {"response", ex.response}}.dump()
                      << "\n";
                }
            }

            // 2. Model + fresh adapters.
            train::TrainModel& m = model();
            for (std::size_t b = 0; b < m.blockCount(); ++b) m.block(b).clearLora();

            // 7.4.5 KL anchor: the base distribution (no adapters yet) on the
            // response rows of every item whose kind is in 'kl_kinds', top-K sparse.
            const double klLambda = a.value("kl_lambda", 0.0);
            const auto   klTopK   = a.value("kl_topk", std::size_t{64});
            const bool   klCe     = a.value("kl_ce", false);
            std::vector<std::string> klKinds;
            if (a.contains("kl_kinds")) {
                for (const auto& k : a["kl_kinds"]) klKinds.push_back(k.get<std::string>());
            }
            const auto anchored = [&](const SftExample& ex) {
                if (klLambda <= 0.0) return false;
                if (klKinds.empty()) return ex.kind != "target";
                return std::find(klKinds.begin(), klKinds.end(), ex.kind) != klKinds.end();
            };
            // Context distillation: items with a teacher_prompt other than their
            // prompt (kinds in 'distill_kinds', default 'target') learn the base's
            // full distribution under the teacher prompt on the same response
            // tokens, weighted 'distill_lambda'; their CE stays unless distill_ce = false.
            const double distillLambda = a.value("distill_lambda", 0.0);
            const bool   distillCe     = a.value("distill_ce", true);
            std::vector<std::string> distillKinds{"target"};
            if (a.contains("distill_kinds")) {
                distillKinds.clear();
                for (const auto& k : a["distill_kinds"]) distillKinds.push_back(k.get<std::string>());
            }
            std::vector<std::optional<train::TrainModel::Anchor>> anchors(data.size());
            std::vector<double> anchorWeight(data.size(), 0.0);
            std::vector<char>   anchorCe(data.size(), 0);
            std::vector<char>   isDistill(data.size(), 0);
            std::size_t distillItems = 0, distillRows = 0;
            double distillTail = 0;
            for (std::size_t i = 0; i < data.size(); ++i) {
                const SftExample& ex = data[i];
                if (distillLambda <= 0.0 || ex.teacherPrompt.empty() ||
                    std::find(distillKinds.begin(), distillKinds.end(), ex.kind) == distillKinds.end()) {
                    continue;
                }
                const std::vector<std::int32_t> prompt(ex.ids.begin(),
                                                       ex.ids.begin() + static_cast<std::ptrdiff_t>(ex.promptTokens));
                if (prompt == ex.teacherPrompt) continue;   // self-teacher: nothing to distil
                // Teacher sequence = teacher prompt + the same response; row j of
                // the response there maps to row promptTokens - 1 + j here.
                std::vector<std::int32_t> tseq = ex.teacherPrompt;
                tseq.insert(tseq.end(), ex.ids.begin() + static_cast<std::ptrdiff_t>(ex.promptTokens), ex.ids.end());
                std::vector<std::size_t> trows, srows;
                for (std::size_t j = 0; j < ex.responseTokens; ++j) {
                    trows.push_back(ex.teacherPrompt.size() - 1 + j);
                    srows.push_back(ex.promptTokens - 1 + j);
                }
                auto anc = m.anchor(tseq, trows, klTopK);
                anc.rows = std::move(srows);
                for (const float tl : anc.tail) distillTail += static_cast<double>(tl);
                anchors[i]      = std::move(anc);
                anchorWeight[i] = distillLambda;
                anchorCe[i]     = distillCe ? 1 : 0;
                isDistill[i]    = 1;
                ++distillItems;
                distillRows += ex.responseTokens;
            }
            if (distillLambda > 0.0) {
                if (distillItems == 0) throw std::runtime_error("distill_lambda > 0 but no item has a distinct teacher_prompt");
                std::cerr << "hugin train_lora_sft: distillation on " << distillItems << " items / " << distillRows
                          << " rows, mean tail mass " << distillTail / static_cast<double>(distillRows) << "\n";
            }
            std::size_t anchorItems = 0, anchorRows = 0;
            double anchorTail = 0;
            for (std::size_t i = 0; i < data.size(); ++i) {
                if (anchors[i] || !anchored(data[i])) continue;
                std::vector<std::size_t> rows;
                for (std::size_t t = 0; t < data[i].targets.size(); ++t) {
                    if (data[i].targets[t] >= 0) rows.push_back(t);
                }
                anchors[i]      = m.anchor(data[i].ids, rows, klTopK);
                anchorWeight[i] = klLambda;
                anchorCe[i]     = klCe ? 1 : 0;
                ++anchorItems;
                anchorRows += rows.size();
                for (const float tl : anchors[i]->tail) anchorTail += static_cast<double>(tl);
            }
            if (klLambda > 0.0) {
                if (anchorItems == 0) throw std::runtime_error("kl_lambda > 0 but no item matches kl_kinds");
                std::cerr << "hugin train_lora_sft: KL anchor on " << anchorItems << " items / " << anchorRows
                          << " rows, top-" << klTopK << " mean tail mass "
                          << anchorTail / static_cast<double>(anchorRows) << " (" << elapsed() << " s)\n";
            }
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
                for (std::size_t b = 0; b < m.blockCount(); ++b) blocks.push_back(b);
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
            json epochLoss = json::array(), stepLoss = json::array(), epochCe = json::array(),
                 epochKl = json::array(), epochDistill = json::array();
            std::size_t steps = 0;
            m.zeroGrad();
            for (std::size_t ep = 0; ep < epochs; ++ep) {
                std::shuffle(order.begin(), order.end(), rng);
                double sum = 0, ceSum = 0, klSum = 0, distSum = 0;
                std::size_t ceN = 0, klN = 0, distN = 0;
                double batchSum = 0;
                std::size_t inBatch = 0;
                for (std::size_t i = 0; i < order.size(); ++i) {
                    const SftExample& ex = data[order[i]];
                    double l = 0;
                    if (const auto& anc = anchors[order[i]]; anc) {
                        // Anchor items: KL to the base (+ CE only with kl_ce).
                        const std::vector<std::int32_t> none(ex.targets.size(), -1);
                        const double w  = anchorWeight[order[i]];
                        const bool   ce = anchorCe[order[i]] != 0;
                        const auto r = m.lossAndGrad(ex.ids, ce ? ex.targets : none, &*anc, w);
                        l = r.total(w);
                        if (isDistill[order[i]] != 0) {
                            distSum += r.kl;
                            ++distN;
                        } else {
                            klSum += r.kl;
                            ++klN;
                        }
                        if (ce) {
                            ceSum += r.ce;
                            ++ceN;
                        }
                    } else {
                        l = m.lossAndGrad(ex.ids, ex.targets);
                        ceSum += l;
                        ++ceN;
                    }
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
                epochCe.push_back(ceN > 0 ? ceSum / static_cast<double>(ceN) : 0.0);
                epochKl.push_back(klN > 0 ? klSum / static_cast<double>(klN) : 0.0);
                epochDistill.push_back(distN > 0 ? distSum / static_cast<double>(distN) : 0.0);
            }

            // 4. Stage as a LowRank rune.
            std::vector<RuneStaging::LowRankSpec> specs;
            for (const auto& s : slots) {
                const auto* l2 = m.block(s.block).lora(s.proj);
                specs.push_back({"blk." + std::to_string(s.block) + "." + projTensor(e.config().architecture, s.proj),
                                 static_cast<std::uint32_t>(rank), static_cast<double>(scale), l2->a, l2->b});
            }
            std::size_t respTok = 0;
            for (const auto& ex : data) respTok += ex.responseTokens;
            json tnames = json::array();
            for (const auto p : projs) tnames.push_back(projName(p));
            json params{{"method", "lora_sft"}, {"examples", data.size()}, {"response_tokens", respTok},
                        {"teacher_truncated", truncatedN}, {"teacher_dropped", droppedN},
                        {"targets", tnames}, {"blocks", blocks},
                        {"rank", rank}, {"alpha", alpha}, {"lr", oc.lr}, {"epochs", epochs},
                        {"batch", batch}, {"clip", clip}, {"seed", a.value("seed", 7403U)},
                        {"epoch_loss", epochLoss}, {"epoch_ce", epochCe}, {"steps", steps},
                        {"note", a.value("note", std::string{})}};
            if (distillLambda > 0.0) {
                params["distill"] = {{"lambda", distillLambda}, {"ce", distillCe}, {"items", distillItems},
                                     {"rows", distillRows}, {"epoch_kl", epochDistill}};
            }
            if (klLambda > 0.0) {
                params["kl"] = {{"lambda", klLambda}, {"topk", klTopK}, {"ce_on_anchors", klCe},
                                {"anchor_items", anchorItems}, {"anchor_rows", anchorRows},
                                {"epoch_kl", epochKl}};
            }
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
            json klInfo;
            if (klLambda > 0.0) {
                klInfo = {{"lambda", klLambda}, {"topk", klTopK}, {"ce_on_anchors", klCe},
                          {"anchor_items", anchorItems}, {"anchor_rows", anchorRows},
                          {"mean_tail_mass", anchorTail / static_cast<double>(anchorRows)},
                          {"epoch_kl", epochKl}};
            }
            json distillInfo;
            if (distillLambda > 0.0) {
                distillInfo = {{"lambda", distillLambda}, {"ce", distillCe}, {"items", distillItems},
                               {"rows", distillRows},
                               {"mean_tail_mass", distillTail / static_cast<double>(distillRows)},
                               {"epoch_kl", epochDistill}};
            }
            return json{{"staged", staged}, {"epoch_loss", epochLoss}, {"epoch_ce", epochCe}, {"kl", klInfo},
                        {"distill", distillInfo},
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
                        {"teacher_dropped", droppedN}, {"teacher_dropped_samples", dropped},
                        {"threads", train::Parallel::threads()}, {"seconds", elapsed()}};
        });
}

} // namespace mimirmind::hugin
