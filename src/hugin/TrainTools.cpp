// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "hugin/TrainTools.hpp"

#include "compute/ComputeOps.hpp"
#include "compute/Dequant.hpp"
#include "hugin/ForwardSupport.hpp"
#include "hugin/ForwardTools.hpp"
#include "hugin/train/DenseBlock.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
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

DenseBlock::Proj projArg(const std::string& s) {
    for (const auto p : {DenseBlock::Proj::Q, DenseBlock::Proj::K, DenseBlock::Proj::V, DenseBlock::Proj::O,
                         DenseBlock::Proj::Gate, DenseBlock::Proj::Up, DenseBlock::Proj::Down}) {
        if (s == projName(p)) return p;
    }
    throw std::runtime_error("unknown LoRA target '" + s + "' (q, k, v, o, gate, up, down)");
}

} // namespace

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
}

} // namespace mimirmind::hugin
