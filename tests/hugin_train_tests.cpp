// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Roadmap 7.4.1: forward parity of the Hugin train ops with the engine's
// compute:: forward, and every analytic gradient against central finite
// differences (no Python oracle).

#include "TestFramework.hpp"

#include "compute/Attention.hpp"
#include "compute/Rope.hpp"
#include "hugin/train/AdamW.hpp"
#include "hugin/train/AttentionGrad.hpp"
#include "hugin/train/CrossEntropyGrad.hpp"
#include "hugin/train/DenseBlock.hpp"
#include "hugin/train/DenseModel.hpp"
#include "hugin/train/Parallel.hpp"
#include "hugin/train/LinearGrad.hpp"
#include "hugin/train/MoeCombineGrad.hpp"
#include "hugin/train/RmsNormGrad.hpp"
#include "hugin/train/RopeGrad.hpp"
#include "hugin/train/SwiGluGrad.hpp"

#include <cmath>
#include <cstdint>
#include <functional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

using namespace mimirmind::hugin::train;

namespace {

std::vector<float> randn(std::size_t n, std::uint32_t seed, float sd = 1.0F) {
    std::mt19937                    rng(seed);
    std::normal_distribution<float> dist(0.0F, sd);
    std::vector<float> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

/// sum_i r_i * y_i in double: the scalar loss whose dL/dy is r.
double dotLoss(const std::vector<float>& y, const std::vector<float>& r) {
    double s = 0;
    for (std::size_t i = 0; i < y.size(); ++i) s += static_cast<double>(y[i]) * static_cast<double>(r[i]);
    return s;
}

/// Central differences of `loss` w.r.t. every element of `x` vs `analytic`.
void checkGrad(const char* what, std::vector<float>& x, const std::vector<float>& analytic,
               const std::function<double()>& loss, double eps = 1e-2, double atol = 2e-3,
               double rtol = 2e-2) {
    if (x.size() != analytic.size()) throw std::runtime_error(std::string(what) + ": size mismatch");
    double worst = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const float keep = x[i];
        x[i] = keep + static_cast<float>(eps);
        const double lp = loss();
        x[i] = keep - static_cast<float>(eps);
        const double lm = loss();
        x[i] = keep;
        const double fd  = (lp - lm) / (2 * eps);
        const double err = std::fabs(fd - static_cast<double>(analytic[i]));
        worst = std::max(worst, err);
        if (err > atol + rtol * std::fabs(fd)) {
            std::ostringstream os;
            os << what << "[" << i << "]: analytic " << analytic[i] << " vs fd " << fd;
            throw std::runtime_error(os.str());
        }
    }
    std::printf("    %s: %zu elems, max |err| %.2e\n", what, x.size(), worst);
}

} // namespace

TEST(linear_lora_gradients) {
    const std::size_t M = 3, K = 5, N = 4, R = 2;
    auto x = randn(M * K, 1), w = randn(N * K, 2), a = randn(R * K, 3), b = randn(N * R, 4);
    const auto rr = randn(M * N, 5);
    const float scale = 0.7F;
    auto fwd = [&] {
        LoraParams lp{a.data(), b.data(), R, scale};
        std::vector<float> y(M * N);
        LinearGrad::forward(x.data(), M, K, w.data(), N, &lp, y.data());
        return dotLoss(y, rr);
    };
    LoraParams lp{a.data(), b.data(), R, scale};
    std::vector<float> dx(M * K), dA(R * K, 0.0F), dB(N * R, 0.0F);
    LinearGrad::backward(rr.data(), x.data(), M, K, w.data(), N, &lp, dx.data(), dA.data(), dB.data());
    checkGrad("dx", x, dx, fwd);
    checkGrad("dA", a, dA, fwd);
    checkGrad("dB", b, dB, fwd);
}

TEST(rmsnorm_gradients_and_per_head) {
    // M = T * nHeads rows of headDim also covers the per-head QK-norm.
    const std::size_t M = 6, K = 8;
    const float eps = 1e-6F;
    auto x = randn(M * K, 11), w = randn(K, 12);
    const auto rr = randn(M * K, 13);
    auto fwd = [&] {
        std::vector<float> y(M * K);
        RmsNormGrad::forward(x.data(), M, K, w.data(), eps, y.data());
        return dotLoss(y, rr);
    };
    std::vector<float> dx(M * K), dw(K, 0.0F);
    RmsNormGrad::backward(rr.data(), x.data(), M, K, w.data(), eps, dx.data(), dw.data());
    checkGrad("dx", x, dx, fwd, 1e-3);
    checkGrad("dw", w, dw, fwd, 1e-3);
}

TEST(rope_forward_parity_and_gradient) {
    const std::size_t T = 4, H = 3, D = 8, start = 5;
    const float base = 10000.0F;
    for (const auto layout : {RopeGrad::Layout::Split, RopeGrad::Layout::Interleaved}) {
        auto x = randn(T * H * D, 21);
        auto ref = x, mine = x;
        if (layout == RopeGrad::Layout::Split) {
            mimirmind::compute::applyRopeInPlace(ref.data(), T, H, D, start, base);
        } else {
            mimirmind::compute::applyRopeInPlaceInterleaved(ref.data(), nullptr, T, H, D, start, base);
        }
        RopeGrad::forward(mine.data(), T, H, D, start, base, layout);
        for (std::size_t i = 0; i < x.size(); ++i) EXPECT_NEAR(mine[i], ref[i], 1e-6F);

        const auto rr = randn(T * H * D, 22);
        auto fwd = [&] {
            auto y = x;
            RopeGrad::forward(y.data(), T, H, D, start, base, layout);
            return dotLoss(y, rr);
        };
        auto dx = rr;
        RopeGrad::backward(dx.data(), T, H, D, start, base, layout);
        checkGrad("dx", x, dx, fwd);
    }
}

TEST(attention_forward_parity_and_gradients) {
    struct Case { std::size_t tq, tk, off, window; float scale; };
    for (const Case c : {Case{5, 5, 0, 0, 0.0F}, Case{5, 5, 0, 3, 0.0F}, Case{2, 6, 4, 0, 0.3F},
                         Case{3, 7, 4, 2, 0.0F}}) {
        AttentionGrad::Shape s{c.tq, c.tk, 4, 2, 4, c.off, c.window, c.scale};
        const std::size_t D = s.headDim;
        auto q = randn(s.tq * s.heads * D, 31), k = randn(s.tk * s.kvHeads * D, 32),
             v = randn(s.tk * s.kvHeads * D, 33);
        std::vector<float> out(s.tq * s.heads * D), ref(out.size()), scratch(s.tk), probs;
        AttentionGrad::forward(q.data(), k.data(), v.data(), s, out.data(), probs);
        mimirmind::compute::multiHeadAttention(q.data(), k.data(), v.data(), s.tq, s.tk, s.heads,
                                               s.kvHeads, D, s.positionOffset, scratch.data(),
                                               ref.data(), s.window, s.scale);
        for (std::size_t i = 0; i < out.size(); ++i) EXPECT_NEAR(out[i], ref[i], 1e-5F);

        const auto rr = randn(out.size(), 34);
        auto fwd = [&] {
            std::vector<float> o(out.size()), p;
            AttentionGrad::forward(q.data(), k.data(), v.data(), s, o.data(), p);
            return dotLoss(o, rr);
        };
        std::vector<float> dq(q.size()), dk(k.size()), dv(v.size());
        AttentionGrad::backward(rr.data(), q.data(), k.data(), v.data(), s, probs,
                                dq.data(), dk.data(), dv.data());
        checkGrad("dq", q, dq, fwd);
        checkGrad("dk", k, dk, fwd);
        checkGrad("dv", v, dv, fwd);
    }
}

TEST(swiglu_gradients) {
    const std::size_t n = 16;
    auto g = randn(n, 41, 2.0F), u = randn(n, 42);
    const auto rr = randn(n, 43);
    auto fwd = [&] {
        std::vector<float> y(n);
        SwiGluGrad::forward(g.data(), u.data(), n, y.data());
        return dotLoss(y, rr);
    };
    std::vector<float> dg(n), du(n);
    SwiGluGrad::backward(rr.data(), g.data(), u.data(), n, dg.data(), du.data());
    checkGrad("dGate", g, dg, fwd);
    checkGrad("dUp", u, du, fwd);
}

TEST(moe_combine_gradients) {
    const std::size_t k = 3, D = 5;
    const float scale = 2.5F;
    auto z = randn(k, 51), x = randn(k * D, 52);
    const auto rr = randn(D, 53);
    auto fwd = [&] {
        std::vector<float> o(D);
        MoeCombineGrad::forward(z.data(), x.data(), k, D, scale, o.data());
        return dotLoss(o, rr);
    };
    std::vector<float> dz(k), dx(k * D);
    MoeCombineGrad::backward(rr.data(), z.data(), x.data(), k, D, scale, dz.data(), dx.data());
    checkGrad("dLogits", z, dz, fwd);
    checkGrad("dExpertOut", x, dx, fwd);
}

TEST(cross_entropy_gradient_and_ignore) {
    const std::size_t T = 4, V = 7;
    auto logits = randn(T * V, 61, 2.0F);
    const std::vector<std::int32_t> tgt{3, -1, 0, 6};
    std::vector<float> d(T * V);
    const double loss = CrossEntropyGrad::lossAndGrad(logits.data(), tgt.data(), T, V, d.data());
    EXPECT_TRUE(loss > 0.0);
    for (std::size_t v = 0; v < V; ++v) EXPECT_NEAR(d[1 * V + v], 0.0F, 0.0F);   // ignored row
    auto fwd = [&] { return CrossEntropyGrad::lossAndGrad(logits.data(), tgt.data(), T, V, nullptr); };
    checkGrad("dLogits", logits, d, fwd);
    const std::vector<std::int32_t> none{-1, -1, -1, -1};
    EXPECT_TRUE(CrossEntropyGrad::lossAndGrad(logits.data(), none.data(), T, V, d.data()) == 0.0);
}

TEST(adamw_matches_hand_computed_steps) {
    AdamW::Config cfg{0.1F, 0.9F, 0.999F, 1e-8F, 0.01F};
    AdamW opt(2, cfg);
    std::vector<float> p{1.0F, -2.0F};
    const std::vector<float> g{0.5F, -1.0F};
    opt.step(p, g);
    // Step 1: mhat = g, vhat = g^2 -> update = sign(g) (+ wd * p).
    EXPECT_NEAR(p[0], 1.0F - 0.1F * (1.0F + 0.01F * 1.0F), 1e-6F);
    EXPECT_NEAR(p[1], -2.0F - 0.1F * (-1.0F + 0.01F * -2.0F), 1e-6F);
    // Step 2 by hand for element 0.
    const double m = 0.9 * 0.05 + 0.1 * 0.5, v = 0.999 * 0.00025 + 0.001 * 0.25;
    const double mhat = m / (1 - 0.81), vhat = v / (1 - 0.999 * 0.999);
    const double p0 = p[0];
    opt.step(p, g);
    EXPECT_NEAR(p[0], static_cast<float>(p0 - 0.1 * (mhat / (std::sqrt(vhat) + 1e-8) + 0.01 * p0)), 1e-6F);
    EXPECT_EQ(opt.steps(), std::size_t{2});
}

TEST(lora_sft_toy_loss_decreases) {
    // End-to-end sanity of the pieces together: frozen linear + LoRA -> CE,
    // AdamW on A/B only; the loss must fall on a fixed tiny batch.
    const std::size_t M = 4, K = 6, V = 5, R = 2;
    auto x = randn(M * K, 71), w = randn(V * K, 72, 0.3F);
    auto a = randn(R * K, 73, 0.1F);
    std::vector<float> b(V * R, 0.0F);   // LoRA init: B = 0 -> starts at the base
    const std::vector<std::int32_t> tgt{1, 4, 2, 0};
    std::vector<float> params(a.size() + b.size());
    AdamW opt(params.size(), AdamW::Config{0.05F});
    double first = 0, last = 0;
    for (int it = 0; it < 60; ++it) {
        LoraParams lp{a.data(), b.data(), R, 1.0F};
        std::vector<float> y(M * V), dy(M * V), dx(M * K), dA(a.size(), 0.0F), dB(b.size(), 0.0F);
        LinearGrad::forward(x.data(), M, K, w.data(), V, &lp, y.data());
        const double loss = CrossEntropyGrad::lossAndGrad(y.data(), tgt.data(), M, V, dy.data());
        if (it == 0) first = loss;
        last = loss;
        LinearGrad::backward(dy.data(), x.data(), M, K, w.data(), V, &lp, dx.data(), dA.data(), dB.data());
        std::copy(a.begin(), a.end(), params.begin());
        std::copy(b.begin(), b.end(), params.begin() + static_cast<std::ptrdiff_t>(a.size()));
        std::vector<float> grads(dA);
        grads.insert(grads.end(), dB.begin(), dB.end());
        opt.step(params, grads);
        std::copy(params.begin(), params.begin() + static_cast<std::ptrdiff_t>(a.size()), a.begin());
        std::copy(params.begin() + static_cast<std::ptrdiff_t>(a.size()), params.end(), b.begin());
    }
    std::printf("    loss %.4f -> %.4f\n", first, last);
    EXPECT_TRUE(last < 0.5 * first);
}

TEST(rope_freq_factors_parity_and_gradient) {
    const std::size_t T = 3, H = 2, D = 8, start = 7;
    const float base = 500000.0F;
    const std::vector<float> ff{1.0F, 2.0F, 8.0F, 3.5F};
    auto x = randn(T * H * D, 25);
    auto ref = x, mine = x;
    mimirmind::compute::applyRopeInPlaceInterleaved(ref.data(), ff.data(), T, H, D, start, base);
    RopeGrad::forward(mine.data(), T, H, D, start, base, RopeGrad::Layout::Interleaved, ff.data());
    for (std::size_t i = 0; i < x.size(); ++i) EXPECT_NEAR(mine[i], ref[i], 1e-6F);
    const auto rr = randn(x.size(), 26);
    auto fwd = [&] {
        auto y = x;
        RopeGrad::forward(y.data(), T, H, D, start, base, RopeGrad::Layout::Interleaved, ff.data());
        return dotLoss(y, rr);
    };
    auto dx = rr;
    RopeGrad::backward(dx.data(), T, H, D, start, base, RopeGrad::Layout::Interleaved, ff.data());
    checkGrad("dx", x, dx, fwd);
}

TEST(dense_block_gradients_all_lora_targets) {
    // Tiny llama-style block (GQA 4/2, biases on q/k/v, interleaved RoPE with
    // factors), LoRA on all seven projections: every input and adapter
    // element against finite differences, through the recompute backward.
    DenseBlock::Config cfg;
    cfg.dModel = 8; cfg.heads = 4; cfg.kvHeads = 2; cfg.headDim = 4; cfg.ff = 12;
    cfg.eps = 1e-5F; cfg.ropeBase = 10000.0F;
    cfg.ropeLayout  = RopeGrad::Layout::Interleaved;
    cfg.ropeFactors = {1.0F, 4.0F};
    const std::size_t d = 8, qd = 16, kd = 8, ff = 12, T = 4;
    std::uint32_t seed = 100;
    DenseBlock::Weights w;
    w.attnNorm = randn(d, seed++); w.ffnNorm = randn(d, seed++);
    w.wq = randn(qd * d, seed++, 0.4F); w.bq = randn(qd, seed++, 0.1F);
    w.wk = randn(kd * d, seed++, 0.4F); w.bk = randn(kd, seed++, 0.1F);
    w.wv = randn(kd * d, seed++, 0.4F); w.bv = randn(kd, seed++, 0.1F);
    w.wo = randn(d * qd, seed++, 0.3F);
    w.wGate = randn(ff * d, seed++, 0.4F); w.wUp = randn(ff * d, seed++, 0.4F);
    w.wDown = randn(d * ff, seed++, 0.3F);
    DenseBlock blk(cfg, w);
    const std::size_t R = 2;
    using P = DenseBlock::Proj;
    for (const P p : {P::Q, P::K, P::V, P::O, P::Gate, P::Up, P::Down}) {
        const auto [in, out] = blk.projShape(p);
        auto a = randn(R * in, seed++, 0.3F);
        auto b = randn(out * R, seed++, 0.3F);
        blk.setLora(p, R, 0.5F, std::move(a), std::move(b));
    }
    auto x = randn(T * d, seed++);
    const auto rr = randn(T * d, seed++);
    auto fwd = [&] {
        std::vector<float> y(T * d);
        blk.forward(x.data(), T, y.data(), 3);
        return dotLoss(y, rr);
    };
    std::vector<float> dx(T * d);
    blk.backward(x.data(), T, rr.data(), dx.data(), 3);
    checkGrad("block dx", x, dx, fwd, 3e-3);
    for (const P p : {P::Q, P::K, P::V, P::O, P::Gate, P::Up, P::Down}) {
        auto* l = blk.lora(p);
        const auto dA = l->dA, dB = l->dB;
        checkGrad("lora dA", l->a, dA, fwd, 3e-3);
        checkGrad("lora dB", l->b, dB, fwd, 3e-3);
    }
    // Gradients accumulate across backward calls; zeroGrad resets them.
    const float before = blk.lora(P::Q)->dA[0];
    blk.backward(x.data(), T, rr.data(), dx.data(), 3);
    EXPECT_NEAR(blk.lora(P::Q)->dA[0], 2.0F * before, 1e-5F + 1e-4F * std::fabs(before));
    blk.zeroGrad();
    EXPECT_TRUE(blk.lora(P::Q)->dA[0] == 0.0F);
}

namespace {

DenseBlock tinyBlock(std::uint32_t& seed) {
    DenseBlock::Config cfg;
    cfg.dModel = 8; cfg.heads = 4; cfg.kvHeads = 2; cfg.headDim = 4; cfg.ff = 12;
    cfg.ropeLayout = RopeGrad::Layout::Interleaved;
    DenseBlock::Weights w;
    w.attnNorm = randn(8, seed++); w.ffnNorm = randn(8, seed++);
    w.wq = randn(16 * 8, seed++, 0.4F); w.wk = randn(8 * 8, seed++, 0.4F);
    w.wv = randn(8 * 8, seed++, 0.4F); w.wo = randn(8 * 16, seed++, 0.3F);
    w.wGate = randn(12 * 8, seed++, 0.4F); w.wUp = randn(12 * 8, seed++, 0.4F);
    w.wDown = randn(8 * 12, seed++, 0.3F);
    return DenseBlock(cfg, w);
}

} // namespace

TEST(dense_model_sft_gradients_through_ce) {
    // Tiny 3-block model, vocab 11, untied head; adapters only from block 1
    // (the backward must stop there), prompt rows ignored by the targets.
    std::uint32_t seed = 300;
    std::vector<DenseBlock> blocks;
    for (int i = 0; i < 3; ++i) blocks.push_back(tinyBlock(seed));
    DenseModel::Head head;
    head.vocab = 11;
    head.embed = randn(11 * 8, seed++);
    head.finalNorm = randn(8, seed++);
    head.lmHead = randn(11 * 8, seed++, 0.5F);
    DenseModel m(std::move(blocks), std::move(head));
    using P = DenseBlock::Proj;
    for (const std::size_t b : {std::size_t{1}, std::size_t{2}}) {
        for (const P p : {P::Q, P::V, P::Down}) {
            const auto [in, out] = m.block(b).projShape(p);
            auto a = randn(2 * in, seed++, 0.3F);
            auto bb = randn(out * 2, seed++, 0.3F);
            m.block(b).setLora(p, 2, 0.5F, std::move(a), std::move(bb));
        }
    }
    const std::vector<std::int32_t> ids{1, 4, 7, 2, 9, 3};
    const std::vector<std::int32_t> tgt{-1, -1, 2, 9, 3, 10};
    auto fwd = [&] {
        // Loss only (gradients of this probe call are discarded below).
        const double l = m.lossAndGrad(ids, tgt);
        return l;
    };
    m.zeroGrad();
    (void)m.lossAndGrad(ids, tgt);
    // Snapshot every gradient first: the FD probes call lossAndGrad again and
    // keep accumulating into the adapters.
    std::vector<std::pair<std::vector<float>, std::vector<float>>> grads;
    for (const std::size_t b : {std::size_t{1}, std::size_t{2}}) {
        for (const P p : {P::Q, P::V, P::Down}) grads.emplace_back(m.block(b).lora(p)->dA, m.block(b).lora(p)->dB);
    }
    std::size_t gi = 0;
    for (const std::size_t b : {std::size_t{1}, std::size_t{2}}) {
        for (const P p : {P::Q, P::V, P::Down}) {
            auto* l = m.block(b).lora(p);
            checkGrad("model dA", l->a, grads[gi].first, fwd, 3e-3);
            checkGrad("model dB", l->b, grads[gi].second, fwd, 3e-3);
            ++gi;
        }
    }
    // Logits of the target rows equal a full forward's rows.
    const std::vector<std::size_t> rows{2, 5};
    const auto part = m.logits(ids, rows);
    const auto all  = m.logits(ids);
    for (std::size_t v = 0; v < 11; ++v) {
        EXPECT_NEAR(part[v], all[2 * 11 + v], 1e-6F);
        EXPECT_NEAR(part[11 + v], all[5 * 11 + v], 1e-6F);
    }
}

TEST(parallel_linear_matches_serial) {
    // Threaded LinearGrad == single-thread result (same chunk math per element).
    const std::size_t M = 5, K = 300, N = 257, R = 3;
    auto x = randn(M * K, 81), w = randn(N * K, 82), a = randn(R * K, 83), b = randn(N * R, 84);
    const auto dy = randn(M * N, 85);
    LoraParams lp{a.data(), b.data(), R, 0.7F};
    auto run = [&](std::size_t threads, std::vector<float>& y, std::vector<float>& dx,
                   std::vector<float>& dA, std::vector<float>& dB) {
        Parallel::setThreads(threads);
        y.assign(M * N, 0.0F); dx.assign(M * K, 0.0F); dA.assign(R * K, 0.0F); dB.assign(N * R, 0.0F);
        LinearGrad::forward(x.data(), M, K, w.data(), N, &lp, y.data());
        LinearGrad::backward(dy.data(), x.data(), M, K, w.data(), N, &lp, dx.data(), dA.data(), dB.data());
    };
    std::vector<float> y1, dx1, dA1, dB1, y8, dx8, dA8, dB8;
    run(1, y1, dx1, dA1, dB1);
    run(8, y8, dx8, dA8, dB8);
    Parallel::setThreads(0);
    for (std::size_t i = 0; i < y1.size(); ++i) EXPECT_NEAR(y1[i], y8[i], 0.0F);
    for (std::size_t i = 0; i < dx1.size(); ++i) EXPECT_NEAR(dx1[i], dx8[i], 0.0F);
    for (std::size_t i = 0; i < dA1.size(); ++i) EXPECT_NEAR(dA1[i], dA8[i], 0.0F);
    for (std::size_t i = 0; i < dB1.size(); ++i) EXPECT_NEAR(dB1[i], dB8[i], 0.0F);
}

int main() { return mm::test::run(); }
