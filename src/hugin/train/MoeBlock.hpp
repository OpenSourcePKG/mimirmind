// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "hugin/train/RopeGrad.hpp"
#include "hugin/train/TrainBlock.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

namespace mimirmind::hugin::train {

/**
 * One Kolibri-1 block (pure-attention MoE) built from the train ops
 * (roadmap 7.4.4), same math as runtime::arch::Kolibri1Backend::runBlock:
 *
 *   n1 = RMS(x) w_in;  q,k,v = n1 W^T;  q,k per-head RMS (q_norm / k_norm)
 *   RoPE(q, k) on sliding layers only (full layers are NoPE)
 *   c  = Attn(q, k, v, window on sliding layers)
 *   h  = x + RMS(c Wo^T) w_post_attn                          (sandwich norm)
 *   m  = RMS(h) w_pre_ffn;  z = m Wr^T  [T, experts]
 *   top-K experts per token on z + bias (selection only, frozen)
 *   f  = sum_k sigmoid(z_k) E_k(m)  +  S(m)     (no renorm; S = shared expert)
 *   y  = h + RMS(f) w_post_ffn
 *
 * E_k / S are SwiGLU MLPs. LoRA slots: Q K V O and Gate Up Down of the shared
 * expert; routed experts and the router stay frozen. The gradient still flows
 * through the router logits of the selected experts (sigmoid weights) into m.
 *
 * Routed experts are never held as a whole: `ExpertLoader` fills one expert's
 * F32 weights on demand (the engine side dequantises FP8 / NVFP4 per expert),
 * experts are processed one at a time over the rows that picked them, so the
 * block needs constant memory however many experts a sequence touches. The
 * backward recomputes the forward (only x is kept) and reloads each used
 * expert once more.
 */
class MoeBlock final : public TrainBlock {
public:
    struct Config {
        std::size_t      dModel{0}, heads{0}, kvHeads{0}, headDim{0};
        std::size_t      experts{0}, topK{0}, expertFf{0}, sharedFf{0};
        float            eps{1e-6F};
        bool             sliding{false};     ///< RoPE + window; false = full attention, NoPE
        std::size_t      window{0};          ///< sliding layers only
        float            ropeBase{10000.0F};
        RopeGrad::Layout ropeLayout{RopeGrad::Layout::Split};
        float            attnScale{0.0F};    ///< <= 0: 1/sqrt(headDim)
    };
    /// Row-major F32, engine layout W [out, in].
    struct Weights {
        std::vector<float> attnNorm, wq, wk, wv, wo, qNorm, kNorm, postAttnNorm;
        std::vector<float> ffnNorm, postFfnNorm, router, routerBias;
        std::vector<float> shGate, shUp, shDown;
    };
    /// Fills routed expert `e`: gate / up [expertFf, dModel], down [dModel, expertFf].
    /// Called from several threads at once (experts run in parallel).
    using ExpertLoader = std::function<void(std::size_t e, float* gate, float* up, float* down)>;

    MoeBlock(Config cfg, Weights w, ExpertLoader experts);

    [[nodiscard]] const Config& config() const noexcept { return _cfg; }
    [[nodiscard]] std::size_t dModel() const noexcept override { return _cfg.dModel; }
    [[nodiscard]] std::pair<std::size_t, std::size_t> projShape(Proj p) const override;

    void forward(const float* x, std::size_t T, float* y, std::size_t startPos = 0) const override;
    void backward(const float* x, std::size_t T, const float* dy, float* dx,
                  std::size_t startPos = 0) override;

    /// The experts picked in the last forward / backward, [T, topK] (diagnostic).
    [[nodiscard]] const std::vector<std::int32_t>& lastSelection() const noexcept { return _lastSel; }

private:
    struct Acts {
        std::vector<float> n1, q, k, v, qn, kn, ctx, probs, o, h, m, z;
        std::vector<std::int32_t> sel;               ///< [T, topK]
        std::vector<float> sg, su, sAct, routedOut, f, y;
    };
    void run(const float* x, std::size_t T, std::size_t startPos, Acts& a) const;
    /// Experts used by `sel`, each with the (token, slot) pairs that chose it.
    struct Use {
        std::size_t              expert{0};
        std::vector<std::size_t> pairs;   ///< t * topK + k
    };
    [[nodiscard]] std::vector<Use> uses(const std::vector<std::int32_t>& sel) const;
    [[nodiscard]] const std::vector<float>& weight(Proj p) const;

    Config       _cfg;
    Weights      _w;
    ExpertLoader _experts;
    mutable std::vector<std::int32_t> _lastSel;
};

} // namespace mimirmind::hugin::train
