// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "hugin/train/RopeGrad.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

namespace mimirmind::hugin::train {

/**
 * One dense pre-norm transformer block (llama / qwen2 layout) built from the
 * train ops, with frozen F32 base weights and optional LoRA adapters on each
 * projection (roadmap 7.4.2):
 *
 *   n1 = RMSNorm(x);  q,k,v = n1 W^T (+ bias);  RoPE(q, k);  c = Attn(q, k, v)
 *   h  = x + c Wo^T;  n2 = RMSNorm(h);  y = h + (silu(n2 Wg^T) * n2 Wu^T) Wd^T
 *
 * backward() recomputes the forward from the block input (activation
 * checkpointing: only x is kept between forward and backward) and
 * accumulates the LoRA gradients; the base weights never get a gradient.
 */
class DenseBlock {
public:
    struct Config {
        std::size_t      dModel{0}, heads{0}, kvHeads{0}, headDim{0}, ff{0};
        float            eps{1e-5F};
        float            ropeBase{10000.0F};
        RopeGrad::Layout ropeLayout{RopeGrad::Layout::Split};
        std::vector<float> ropeFactors;   ///< empty or headDim/2 values
        float            attnScale{0.0F}; ///< <= 0: 1/sqrt(headDim)
    };
    /// Row-major F32, engine layout W [out, in]. Empty bias = none.
    struct Weights {
        std::vector<float> attnNorm, wq, bq, wk, bk, wv, bv, wo;
        std::vector<float> ffnNorm, wGate, wUp, wDown;
    };
    enum class Proj { Q, K, V, O, Gate, Up, Down };
    static constexpr std::size_t kProjCount = 7;
    struct Lora {
        std::size_t        rank{0};
        float              scale{1.0F};
        std::vector<float> a, b;     ///< [rank, in], [out, rank]
        std::vector<float> dA, dB;   ///< accumulated gradients, same shapes
    };

    DenseBlock(Config cfg, Weights w);

    [[nodiscard]] const Config& config() const noexcept { return _cfg; }
    /// (in, out) feature counts of a projection.
    [[nodiscard]] std::pair<std::size_t, std::size_t> projShape(Proj p) const;

    /// Installs a LoRA on `p` (a [rank, in], b [out, rank]); gradients zeroed.
    void setLora(Proj p, std::size_t rank, float scale, std::vector<float> a, std::vector<float> b);
    [[nodiscard]] Lora*       lora(Proj p);
    [[nodiscard]] const Lora* lora(Proj p) const;
    void zeroGrad();
    /// Removes every adapter (back to the frozen base block).
    void clearLora();

    /// x, y [T, dModel]; the block input sits at absolute position startPos.
    void forward(const float* x, std::size_t T, float* y, std::size_t startPos = 0) const;

    /// dx [T, dModel] written; LoRA dA / dB accumulated.
    void backward(const float* x, std::size_t T, const float* dy, float* dx,
                  std::size_t startPos = 0);

private:
    struct Acts {
        std::vector<float> n1, q, k, v, ctx, probs, h, n2, g, u, act, y;
    };
    void run(const float* x, std::size_t T, std::size_t startPos, Acts& a) const;
    void linear(Proj p, const float* in, std::size_t T, float* out) const;
    /// dIn written (+= when accumulate).
    void linearBack(Proj p, const float* dOut, const float* in, std::size_t T, float* dIn,
                    bool accumulate);
    [[nodiscard]] const std::vector<float>& weight(Proj p) const;
    [[nodiscard]] const std::vector<float>* bias(Proj p) const;

    Config                                     _cfg;
    Weights                                    _w;
    std::array<std::optional<Lora>, kProjCount> _lora;
};

} // namespace mimirmind::hugin::train
