// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

namespace mimirmind::hugin::train {

/**
 * One trainable transformer block built from the train ops (roadmap 7.4):
 * frozen base weights, optional LoRA adapters on up to seven projections,
 * forward and a recompute backward (only the block input is kept between
 * the two). DenseBlock (llama / qwen2) and MoeBlock (Kolibri-1) implement it;
 * TrainModel stacks them.
 *
 * Projection slots: Q / K / V / O of the attention and Gate / Up / Down of
 * the block's dense MLP — the whole FFN of a dense block, the always-on
 * shared expert of a MoE block (routed experts stay frozen).
 */
class TrainBlock {
public:
    enum class Proj { Q, K, V, O, Gate, Up, Down };
    static constexpr std::size_t kProjCount = 7;
    struct Lora {
        std::size_t        rank{0};
        float              scale{1.0F};
        std::vector<float> a, b;     ///< [rank, in], [out, rank]
        std::vector<float> dA, dB;   ///< accumulated gradients, same shapes
    };

    TrainBlock()                                 = default;
    TrainBlock(const TrainBlock&)                = default;
    TrainBlock(TrainBlock&&) noexcept            = default;
    TrainBlock& operator=(const TrainBlock&)     = default;
    TrainBlock& operator=(TrainBlock&&) noexcept = default;
    virtual ~TrainBlock()                        = default;

    [[nodiscard]] virtual std::size_t dModel() const noexcept = 0;
    /// (in, out) feature counts of a projection.
    [[nodiscard]] virtual std::pair<std::size_t, std::size_t> projShape(Proj p) const = 0;

    /// x, y [T, dModel]; the block input sits at absolute position startPos.
    virtual void forward(const float* x, std::size_t T, float* y, std::size_t startPos = 0) const = 0;
    /// dx [T, dModel] written; LoRA dA / dB accumulated.
    virtual void backward(const float* x, std::size_t T, const float* dy, float* dx,
                          std::size_t startPos = 0) = 0;

    /// Installs a LoRA on `p` (a [rank, in], b [out, rank]); gradients zeroed.
    void setLora(Proj p, std::size_t rank, float scale, std::vector<float> a, std::vector<float> b);
    [[nodiscard]] Lora*       lora(Proj p);
    [[nodiscard]] const Lora* lora(Proj p) const;
    [[nodiscard]] bool        hasLora() const noexcept;
    void zeroGrad();
    /// Removes every adapter (back to the frozen base block).
    void clearLora();

protected:
    /// out [T, N] = in [T, K] W^T (+ bias) (+ the LoRA of `p`), (K, N) = projShape(p).
    void loraLinear(Proj p, const float* w, const float* bias, const float* in, std::size_t T,
                    float* out) const;
    /// dIn [T, K] written (+= when accumulate); the LoRA gradients of `p` accumulated.
    void loraLinearBack(Proj p, const float* w, const float* dOut, const float* in, std::size_t T,
                        float* dIn, bool accumulate);

private:
    std::array<std::optional<Lora>, kProjCount> _lora;
};

} // namespace mimirmind::hugin::train
