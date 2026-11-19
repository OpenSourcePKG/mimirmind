// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstddef>
#include <vector>

namespace mimirmind::hugin::train {

/**
 * Naive (non-flash) causal GQA attention with optional sliding window, and
 * its backward (roadmap 7.4.1). Semantics are those of
 * compute::multiHeadAttention: q [Tq, H, D], k/v [Tk, Hkv, D], query row p
 * sits at absolute position positionOffset + p and sees keys
 * [kMax - window, kMax) with kMax = min(positionOffset + p + 1, Tk);
 * query head h reads KV head h * Hkv / H; scale <= 0 means 1/sqrt(D).
 *
 * The forward keeps the probabilities P [H, Tq, Tk] for the backward (short
 * training sequences; recompute in 7.4.2 keeps only the block inputs).
 */
class AttentionGrad {
public:
    struct Shape {
        std::size_t tq{0}, tk{0}, heads{0}, kvHeads{0}, headDim{0};
        std::size_t positionOffset{0};
        std::size_t window{0};
        float       scale{0.0F};
    };

    /// out [Tq, H, D]; probs is resized to H * Tq * Tk.
    static void forward(const float* q, const float* k, const float* v, const Shape& s,
                        float* out, std::vector<float>& probs);

    /// dq written; dk / dv [Tk, Hkv, D] written (summed over the query heads of a group).
    static void backward(const float* dOut, const float* q, const float* k, const float* v,
                         const Shape& s, const std::vector<float>& probs,
                         float* dq, float* dk, float* dv);

private:
    static float effectiveScale(const Shape& s);
    static void  keyRange(const Shape& s, std::size_t p, std::size_t& kMin, std::size_t& kMax);
};

} // namespace mimirmind::hugin::train
