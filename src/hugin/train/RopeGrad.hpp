// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstddef>

namespace mimirmind::hugin::train {

/**
 * RoPE forward / backward on [T, nHeads, headDim] (roadmap 7.4.1).
 *
 * Same angles as compute::applyRopeInPlace (split / NEOX pairs (i, i+D/2))
 * and compute::applyRopeInPlaceInterleaved (adjacent pairs (2i, 2i+1)).
 * `freqFactors` (nullable, headDim/2 values) divides each pair's frequency
 * like the engine's llama3 / YaRN factor path. RoPE is an orthogonal rotation
 * per pair, so the backward is the rotation by -theta of the upstream gradient.
 */
class RopeGrad {
public:
    enum class Layout { Split, Interleaved };

    static void forward(float* x, std::size_t T, std::size_t nHeads, std::size_t headDim,
                        std::size_t startPos, float base, Layout layout,
                        const float* freqFactors = nullptr);

    /// In place: dy -> dx.
    static void backward(float* d, std::size_t T, std::size_t nHeads, std::size_t headDim,
                         std::size_t startPos, float base, Layout layout,
                         const float* freqFactors = nullptr);

private:
    static void rotate(float* x, std::size_t T, std::size_t nHeads, std::size_t headDim,
                       std::size_t startPos, float base, Layout layout, const float* freqFactors,
                       float sign);
};

} // namespace mimirmind::hugin::train
