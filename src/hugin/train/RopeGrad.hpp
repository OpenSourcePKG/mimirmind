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
 * RoPE is an orthogonal rotation per pair, so the backward is the rotation
 * by -theta applied to the upstream gradient.
 */
class RopeGrad {
public:
    enum class Layout { Split, Interleaved };

    static void forward(float* x, std::size_t T, std::size_t nHeads, std::size_t headDim,
                        std::size_t startPos, float base, Layout layout);

    /// In place: dy -> dx.
    static void backward(float* d, std::size_t T, std::size_t nHeads, std::size_t headDim,
                         std::size_t startPos, float base, Layout layout);

private:
    static void rotate(float* x, std::size_t T, std::size_t nHeads, std::size_t headDim,
                       std::size_t startPos, float base, Layout layout, float sign);
};

} // namespace mimirmind::hugin::train
