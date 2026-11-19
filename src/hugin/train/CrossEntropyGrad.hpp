// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstddef>
#include <cstdint>

namespace mimirmind::hugin::train {

/**
 * Mean token cross-entropy over logits [T, V] with targets [T] (a negative
 * target = ignored row, e.g. prompt tokens in SFT) and its gradient (7.4.1).
 */
class CrossEntropyGrad {
public:
    /// Returns the mean loss over the counted rows (0 if none); dLogits [T, V]
    /// written (zero on ignored rows) when non-null.
    static double lossAndGrad(const float* logits, const std::int32_t* targets,
                              std::size_t T, std::size_t V, float* dLogits);
};

} // namespace mimirmind::hugin::train
