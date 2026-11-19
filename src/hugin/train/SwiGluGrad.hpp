// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstddef>

namespace mimirmind::hugin::train {

/// y = silu(gate) * up, elementwise over n values, and its backward (7.4.1).
class SwiGluGrad {
public:
    static void forward(const float* gate, const float* up, std::size_t n, float* y);
    /// dGate / dUp written.
    static void backward(const float* dy, const float* gate, const float* up, std::size_t n,
                         float* dGate, float* dUp);
};

} // namespace mimirmind::hugin::train
