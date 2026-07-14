// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstddef>

namespace mimirmind::compute {

/// In-place SiLU (a.k.a. Swish): x[i] = x[i] * sigmoid(x[i])
///                                    = x[i] / (1 + exp(-x[i]))
/// Llama / Qwen / Gemma FFN gate activation.
void siluInPlace(float* x, std::size_t n) noexcept;

/// In-place element-wise multiply: a[i] *= b[i] for i in [0, n).
/// Used as the "* up" half of SwiGLU after silu(gate).
void mulInPlace(float* a, const float* b, std::size_t n) noexcept;

} // namespace mimirmind::compute