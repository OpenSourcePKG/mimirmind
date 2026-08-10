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

/// In-place exact (erf) GELU — PyTorch/HF nn.GELU default, as BERT /
/// RoBERTa / XLM-R use it in the encoder FFN (the EncoderRunner /
/// cross-encoder reranker path):
///   x[i] = 0.5 * x[i] * (1 + erf(x[i] / sqrt(2)))
/// Distinct from the tanh-approximation GELU in gelu_mul (Gemma FFN).
void geluErfInPlace(float* x, std::size_t n) noexcept;

/// In-place tanh: x[i] = tanh(x[i]). The activation in the RoBERTa/XLM-R
/// sequence-classification head (dense -> tanh -> out_proj) used by the
/// cross-encoder reranker (EncoderRunner).
void tanhInPlace(float* x, std::size_t n) noexcept;

} // namespace mimirmind::compute