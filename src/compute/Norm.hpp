// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstddef>

namespace mimirmind::compute {

/**
 * Per-row RMSNorm as Llama / Gemma / Qwen use it:
 *
 *   y[i] = x[i] * w[i] / sqrt( mean(x^2) + eps )
 *
 * Inputs:
 *   x:      [M, K] F32 row-major
 *   weight: [K]    F32 — the per-feature scale (γ); no β (no bias).
 *   eps:    additive term inside the sqrt
 *
 * Output:
 *   y: [M, K] F32 row-major
 *
 * Sum-of-squares is accumulated in double for numerical stability;
 * everything else is single precision.
 */
void rmsNorm(const float* x,
             std::size_t  M,
             std::size_t  K,
             const float* weight,
             float        eps,
             float*       y);

/**
 * Per-row LayerNorm as BERT / RoBERTa / XLM-R use it (the cross-encoder
 * reranker path — EncoderRunner):
 *
 *   mean   = mean(x[m, :])
 *   var    = mean( (x[m, :] - mean)^2 )          (biased, two-pass)
 *   y[i]   = (x[i] - mean) / sqrt(var + eps) * weight[i] + bias[i]
 *
 * Inputs:
 *   x:      [M, K] F32 row-major
 *   weight: [K]    F32 — per-feature scale (γ)
 *   bias:   [K]    F32 — per-feature shift (β)
 *   eps:    additive term inside the sqrt
 *
 * Output:
 *   y: [M, K] F32 row-major
 *
 * Two-pass (mean, then variance from (x-mean)^2) to match PyTorch/HF
 * LayerNorm numerics exactly — matters for the cross-encoder parity gate.
 * Mean and variance are accumulated in double for stability.
 */
void layerNorm(const float* x,
               std::size_t  M,
               std::size_t  K,
               const float* weight,
               const float* bias,
               float        eps,
               float*       y);

} // namespace mimirmind::compute