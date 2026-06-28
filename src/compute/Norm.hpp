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

} // namespace mimirmind::compute