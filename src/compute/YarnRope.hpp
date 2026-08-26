// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace mimirmind::compute {

/**
 * YaRN (NTK-by-parts) long-context RoPE scaling (roadmap 8.8).
 *
 * Extends a model's usable context beyond its trained `original_max_position`
 * by interpolating the RoPE frequencies. It maps onto the EXISTING freq_factors
 * kernel (ropeInPlaceWithFactorsAsync: `theta_i = pos * base^(-2i/d) /
 * factor[i]`): the returned factor[i] ramps from 1.0 on the high-frequency dims
 * (extrapolate — keep the base angle) to `factor` (== the extension scale) on
 * the low-frequency dims (interpolate — stretch the angle by 1/scale), with a
 * linear ramp between the beta_fast/beta_slow "correction dimensions". This is
 * the standard YaRN blend (HF `_compute_yarn_parameters` / the YaRN paper).
 *
 * DYNAMIC use: the caller applies these factors ONLY when the sequence length
 * exceeds `original_max_position`. Below that, use base RoPE (factor == 1
 * everywhere) → bit-identical to today → zero short-context regression.
 */
[[nodiscard]] inline std::vector<float> computeYarnFreqFactors(
    std::size_t headDim,
    double      base,            // rope_theta
    double      factor,          // extension scale (e.g. 4 → 4x context)
    double      origMaxPos,      // original_max_position_embeddings
    double      betaFast = 32.0, // high-freq correction (fewer rotations)
    double      betaSlow = 1.0)  // low-freq correction (more rotations)
{
    const std::size_t half = headDim / 2;
    std::vector<float> ff(half, 1.0F);
    if (factor <= 1.0 || origMaxPos <= 0.0 || headDim == 0) {
        return ff;   // no extension → identity factors (base RoPE)
    }

    constexpr double kTwoPi = 6.283185307179586;
    // The dimension index whose wavelength completes `numRotations` full turns
    // over `origMaxPos` positions. Higher numRotations (beta_fast) → higher
    // frequency → smaller index.
    const auto correctionDim = [&](double numRotations) {
        return (static_cast<double>(headDim) *
                std::log(origMaxPos / (numRotations * kTwoPi))) /
               (2.0 * std::log(base));
    };
    double low  = std::floor(correctionDim(betaFast));
    double high = std::ceil(correctionDim(betaSlow));
    low  = std::max(low, 0.0);
    high = std::min(high, static_cast<double>(half > 0 ? half - 1 : 0));
    const double span = high - low;

    for (std::size_t i = 0; i < half; ++i) {
        double ramp = (span > 0.0)
            ? (static_cast<double>(i) - low) / span
            : (static_cast<double>(i) < low ? 0.0 : 1.0);
        ramp = std::clamp(ramp, 0.0, 1.0);
        // ramp 0 (high-freq, i<low): extrapolate → factor 1 (unchanged angle).
        // ramp 1 (low-freq,  i>high): interpolate → factor == scale (angle/scale).
        ff[i] = static_cast<float>(1.0 * (1.0 - ramp) + factor * ramp);
    }
    return ff;
}

/// YaRN attention temperature (mscale): applied as a multiplier on the
/// attention softmax scale to compensate for the frequency stretch. 1.0 when
/// there is no extension.
[[nodiscard]] inline double yarnAttentionMscale(double factor) {
    return (factor <= 1.0) ? 1.0 : (0.1 * std::log(factor) + 1.0);
}

} // namespace mimirmind::compute
