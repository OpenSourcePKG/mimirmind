// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstddef>

namespace mimirmind::compute {

/// Qwen4-Exp PLE (per-layer-embedding) device-forward ops (5.27 I-4): the
/// signed-sqrt stream gate + the dilated depthwise causal conv1d+silu.
/// Segregated out of the backend-neutral `ComputeOps` (8.30.6 ISP-split)
/// because only the CUDA/Bragi backend implements them — qwen4_exp is a
/// CUDA-only arch. A backend that supports them returns a non-null `IPleOps*`
/// from `ComputeOps::pleOps()`; every other backend returns nullptr, so the
/// neutral base no longer carries throw-default stubs for this concern.
/// See kernels/cuda/llm/ple_forward.cu.
class IPleOps {
public:
    virtual ~IPleOps() = default;

    /// PLE signed-sqrt stream gate: gated[t,g*d+j] =
    /// sigmoid(sign(g)*sqrt(max(|g|,1e-6))) * value[t,j], g =
    /// (keyNormed·queryNormed over d)/sqrt(d) per stream. keyNormed/queryNormed
    /// are [T, hc*d]; value [T, d]; gated [T, hc*d].
    virtual void pleGateAsync(const float* keyNormed, const float* queryNormed,
                              const float* value, float* gated,
                              std::size_t T, std::size_t hc, std::size_t d) = 0;

    /// PLE dilated depthwise causal conv1d + silu over [T, hcd] with a decode
    /// state [stateLen, hcd] (nullptr = zero state / prefill). w is [hcd, K].
    virtual void pleConvSiluAsync(const float* x, const float* state,
                                  const float* w, float* out, std::size_t T,
                                  std::size_t hcd, std::size_t K,
                                  std::size_t dilation, std::size_t stateLen) = 0;
};

} // namespace mimirmind::compute
