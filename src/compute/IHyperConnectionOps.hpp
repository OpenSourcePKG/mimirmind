// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstddef>

namespace mimirmind::compute {

/// Qwen4-Exp Hyper-Connections device glue (5.27 I-3): the elementwise ops for
/// the 4-stream GatedResidual. Segregated out of the backend-neutral
/// `ComputeOps` (8.30.6 ISP-split) because only the CUDA/Bragi backend
/// implements them — qwen4_exp is a CUDA-only arch. A backend that supports
/// them returns a non-null `IHyperConnectionOps*` from
/// `ComputeOps::hyperConnectionOps()`; every other backend returns nullptr, so
/// the neutral base no longer carries throw-default stubs for this concern.
/// See kernels/cuda/llm/hyper_connection.cu; math verified by the q4e_hc parity
/// harness.
class IHyperConnectionOps {
public:
    virtual ~IHyperConnectionOps() = default;

    /// Grouped RMSNorm over a [T, hc*d] token-major stream tensor: normalise
    /// each `d`-chunk (stream) independently, then scale by the (1+w)-baked
    /// weight `wBaked` [hc*d]. normed = groupRMS(x) * wBaked.
    virtual void hcGroupedRmsNormAsync(const float* x, const float* wBaked,
                                       float* normed, std::size_t T,
                                       std::size_t hc, std::size_t d,
                                       float eps) = 0;

    /// In-place x = silu(x * scale) over n elements.
    virtual void hcSiluScaleAsync(float* x, std::size_t n, float scale) = 0;

    /// Weighted stream mean: mixed[t,j] = (1/hc) * sum_g w2[t,g*d+j]*normed[t,g*d+j].
    virtual void hcWeightedMeanStreamsAsync(const float* w2, const float* normed,
                                            float* mixed, std::size_t T,
                                            std::size_t hc, std::size_t d) = 0;

    /// Injection scatter (HC residual add): x[t,g*d+j] += inj[t,g]*moduleOut[t,j].
    virtual void hcInjectScatterAsync(float* x, const float* moduleOut,
                                      const float* inj, std::size_t T,
                                      std::size_t hc, std::size_t d) = 0;

    /// Stream broadcast (embed repeat x hc at forward start): dst[t,g*d+j]=src[t,j].
    virtual void hcStreamBroadcastAsync(const float* src, float* dst,
                                        std::size_t T, std::size_t hc,
                                        std::size_t d) = 0;
};

} // namespace mimirmind::compute
