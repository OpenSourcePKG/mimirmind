// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// l0_moe_topk_probe — parity check for the M-Q3N.5 device-side MoE top-K
// router (kernels/moe_topk.cl -> moe_topk.spv) on Xe-LPG (the NUC target).
// Routes random logits through compute::l0::MoeTopKRouteDevice and compares
// the kept expert indices and renormalised weights against the CPU golden
// reference compute::moeTopKRoute.
//
// This is the parity gate for the L0 kernel + launcher before they get wired
// into the backend — the primary path for the 100 tok/s NUC goal. The kernel
// is the same algorithm as the CUDA/HIP ports (warp/SIMD-size agnostic), so a
// green here corroborates all three.
//
// Indices must match exactly; weights use the combined-tolerance pattern
// (|diff| <= abs_tol + rel_tol * |ref|). Tie-break note: moeTopKRoute leaves
// equal-probability ties implementation-defined, the kernel fixes them to
// lowest-index; with continuous random logits an exact tie has ~zero
// probability, so index parity holds regardless.

#include "compute/MoeRouting.hpp"
#include "compute/l0/GpuOps.hpp"
#include "compute/l0/MoeTopKRouteDevice.hpp"
#include "core/gpu/l0/L0ComputeContext.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <vector>

namespace {

using ::mimirmind::compute::ComputeBuffer;
using ::mimirmind::compute::l0::GpuOps;
using ::mimirmind::compute::l0::MoeTopKRouteDevice;
using ::mimirmind::core::l0::L0ComputeContext;

// Deterministic pseudo-random in [-1, 1) — reproducible, no rand/Date.
std::vector<float> randVec(std::size_t n, std::uint32_t seed) {
    std::uint32_t x = seed;
    std::vector<float> v(n);
    for (auto& f : v) {
        x = x * 1664525u + 1013904223u;
        f = static_cast<float>((x >> 8) & 0xFFFFu) / 32768.0f - 1.0f;
    }
    return v;
}

struct Config {
    std::size_t   T;
    std::size_t   nExperts;
    std::size_t   K;
    float         wScale;
    std::uint32_t seed;
};

bool runConfig(GpuOps& ops, MoeTopKRouteDevice& router, const Config& c) {
    const std::size_t nLogit = c.T * c.nExperts;
    const std::size_t nOut   = c.T * c.K;

    const auto logits = randVec(nLogit, c.seed);

    // CPU golden reference (weights renormalise to 1, no wScale applied).
    std::vector<std::int32_t> refIdx(nOut);
    std::vector<float>        refW(nOut);
    ::mimirmind::compute::moeTopKRoute(logits.data(), c.T, c.nExperts, c.K,
                                       refIdx.data(), refW.data());

    ComputeBuffer dLogits = ops.allocate(nLogit * sizeof(float));
    ops.uploadHostBytes(dLogits.get(), logits.data(), nLogit * sizeof(float));
    ComputeBuffer dIdx = ops.allocate(nOut * sizeof(std::int32_t));
    ComputeBuffer dW   = ops.allocate(nOut * sizeof(float));

    router.launch(static_cast<const float*>(dLogits.get()),
                  static_cast<std::int32_t*>(dIdx.get()),
                  static_cast<float*>(dW.get()),
                  c.T, c.nExperts, c.K, c.wScale);
    ops.flush();

    std::vector<std::int32_t> gpuIdx(nOut);
    std::vector<float>        gpuW(nOut);
    ops.readbackToHost(gpuIdx.data(), dIdx.get(), nOut * sizeof(std::int32_t));
    ops.readbackToHost(gpuW.data(),   dW.get(),   nOut * sizeof(float));

    constexpr float kAbsTol = 1e-4f;
    constexpr float kRelTol = 1e-4f;

    std::size_t idxMismatch = 0;
    float       maxWRatio   = 0.0f;
    for (std::size_t i = 0; i < nOut; ++i) {
        if (gpuIdx[i] != refIdx[i]) {
            ++idxMismatch;
        }
        const float ref = refW[i] * c.wScale;
        const float d   = std::fabs(gpuW[i] - ref);
        const float thr = kAbsTol + kRelTol * std::fabs(ref);
        const float r   = d / thr;
        if (r > maxWRatio) {
            maxWRatio = r;
        }
    }

    const bool ok = (idxMismatch == 0) && (maxWRatio <= 1.0f);
    std::printf("  T=%zu nE=%zu K=%zu wScale=%.3g seed=%#x : "
                "idxMismatch=%zu  maxW/tol=%.3f  %s\n",
                c.T, c.nExperts, c.K, static_cast<double>(c.wScale), c.seed,
                idxMismatch, static_cast<double>(maxWRatio),
                ok ? "OK" : "FAIL");
    return ok;
}

} // namespace

int main() {
    std::printf("l0_moe_topk_probe:\n");

    try {
        L0ComputeContext   ctx{};
        GpuOps             ops{ctx};
        MoeTopKRouteDevice router{ctx};

        const Config configs[] = {
            {/*T=*/6, /*nE=*/128, /*K=*/8, /*wScale=*/1.0f, 0xA11Cu},
            {/*T=*/4, /*nE=*/ 64, /*K=*/4, /*wScale=*/2.5f, 0xBEEFu},
            {/*T=*/1, /*nE=*/128, /*K=*/8, /*wScale=*/1.0f, 0x5EEDu},
        };

        bool allOk = true;
        for (const auto& c : configs) {
            allOk &= runConfig(ops, router, c);
        }

        std::printf("\nl0_moe_topk_probe: %s\n", allOk ? "OK" : "FAIL");
        return allOk ? 0 : 1;

    } catch (const std::exception& e) {
        std::fprintf(stderr, "l0_moe_topk_probe: threw: %s\n", e.what());
        return 2;
    }
}