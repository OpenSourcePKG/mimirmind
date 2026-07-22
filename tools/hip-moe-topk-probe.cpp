// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// hip_moe_topk_probe — parity check for the M-Q3N.5 device-side MoE top-K
// router (kernels_hip/moe_topk.hip) on gfx1101 (radeon-testbed). Loads
// moe_topk.hsaco, routes random logits, and compares the kept expert indices
// and renormalised weights against the CPU golden reference
// compute::moeTopKRoute.
//
// This is the parity gate for the HIP kernel before it gets wired into the
// backend. The kernel is warp-size agnostic (single work-item per block, no
// shuffles), so a green here also validates the CUDA/L0 ports of the same
// algorithm.
//
// Indices must match exactly; weights use the combined-tolerance pattern
// (|diff| <= abs_tol + rel_tol * |ref|). Tie-break note: moeTopKRoute leaves
// equal-probability ties implementation-defined, the kernel fixes them to
// lowest-index; with continuous random logits an exact tie has ~zero
// probability, so index parity holds regardless.

#include "compute/MoeRouting.hpp"

#include "core/gpu/hip/HipContext.hpp"
#include "core/gpu/hip/HipKernel.hpp"
#include "core/gpu/hip/HipMemoryAllocator.hpp"
#include "core/gpu/hip/HipModule.hpp"
#include "core/gpu/hip/HipStream.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <string>
#include <vector>

namespace {

using namespace mimirmind::core::hip;

constexpr std::uint32_t kBlock = 32;   // warp-aligned; only thread 0 computes

std::string defaultHsacoPath(const char* argv0) {
    std::filesystem::path exe = std::filesystem::canonical(argv0);
    return (exe.parent_path() / "hsaco" / "moe_topk.hsaco").string();
}

// Deterministic pseudo-random in [-1, 1) — reproducible, no rand/Date.
void fillRandom(std::vector<float>& v, std::uint32_t seed) {
    std::uint32_t x = seed;
    for (auto& f : v) {
        x = x * 1664525u + 1013904223u;
        f = static_cast<float>((x >> 8) & 0xFFFFu) / 32768.0f - 1.0f;
    }
}

struct Config {
    std::size_t   T;
    std::size_t   nExperts;
    std::size_t   K;
    float         wScale;
    std::uint32_t seed;
};

// Run one config; returns true on parity pass. Prints a per-config line.
bool runConfig(HipMemoryAllocator& alloc, HipStream& stream, HipKernel& kernel,
               const Config& c) {
    const std::size_t nLogit = c.T * c.nExperts;
    const std::size_t nOut   = c.T * c.K;

    std::vector<float> logits(nLogit);
    fillRandom(logits, c.seed);

    // CPU golden reference (weights renormalise to 1, no wScale applied).
    std::vector<std::int32_t> refIdx(nOut);
    std::vector<float>        refW(nOut);
    mimirmind::compute::moeTopKRoute(logits.data(), c.T, c.nExperts, c.K,
                                     refIdx.data(), refW.data());

    // Device buffers.
    HipBuffer devLogits{alloc, nLogit * sizeof(float)};
    HipBuffer devIdx   {alloc, nOut   * sizeof(std::int32_t)};
    HipBuffer devW     {alloc, nOut   * sizeof(float)};
    alloc.copyH2D(devLogits.data(), logits.data(), nLogit * sizeof(float));

    kernel.setPtr  (0, devLogits.data());
    kernel.setPtr  (1, devIdx.data());
    kernel.setPtr  (2, devW.data());
    kernel.setValue(3, static_cast<std::int32_t>(c.nExperts));
    kernel.setValue(4, static_cast<std::int32_t>(c.K));
    kernel.setValue(5, c.wScale);
    kernel.launch(stream,
                  static_cast<std::uint32_t>(c.T), 1, 1,
                  kBlock, 1, 1, /*shared=*/0);
    stream.synchronize();

    std::vector<std::int32_t> gpuIdx(nOut);
    std::vector<float>        gpuW(nOut);
    alloc.copyD2H(gpuIdx.data(), devIdx.data(), nOut * sizeof(std::int32_t));
    alloc.copyD2H(gpuW.data(),   devW.data(),   nOut * sizeof(float));

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

int main(int argc, char** argv) {
    const std::string hsacoPath =
        (argc > 1) ? std::string{argv[1]} : defaultHsacoPath(argv[0]);

    std::printf("hip_moe_topk_probe:\n  hsaco: %s\n", hsacoPath.c_str());

    try {
        HipContext         ctx{};
        HipMemoryAllocator alloc{ctx};
        HipStream          stream{ctx, HipStreamKind::BlockingDefault};

        HipModule mod    = HipModule::fromFile(ctx, hsacoPath);
        HipKernel kernel = mod.getKernel("moe_topk");

        const Config configs[] = {
            {/*T=*/6, /*nE=*/128, /*K=*/8, /*wScale=*/1.0f, 0xA11Cu},
            {/*T=*/4, /*nE=*/ 64, /*K=*/4, /*wScale=*/2.5f, 0xBEEFu},
            {/*T=*/1, /*nE=*/128, /*K=*/8, /*wScale=*/1.0f, 0x5EEDu},
        };

        bool allOk = true;
        for (const auto& c : configs) {
            allOk &= runConfig(alloc, stream, kernel, c);
        }

        std::printf("\nhip_moe_topk_probe: %s\n", allOk ? "OK" : "FAIL");
        return allOk ? 0 : 1;

    } catch (const std::exception& e) {
        std::fprintf(stderr, "hip_moe_topk_probe: threw: %s\n", e.what());
        return 2;
    }
}