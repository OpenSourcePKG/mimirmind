// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// hip_add_bias_probe — parity check for the broadcast bias add.
// Loads add_bias.hsaco, applies y[m,k] += bias[k] for a row-major (M,K)
// tensor, compares against an inline double-precision CPU reference.
//
// Uses the combined-tolerance pattern
// (|diff| <= abs_tol + rel_tol * |ref|). Same reasoning as add_residual:
// a single float add per element, no reduction — tight tol (1e-6 / 1e-6)
// catches any accidental precision loss on the broadcast index math.

#include "core/gpu/hip/HipContext.hpp"
#include "core/gpu/hip/HipEvent.hpp"
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

// Shape with M > 1 so the broadcast is actually exercised (bias[k]
// added to every row). Total = 32k, multi-block launch.
constexpr int          kM     = 8;
constexpr int          kK     = 4096;
constexpr int          kN     = kM * kK;
constexpr std::uint32_t kBlock = 256;   // == ADD_BIAS_LOCAL

std::string defaultHsacoPath(const char* argv0) {
    std::filesystem::path exe = std::filesystem::canonical(argv0);
    return (exe.parent_path() / "hsaco" / "add_bias.hsaco").string();
}

void fillRandom(std::vector<float>& v, std::uint32_t seed, float scale) {
    std::uint32_t x = seed;
    for (auto& f : v) {
        x = x * 1664525u + 1013904223u;
        const std::int32_t s = static_cast<std::int32_t>(x >> 8) - (1 << 23);
        f = scale * static_cast<float>(s) / static_cast<float>(1 << 23);
    }
}

void addBiasCpuRef(std::vector<float>&       y,   // in+out, shape (M,K)
                   const std::vector<float>& bias,
                   int M, int K) {
    for (int m = 0; m < M; ++m) {
        float* yr = y.data() + static_cast<std::size_t>(m) * K;
        for (int k = 0; k < K; ++k) {
            const double a = static_cast<double>(yr[k]);
            const double b = static_cast<double>(bias[k]);
            yr[k] = static_cast<float>(a + b);
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    const std::string hsacoPath =
        (argc > 1) ? std::string{argv[1]} : defaultHsacoPath(argv[0]);

    std::printf("hip_add_bias_probe:\n  hsaco: %s\n  M=%d K=%d block=%u\n",
                hsacoPath.c_str(), kM, kK, kBlock);

    try {
        HipContext         ctx{};
        HipMemoryAllocator alloc{ctx};
        HipStream          stream{ctx, HipStreamKind::BlockingDefault};
        HipEvent           evStart{ctx};
        HipEvent           evEnd  {ctx};

        HipModule mod    = HipModule::fromFile(ctx, hsacoPath);
        HipKernel kernel = mod.getKernel("add_bias");

        // ---- host tensors -----------------------------------------------
        std::vector<float> hostYIn(static_cast<std::size_t>(kN));
        std::vector<float> hostBias(static_cast<std::size_t>(kK));
        fillRandom(hostYIn,  /*seed=*/0xC0FFEE00u, /*scale=*/1.0f);
        fillRandom(hostBias, /*seed=*/0xDEADFA11u, /*scale=*/1.0f);

        std::vector<float> hostRef = hostYIn;
        addBiasCpuRef(hostRef, hostBias, kM, kK);

        std::vector<float> hostGpu(static_cast<std::size_t>(kN), 0.0f);

        // ---- device tensors ---------------------------------------------
        const std::size_t yBytes = static_cast<std::size_t>(kN) * sizeof(float);
        const std::size_t bBytes = static_cast<std::size_t>(kK) * sizeof(float);
        HipBuffer devY   {alloc, yBytes};
        HipBuffer devBias{alloc, bBytes};

        alloc.copyH2D(devY.data(),    hostYIn .data(), yBytes);
        alloc.copyH2D(devBias.data(), hostBias.data(), bBytes);

        // ---- launch -----------------------------------------------------
        kernel.setPtr  (0, devY   .data());
        kernel.setPtr  (1, devBias.data());
        kernel.setValue(2, kM);
        kernel.setValue(3, kK);

        const std::uint32_t grid =
            (static_cast<std::uint32_t>(kN) + kBlock - 1) / kBlock;

        evStart.record(stream);
        kernel.launch(stream, grid, 1, 1, kBlock, 1, 1, /*shared=*/0);
        evEnd.record(stream);
        stream.synchronize();

        const float kernelMs = evEnd.elapsedMs(evStart);

        // ---- readback + combined-tolerance compare ----------------------
        alloc.copyD2H(hostGpu.data(), devY.data(), yBytes);

        constexpr float kAbsTol = 1e-6f;
        constexpr float kRelTol = 1e-6f;

        float       maxAbs   = 0.0f;
        float       maxRatio = 0.0f;
        std::size_t badIdx   = SIZE_MAX;
        for (std::size_t i = 0; i < static_cast<std::size_t>(kN); ++i) {
            const float d         = std::fabs(hostGpu[i] - hostRef[i]);
            const float threshold = kAbsTol + kRelTol * std::fabs(hostRef[i]);
            const float ratio     = d / threshold;
            if (ratio > maxRatio) { maxRatio = ratio; badIdx = i; }
            if (d > maxAbs) maxAbs = d;
        }

        std::printf("\n  kernel:        %.3f ms\n", static_cast<double>(kernelMs));
        std::printf("  max abs err:   %.3e\n", static_cast<double>(maxAbs));
        std::printf("  max err / tol: %.3f (fails if > 1.0)\n",
                    static_cast<double>(maxRatio));
        std::printf("  tol formula:   abs %.1e + rel %.1e * |ref|\n",
                    static_cast<double>(kAbsTol), static_cast<double>(kRelTol));
        if (badIdx != SIZE_MAX) {
            const std::size_t m = badIdx / static_cast<std::size_t>(kK);
            const std::size_t k = badIdx % static_cast<std::size_t>(kK);
            std::printf("  worst @ m=%zu k=%zu: gpu=%.6g cpu=%.6g\n",
                        m, k,
                        static_cast<double>(hostGpu[badIdx]),
                        static_cast<double>(hostRef[badIdx]));
        }

        const bool ok = (maxRatio <= 1.0f);
        std::printf("\nhip_add_bias_probe: %s\n", ok ? "OK" : "FAIL");
        return ok ? 0 : 1;

    } catch (const std::exception& e) {
        std::fprintf(stderr, "hip_add_bias_probe: threw: %s\n", e.what());
        return 2;
    }
}