// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// hip_gelu_mul_probe — parity check for the GELU-tanh SwiGLU fused
// step. Loads gelu_mul.hsaco, applies (gelu_tanh(gate) * up)
// element-wise, compares against an inline double-precision CPU
// reference that runs the same tanh-approx formula.
//
// Combined-tolerance pattern: gelu passes through zero for negative
// inputs, so a strict-relative check would false-fail on near-zero
// outputs.

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

// Gemma-class FFN intermediate widths as a shape stand-in.
constexpr int          kRows    = 8;
constexpr int          kWidth   = 4096;
constexpr int          kN       = kRows * kWidth;
constexpr std::uint32_t kBlock  = 256;   // == GELU_MUL_LOCAL

std::string defaultHsacoPath(const char* argv0) {
    std::filesystem::path exe = std::filesystem::canonical(argv0);
    return (exe.parent_path() / "hsaco" / "gelu_mul.hsaco").string();
}

void fillRandom(std::vector<float>& v, std::uint32_t seed, float scale) {
    std::uint32_t x = seed;
    for (auto& f : v) {
        x = x * 1664525u + 1013904223u;
        const std::int32_t s = static_cast<std::int32_t>(x >> 8) - (1 << 23);
        f = scale * static_cast<float>(s) / static_cast<float>(1 << 23);
    }
}

// CPU reference in double precision. Same tanh-approx formula the
// kernel uses; the round-trip through tanh(fp64) sets a legitimate
// per-element target the fp32 GPU result can be compared against with
// tight tolerance.
void geluMulCpuRef(std::vector<float>&       gate,   // in+out
                   const std::vector<float>& up) {
    for (std::size_t i = 0; i < gate.size(); ++i) {
        const double g  = static_cast<double>(gate[i]);
        const double g3 = g * g * g;
        const double t  = std::tanh(0.7978845608028654 * (g + 0.044715 * g3));
        const double ge = 0.5 * g * (1.0 + t);
        gate[i] = static_cast<float>(ge * static_cast<double>(up[i]));
    }
}

} // namespace

int main(int argc, char** argv) {
    const std::string hsacoPath =
        (argc > 1) ? std::string{argv[1]} : defaultHsacoPath(argv[0]);

    std::printf("hip_gelu_mul_probe:\n  hsaco: %s\n  n=%d block=%u\n",
                hsacoPath.c_str(), kN, kBlock);

    try {
        HipContext         ctx{};
        HipMemoryAllocator alloc{ctx};
        HipStream          stream{ctx, HipStreamKind::BlockingDefault};
        HipEvent           evStart{ctx};
        HipEvent           evEnd  {ctx};

        HipModule mod    = HipModule::fromFile(ctx, hsacoPath);
        HipKernel kernel = mod.getKernel("gelu_mul");

        // gate ∈ [-4, 4]: exercises gelu across negative-flat and
        // positive-linear regimes plus the tanh non-linearity in the
        // middle. up ∈ [-1, 1] keeps output magnitudes moderate.
        std::vector<float> hostGateIn(static_cast<std::size_t>(kN));
        std::vector<float> hostUp    (static_cast<std::size_t>(kN));
        fillRandom(hostGateIn, /*seed=*/0xC0FFEE00u, /*scale=*/4.0f);
        fillRandom(hostUp,     /*seed=*/0xDEADBEEFu, /*scale=*/1.0f);

        std::vector<float> hostRef = hostGateIn;
        geluMulCpuRef(hostRef, hostUp);

        std::vector<float> hostGpu(static_cast<std::size_t>(kN), 0.0f);

        const std::size_t nBytes = static_cast<std::size_t>(kN) * sizeof(float);
        HipBuffer devGate{alloc, nBytes};
        HipBuffer devUp  {alloc, nBytes};

        alloc.copyH2D(devGate.data(), hostGateIn.data(), nBytes);
        alloc.copyH2D(devUp  .data(), hostUp    .data(), nBytes);

        kernel.setPtr  (0, devGate.data());
        kernel.setPtr  (1, devUp  .data());
        kernel.setValue(2, kN);

        const std::uint32_t grid =
            (static_cast<std::uint32_t>(kN) + kBlock - 1) / kBlock;

        evStart.record(stream);
        kernel.launch(stream, grid, 1, 1, kBlock, 1, 1, /*shared=*/0);
        evEnd.record(stream);
        stream.synchronize();

        const float kernelMs = evEnd.elapsedMs(evStart);

        alloc.copyD2H(hostGpu.data(), devGate.data(), nBytes);

        // Slightly looser than silu_mul because tanh is a transcendental
        // and fp32 vs fp64 tanh drift is ~1-2 ULP on values near the
        // steep part of the curve.
        constexpr float kAbsTol = 1e-6f;
        constexpr float kRelTol = 1e-5f;

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
            std::printf("  worst @ idx=%zu: gpu=%.6g cpu=%.6g\n",
                        badIdx,
                        static_cast<double>(hostGpu[badIdx]),
                        static_cast<double>(hostRef[badIdx]));
        }

        const bool ok = (maxRatio <= 1.0f);
        std::printf("\nhip_gelu_mul_probe: %s\n", ok ? "OK" : "FAIL");
        return ok ? 0 : 1;

    } catch (const std::exception& e) {
        std::fprintf(stderr, "hip_gelu_mul_probe: threw: %s\n", e.what());
        return 2;
    }
}