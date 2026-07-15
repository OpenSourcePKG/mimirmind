// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// hip_mul_scalar_probe — parity check for the in-place scalar multiply.
// Loads mul_scalar.hsaco, applies y[i] *= s element-wise, compares
// against an inline double-precision CPU reference.
//
// First HIP probe to bind a float POD via HipKernel::setValue<float> —
// exercises the by-value scalar arg path that Gemma 4's layer_output_
// scale relies on.
//
// Uses the combined-tolerance pattern
// (|diff| <= abs_tol + rel_tol * |ref|). One multiply per element, no
// reduction — tight tol (1e-6 abs, 1e-6 rel) catches drift.

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

constexpr int          kRows   = 8;
constexpr int          kWidth  = 4096;
constexpr int          kN      = kRows * kWidth;
constexpr std::uint32_t kBlock = 256;   // == MUL_SCALAR_LOCAL

// Non-trivial scalar (not 1, not a power of 2) — exposes any accidental
// identity fastpath and forces a real fp32 multiply.
constexpr float        kScalar = 0.3125f;

std::string defaultHsacoPath(const char* argv0) {
    std::filesystem::path exe = std::filesystem::canonical(argv0);
    return (exe.parent_path() / "hsaco" / "mul_scalar.hsaco").string();
}

void fillRandom(std::vector<float>& v, std::uint32_t seed, float scale) {
    std::uint32_t x = seed;
    for (auto& f : v) {
        x = x * 1664525u + 1013904223u;
        const std::int32_t s = static_cast<std::int32_t>(x >> 8) - (1 << 23);
        f = scale * static_cast<float>(s) / static_cast<float>(1 << 23);
    }
}

void mulScalarCpuRef(std::vector<float>& y, float s) {
    const double sd = static_cast<double>(s);
    for (auto& f : y) {
        f = static_cast<float>(static_cast<double>(f) * sd);
    }
}

} // namespace

int main(int argc, char** argv) {
    const std::string hsacoPath =
        (argc > 1) ? std::string{argv[1]} : defaultHsacoPath(argv[0]);

    std::printf("hip_mul_scalar_probe:\n  hsaco: %s\n  n=%d s=%g block=%u\n",
                hsacoPath.c_str(), kN, static_cast<double>(kScalar), kBlock);

    try {
        HipContext         ctx{};
        HipMemoryAllocator alloc{ctx};
        HipStream          stream{ctx, HipStreamKind::BlockingDefault};
        HipEvent           evStart{ctx};
        HipEvent           evEnd  {ctx};

        HipModule mod    = HipModule::fromFile(ctx, hsacoPath);
        HipKernel kernel = mod.getKernel("mul_scalar");

        // ---- host tensors -----------------------------------------------
        std::vector<float> hostYIn(static_cast<std::size_t>(kN));
        fillRandom(hostYIn, /*seed=*/0xA5A5A5A5u, /*scale=*/1.0f);

        std::vector<float> hostRef = hostYIn;
        mulScalarCpuRef(hostRef, kScalar);

        std::vector<float> hostGpu(static_cast<std::size_t>(kN), 0.0f);

        // ---- device tensors ---------------------------------------------
        const std::size_t nBytes = static_cast<std::size_t>(kN) * sizeof(float);
        HipBuffer devY{alloc, nBytes};

        alloc.copyH2D(devY.data(), hostYIn.data(), nBytes);

        // ---- launch -----------------------------------------------------
        kernel.setPtr  (0, devY.data());
        kernel.setValue(1, kScalar);
        kernel.setValue(2, kN);

        const std::uint32_t grid =
            (static_cast<std::uint32_t>(kN) + kBlock - 1) / kBlock;

        evStart.record(stream);
        kernel.launch(stream, grid, 1, 1, kBlock, 1, 1, /*shared=*/0);
        evEnd.record(stream);
        stream.synchronize();

        const float kernelMs = evEnd.elapsedMs(evStart);

        // ---- readback + combined-tolerance compare ----------------------
        alloc.copyD2H(hostGpu.data(), devY.data(), nBytes);

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
            std::printf("  worst @ idx=%zu: gpu=%.6g cpu=%.6g\n",
                        badIdx,
                        static_cast<double>(hostGpu[badIdx]),
                        static_cast<double>(hostRef[badIdx]));
        }

        const bool ok = (maxRatio <= 1.0f);
        std::printf("\nhip_mul_scalar_probe: %s\n", ok ? "OK" : "FAIL");
        return ok ? 0 : 1;

    } catch (const std::exception& e) {
        std::fprintf(stderr, "hip_mul_scalar_probe: threw: %s\n", e.what());
        return 2;
    }
}