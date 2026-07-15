// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// hip_scaled_add_residual_probe — parity check for the fused scaled-
// accumulate kernel. Loads scaled_add_residual.hsaco, applies
// (dst += scale * src) in place, compares against inline double-
// precision CPU reference. Combined-tolerance since the fp32 FMA vs
// fp64 multiply-then-add produces sub-ULP drift on outputs whose
// magnitudes cross the near-zero band.

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

constexpr int          kN     = 8 * 4096;   // MoE per-expert accum shape
constexpr float        kScale = 0.1875f;    // representative MoE gate weight
constexpr std::uint32_t kBlock = 256;       // == SCALED_ADD_RESIDUAL_LOCAL

std::string defaultHsacoPath(const char* argv0) {
    std::filesystem::path exe = std::filesystem::canonical(argv0);
    return (exe.parent_path() / "hsaco" / "scaled_add_residual.hsaco").string();
}

void fillRandom(std::vector<float>& v, std::uint32_t seed, float scale) {
    std::uint32_t x = seed;
    for (auto& f : v) {
        x = x * 1664525u + 1013904223u;
        const std::int32_t s = static_cast<std::int32_t>(x >> 8) - (1 << 23);
        f = scale * static_cast<float>(s) / static_cast<float>(1 << 23);
    }
}

void scaledAddResidualCpuRef(std::vector<float>&       dst,   // in+out
                             const std::vector<float>& src,
                             float                     scale) {
    for (std::size_t i = 0; i < dst.size(); ++i) {
        // std::fma promises the fp64 fused primitive, so the reference
        // matches the kernel's fmaf() semantics up to fp32 rounding.
        const double a  = static_cast<double>(scale);
        const double b  = static_cast<double>(src[i]);
        const double c  = static_cast<double>(dst[i]);
        dst[i] = static_cast<float>(std::fma(a, b, c));
    }
}

} // namespace

int main(int argc, char** argv) {
    const std::string hsacoPath =
        (argc > 1) ? std::string{argv[1]} : defaultHsacoPath(argv[0]);

    std::printf("hip_scaled_add_residual_probe:\n  hsaco: %s\n"
                "  n=%d scale=%.4f block=%u\n",
                hsacoPath.c_str(), kN, static_cast<double>(kScale), kBlock);

    try {
        HipContext         ctx{};
        HipMemoryAllocator alloc{ctx};
        HipStream          stream{ctx, HipStreamKind::BlockingDefault};
        HipEvent           evStart{ctx};
        HipEvent           evEnd  {ctx};

        HipModule mod    = HipModule::fromFile(ctx, hsacoPath);
        HipKernel kernel = mod.getKernel("scaled_add_residual");

        // Both operands in [-1, 1] — output stays well-behaved.
        std::vector<float> hostDstIn(static_cast<std::size_t>(kN));
        std::vector<float> hostSrc  (static_cast<std::size_t>(kN));
        fillRandom(hostDstIn, /*seed=*/0xA11CE001u, /*scale=*/1.0f);
        fillRandom(hostSrc,   /*seed=*/0xB0B0F00Du, /*scale=*/1.0f);

        std::vector<float> hostRef = hostDstIn;
        scaledAddResidualCpuRef(hostRef, hostSrc, kScale);

        std::vector<float> hostGpu(static_cast<std::size_t>(kN), 0.0f);

        const std::size_t nBytes = static_cast<std::size_t>(kN) * sizeof(float);
        HipBuffer devDst{alloc, nBytes};
        HipBuffer devSrc{alloc, nBytes};

        alloc.copyH2D(devDst.data(), hostDstIn.data(), nBytes);
        alloc.copyH2D(devSrc.data(), hostSrc  .data(), nBytes);

        kernel.setPtr  (0, devDst.data());
        kernel.setPtr  (1, devSrc.data());
        kernel.setValue(2, kScale);
        kernel.setValue(3, kN);

        const std::uint32_t grid =
            (static_cast<std::uint32_t>(kN) + kBlock - 1) / kBlock;

        evStart.record(stream);
        kernel.launch(stream, grid, 1, 1, kBlock, 1, 1, /*shared=*/0);
        evEnd.record(stream);
        stream.synchronize();

        const float kernelMs = evEnd.elapsedMs(evStart);

        alloc.copyD2H(hostGpu.data(), devDst.data(), nBytes);

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
        std::printf("\nhip_scaled_add_residual_probe: %s\n",
                    ok ? "OK" : "FAIL");
        return ok ? 0 : 1;

    } catch (const std::exception& e) {
        std::fprintf(stderr, "hip_scaled_add_residual_probe: threw: %s\n",
                     e.what());
        return 2;
    }
}