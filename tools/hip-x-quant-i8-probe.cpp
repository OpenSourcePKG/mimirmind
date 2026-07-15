// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// hip_x_quant_i8_probe — parity check for per-row symmetric int8
// quantisation. Loads x_quant_i8.hsaco, runs on a mix of typical
// activation rows (varied magnitudes) plus one deliberately-zero row
// to exercise the amax=0 sentinel path, and compares against an
// inline double-precision CPU reference.
//
// Comparison policy:
//   - Scales: combined-tolerance (fp32 vs fp64 through a divide).
//   - Quants: byte-exact expected. If the fp32 GPU-side `roundf`
//     drifts by 1 ULP on a tie, we allow at most `kQuantSlackPerRow`
//     off-by-one differences per row and log the worst row. The L0
//     side is byte-exact against ggml; the HIP side is expected to
//     match — this slack is a safety net for driver differences,
//     not a licence for silent divergence.

#include "core/gpu/hip/HipContext.hpp"
#include "core/gpu/hip/HipEvent.hpp"
#include "core/gpu/hip/HipKernel.hpp"
#include "core/gpu/hip/HipMemoryAllocator.hpp"
#include "core/gpu/hip/HipModule.hpp"
#include "core/gpu/hip/HipStream.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <string>
#include <vector>

namespace {

using namespace mimirmind::core::hip;

constexpr int          kM         = 8;
constexpr int          kK         = 4096;
constexpr std::uint32_t kBlock    = 128;   // == X_QUANT_I8_LOCAL

// Post-M8.H.1 lessons: fp32 round-to-nearest-tie-away can produce
// off-by-one ints against a fp64 reference for values landing on
// exact halves. In practice the parity is byte-exact; this cap is
// a driver-hedge, not a spec allowance.
constexpr std::size_t   kQuantSlackPerRow = 0;

std::string defaultHsacoPath(const char* argv0) {
    std::filesystem::path exe = std::filesystem::canonical(argv0);
    return (exe.parent_path() / "hsaco" / "x_quant_i8.hsaco").string();
}

void fillRandom(std::vector<float>& v, std::uint32_t seed, float scale) {
    std::uint32_t x = seed;
    for (auto& f : v) {
        x = x * 1664525u + 1013904223u;
        const std::int32_t s = static_cast<std::int32_t>(x >> 8) - (1 << 23);
        f = scale * static_cast<float>(s) / static_cast<float>(1 << 23);
    }
}

void xQuantI8CpuRef(const std::vector<float>&    x,
                    std::vector<std::int8_t>&    yQuants,
                    std::vector<float>&          scales,
                    int M, int K) {
    for (int m = 0; m < M; ++m) {
        const float* xr = x.data() + static_cast<std::size_t>(m) * K;
        // amax in fp32 to match the kernel's per-thread accumulator.
        float amax = 0.0f;
        for (int k = 0; k < K; ++k) {
            amax = std::fmax(amax, std::fabs(xr[k]));
        }
        const float s    = amax * (1.0f / 127.0f);
        const float invS = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        scales[m] = s;

        std::int8_t* yr = yQuants.data() + static_cast<std::size_t>(m) * K;
        for (int k = 0; k < K; ++k) {
            const float q  = std::round(xr[k] * invS);
            const float qc = std::min(std::max(q, -127.0f), 127.0f);
            yr[k] = static_cast<std::int8_t>(static_cast<int>(qc));
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    const std::string hsacoPath =
        (argc > 1) ? std::string{argv[1]} : defaultHsacoPath(argv[0]);

    std::printf("hip_x_quant_i8_probe:\n  hsaco: %s\n  M=%d K=%d block=%u\n",
                hsacoPath.c_str(), kM, kK, kBlock);

    try {
        HipContext         ctx{};
        HipMemoryAllocator alloc{ctx};
        HipStream          stream{ctx, HipStreamKind::BlockingDefault};
        HipEvent           evStart{ctx};
        HipEvent           evEnd  {ctx};

        HipModule mod    = HipModule::fromFile(ctx, hsacoPath);
        HipKernel kernel = mod.getKernel("x_quant_i8");

        const std::size_t xElems = static_cast<std::size_t>(kM) * kK;
        std::vector<float>       hostX(xElems);
        std::vector<std::int8_t> hostYRef(xElems, 0);
        std::vector<float>       hostSRef(static_cast<std::size_t>(kM), 0.0f);
        std::vector<std::int8_t> hostYGpu(xElems, 0);
        std::vector<float>       hostSGpu(static_cast<std::size_t>(kM), 0.0f);

        // Row 0..M-2: random floats in [-2, 2].
        // Row M-1: all zeros (exercises the amax=0 sentinel path).
        fillRandom(hostX, /*seed=*/0xB16BEEFu, /*scale=*/2.0f);
        for (int k = 0; k < kK; ++k) {
            hostX[static_cast<std::size_t>(kM - 1) * kK + k] = 0.0f;
        }

        xQuantI8CpuRef(hostX, hostYRef, hostSRef, kM, kK);

        const std::size_t xBytes = xElems * sizeof(float);
        const std::size_t yBytes = xElems * sizeof(std::int8_t);
        const std::size_t sBytes = static_cast<std::size_t>(kM) * sizeof(float);
        HipBuffer devX{alloc, xBytes};
        HipBuffer devY{alloc, yBytes};
        HipBuffer devS{alloc, sBytes};

        alloc.copyH2D(devX.data(), hostX.data(), xBytes);

        kernel.setPtr  (0, devX.data());
        kernel.setPtr  (1, devY.data());
        kernel.setPtr  (2, devS.data());
        kernel.setValue(3, kK);

        evStart.record(stream);
        kernel.launch(stream, static_cast<std::uint32_t>(kM), 1, 1,
                      kBlock, 1, 1, /*shared=*/0);
        evEnd.record(stream);
        stream.synchronize();

        const float kernelMs = evEnd.elapsedMs(evStart);

        alloc.copyD2H(hostYGpu.data(), devY.data(), yBytes);
        alloc.copyD2H(hostSGpu.data(), devS.data(), sBytes);

        // Scales: combined tolerance.
        constexpr float kAbsTol = 1e-8f;
        constexpr float kRelTol = 1e-6f;
        float sMaxRatio = 0.0f;
        int   sBadRow   = -1;
        for (int m = 0; m < kM; ++m) {
            const float d         = std::fabs(hostSGpu[m] - hostSRef[m]);
            const float threshold = kAbsTol + kRelTol * std::fabs(hostSRef[m]);
            const float ratio     = d / threshold;
            if (ratio > sMaxRatio) { sMaxRatio = ratio; sBadRow = m; }
        }

        // Quants: strict byte compare with per-row slack budget.
        std::size_t totalDiff = 0;
        int         worstRow  = -1;
        std::size_t worstRowDiff = 0;
        for (int m = 0; m < kM; ++m) {
            std::size_t rowDiff = 0;
            for (int k = 0; k < kK; ++k) {
                const std::size_t i =
                    static_cast<std::size_t>(m) * kK + k;
                if (hostYGpu[i] != hostYRef[i]) {
                    ++rowDiff;
                    ++totalDiff;
                }
            }
            if (rowDiff > worstRowDiff) {
                worstRowDiff = rowDiff;
                worstRow     = m;
            }
        }

        std::printf("\n  kernel:              %.3f ms\n",
                    static_cast<double>(kernelMs));
        std::printf("  scale max err/tol:   %.3f (fails if > 1.0)\n",
                    static_cast<double>(sMaxRatio));
        if (sBadRow >= 0) {
            std::printf("  worst scale @ m=%d: gpu=%.6g cpu=%.6g\n",
                        sBadRow, static_cast<double>(hostSGpu[sBadRow]),
                        static_cast<double>(hostSRef[sBadRow]));
        }
        std::printf("  quant total diffs:   %zu bytes\n", totalDiff);
        std::printf("  worst row diffs:     %zu (row %d, slack=%zu)\n",
                    worstRowDiff, worstRow, kQuantSlackPerRow);

        const bool scalesOk = (sMaxRatio <= 1.0f);
        const bool quantsOk = (worstRowDiff <= kQuantSlackPerRow);
        const bool ok       = scalesOk && quantsOk;
        std::printf("\nhip_x_quant_i8_probe: %s%s%s\n",
                    ok ? "OK" : "FAIL",
                    scalesOk ? "" : " [scales fail]",
                    quantsOk ? "" : " [quants fail]");
        return ok ? 0 : 1;

    } catch (const std::exception& e) {
        std::fprintf(stderr, "hip_x_quant_i8_probe: threw: %s\n", e.what());
        return 2;
    }
}