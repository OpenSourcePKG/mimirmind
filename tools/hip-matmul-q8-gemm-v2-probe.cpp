// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// hip_matmul_q8_gemm_v2_probe — parity check for the v2-revised GEMM
// with 4× smaller SLM footprint (X_TILE_ELEMENTS shrunk from 1024 to
// 256). Otherwise identical semantics to matmul_q8_0_gemm.
//
// The tighter SLM means bench-comparisons on the same shape will
// touch the X-tile 4× more often — the parity output stays identical
// but the load pattern differs.

#include "core/gpu/hip/HipContext.hpp"
#include "core/gpu/hip/HipEvent.hpp"
#include "core/gpu/hip/HipKernel.hpp"
#include "core/gpu/hip/HipMemoryAllocator.hpp"
#include "core/gpu/hip/HipModule.hpp"
#include "core/gpu/hip/HipStream.hpp"

#include <hip/hip_fp16.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <string>
#include <vector>

namespace {

using namespace mimirmind::core::hip;

constexpr int          kM              = 8;
constexpr int          kN              = 128;
constexpr int          kK              = 1024;
constexpr int          kM_TILE         = 8;
constexpr int          kOutputsPerWG   = 4;
constexpr int          kBlockElems     = 32;
constexpr int          kBlockBytes     = 34;
constexpr int          kNBlocksPerRow  = kK / kBlockElems;
constexpr std::uint32_t kBlock         = 64;

std::string defaultHsacoPath(const char* argv0) {
    std::filesystem::path exe = std::filesystem::canonical(argv0);
    return (exe.parent_path() / "hsaco" / "matmul_q8_0_gemm_v2.hsaco").string();
}

void fillRandomFloat(std::vector<float>& v, std::uint32_t seed, float scale) {
    std::uint32_t x = seed;
    for (auto& f : v) {
        x = x * 1664525u + 1013904223u;
        const std::int32_t s = static_cast<std::int32_t>(x >> 8) - (1 << 23);
        f = scale * static_cast<float>(s) / static_cast<float>(1 << 23);
    }
}

void fillRandomQ8_0Native(std::vector<std::uint8_t>& bytes,
                          std::uint32_t seed, int nRows) {
    std::uint32_t x = seed;
    const std::size_t rowBytes = static_cast<std::size_t>(kNBlocksPerRow)
                               * static_cast<std::size_t>(kBlockBytes);
    for (int r = 0; r < nRows; ++r) {
        std::uint8_t* row = bytes.data() + static_cast<std::size_t>(r) * rowBytes;
        for (int b = 0; b < kNBlocksPerRow; ++b) {
            std::uint8_t* blk = row + b * kBlockBytes;
            x = x * 1664525u + 1013904223u;
            const float dScalar =
                0.001f + 0.02f * (static_cast<float>(x & 0xFFFF) / 65535.0f);
            const __half dHalf = __float2half(dScalar);
            std::memcpy(blk, &dHalf, sizeof(__half));
            for (int i = 0; i < kBlockElems; ++i) {
                x = x * 1664525u + 1013904223u;
                const std::int8_t v = static_cast<std::int8_t>(
                    (static_cast<std::int32_t>(x >> 24)) - 128);
                reinterpret_cast<signed char*>(blk)[2 + i] = v;
            }
        }
    }
}

inline float dequantElemNative(const std::uint8_t* rowBase, int k) {
    const int b  = k / kBlockElems;
    const int in = k % kBlockElems;
    const std::uint8_t* blk = rowBase + static_cast<std::size_t>(b) * kBlockBytes;
    __half hScale;
    std::memcpy(&hScale, blk, sizeof(__half));
    const float d = __half2float(hScale);
    const signed char qi = reinterpret_cast<const signed char*>(blk)[2 + in];
    return d * static_cast<float>(qi);
}

void gemmCpuRef(const std::vector<float>&        X,
                const std::vector<std::uint8_t>& W,
                std::vector<float>&              Y,
                int M, int N, int K)
{
    const std::size_t rowBytes = static_cast<std::size_t>(kNBlocksPerRow)
                               * static_cast<std::size_t>(kBlockBytes);
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            const std::uint8_t* rowW = W.data()
                                     + static_cast<std::size_t>(n) * rowBytes;
            double acc = 0.0;
            for (int k = 0; k < K; ++k) {
                const float wVal = dequantElemNative(rowW, k);
                acc += static_cast<double>(X[static_cast<std::size_t>(m) * K + k])
                     * static_cast<double>(wVal);
            }
            Y[static_cast<std::size_t>(m) * N + n] = static_cast<float>(acc);
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    const std::string hsacoPath =
        (argc > 1) ? std::string{argv[1]} : defaultHsacoPath(argv[0]);

    std::printf("hip_matmul_q8_gemm_v2_probe:\n  hsaco: %s\n"
                "  M=%d N=%d K=%d M_TILE=%d block=%u outputs_per_wg=%d\n",
                hsacoPath.c_str(),
                kM, kN, kK, kM_TILE, kBlock, kOutputsPerWG);

    try {
        HipContext         ctx{};
        HipMemoryAllocator alloc{ctx};
        HipStream          stream{ctx, HipStreamKind::BlockingDefault};
        HipEvent           evStart{ctx};
        HipEvent           evEnd  {ctx};

        HipModule mod    = HipModule::fromFile(ctx, hsacoPath);
        HipKernel kernel = mod.getKernel("matmul_q8_0_gemm_v2");

        const std::size_t xElems  = static_cast<std::size_t>(kM) * kK;
        const std::size_t yElems  = static_cast<std::size_t>(kM) * kN;
        const std::size_t wBytes  = static_cast<std::size_t>(kN)
                                  * kNBlocksPerRow * kBlockBytes;

        std::vector<float>        hostX(xElems);
        std::vector<std::uint8_t> hostW(wBytes);
        std::vector<float>        hostYRef(yElems);
        std::vector<float>        hostYGpu(yElems, 0.0f);

        fillRandomFloat(hostX, /*seed=*/0x6E120001u, /*scale=*/0.5f);
        fillRandomQ8_0Native(hostW, /*seed=*/0x6E120002u, /*nRows=*/kN);

        gemmCpuRef(hostX, hostW, hostYRef, kM, kN, kK);

        const std::size_t xBytes = xElems * sizeof(float);
        const std::size_t yBytes = yElems * sizeof(float);
        HipBuffer devX{alloc, xBytes};
        HipBuffer devW{alloc, wBytes};
        HipBuffer devY{alloc, yBytes};

        alloc.copyH2D(devX.data(), hostX.data(), xBytes);
        alloc.copyH2D(devW.data(), hostW.data(), wBytes);

        kernel.setPtr  (0, devX.data());
        kernel.setPtr  (1, devW.data());
        kernel.setPtr  (2, devY.data());
        kernel.setValue(3, kK);
        kernel.setValue(4, kN);
        kernel.setValue(5, kM);

        const std::uint32_t gridX =
            (static_cast<std::uint32_t>(kN) + kOutputsPerWG - 1) / kOutputsPerWG;
        const std::uint32_t gridY =
            (static_cast<std::uint32_t>(kM) + kM_TILE - 1) / kM_TILE;

        evStart.record(stream);
        kernel.launch(stream, gridX, gridY, 1,
                      kBlock, 1, 1, /*shared=*/0);
        evEnd.record(stream);
        stream.synchronize();
        const float kernelMs = evEnd.elapsedMs(evStart);

        alloc.copyD2H(hostYGpu.data(), devY.data(), yBytes);

        constexpr float kAbsTol = 1e-3f;
        constexpr float kRelTol = 1e-3f;

        float       maxAbs   = 0.0f;
        float       maxRatio = 0.0f;
        std::size_t badIdx   = SIZE_MAX;
        for (std::size_t i = 0; i < yElems; ++i) {
            const float d         = std::fabs(hostYGpu[i] - hostYRef[i]);
            const float threshold = kAbsTol + kRelTol * std::fabs(hostYRef[i]);
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
        if (badIdx != SIZE_MAX && maxRatio > 1e-3f) {
            const std::size_t m = badIdx / kN;
            const std::size_t n = badIdx % kN;
            std::printf("  worst @ m=%zu n=%zu: gpu=%.6g cpu=%.6g\n",
                        m, n,
                        static_cast<double>(hostYGpu[badIdx]),
                        static_cast<double>(hostYRef[badIdx]));
        }

        const bool ok = (maxRatio <= 1.0f);
        std::printf("\nhip_matmul_q8_gemm_v2_probe: %s\n", ok ? "OK" : "FAIL");
        return ok ? 0 : 1;

    } catch (const std::exception& e) {
        std::fprintf(stderr, "hip_matmul_q8_gemm_v2_probe: threw: %s\n", e.what());
        return 2;
    }
}