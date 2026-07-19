// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// hip_matmul_q8_vec_dp4a_probe — parity check for the DP4A matvec.
// First test of a HIP compiler intrinsic in the port
// (__builtin_amdgcn_sdot4 on gfx1101).
//
// Xq is a pre-quantised int8 activation vector; Xscale is a single
// fp32 the CPU-ref multiplies onto the reconstructed X. W is native-
// layout Q8_0. CPU reference does the equivalent fp64 matmul against
// (Xq[k] * Xscale) and dequantised W, so any diff is a DP4A-intrinsic
// or lane-mapping bug.

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

constexpr int          kN              = 128;
constexpr int          kK              = 1024;
constexpr int          kOutputsPerWG   = 4;
constexpr int          kBlockElems     = 32;
constexpr int          kBlockBytes     = 34;
constexpr int          kNBlocksPerRow  = kK / kBlockElems;   // 32
constexpr std::uint32_t kBlock         = 64;

// Pre-quant scale for X. Value doesn't matter for the parity check,
// just picked to keep dequantised X in a sensible fp32 range.
constexpr float        kXScale         = 0.007874f;   // ~= 1/127

std::string defaultHsacoPath(const char* argv0) {
    std::filesystem::path exe = std::filesystem::canonical(argv0);
    return (exe.parent_path() / "hsaco" / "matmul_q8_0_vec_dp4a.hsaco").string();
}

// Random int8 vector: full [-127, 127] range.
void fillRandomInt8(std::vector<std::int8_t>& v, std::uint32_t seed) {
    std::uint32_t x = seed;
    for (auto& b : v) {
        x = x * 1664525u + 1013904223u;
        b = static_cast<std::int8_t>(
            (static_cast<std::int32_t>(x >> 24)) - 128);
    }
}

// Native-layout Q8_0 blob (34 B blocks).
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

// CPU reference: Y[n] = sum_k (Xq[k] * xScale) * dequant_w(n, k),
// computed in fp64 to match the algebraic semantics of the DP4A
// kernel exactly.
void dp4aCpuRef(const std::vector<std::int8_t>& Xq,
                float                            xScale,
                const std::vector<std::uint8_t>& W,
                std::vector<float>&              Y,
                int N, int K)
{
    const std::size_t rowBytes = static_cast<std::size_t>(kNBlocksPerRow)
                               * static_cast<std::size_t>(kBlockBytes);
    for (int n = 0; n < N; ++n) {
        const std::uint8_t* rowW = W.data()
                                 + static_cast<std::size_t>(n) * rowBytes;
        double acc = 0.0;
        for (int k = 0; k < K; ++k) {
            const float wVal = dequantElemNative(rowW, k);
            const float xVal = static_cast<float>(Xq[k]) * xScale;
            acc += static_cast<double>(xVal) * static_cast<double>(wVal);
        }
        Y[n] = static_cast<float>(acc);
    }
}

} // namespace

int main(int argc, char** argv) {
    const std::string hsacoPath =
        (argc > 1) ? std::string{argv[1]} : defaultHsacoPath(argv[0]);

    std::printf("hip_matmul_q8_vec_dp4a_probe:\n  hsaco: %s\n"
                "  N=%d K=%d xScale=%g block=%u outputs_per_wg=%d\n",
                hsacoPath.c_str(),
                kN, kK, static_cast<double>(kXScale), kBlock, kOutputsPerWG);

    try {
        HipContext         ctx{};
        HipMemoryAllocator alloc{ctx};
        HipStream          stream{ctx, HipStreamKind::BlockingDefault};
        HipEvent           evStart{ctx};
        HipEvent           evEnd  {ctx};

        HipModule mod    = HipModule::fromFile(ctx, hsacoPath);
        HipKernel kernel = mod.getKernel("matmul_q8_0_vec_dp4a");

        const std::size_t xBytes = kK * sizeof(std::int8_t);
        const std::size_t yBytes = kN * sizeof(float);
        const std::size_t wBytes = static_cast<std::size_t>(kN)
                                 * kNBlocksPerRow * kBlockBytes;

        std::vector<std::int8_t>  hostXq(kK);
        std::vector<std::uint8_t> hostW(wBytes);
        std::vector<float>        hostYRef(kN);
        std::vector<float>        hostYGpu(kN, 0.0f);

        fillRandomInt8       (hostXq, /*seed=*/0xD4A00001u);
        fillRandomQ8_0Native(hostW,  /*seed=*/0xD4A00002u, /*nRows=*/kN);

        dp4aCpuRef(hostXq, kXScale, hostW, hostYRef, kN, kK);

        HipBuffer devXq    {alloc, xBytes};
        HipBuffer devXscale{alloc, sizeof(float), HipAllocKind::Device};
        HipBuffer devW     {alloc, wBytes};
        HipBuffer devY     {alloc, yBytes};

        alloc.copyH2D(devXq    .data(), hostXq.data(), xBytes);
        alloc.copyH2D(devXscale.data(), &kXScale,      sizeof(float));
        alloc.copyH2D(devW     .data(), hostW.data(),  wBytes);

        kernel.setPtr  (0, devXq    .data());
        kernel.setPtr  (1, devXscale.data());
        kernel.setPtr  (2, devW     .data());
        kernel.setPtr  (3, devY     .data());
        kernel.setValue(4, kK);
        kernel.setValue(5, kN);

        const std::uint32_t gridX =
            (static_cast<std::uint32_t>(kN) + kOutputsPerWG - 1) / kOutputsPerWG;

        evStart.record(stream);
        kernel.launch(stream, gridX, 1, 1,
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
        for (std::size_t i = 0; i < static_cast<std::size_t>(kN); ++i) {
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
            std::printf("  worst @ n=%zu: gpu=%.6g cpu=%.6g\n",
                        badIdx,
                        static_cast<double>(hostYGpu[badIdx]),
                        static_cast<double>(hostYRef[badIdx]));
        }

        const bool ok = (maxRatio <= 1.0f);
        std::printf("\nhip_matmul_q8_vec_dp4a_probe: %s\n", ok ? "OK" : "FAIL");
        return ok ? 0 : 1;

    } catch (const std::exception& e) {
        std::fprintf(stderr, "hip_matmul_q8_vec_dp4a_probe: threw: %s\n", e.what());
        return 2;
    }
}