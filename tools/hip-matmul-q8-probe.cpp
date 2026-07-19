// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// hip_matmul_q8_probe — parity check for the second real HIP kernel
// port: matmul_q8_0_vec. Loads matmul_q8_0_vec.hsaco, runs it on
// synthetic Q8_0-quantized weights + random X, compares against an
// inline CPU reference that dequantizes the same weights + does the
// dot product in double precision.
//
// Test shape: N=128 outputs × K=1024 elements (32 Q8_0 blocks per
// row). Small enough to run fast, big enough to exercise the tiling
// path (X_TILE_ELEMENTS=1024 so one full tile pass), and small
// enough that the double-precision CPU reference is trivial.
//
// We DON'T round-trip real floats through Q8_0 packing. Instead we
// generate random block scales + random int8 quants directly and
// give both the GPU (packed bytes) and the CPU (dequant to fp32)
// exactly the same data. That way any diff is a matmul bug, not
// packing noise.
//
// Uses the combined-tolerance pattern
// (|diff| <= abs_tol + rel_tol * |ref|). Q8_0 dequant + 1024-term
// parallel FMA adds a bit more noise than pure fp32, so tol is looser
// (1e-3 abs + 1e-3 rel) than the pure-fp32 element-wise probes.

#include "core/gpu/hip/HipContext.hpp"
#include "core/gpu/hip/HipEvent.hpp"
#include "core/gpu/hip/HipKernel.hpp"
#include "core/gpu/hip/HipMemoryAllocator.hpp"
#include "core/gpu/hip/HipModule.hpp"
#include "core/gpu/hip/HipStream.hpp"

#include <hip/hip_fp16.h>

#include <algorithm>
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

// Test shape.
constexpr int          kN                       = 128;
constexpr int          kK                       = 1024;
constexpr int          kBlockElems              = 32;
constexpr int          kBlockBytes              = 34;    // 2B half + 32B int8
constexpr int          kBlocksPerRow            = kK / kBlockElems;
constexpr std::uint32_t kThreadsPerBlock        = 128;   // == MATMUL_Q8_0_LOCAL
constexpr std::uint32_t kOutputsPerGroup        = 4;     // == LOCAL/32

std::string defaultHsacoPath(const char* argv0) {
    std::filesystem::path exe = std::filesystem::canonical(argv0);
    return (exe.parent_path() / "hsaco" / "matmul_q8_0_vec.hsaco").string();
}

// Deterministic PRNG so the probe is reproducible.
std::uint32_t nextRand(std::uint32_t& state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

void fillFloats(std::vector<float>& v, std::uint32_t seed, float scale) {
    std::uint32_t s = seed;
    for (auto& f : v) {
        const std::int32_t r = static_cast<std::int32_t>(nextRand(s) >> 8) - (1 << 23);
        f = scale * static_cast<float>(r) / static_cast<float>(1 << 23);
    }
}

// Build a random Q8_0 weight blob (N rows × K/32 blocks × 34 bytes).
// Also returns the dequantized fp32 weight matrix for the CPU
// reference — same numbers on both sides.
void makeRandomQ8_0(std::vector<std::uint8_t>& packed,
                    std::vector<float>&        dequant,
                    std::uint32_t              seed)
{
    const std::size_t rowBytes = static_cast<std::size_t>(kBlocksPerRow) * kBlockBytes;
    packed .assign(static_cast<std::size_t>(kN) * rowBytes, 0);
    dequant.assign(static_cast<std::size_t>(kN) * kK, 0.0f);

    std::uint32_t s = seed;
    for (int n = 0; n < kN; ++n) {
        std::uint8_t* row = packed.data() + static_cast<std::size_t>(n) * rowBytes;
        for (int b = 0; b < kBlocksPerRow; ++b) {
            std::uint8_t* blk = row + b * kBlockBytes;

            // Scale in a reasonable range so dequantized values live
            // in [-1, 1] approx (127 * d ≈ 1 → d ≈ 0.008).
            const std::uint32_t r = nextRand(s);
            const float dScalar =
                0.001f + 0.02f * (static_cast<float>(r & 0xFFFF) / 65535.0f);
            const __half dHalf = __float2half(dScalar);
            std::memcpy(blk, &dHalf, 2);

            // Round-trip the scale through fp16 for the CPU reference,
            // matching what the GPU kernel actually sees. Skipping this
            // step makes the CPU use ~10 extra bits of precision that
            // the fp16 storage doesn't have — accumulates to a few %
            // error over 1024 terms and produces a false-positive FAIL.
            const float dFromHalf = __half2float(dHalf);

            // int8 quants
            for (int i = 0; i < kBlockElems; ++i) {
                const std::uint32_t rq = nextRand(s);
                const std::int8_t v = static_cast<std::int8_t>(
                    (static_cast<std::int32_t>(rq >> 24)) - 128);
                blk[2 + i] = static_cast<std::uint8_t>(v);

                const int k = b * kBlockElems + i;
                dequant[static_cast<std::size_t>(n) * kK + k] =
                    dFromHalf * static_cast<float>(v);
            }
        }
    }
}

// CPU reference: Y[n] = sum_k X[k] * W_dequant[n, k] in fp64.
void matmulCpuRef(const std::vector<float>& X,
                  const std::vector<float>& Wdequant,
                  std::vector<float>&       Y)
{
    for (std::size_t n = 0; n < static_cast<std::size_t>(kN); ++n) {
        double acc = 0.0;
        const float* row = Wdequant.data() + n * static_cast<std::size_t>(kK);
        for (std::size_t k = 0; k < static_cast<std::size_t>(kK); ++k) {
            acc += static_cast<double>(X[k]) * static_cast<double>(row[k]);
        }
        Y[n] = static_cast<float>(acc);
    }
}

} // namespace

int main(int argc, char** argv) {
    const std::string hsacoPath =
        (argc > 1) ? std::string{argv[1]} : defaultHsacoPath(argv[0]);

    std::printf("hip_matmul_q8_probe:\n  hsaco: %s\n  N=%d K=%d block=%u outputs_per_group=%u\n",
                hsacoPath.c_str(), kN, kK, kThreadsPerBlock, kOutputsPerGroup);

    try {
        HipContext         ctx{};
        HipMemoryAllocator alloc{ctx};
        HipStream          stream{ctx, HipStreamKind::BlockingDefault};
        HipEvent           evStart{ctx};
        HipEvent           evEnd  {ctx};

        HipModule mod    = HipModule::fromFile(ctx, hsacoPath);
        HipKernel kernel = mod.getKernel("matmul_q8_0_vec");

        // ---- host tensors -----------------------------------------------
        std::vector<float>        hostX(kK);
        std::vector<std::uint8_t> hostW;
        std::vector<float>        hostWDequant;
        std::vector<float>        hostYRef(kN);
        std::vector<float>        hostYGpu(kN, 0.0f);

        fillFloats(hostX, /*seed=*/0xA5A5A5A5u, /*scale=*/0.5f);
        makeRandomQ8_0(hostW, hostWDequant, /*seed=*/0x5A5A5A5Au);

        matmulCpuRef(hostX, hostWDequant, hostYRef);

        // ---- device tensors ---------------------------------------------
        const std::size_t xBytes = static_cast<std::size_t>(kK) * sizeof(float);
        const std::size_t wBytes = hostW.size();
        const std::size_t yBytes = static_cast<std::size_t>(kN) * sizeof(float);
        HipBuffer devX{alloc, xBytes};
        HipBuffer devW{alloc, wBytes};
        HipBuffer devY{alloc, yBytes};

        alloc.copyH2D(devX.data(), hostX.data(), xBytes);
        alloc.copyH2D(devW.data(), hostW.data(), wBytes);

        // ---- launch -----------------------------------------------------
        kernel.setPtr  (0, devX.data());
        kernel.setPtr  (1, devW.data());
        kernel.setPtr  (2, devY.data());
        kernel.setValue(3, kK);
        kernel.setValue(4, kN);

        const std::uint32_t grid =
            (static_cast<std::uint32_t>(kN) + kOutputsPerGroup - 1) / kOutputsPerGroup;

        evStart.record(stream);
        kernel.launch(stream, grid, 1, 1,
                      kThreadsPerBlock, 1, 1,
                      /*shared=*/0);
        evEnd.record(stream);
        stream.synchronize();

        const float kernelMs = evEnd.elapsedMs(evStart);

        // ---- readback + combined-tolerance compare ----------------------
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
        if (badIdx != SIZE_MAX) {
            std::printf("  worst @ n=%zu: gpu=%.6g cpu=%.6g\n",
                        badIdx,
                        static_cast<double>(hostYGpu[badIdx]),
                        static_cast<double>(hostYRef[badIdx]));
        }

        const bool ok = (maxRatio <= 1.0f);
        std::printf("\nhip_matmul_q8_probe: %s\n", ok ? "OK" : "FAIL");
        return ok ? 0 : 1;

    } catch (const std::exception& e) {
        std::fprintf(stderr, "hip_matmul_q8_probe: threw: %s\n", e.what());
        return 2;
    }
}