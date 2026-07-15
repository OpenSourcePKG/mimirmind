// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// hip_kv_quant_commit_q8_0_probe — parity check for the fp32 → Q8_0
// KV write kernel. First write-side quantization kernel in the port.
//
// Feeds a random fp32 workspace [T, kvDim] and verifies byte-exact
// output against a CPU reference that does the same absmax +
// __float2half + roundf-to-int8 dance. Uses full-wave WgSize=32 —
// no width scoping in the reduction.
//
// Also verifies (cache-stomp check) that Q8_0 slots before curLen
// stay bit-identical to their pre-seeded sentinel.

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

constexpr int          kT              = 4;
constexpr int          kNumKvHeads     = 2;
constexpr int          kHeadDim        = 128;
constexpr int          kKvDim          = kNumKvHeads * kHeadDim;      // 256
constexpr int          kBlockElems     = 32;
constexpr int          kBlockBytes     = 34;
constexpr int          kNBlocksPerRow  = kKvDim / kBlockElems;         // 8
constexpr int          kCurLen         = 8;
constexpr int          kCacheSlots     = 32;   // > curLen + T; check untouched tail
constexpr std::uint32_t kBlock         = 32;   // KV_QUANT_COMMIT_LOCAL, full wave

std::string defaultHsacoPath(const char* argv0) {
    std::filesystem::path exe = std::filesystem::canonical(argv0);
    return (exe.parent_path() / "hsaco" / "kv_quant_commit_q8_0.hsaco").string();
}

void fillRandomFloat(std::vector<float>& v, std::uint32_t seed, float scale) {
    std::uint32_t x = seed;
    for (auto& f : v) {
        x = x * 1664525u + 1013904223u;
        const std::int32_t s = static_cast<std::int32_t>(x >> 8) - (1 << 23);
        f = scale * static_cast<float>(s) / static_cast<float>(1 << 23);
    }
}

void fillRandomBytes(std::vector<std::uint8_t>& v, std::uint32_t seed) {
    std::uint32_t x = seed;
    for (auto& b : v) {
        x = x * 1664525u + 1013904223u;
        b = static_cast<std::uint8_t>((x >> 24) & 0xFFu);
    }
}

// CPU reference — one row of Q8_0 quantization. Byte-exact target for
// the GPU kernel. Matches the semantics of compute::quant::Q8_0.
void quantizeRowQ8_0(const float* src, std::uint8_t* dst, int kvDim) {
    const int nBlocks = kvDim / kBlockElems;
    for (int b = 0; b < nBlocks; ++b) {
        const float* srcBlk = src + static_cast<std::size_t>(b) * kBlockElems;
        std::uint8_t* dstBlk = dst + static_cast<std::size_t>(b) * kBlockBytes;

        float absMax = 0.0f;
        for (int i = 0; i < kBlockElems; ++i) {
            const float a = std::fabs(srcBlk[i]);
            if (a > absMax) absMax = a;
        }
        const float scale    = (absMax > 0.0f) ? (absMax / 127.0f) : 0.0f;
        const float invScale = (absMax > 0.0f) ? (127.0f / absMax) : 0.0f;

        // Write fp16 scale into the 2-byte header.
        const __half hScale = __float2half(scale);
        std::memcpy(dstBlk, &hScale, sizeof(__half));

        // int8 quants via roundf + clamp — same rounding as HIP roundf.
        for (int i = 0; i < kBlockElems; ++i) {
            const float qf = std::round(srcBlk[i] * invScale);
            const float qc = std::fmin(std::fmax(qf, -127.0f), 127.0f);
            reinterpret_cast<signed char*>(dstBlk)[2 + i] =
                static_cast<signed char>(static_cast<int>(qc));
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    const std::string hsacoPath =
        (argc > 1) ? std::string{argv[1]} : defaultHsacoPath(argv[0]);

    std::printf("hip_kv_quant_commit_q8_0_probe:\n  hsaco: %s\n"
                "  T=%d kvDim=%d nBlocksPerRow=%d curLen=%d block=%u\n",
                hsacoPath.c_str(),
                kT, kKvDim, kNBlocksPerRow, kCurLen, kBlock);

    try {
        HipContext         ctx{};
        HipMemoryAllocator alloc{ctx};
        HipStream          stream{ctx, HipStreamKind::BlockingDefault};
        HipEvent           evStart{ctx};
        HipEvent           evEnd  {ctx};

        HipModule mod    = HipModule::fromFile(ctx, hsacoPath);
        HipKernel kernel = mod.getKernel("kv_quant_commit_q8_0");

        // ---- host tensors -----------------------------------------------
        const std::size_t srcElems  = static_cast<std::size_t>(kT) * kKvDim;
        const std::size_t srcBytes  = srcElems * sizeof(float);
        const std::size_t cacheBytes = static_cast<std::size_t>(kCacheSlots)
                                     * static_cast<std::size_t>(kNBlocksPerRow)
                                     * static_cast<std::size_t>(kBlockBytes);

        std::vector<float>        hostSrc(srcElems);
        std::vector<std::uint8_t> hostCacheSeed(cacheBytes);
        fillRandomFloat(hostSrc,       /*seed=*/0xAA55AA55u, /*scale=*/1.0f);
        fillRandomBytes(hostCacheSeed, /*seed=*/0xF00DFACEu);

        // ---- CPU reference ----------------------------------------------
        std::vector<std::uint8_t> hostCacheRef = hostCacheSeed;
        for (int t = 0; t < kT; ++t) {
            const float* srcRow = hostSrc.data()
                                + static_cast<std::size_t>(t) * kKvDim;
            std::uint8_t* dstRow = hostCacheRef.data()
                                 + static_cast<std::size_t>(kCurLen + t)
                                 * kNBlocksPerRow * kBlockBytes;
            quantizeRowQ8_0(srcRow, dstRow, kKvDim);
        }

        // ---- device tensors ---------------------------------------------
        HipBuffer devSrc     {alloc, srcBytes};
        HipBuffer devCache   {alloc, cacheBytes};
        HipBuffer devCurLen  {alloc, sizeof(int), HipAllocKind::Device};

        alloc.copyH2D(devSrc.data(),     hostSrc.data(),       srcBytes);
        alloc.copyH2D(devCache.data(),   hostCacheSeed.data(), cacheBytes);
        alloc.copyH2D(devCurLen.data(),  &kCurLen,             sizeof(int));

        // ---- launch -----------------------------------------------------
        kernel.setPtr  (0, devSrc.data());
        kernel.setPtr  (1, devCache.data());
        kernel.setValue(2, kKvDim);
        kernel.setPtr  (3, devCurLen.data());

        evStart.record(stream);
        kernel.launch(stream,
                      /*grid=*/  static_cast<std::uint32_t>(kT),
                                 static_cast<std::uint32_t>(kNBlocksPerRow), 1,
                      /*block=*/ kBlock, 1, 1,
                      /*shared=*/0);
        evEnd.record(stream);
        stream.synchronize();
        const float kernelMs = evEnd.elapsedMs(evStart);

        // ---- readback + byte-exact compare ------------------------------
        std::vector<std::uint8_t> gpu(cacheBytes, 0u);
        alloc.copyD2H(gpu.data(), devCache.data(), cacheBytes);

        std::size_t mismatches = 0;
        std::size_t firstBad   = SIZE_MAX;
        for (std::size_t i = 0; i < cacheBytes; ++i) {
            if (gpu[i] != hostCacheRef[i]) {
                if (mismatches == 0) firstBad = i;
                ++mismatches;
            }
        }

        // Cache-stomp: bytes in pre-curLen rows must be bit-identical
        // to the seed. mismatches above catches this too; we report it
        // separately for readability.
        const std::size_t preCurBytes = static_cast<std::size_t>(kCurLen)
                                      * kNBlocksPerRow * kBlockBytes;
        bool untouched = true;
        for (std::size_t i = 0; i < preCurBytes; ++i) {
            if (gpu[i] != hostCacheSeed[i]) { untouched = false; break; }
        }

        std::printf("\n  kernel:        %.3f ms\n", static_cast<double>(kernelMs));
        std::printf("  tol:           byte-exact (bit-parity vs CPU Q8_0)\n");
        std::printf("  mismatches:    %zu / %zu bytes",
                    mismatches, cacheBytes);
        if (firstBad != SIZE_MAX) {
            std::printf("   first @ %zu: gpu=0x%02X cpu=0x%02X",
                        firstBad,
                        static_cast<unsigned>(gpu[firstBad]),
                        static_cast<unsigned>(hostCacheRef[firstBad]));
        }
        std::printf("\n");
        std::printf("  pre-curLen cache bytes untouched: %s\n",
                    untouched ? "yes" : "NO");

        const bool ok = (mismatches == 0) && untouched;
        std::printf("\nhip_kv_quant_commit_q8_0_probe: %s\n", ok ? "OK" : "FAIL");
        return ok ? 0 : 1;

    } catch (const std::exception& e) {
        std::fprintf(stderr, "hip_kv_quant_commit_q8_0_probe: threw: %s\n", e.what());
        return 2;
    }
}