// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// hip_rmsnorm_qkv_fp16_probe — parity check for the fp16-KV variant
// of the fused Q+K+V rmsnorm.
//
// Q branch is bit-identical to rmsnorm_qkv (fp32 workspace + fp32
// weights, same math). K/V branches promote fp16 cache reads to fp32,
// compute the reduction and scale in fp32, then round-trip results
// back to fp16 via __float2half.
//
// Combined-tolerance harness:
//   • Q:   abs 1e-4 + rel 1e-4  (same as rmsnorm_qkv f32)
//   • K/V: abs 1e-3 + rel 1e-3  (fp16 store precision + fp32 drift)
// K/V are compared as fp32 after __half2float on the GPU output —
// the CPU reference already stores fp32-quantized-to-fp16-and-back
// values, so equal rounding modes should keep the diff near zero.
//
// Cache-stomp check verifies that fp16 cache slots below curLen*kvDim
// stay bit-identical to their pre-seeded sentinels.

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

constexpr int          kT         = 4;
constexpr int          kNumHeads  = 8;
constexpr int          kNumKvHeads = 2;
constexpr int          kHeadDim   = 128;   // == K
constexpr int          kQRows     = kT * kNumHeads;         // 32
constexpr int          kKRows     = kT * kNumKvHeads;       // 8
constexpr int          kKvDim     = kNumKvHeads * kHeadDim; // 256
constexpr int          kCurLen    = 8;
constexpr int          kCacheSlots = 32;
constexpr float        kEps       = 1e-5f;
constexpr std::uint32_t kBlock    = 128;   // == RMSNORM_QKV_FP16_LOCAL_SIZE

std::string defaultHsacoPath(const char* argv0) {
    std::filesystem::path exe = std::filesystem::canonical(argv0);
    return (exe.parent_path() / "hsaco" / "rmsnorm_qkv_fp16.hsaco").string();
}

void fillRandomFloat(std::vector<float>& v, std::uint32_t seed, float scale) {
    std::uint32_t x = seed;
    for (auto& f : v) {
        x = x * 1664525u + 1013904223u;
        const std::int32_t s = static_cast<std::int32_t>(x >> 8) - (1 << 23);
        f = scale * static_cast<float>(s) / static_cast<float>(1 << 23);
    }
}

void fillRandomHalf(std::vector<__half>& v, std::uint32_t seed, float scale) {
    std::uint32_t x = seed;
    for (auto& h : v) {
        x = x * 1664525u + 1013904223u;
        const std::int32_t s = static_cast<std::int32_t>(x >> 8) - (1 << 23);
        h = __float2half(scale * static_cast<float>(s) / static_cast<float>(1 << 23));
    }
}

bool halfBitsEqual(__half a, __half b) {
    std::uint16_t ba;
    std::uint16_t bb;
    std::memcpy(&ba, &a, sizeof(ba));
    std::memcpy(&bb, &b, sizeof(bb));
    return ba == bb;
}

// One row of rmsnorm. `wr == nullptr` → skip weight (V path).
// Reads fp32 xr; writes fp32 to yr. K/V paths convert around this.
void rmsnormRowF32(const float* xr,
                   const float* wr,
                   float*       yr,
                   int K, float eps) {
    double sumsq = 0.0;
    for (int k = 0; k < K; ++k) {
        const double v = static_cast<double>(xr[k]);
        sumsq += v * v;
    }
    const double invRms = 1.0 / std::sqrt(sumsq / static_cast<double>(K)
                                        + static_cast<double>(eps));
    if (wr != nullptr) {
        for (int k = 0; k < K; ++k) {
            yr[k] = static_cast<float>(
                        static_cast<double>(xr[k])
                      * static_cast<double>(wr[k])
                      * invRms);
        }
    } else {
        for (int k = 0; k < K; ++k) {
            yr[k] = static_cast<float>(
                        static_cast<double>(xr[k]) * invRms);
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    const std::string hsacoPath =
        (argc > 1) ? std::string{argv[1]} : defaultHsacoPath(argv[0]);

    std::printf("hip_rmsnorm_qkv_fp16_probe:\n  hsaco: %s\n"
                "  T=%d nHeads=%d nKvHeads=%d headDim=%d curLen=%d kvDim=%d\n"
                "  qRows=%d kRows=%d grid=%d block=%u\n",
                hsacoPath.c_str(),
                kT, kNumHeads, kNumKvHeads, kHeadDim, kCurLen, kKvDim,
                kQRows, kKRows, kQRows + 2 * kKRows, kBlock);

    try {
        HipContext         ctx{};
        HipMemoryAllocator alloc{ctx};
        HipStream          stream{ctx, HipStreamKind::BlockingDefault};
        HipEvent           evStart{ctx};
        HipEvent           evEnd  {ctx};

        HipModule mod    = HipModule::fromFile(ctx, hsacoPath);
        HipKernel kernel = mod.getKernel("rmsnorm_qkv_fp16");

        // ---- host tensors -----------------------------------------------
        const std::size_t qElems  = static_cast<std::size_t>(kQRows) * kHeadDim;
        const std::size_t kvElems = static_cast<std::size_t>(kCacheSlots) * kKvDim;
        const std::size_t wElems  = static_cast<std::size_t>(kHeadDim);

        std::vector<float>  hostQx(qElems);
        std::vector<float>  hostQw(wElems);
        std::vector<__half> hostKx(kvElems);
        std::vector<float>  hostKw(wElems);
        std::vector<__half> hostVx(kvElems);
        std::vector<__half> hostKySeed(kvElems);
        std::vector<__half> hostVySeed(kvElems);

        fillRandomFloat(hostQx, /*seed=*/0xBEEF1001u, /*scale=*/1.0f);
        fillRandomFloat(hostQw, /*seed=*/0xBEEF1002u, /*scale=*/1.0f);
        fillRandomHalf (hostKx, /*seed=*/0xBEEF1003u, /*scale=*/1.0f);
        fillRandomFloat(hostKw, /*seed=*/0xBEEF1004u, /*scale=*/1.0f);
        fillRandomHalf (hostVx, /*seed=*/0xBEEF1005u, /*scale=*/1.0f);
        fillRandomHalf (hostKySeed, /*seed=*/0xBEEFFA11u, /*scale=*/1.0f);
        fillRandomHalf (hostVySeed, /*seed=*/0xBEEFB00Bu, /*scale=*/1.0f);

        // ---- CPU reference ----------------------------------------------
        std::vector<float>  hostQyRef(qElems, 0.0f);
        std::vector<__half> hostKyRef = hostKySeed;
        std::vector<__half> hostVyRef = hostVySeed;

        const std::size_t kvBase = static_cast<std::size_t>(kCurLen) * kKvDim;

        // Q branch: fp32.
        for (int row = 0; row < kQRows; ++row) {
            const float* xr = hostQx.data() + static_cast<std::size_t>(row) * kHeadDim;
            float*       yr = hostQyRef.data() + static_cast<std::size_t>(row) * kHeadDim;
            rmsnormRowF32(xr, hostQw.data(), yr, kHeadDim, kEps);
        }

        // K branch: fp16 in → fp32 compute → fp16 out.
        for (int row = 0; row < kKRows; ++row) {
            const std::size_t off = kvBase + static_cast<std::size_t>(row) * kHeadDim;
            std::vector<float> xr(kHeadDim);
            std::vector<float> yr(kHeadDim);
            for (int k = 0; k < kHeadDim; ++k) {
                xr[k] = __half2float(hostKx[off + k]);
            }
            rmsnormRowF32(xr.data(), hostKw.data(), yr.data(), kHeadDim, kEps);
            for (int k = 0; k < kHeadDim; ++k) {
                hostKyRef[off + k] = __float2half(yr[k]);
            }
        }

        // V branch: fp16 in → fp32 compute → fp16 out, no weight.
        for (int row = 0; row < kKRows; ++row) {
            const std::size_t off = kvBase + static_cast<std::size_t>(row) * kHeadDim;
            std::vector<float> xr(kHeadDim);
            std::vector<float> yr(kHeadDim);
            for (int k = 0; k < kHeadDim; ++k) {
                xr[k] = __half2float(hostVx[off + k]);
            }
            rmsnormRowF32(xr.data(), /*wr=*/nullptr, yr.data(), kHeadDim, kEps);
            for (int k = 0; k < kHeadDim; ++k) {
                hostVyRef[off + k] = __float2half(yr[k]);
            }
        }

        // ---- device tensors ---------------------------------------------
        const std::size_t qBytes  = qElems  * sizeof(float);
        const std::size_t kvBytes = kvElems * sizeof(__half);
        const std::size_t wBytes  = wElems  * sizeof(float);

        HipBuffer devQx {alloc, qBytes};
        HipBuffer devQw {alloc, wBytes};
        HipBuffer devQy {alloc, qBytes};
        HipBuffer devKx {alloc, kvBytes};
        HipBuffer devKw {alloc, wBytes};
        HipBuffer devKy {alloc, kvBytes};
        HipBuffer devVx {alloc, kvBytes};
        HipBuffer devVy {alloc, kvBytes};
        HipBuffer devCurLen{alloc, sizeof(int), HipAllocKind::Device};

        alloc.copyH2D(devQx    .data(), hostQx    .data(), qBytes);
        alloc.copyH2D(devQw    .data(), hostQw    .data(), wBytes);
        alloc.copyH2D(devKx    .data(), hostKx    .data(), kvBytes);
        alloc.copyH2D(devKw    .data(), hostKw    .data(), wBytes);
        alloc.copyH2D(devVx    .data(), hostVx    .data(), kvBytes);
        alloc.copyH2D(devKy    .data(), hostKySeed.data(), kvBytes);
        alloc.copyH2D(devVy    .data(), hostVySeed.data(), kvBytes);
        alloc.copyH2D(devCurLen.data(), &kCurLen,          sizeof(int));

        // ---- launch -----------------------------------------------------
        kernel.setPtr  (0,  devQx .data());
        kernel.setPtr  (1,  devQw .data());
        kernel.setPtr  (2,  devQy .data());
        kernel.setPtr  (3,  devKx .data());
        kernel.setPtr  (4,  devKw .data());
        kernel.setPtr  (5,  devKy .data());
        kernel.setPtr  (6,  devVx .data());
        kernel.setPtr  (7,  devVy .data());
        kernel.setValue(8,  kQRows);
        kernel.setValue(9,  kKRows);
        kernel.setValue(10, kHeadDim);
        kernel.setValue(11, kEps);
        kernel.setPtr  (12, devCurLen.data());
        kernel.setValue(13, kKvDim);

        const std::uint32_t grid =
            static_cast<std::uint32_t>(kQRows + 2 * kKRows);

        evStart.record(stream);
        kernel.launch(stream, grid, 1, 1, kBlock, 1, 1, /*shared=*/0);
        evEnd.record(stream);
        stream.synchronize();

        const float kernelMs = evEnd.elapsedMs(evStart);

        // ---- readback + tolerance compare -------------------------------
        std::vector<float>  gpuQy(qElems);
        std::vector<__half> gpuKy(kvElems);
        std::vector<__half> gpuVy(kvElems);
        alloc.copyD2H(gpuQy.data(), devQy.data(), qBytes);
        alloc.copyD2H(gpuKy.data(), devKy.data(), kvBytes);
        alloc.copyD2H(gpuVy.data(), devVy.data(), kvBytes);

        auto checkF32 = [&](const char* label,
                            const std::vector<float>& gpu,
                            const std::vector<float>& ref,
                            float absTol, float relTol) {
            float       maxAbs   = 0.0f;
            float       maxRatio = 0.0f;
            std::size_t badIdx   = SIZE_MAX;
            for (std::size_t i = 0; i < gpu.size(); ++i) {
                const float d         = std::fabs(gpu[i] - ref[i]);
                const float threshold = absTol + relTol * std::fabs(ref[i]);
                const float ratio     = d / threshold;
                if (ratio > maxRatio) { maxRatio = ratio; badIdx = i; }
                if (d > maxAbs) maxAbs = d;
            }
            std::printf("  %-3s max abs %.3e   max err/tol %.3f"
                        "   (abs %.1e + rel %.1e * |ref|)",
                        label,
                        static_cast<double>(maxAbs),
                        static_cast<double>(maxRatio),
                        static_cast<double>(absTol),
                        static_cast<double>(relTol));
            if (badIdx != SIZE_MAX && maxRatio > 0.0f) {
                std::printf("   worst @ %zu: gpu=%.6g cpu=%.6g",
                            badIdx,
                            static_cast<double>(gpu[badIdx]),
                            static_cast<double>(ref[badIdx]));
            }
            std::printf("\n");
            return maxRatio <= 1.0f;
        };

        // Promote fp16 outputs to fp32 for combined-tolerance compare.
        auto promote = [](const std::vector<__half>& src) {
            std::vector<float> dst(src.size());
            for (std::size_t i = 0; i < src.size(); ++i) {
                dst[i] = __half2float(src[i]);
            }
            return dst;
        };
        const std::vector<float> gpuKyF32 = promote(gpuKy);
        const std::vector<float> gpuVyF32 = promote(gpuVy);
        const std::vector<float> refKyF32 = promote(hostKyRef);
        const std::vector<float> refVyF32 = promote(hostVyRef);

        std::printf("\n  kernel:        %.3f ms\n", static_cast<double>(kernelMs));

        const bool okQ = checkF32("Q", gpuQy,    hostQyRef, 1e-4f, 1e-4f);
        const bool okK = checkF32("K", gpuKyF32, refKyF32,  1e-3f, 1e-3f);
        const bool okV = checkF32("V", gpuVyF32, refVyF32,  1e-3f, 1e-3f);

        // Cache-stomp check: pre-curLen fp16 slots bit-identical to seed.
        bool untouched = true;
        for (std::size_t i = 0; i < kvBase; ++i) {
            if (!halfBitsEqual(gpuKy[i], hostKySeed[i]) ||
                !halfBitsEqual(gpuVy[i], hostVySeed[i])) {
                untouched = false;
                break;
            }
        }
        std::printf("  pre-curLen cache slots untouched: %s\n",
                    untouched ? "yes" : "NO");

        const bool ok = okQ && okK && okV && untouched;
        std::printf("\nhip_rmsnorm_qkv_fp16_probe: %s\n", ok ? "OK" : "FAIL");
        return ok ? 0 : 1;

    } catch (const std::exception& e) {
        std::fprintf(stderr, "hip_rmsnorm_qkv_fp16_probe: threw: %s\n", e.what());
        return 2;
    }
}