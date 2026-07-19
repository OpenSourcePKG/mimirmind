// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// hip_rmsnorm_qkv_probe — parity check for the fused Q+K+V RMSNorm.
// One dispatch normalizes all three streams into their proper
// destinations:
//   • Q rows into a plain workspace at natural row offsets
//   • K rows into the K cache at `curLen * kvDim` offset (with weight)
//   • V rows into the V cache at `curLen * kvDim` offset (no weight)
//
// The probe additionally verifies that K/V cache slots BEFORE curLen
// are untouched — a common bug-class for cache-writing kernels is
// stomping on prior tokens.
//
// `curLenPtr` is a device int slot (the RoPE-style pattern already
// exercised in hip_rope_probe).

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

// Test shape — Gemma-flavored small: nHeads=8 / nKvHeads=2 (4:1 GQA
// ratio), head_dim=128. Small enough to run in a blink, exercises the
// three-branch grid routing.
constexpr int          kT         = 4;
constexpr int          kNumHeads  = 8;
constexpr int          kNumKvHeads = 2;
constexpr int          kHeadDim   = 128;   // == K
constexpr int          kQRows     = kT * kNumHeads;      // 32
constexpr int          kKRows     = kT * kNumKvHeads;    // 8
constexpr int          kKvDim     = kNumKvHeads * kHeadDim;  // 256
constexpr int          kCurLen    = 8;     // non-zero → exercise offset path
constexpr int          kCacheSlots = 32;   // > curLen + T so we can check untouched tail
constexpr float        kEps       = 1e-5f;
constexpr std::uint32_t kBlock    = 128;   // == RMSNORM_QKV_LOCAL_SIZE

std::string defaultHsacoPath(const char* argv0) {
    std::filesystem::path exe = std::filesystem::canonical(argv0);
    return (exe.parent_path() / "hsaco" / "rmsnorm_qkv.hsaco").string();
}

void fillRandom(std::vector<float>& v, std::uint32_t seed, float scale) {
    std::uint32_t x = seed;
    for (auto& f : v) {
        x = x * 1664525u + 1013904223u;
        const std::int32_t s = static_cast<std::int32_t>(x >> 8) - (1 << 23);
        f = scale * static_cast<float>(s) / static_cast<float>(1 << 23);
    }
}

// Normalize one row using the sum-of-squares formula. If `weight` is
// nullptr, skip the multiplicative weight (V path).
void rmsnormRow(const float* xr,
                const float* weight,
                float*       yr,
                int K, float eps) {
    double sumsq = 0.0;
    for (int k = 0; k < K; ++k) {
        const double v = static_cast<double>(xr[k]);
        sumsq += v * v;
    }
    const double mean   = sumsq / static_cast<double>(K);
    const double invRms = 1.0 / std::sqrt(mean + static_cast<double>(eps));
    if (weight != nullptr) {
        for (int k = 0; k < K; ++k) {
            yr[k] = static_cast<float>(
                        static_cast<double>(xr[k])
                      * static_cast<double>(weight[k])
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

    std::printf("hip_rmsnorm_qkv_probe:\n  hsaco: %s\n"
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
        HipKernel kernel = mod.getKernel("rmsnorm_qkv");

        // ---- host tensors -----------------------------------------------
        const std::size_t qElems  = static_cast<std::size_t>(kQRows) * kHeadDim;
        const std::size_t kvElems = static_cast<std::size_t>(kCacheSlots) * kKvDim;
        const std::size_t wElems  = static_cast<std::size_t>(kHeadDim);

        std::vector<float> hostQx  (qElems);
        std::vector<float> hostQw  (wElems);
        std::vector<float> hostKx  (kvElems);
        std::vector<float> hostKw  (wElems);
        std::vector<float> hostVx  (kvElems);

        fillRandom(hostQx, /*seed=*/0xBEEF0001u, /*scale=*/1.0f);
        fillRandom(hostQw, /*seed=*/0xBEEF0002u, /*scale=*/1.0f);
        fillRandom(hostKx, /*seed=*/0xBEEF0003u, /*scale=*/1.0f);
        fillRandom(hostKw, /*seed=*/0xBEEF0004u, /*scale=*/1.0f);
        fillRandom(hostVx, /*seed=*/0xBEEF0005u, /*scale=*/1.0f);

        // Prior K/V cache content — sentinel pattern so we can verify
        // that slots < curLen are NOT touched by the kernel.
        std::vector<float> hostKy = hostKx;   // start identical
        std::vector<float> hostVy = hostVx;

        // ---- CPU reference ----------------------------------------------
        // Q: rows [0..qRows) → hostQy at row offset row*K
        // K: rows [0..kRows) → hostKy at (curLen*kvDim + row*K)
        // V: rows [0..kRows) → hostVy at (curLen*kvDim + row*K), no weight
        std::vector<float> hostQyRef(qElems, 0.0f);
        std::vector<float> hostKyRef = hostKy;
        std::vector<float> hostVyRef = hostVy;
        {
            const std::size_t kvBase = static_cast<std::size_t>(kCurLen) * kKvDim;
            for (int row = 0; row < kQRows; ++row) {
                const float* xr = hostQx.data() + static_cast<std::size_t>(row) * kHeadDim;
                float*       yr = hostQyRef.data() + static_cast<std::size_t>(row) * kHeadDim;
                rmsnormRow(xr, hostQw.data(), yr, kHeadDim, kEps);
            }
            for (int row = 0; row < kKRows; ++row) {
                const std::size_t off = kvBase + static_cast<std::size_t>(row) * kHeadDim;
                rmsnormRow(hostKx.data() + off, hostKw.data(),
                           hostKyRef.data() + off, kHeadDim, kEps);
            }
            for (int row = 0; row < kKRows; ++row) {
                const std::size_t off = kvBase + static_cast<std::size_t>(row) * kHeadDim;
                rmsnormRow(hostVx.data() + off, /*weight=*/nullptr,
                           hostVyRef.data() + off, kHeadDim, kEps);
            }
        }

        // ---- device tensors ---------------------------------------------
        const std::size_t qBytes  = qElems  * sizeof(float);
        const std::size_t kvBytes = kvElems * sizeof(float);
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

        alloc.copyH2D(devQx.data(),  hostQx.data(), qBytes);
        alloc.copyH2D(devQw.data(),  hostQw.data(), wBytes);
        alloc.copyH2D(devKx.data(),  hostKx.data(), kvBytes);
        alloc.copyH2D(devKw.data(),  hostKw.data(), wBytes);
        alloc.copyH2D(devVx.data(),  hostVx.data(), kvBytes);
        // Pre-seed K/V _y_ caches with the sentinel content so the
        // < curLen slots have known values to check against.
        alloc.copyH2D(devKy.data(),  hostKy.data(), kvBytes);
        alloc.copyH2D(devVy.data(),  hostVy.data(), kvBytes);
        alloc.copyH2D(devCurLen.data(), &kCurLen, sizeof(int));

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

        // ---- readback + combined-tolerance compare ----------------------
        std::vector<float> gpuQy(qElems);
        std::vector<float> gpuKy(kvElems);
        std::vector<float> gpuVy(kvElems);
        alloc.copyD2H(gpuQy.data(), devQy.data(), qBytes);
        alloc.copyD2H(gpuKy.data(), devKy.data(), kvBytes);
        alloc.copyD2H(gpuVy.data(), devVy.data(), kvBytes);

        constexpr float kAbsTol = 1e-4f;
        constexpr float kRelTol = 1e-4f;

        auto check = [&](const char* label,
                         const std::vector<float>& gpu,
                         const std::vector<float>& ref) {
            float       maxAbs   = 0.0f;
            float       maxRatio = 0.0f;
            std::size_t badIdx   = SIZE_MAX;
            for (std::size_t i = 0; i < gpu.size(); ++i) {
                const float d         = std::fabs(gpu[i] - ref[i]);
                const float threshold = kAbsTol + kRelTol * std::fabs(ref[i]);
                const float ratio     = d / threshold;
                if (ratio > maxRatio) { maxRatio = ratio; badIdx = i; }
                if (d > maxAbs) maxAbs = d;
            }
            std::printf("  %-3s max abs %.3e   max err/tol %.3f",
                        label,
                        static_cast<double>(maxAbs),
                        static_cast<double>(maxRatio));
            if (badIdx != SIZE_MAX && maxRatio > 0.0f) {
                std::printf("   worst @ %zu: gpu=%.6g cpu=%.6g",
                            badIdx,
                            static_cast<double>(gpu[badIdx]),
                            static_cast<double>(ref[badIdx]));
            }
            std::printf("\n");
            return maxRatio <= 1.0f;
        };

        std::printf("\n  kernel:        %.3f ms\n", static_cast<double>(kernelMs));
        std::printf("  tol formula:   abs %.1e + rel %.1e * |ref|\n",
                    static_cast<double>(kAbsTol), static_cast<double>(kRelTol));

        const bool okQ = check("Q",  gpuQy, hostQyRef);
        const bool okK = check("K",  gpuKy, hostKyRef);
        const bool okV = check("V",  gpuVy, hostVyRef);

        // Explicit sub-check: the region strictly before curLen*kvDim
        // must be bit-identical to the pre-seeded sentinel — no drift
        // tolerance, cache stomping is a bug not a numerical issue.
        const std::size_t kvBase = static_cast<std::size_t>(kCurLen) * kKvDim;
        bool untouched = true;
        for (std::size_t i = 0; i < kvBase; ++i) {
            if (gpuKy[i] != hostKy[i] || gpuVy[i] != hostVy[i]) {
                untouched = false;
                std::printf("  cache stomp @ i=%zu (< kvBase=%zu):"
                            " K gpu=%.6g seed=%.6g   V gpu=%.6g seed=%.6g\n",
                            i, kvBase,
                            static_cast<double>(gpuKy[i]),
                            static_cast<double>(hostKy[i]),
                            static_cast<double>(gpuVy[i]),
                            static_cast<double>(hostVy[i]));
                break;
            }
        }
        std::printf("  pre-curLen cache slots untouched: %s\n",
                    untouched ? "yes" : "NO");

        const bool ok = okQ && okK && okV && untouched;
        std::printf("\nhip_rmsnorm_qkv_probe: %s\n", ok ? "OK" : "FAIL");
        return ok ? 0 : 1;

    } catch (const std::exception& e) {
        std::fprintf(stderr, "hip_rmsnorm_qkv_probe: threw: %s\n", e.what());
        return 2;
    }
}