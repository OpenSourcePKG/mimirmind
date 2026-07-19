// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// hip_rmsnorm_probe — bit-parity check for the first real HIP kernel
// port. Loads kernels_hip/rmsnorm.hsaco (compiled from rmsnorm.hip via
// hipcc --genco at CMake time), runs it on synthetic input, and
// compares against an inline CPU reference that computes the same
// math in double precision.
//
// Deliberately self-contained on the reference side — we don't pull
// compute::rmsNorm from mimirmind_core because that would drag the
// full compute/ dependency graph in. The reference is 8 lines of
// arithmetic, easier to audit inline.
//
// Tolerance: combined-tolerance pattern
// (|diff| <= abs_tol + rel_tol * |ref|). RMSNorm involves a
// sum-of-squares reduction + rsqrt, and the parallel reduction order
// on the GPU differs from the sequential CPU order — expect ULP-level
// drift, not bit-exact. 1e-4 abs + 1e-4 rel is comfortable for fp32
// with K in the low thousands.

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

using mimirmind::core::hip::HipAllocKind;
using mimirmind::core::hip::HipBuffer;
using mimirmind::core::hip::HipContext;
using mimirmind::core::hip::HipEvent;
using mimirmind::core::hip::HipKernel;
using mimirmind::core::hip::HipMemoryAllocator;
using mimirmind::core::hip::HipModule;
using mimirmind::core::hip::HipStream;
using mimirmind::core::hip::HipStreamKind;

// Test shape — a couple of representative sizes for early Llama-like
// models. Not the full 4096 hidden dim yet, just enough that the
// per-workgroup reduction is exercised across multiple strides.
constexpr int          kM        = 8;
constexpr int          kK        = 4096;
constexpr float        kEps      = 1e-5f;
constexpr std::uint32_t kBlock   = 128;   // must match RMSNORM_LOCAL_SIZE

std::string defaultHsacoPath(const char* argv0) {
    std::filesystem::path exe = std::filesystem::canonical(argv0);
    return (exe.parent_path() / "hsaco" / "rmsnorm.hsaco").string();
}

// Deterministic pseudo-random fill so the probe is reproducible.
void fillRandom(std::vector<float>& v, std::uint32_t seed) {
    std::uint32_t x = seed;
    for (auto& f : v) {
        x = x * 1664525u + 1013904223u;
        // Map top 24 bits to [-1, 1).
        const std::int32_t s = static_cast<std::int32_t>(x >> 8) - (1 << 23);
        f = static_cast<float>(s) / static_cast<float>(1 << 23);
    }
}

// CPU reference: per-row RMSNorm in double precision. Same formula as
// the kernel, no parallelism, deterministic order.
void rmsnormCpuRef(const std::vector<float>& x,
                   const std::vector<float>& weight,
                   std::vector<float>&       y,
                   int M, int K, float eps) {
    for (int m = 0; m < M; ++m) {
        const float* xr = x.data() + static_cast<std::size_t>(m) * K;
        float*       yr = y.data() + static_cast<std::size_t>(m) * K;
        double sumsq = 0.0;
        for (int k = 0; k < K; ++k) {
            const double v = static_cast<double>(xr[k]);
            sumsq += v * v;
        }
        const double mean   = sumsq / static_cast<double>(K);
        const double invRms = 1.0 / std::sqrt(mean + static_cast<double>(eps));
        for (int k = 0; k < K; ++k) {
            yr[k] = static_cast<float>(
                        static_cast<double>(xr[k])
                      * static_cast<double>(weight[k])
                      * invRms);
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    const std::string hsacoPath =
        (argc > 1) ? std::string{argv[1]} : defaultHsacoPath(argv[0]);

    std::printf("hip_rmsnorm_probe:\n  hsaco: %s\n  M=%d K=%d eps=%g block=%u\n",
                hsacoPath.c_str(), kM, kK, static_cast<double>(kEps), kBlock);

    try {
        HipContext         ctx{};
        HipMemoryAllocator alloc{ctx};
        HipStream          stream{ctx, HipStreamKind::BlockingDefault};
        HipEvent           evStart{ctx};
        HipEvent           evEnd  {ctx};

        HipModule mod = HipModule::fromFile(ctx, hsacoPath);
        HipKernel kernel = mod.getKernel("rmsnorm");

        // ---- host tensors -----------------------------------------------
        const std::size_t xElems = static_cast<std::size_t>(kM) * kK;
        std::vector<float> hostX(xElems);
        std::vector<float> hostW(kK);
        std::vector<float> hostYRef(xElems);
        std::vector<float> hostYGpu(xElems, 0.0f);

        fillRandom(hostX, /*seed=*/0xB16BEEFu);
        fillRandom(hostW, /*seed=*/0x51EEF00Du);

        // ---- CPU reference ----------------------------------------------
        rmsnormCpuRef(hostX, hostW, hostYRef, kM, kK, kEps);

        // ---- device tensors ---------------------------------------------
        const std::size_t xBytes = xElems * sizeof(float);
        const std::size_t wBytes = static_cast<std::size_t>(kK) * sizeof(float);
        HipBuffer devX{alloc, xBytes};
        HipBuffer devW{alloc, wBytes};
        HipBuffer devY{alloc, xBytes};

        alloc.copyH2D(devX.data(), hostX.data(), xBytes);
        alloc.copyH2D(devW.data(), hostW.data(), wBytes);

        // ---- launch -----------------------------------------------------
        kernel.setPtr  (0, devX.data());
        kernel.setPtr  (1, devW.data());
        kernel.setPtr  (2, devY.data());
        kernel.setValue(3, kEps);
        kernel.setValue(4, kK);

        evStart.record(stream);
        kernel.launch(stream,
                      /*grid=*/  static_cast<std::uint32_t>(kM), 1, 1,
                      /*block=*/ kBlock, 1, 1,
                      /*shared=*/0);
        evEnd.record(stream);
        stream.synchronize();

        const float kernelMs = evEnd.elapsedMs(evStart);

        // ---- readback + combined-tolerance compare ----------------------
        alloc.copyD2H(hostYGpu.data(), devY.data(), xBytes);

        constexpr float kAbsTol = 1e-4f;
        constexpr float kRelTol = 1e-4f;

        float       maxAbs   = 0.0f;
        float       maxRatio = 0.0f;
        std::size_t badIdx   = SIZE_MAX;
        for (std::size_t i = 0; i < xElems; ++i) {
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
            const std::size_t m = badIdx / static_cast<std::size_t>(kK);
            const std::size_t k = badIdx % static_cast<std::size_t>(kK);
            std::printf("  worst @ m=%zu k=%zu: gpu=%.6g cpu=%.6g\n",
                        m, k,
                        static_cast<double>(hostYGpu[badIdx]),
                        static_cast<double>(hostYRef[badIdx]));
        }

        const bool ok = (maxRatio <= 1.0f);
        std::printf("\nhip_rmsnorm_probe: %s\n", ok ? "OK" : "FAIL");
        return ok ? 0 : 1;

    } catch (const std::exception& e) {
        std::fprintf(stderr, "hip_rmsnorm_probe: threw: %s\n", e.what());
        return 2;
    }
}