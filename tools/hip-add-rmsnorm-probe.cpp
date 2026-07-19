// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// hip_add_rmsnorm_probe — parity check for the fused
// residual-add + RMSNorm kernel. Verifies both outputs: the in-place
// updated `x` (post-add), and `y` = (x_updated * weight * invRms).
// Same combined-tolerance envelope as rmsnorm_probe since the RMS
// reduction + rsqrt is the dominant fp32-vs-fp64 drift source.

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

constexpr int          kM      = 8;
constexpr int          kK      = 4096;
constexpr float        kEps    = 1e-5f;
constexpr std::uint32_t kBlock = 128;   // == ADD_RMSNORM_LOCAL_SIZE

std::string defaultHsacoPath(const char* argv0) {
    std::filesystem::path exe = std::filesystem::canonical(argv0);
    return (exe.parent_path() / "hsaco" / "add_rmsnorm.hsaco").string();
}

void fillRandom(std::vector<float>& v, std::uint32_t seed, float scale) {
    std::uint32_t x = seed;
    for (auto& f : v) {
        x = x * 1664525u + 1013904223u;
        const std::int32_t s = static_cast<std::int32_t>(x >> 8) - (1 << 23);
        f = scale * static_cast<float>(s) / static_cast<float>(1 << 23);
    }
}

// Double-precision reference: computes both outputs the kernel is
// expected to write. Both `xUpdated` and `yOut` come back to the caller.
void addRmsNormCpuRef(const std::vector<float>& xIn,
                      const std::vector<float>& delta,
                      const std::vector<float>& weight,
                      std::vector<float>&       xUpdated,
                      std::vector<float>&       yOut,
                      int M, int K, float eps) {
    for (int m = 0; m < M; ++m) {
        const float* xr = xIn   .data() + static_cast<std::size_t>(m) * K;
        const float* dr = delta .data() + static_cast<std::size_t>(m) * K;
              float* xu = xUpdated.data() + static_cast<std::size_t>(m) * K;
              float* yr = yOut    .data() + static_cast<std::size_t>(m) * K;

        double sumsq = 0.0;
        for (int k = 0; k < K; ++k) {
            const double v = static_cast<double>(xr[k]) +
                             static_cast<double>(dr[k]);
            xu[k] = static_cast<float>(v);
            sumsq += v * v;
        }
        const double mean   = sumsq / static_cast<double>(K);
        const double invRms = 1.0 / std::sqrt(mean + static_cast<double>(eps));
        for (int k = 0; k < K; ++k) {
            yr[k] = static_cast<float>(
                        static_cast<double>(xu[k])
                      * static_cast<double>(weight[k])
                      * invRms);
        }
    }
}

struct CmpResult {
    float       maxAbs;
    float       maxRatio;
    std::size_t badIdx;
};

CmpResult compareCombined(const std::vector<float>& gpu,
                          const std::vector<float>& ref,
                          float absTol, float relTol) {
    CmpResult r{0.0f, 0.0f, SIZE_MAX};
    for (std::size_t i = 0; i < gpu.size(); ++i) {
        const float d         = std::fabs(gpu[i] - ref[i]);
        const float threshold = absTol + relTol * std::fabs(ref[i]);
        const float ratio     = d / threshold;
        if (ratio > r.maxRatio) { r.maxRatio = ratio; r.badIdx = i; }
        if (d > r.maxAbs) r.maxAbs = d;
    }
    return r;
}

} // namespace

int main(int argc, char** argv) {
    const std::string hsacoPath =
        (argc > 1) ? std::string{argv[1]} : defaultHsacoPath(argv[0]);

    std::printf("hip_add_rmsnorm_probe:\n  hsaco: %s\n"
                "  M=%d K=%d eps=%g block=%u\n",
                hsacoPath.c_str(), kM, kK,
                static_cast<double>(kEps), kBlock);

    try {
        HipContext         ctx{};
        HipMemoryAllocator alloc{ctx};
        HipStream          stream{ctx, HipStreamKind::BlockingDefault};
        HipEvent           evStart{ctx};
        HipEvent           evEnd  {ctx};

        HipModule mod    = HipModule::fromFile(ctx, hsacoPath);
        HipKernel kernel = mod.getKernel("add_rmsnorm");

        const std::size_t xElems = static_cast<std::size_t>(kM) * kK;
        std::vector<float> hostXIn (xElems);
        std::vector<float> hostD   (xElems);
        std::vector<float> hostW   (static_cast<std::size_t>(kK));

        fillRandom(hostXIn, /*seed=*/0x1234ABCDu, /*scale=*/1.0f);
        fillRandom(hostD,   /*seed=*/0x5678EF01u, /*scale=*/1.0f);
        fillRandom(hostW,   /*seed=*/0x9ABC2345u, /*scale=*/1.0f);

        std::vector<float> hostXRef(xElems);
        std::vector<float> hostYRef(xElems);
        addRmsNormCpuRef(hostXIn, hostD, hostW,
                         hostXRef, hostYRef, kM, kK, kEps);

        std::vector<float> hostXGpu(xElems, 0.0f);
        std::vector<float> hostYGpu(xElems, 0.0f);

        const std::size_t xBytes = xElems * sizeof(float);
        const std::size_t wBytes = static_cast<std::size_t>(kK) * sizeof(float);
        HipBuffer devX{alloc, xBytes};
        HipBuffer devD{alloc, xBytes};
        HipBuffer devW{alloc, wBytes};
        HipBuffer devY{alloc, xBytes};

        alloc.copyH2D(devX.data(), hostXIn.data(), xBytes);
        alloc.copyH2D(devD.data(), hostD  .data(), xBytes);
        alloc.copyH2D(devW.data(), hostW  .data(), wBytes);

        kernel.setPtr  (0, devX.data());
        kernel.setPtr  (1, devD.data());
        kernel.setPtr  (2, devW.data());
        kernel.setPtr  (3, devY.data());
        kernel.setValue(4, kEps);
        kernel.setValue(5, kK);

        evStart.record(stream);
        kernel.launch(stream, static_cast<std::uint32_t>(kM), 1, 1,
                      kBlock, 1, 1, /*shared=*/0);
        evEnd.record(stream);
        stream.synchronize();

        const float kernelMs = evEnd.elapsedMs(evStart);

        alloc.copyD2H(hostXGpu.data(), devX.data(), xBytes);
        alloc.copyD2H(hostYGpu.data(), devY.data(), xBytes);

        // The in-place x add is a plain sum → tight tolerance suffices.
        constexpr float kXAbsTol = 1e-6f;
        constexpr float kXRelTol = 1e-6f;
        // The y output rides through the RMS reduction + rsqrt →
        // looser combined-tolerance like the base rmsnorm probe.
        constexpr float kYAbsTol = 1e-4f;
        constexpr float kYRelTol = 1e-4f;

        const auto xCmp = compareCombined(hostXGpu, hostXRef,
                                          kXAbsTol, kXRelTol);
        const auto yCmp = compareCombined(hostYGpu, hostYRef,
                                          kYAbsTol, kYRelTol);

        std::printf("\n  kernel:              %.3f ms\n",
                    static_cast<double>(kernelMs));
        std::printf("  x max abs err:       %.3e\n",
                    static_cast<double>(xCmp.maxAbs));
        std::printf("  x max err / tol:     %.3f (fails if > 1.0)\n",
                    static_cast<double>(xCmp.maxRatio));
        std::printf("  y max abs err:       %.3e\n",
                    static_cast<double>(yCmp.maxAbs));
        std::printf("  y max err / tol:     %.3f (fails if > 1.0)\n",
                    static_cast<double>(yCmp.maxRatio));

        const bool xOk = (xCmp.maxRatio <= 1.0f);
        const bool yOk = (yCmp.maxRatio <= 1.0f);
        const bool ok  = xOk && yOk;
        std::printf("\nhip_add_rmsnorm_probe: %s%s%s\n",
                    ok ? "OK" : "FAIL",
                    xOk ? "" : " [x fail]",
                    yOk ? "" : " [y fail]");
        return ok ? 0 : 1;

    } catch (const std::exception& e) {
        std::fprintf(stderr, "hip_add_rmsnorm_probe: threw: %s\n", e.what());
        return 2;
    }
}