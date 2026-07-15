// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// hip_rmsnorm_no_weight_probe — parity check for the weight-less RMSNorm
// variant used by Gemma 4's V-norm path (V passes through a bare
// ggml_rms_norm before entering the KV cache).
//
// Uses the combined-tolerance harness
// (|diff| <= abs_tol + rel_tol * |ref|). Same tolerance as vanilla
// rmsnorm — same reduction path, same numerical drift budget.

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

constexpr int          kM     = 8;
constexpr int          kK     = 4096;
constexpr float        kEps   = 1e-5f;
constexpr std::uint32_t kBlock = 128;   // == RMSNORM_NW_LOCAL_SIZE

std::string defaultHsacoPath(const char* argv0) {
    std::filesystem::path exe = std::filesystem::canonical(argv0);
    return (exe.parent_path() / "hsaco" / "rmsnorm_no_weight.hsaco").string();
}

void fillRandom(std::vector<float>& v, std::uint32_t seed) {
    std::uint32_t x = seed;
    for (auto& f : v) {
        x = x * 1664525u + 1013904223u;
        const std::int32_t s = static_cast<std::int32_t>(x >> 8) - (1 << 23);
        f = static_cast<float>(s) / static_cast<float>(1 << 23);
    }
}

void rmsnormNoWeightCpuRef(const std::vector<float>& x,
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
            yr[k] = static_cast<float>(static_cast<double>(xr[k]) * invRms);
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    const std::string hsacoPath =
        (argc > 1) ? std::string{argv[1]} : defaultHsacoPath(argv[0]);

    std::printf("hip_rmsnorm_no_weight_probe:\n  hsaco: %s\n  M=%d K=%d eps=%g block=%u\n",
                hsacoPath.c_str(), kM, kK, static_cast<double>(kEps), kBlock);

    try {
        HipContext         ctx{};
        HipMemoryAllocator alloc{ctx};
        HipStream          stream{ctx, HipStreamKind::BlockingDefault};
        HipEvent           evStart{ctx};
        HipEvent           evEnd  {ctx};

        HipModule mod    = HipModule::fromFile(ctx, hsacoPath);
        HipKernel kernel = mod.getKernel("rmsnorm_no_weight");

        // ---- host tensors -----------------------------------------------
        const std::size_t xElems = static_cast<std::size_t>(kM) * kK;
        std::vector<float> hostX(xElems);
        std::vector<float> hostYRef(xElems);
        std::vector<float> hostYGpu(xElems, 0.0f);

        fillRandom(hostX, /*seed=*/0xB16BEEFu);

        rmsnormNoWeightCpuRef(hostX, hostYRef, kM, kK, kEps);

        // ---- device tensors ---------------------------------------------
        const std::size_t xBytes = xElems * sizeof(float);
        HipBuffer devX{alloc, xBytes};
        HipBuffer devY{alloc, xBytes};

        alloc.copyH2D(devX.data(), hostX.data(), xBytes);

        // ---- launch -----------------------------------------------------
        kernel.setPtr  (0, devX.data());
        kernel.setPtr  (1, devY.data());
        kernel.setValue(2, kEps);
        kernel.setValue(3, kK);

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
        std::printf("\nhip_rmsnorm_no_weight_probe: %s\n", ok ? "OK" : "FAIL");
        return ok ? 0 : 1;

    } catch (const std::exception& e) {
        std::fprintf(stderr, "hip_rmsnorm_no_weight_probe: threw: %s\n", e.what());
        return 2;
    }
}