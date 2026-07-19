// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// hip_qkv_split_probe — parity check for the fused-QKV output scatter.
// Runs the kernel with hasV=1 (the common case) on a random fused
// tensor, scatters into Q workspace + K cache + V cache, verifies:
//
//   • Q workspace: bit-exact copy of fused[m, 0..Nq-1]
//   • K cache:     bit-exact copy of fused[m, Nq..Nq+Nkv-1] at
//                  (curLen+m)*Nkv offset
//   • V cache:     bit-exact copy of fused[m, Nq+Nkv..Nq+2*Nkv-1] at
//                  (curLen+m)*Nkv offset
//   • Pre-curLen K/V cache slots untouched (sentinel check — the same
//     cache-stomp guard rmsnorm_qkv established as a standard).
//
// Scatter is a pure copy — no arithmetic, so the tolerance is 0.
// Any mismatch is a routing/offset bug, never numerical drift.

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

// Test shape — Gemma-ish: nHeads=8 / nKvHeads=2 / head_dim=128.
constexpr int          kM         = 4;                        // tokens
constexpr int          kHeadDim   = 128;
constexpr int          kNumHeads  = 8;
constexpr int          kNumKvHeads = 2;
constexpr int          kNq        = kNumHeads   * kHeadDim;   // 1024
constexpr int          kNkv       = kNumKvHeads * kHeadDim;   // 256
constexpr int          kHasV      = 1;
constexpr int          kNfused    = kNq + kNkv * (1 + kHasV); // 1536
constexpr int          kCurLen    = 8;   // non-zero → exercise offset path
constexpr int          kCacheSlots = 32; // > curLen + M so we can check untouched region
constexpr std::uint32_t kBlock    = 256; // == QKV_SPLIT_LOCAL

std::string defaultHsacoPath(const char* argv0) {
    std::filesystem::path exe = std::filesystem::canonical(argv0);
    return (exe.parent_path() / "hsaco" / "qkv_split.hsaco").string();
}

void fillRandom(std::vector<float>& v, std::uint32_t seed, float scale) {
    std::uint32_t x = seed;
    for (auto& f : v) {
        x = x * 1664525u + 1013904223u;
        const std::int32_t s = static_cast<std::int32_t>(x >> 8) - (1 << 23);
        f = scale * static_cast<float>(s) / static_cast<float>(1 << 23);
    }
}

} // namespace

int main(int argc, char** argv) {
    const std::string hsacoPath =
        (argc > 1) ? std::string{argv[1]} : defaultHsacoPath(argv[0]);

    std::printf("hip_qkv_split_probe:\n  hsaco: %s\n"
                "  M=%d Nq=%d Nkv=%d hasV=%d Nfused=%d curLen=%d block=%u\n",
                hsacoPath.c_str(),
                kM, kNq, kNkv, kHasV, kNfused, kCurLen, kBlock);

    try {
        HipContext         ctx{};
        HipMemoryAllocator alloc{ctx};
        HipStream          stream{ctx, HipStreamKind::BlockingDefault};
        HipEvent           evStart{ctx};
        HipEvent           evEnd  {ctx};

        HipModule mod    = HipModule::fromFile(ctx, hsacoPath);
        HipKernel kernel = mod.getKernel("qkv_split");

        // ---- host tensors -----------------------------------------------
        const std::size_t fusedElems = static_cast<std::size_t>(kM)
                                     * static_cast<std::size_t>(kNfused);
        const std::size_t qElems     = static_cast<std::size_t>(kM)
                                     * static_cast<std::size_t>(kNq);
        const std::size_t kvElems    = static_cast<std::size_t>(kCacheSlots)
                                     * static_cast<std::size_t>(kNkv);

        std::vector<float> hostFused(fusedElems);
        fillRandom(hostFused, /*seed=*/0xC0FFEEEEu, /*scale=*/1.0f);

        // Pre-seed Yq / Yk / Yv with distinct sentinels — makes it
        // easy to spot any thread that wrote to the wrong slot.
        std::vector<float> hostYqSeed(qElems);
        std::vector<float> hostYkSeed(kvElems);
        std::vector<float> hostYvSeed(kvElems);
        fillRandom(hostYqSeed, /*seed=*/0xDEADFA11u, /*scale=*/1.0f);
        fillRandom(hostYkSeed, /*seed=*/0xF00DFA11u, /*scale=*/1.0f);
        fillRandom(hostYvSeed, /*seed=*/0xB00BFA11u, /*scale=*/1.0f);

        // ---- CPU reference ----------------------------------------------
        // Start refs from the sentinels — only slots the kernel writes
        // should end up differing.
        std::vector<float> hostYqRef = hostYqSeed;
        std::vector<float> hostYkRef = hostYkSeed;
        std::vector<float> hostYvRef = hostYvSeed;
        for (int m = 0; m < kM; ++m) {
            const int rowBase = m * kNfused;
            // Q slice
            for (int i = 0; i < kNq; ++i) {
                hostYqRef[static_cast<std::size_t>(m) * kNq + i] =
                    hostFused[static_cast<std::size_t>(rowBase + i)];
            }
            // K slice → K cache at (curLen + m) * Nkv
            const std::size_t kRow = static_cast<std::size_t>(kCurLen + m)
                                   * static_cast<std::size_t>(kNkv);
            for (int i = 0; i < kNkv; ++i) {
                hostYkRef[kRow + i] =
                    hostFused[static_cast<std::size_t>(rowBase + kNq + i)];
            }
            // V slice → V cache at (curLen + m) * Nkv
            if (kHasV) {
                const std::size_t vRow = kRow;   // same offset formula
                for (int i = 0; i < kNkv; ++i) {
                    hostYvRef[vRow + i] =
                        hostFused[static_cast<std::size_t>(rowBase + kNq + kNkv + i)];
                }
            }
        }

        // ---- device tensors ---------------------------------------------
        const std::size_t fusedBytes = fusedElems * sizeof(float);
        const std::size_t qBytes     = qElems     * sizeof(float);
        const std::size_t kvBytes    = kvElems    * sizeof(float);

        HipBuffer devFused {alloc, fusedBytes};
        HipBuffer devYq    {alloc, qBytes};
        HipBuffer devYk    {alloc, kvBytes};
        HipBuffer devYv    {alloc, kvBytes};
        HipBuffer devCurLen{alloc, sizeof(int), HipAllocKind::Device};

        alloc.copyH2D(devFused .data(), hostFused .data(), fusedBytes);
        alloc.copyH2D(devYq    .data(), hostYqSeed.data(), qBytes);
        alloc.copyH2D(devYk    .data(), hostYkSeed.data(), kvBytes);
        alloc.copyH2D(devYv    .data(), hostYvSeed.data(), kvBytes);
        alloc.copyH2D(devCurLen.data(), &kCurLen,          sizeof(int));

        // ---- launch -----------------------------------------------------
        kernel.setPtr  (0, devFused .data());
        kernel.setPtr  (1, devYq    .data());
        kernel.setPtr  (2, devYk    .data());
        kernel.setPtr  (3, devYv    .data());
        kernel.setValue(4, kM);
        kernel.setValue(5, kNq);
        kernel.setValue(6, kNkv);
        kernel.setValue(7, kHasV);
        kernel.setValue(8, kNfused);
        kernel.setPtr  (9, devCurLen.data());

        const std::uint32_t totalWork =
            static_cast<std::uint32_t>(kM) * static_cast<std::uint32_t>(kNfused);
        const std::uint32_t grid = (totalWork + kBlock - 1) / kBlock;

        evStart.record(stream);
        kernel.launch(stream, grid, 1, 1, kBlock, 1, 1, /*shared=*/0);
        evEnd.record(stream);
        stream.synchronize();

        const float kernelMs = evEnd.elapsedMs(evStart);

        // ---- readback + bit-exact compare -------------------------------
        std::vector<float> gpuYq(qElems);
        std::vector<float> gpuYk(kvElems);
        std::vector<float> gpuYv(kvElems);
        alloc.copyD2H(gpuYq.data(), devYq.data(), qBytes);
        alloc.copyD2H(gpuYk.data(), devYk.data(), kvBytes);
        alloc.copyD2H(gpuYv.data(), devYv.data(), kvBytes);

        auto checkExact = [&](const char* label,
                              const std::vector<float>& gpu,
                              const std::vector<float>& ref) {
            std::size_t mismatches = 0;
            std::size_t firstBad   = SIZE_MAX;
            for (std::size_t i = 0; i < gpu.size(); ++i) {
                if (gpu[i] != ref[i]) {
                    if (mismatches == 0) firstBad = i;
                    ++mismatches;
                }
            }
            std::printf("  %-3s mismatches: %zu / %zu",
                        label, mismatches, gpu.size());
            if (firstBad != SIZE_MAX) {
                std::printf("   first @ %zu: gpu=%.6g cpu=%.6g",
                            firstBad,
                            static_cast<double>(gpu[firstBad]),
                            static_cast<double>(ref[firstBad]));
            }
            std::printf("\n");
            return mismatches == 0;
        };

        std::printf("\n  kernel:        %.3f ms\n", static_cast<double>(kernelMs));
        std::printf("  tol:           bit-exact (scatter is a pure copy)\n");

        const bool okQ = checkExact("Q", gpuYq, hostYqRef);
        const bool okK = checkExact("K", gpuYk, hostYkRef);
        const bool okV = checkExact("V", gpuYv, hostYvRef);

        // Cache-stomp check: slots strictly before curLen must be
        // bit-identical to the seed. `hostYkRef` / `hostYvRef` already
        // preserve the seed in that region — checkExact above catches
        // stomps too. This explicit report just calls out the property
        // for readability.
        const std::size_t kvBase = static_cast<std::size_t>(kCurLen) * kNkv;
        bool untouched = true;
        for (std::size_t i = 0; i < kvBase; ++i) {
            if (gpuYk[i] != hostYkSeed[i] || gpuYv[i] != hostYvSeed[i]) {
                untouched = false;
                break;
            }
        }
        std::printf("  pre-curLen cache slots untouched: %s\n",
                    untouched ? "yes" : "NO");

        const bool ok = okQ && okK && okV && untouched;
        std::printf("\nhip_qkv_split_probe: %s\n", ok ? "OK" : "FAIL");
        return ok ? 0 : 1;

    } catch (const std::exception& e) {
        std::fprintf(stderr, "hip_qkv_split_probe: threw: %s\n", e.what());
        return 2;
    }
}