// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// hip_qkv_split_fp16_probe — parity check for the fp16-KV variant of
// the QKV scatter. Q workspace stays fp32; K and V land in the fp16
// KV cache. Same routing as qkv_split, so the same offset math is
// exercised — the only new bit is the fp16 store.
//
// Bit-exact expectations:
//   • Q: unchanged from qkv_split — pure fp32 copy.
//   • K/V: bit-exact against a CPU-computed __float2half of the
//     corresponding fused slice. Both GPU and CPU use round-to-
//     nearest-even, so any diff is a routing/rounding-mode bug.
//   • Pre-curLen K/V slots: bit-identical to the fp16 sentinel.

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

constexpr int          kM         = 4;
constexpr int          kHeadDim   = 128;
constexpr int          kNumHeads  = 8;
constexpr int          kNumKvHeads = 2;
constexpr int          kNq        = kNumHeads   * kHeadDim;    // 1024
constexpr int          kNkv       = kNumKvHeads * kHeadDim;    // 256
constexpr int          kHasV      = 1;
constexpr int          kNfused    = kNq + kNkv * (1 + kHasV);  // 1536
constexpr int          kCurLen    = 8;
constexpr int          kCacheSlots = 32;
constexpr std::uint32_t kBlock    = 256;   // == QKV_SPLIT_FP16_LOCAL

std::string defaultHsacoPath(const char* argv0) {
    std::filesystem::path exe = std::filesystem::canonical(argv0);
    return (exe.parent_path() / "hsaco" / "qkv_split_fp16.hsaco").string();
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

// Bit-compare two half arrays using their underlying 16-bit storage.
bool halfBitsEqual(__half a, __half b) {
    std::uint16_t ba;
    std::uint16_t bb;
    std::memcpy(&ba, &a, sizeof(ba));
    std::memcpy(&bb, &b, sizeof(bb));
    return ba == bb;
}

} // namespace

int main(int argc, char** argv) {
    const std::string hsacoPath =
        (argc > 1) ? std::string{argv[1]} : defaultHsacoPath(argv[0]);

    std::printf("hip_qkv_split_fp16_probe:\n  hsaco: %s\n"
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
        HipKernel kernel = mod.getKernel("qkv_split_fp16");

        // ---- host tensors -----------------------------------------------
        const std::size_t fusedElems = static_cast<std::size_t>(kM)
                                     * static_cast<std::size_t>(kNfused);
        const std::size_t qElems     = static_cast<std::size_t>(kM)
                                     * static_cast<std::size_t>(kNq);
        const std::size_t kvElems    = static_cast<std::size_t>(kCacheSlots)
                                     * static_cast<std::size_t>(kNkv);

        std::vector<float>  hostFused (fusedElems);
        std::vector<float>  hostYqSeed(qElems);
        std::vector<__half> hostYkSeed(kvElems);
        std::vector<__half> hostYvSeed(kvElems);
        fillRandomFloat(hostFused,  /*seed=*/0xC0FFEEEEu, /*scale=*/1.0f);
        fillRandomFloat(hostYqSeed, /*seed=*/0xDEADFA11u, /*scale=*/1.0f);
        fillRandomHalf (hostYkSeed, /*seed=*/0xF00DFA11u, /*scale=*/1.0f);
        fillRandomHalf (hostYvSeed, /*seed=*/0xB00BFA11u, /*scale=*/1.0f);

        // ---- CPU reference ----------------------------------------------
        std::vector<float>  hostYqRef = hostYqSeed;
        std::vector<__half> hostYkRef = hostYkSeed;
        std::vector<__half> hostYvRef = hostYvSeed;
        for (int m = 0; m < kM; ++m) {
            const int rowBase = m * kNfused;
            for (int i = 0; i < kNq; ++i) {
                hostYqRef[static_cast<std::size_t>(m) * kNq + i] =
                    hostFused[static_cast<std::size_t>(rowBase + i)];
            }
            const std::size_t cacheRow = static_cast<std::size_t>(kCurLen + m)
                                       * static_cast<std::size_t>(kNkv);
            for (int i = 0; i < kNkv; ++i) {
                hostYkRef[cacheRow + i] = __float2half(
                    hostFused[static_cast<std::size_t>(rowBase + kNq + i)]);
            }
            if (kHasV) {
                for (int i = 0; i < kNkv; ++i) {
                    hostYvRef[cacheRow + i] = __float2half(
                        hostFused[static_cast<std::size_t>(rowBase + kNq + kNkv + i)]);
                }
            }
        }

        // ---- device tensors ---------------------------------------------
        const std::size_t fusedBytes = fusedElems * sizeof(float);
        const std::size_t qBytes     = qElems     * sizeof(float);
        const std::size_t kvBytes    = kvElems    * sizeof(__half);

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
        std::vector<float>  gpuYq(qElems);
        std::vector<__half> gpuYk(kvElems);
        std::vector<__half> gpuYv(kvElems);
        alloc.copyD2H(gpuYq.data(), devYq.data(), qBytes);
        alloc.copyD2H(gpuYk.data(), devYk.data(), kvBytes);
        alloc.copyD2H(gpuYv.data(), devYv.data(), kvBytes);

        std::printf("\n  kernel:        %.3f ms\n", static_cast<double>(kernelMs));
        std::printf("  tol:           bit-exact (fp32 copy for Q, "
                    "fp16 round-trip for K/V)\n");

        // Q: fp32 bit-exact
        std::size_t qMiss = 0;
        std::size_t qFirst = SIZE_MAX;
        for (std::size_t i = 0; i < qElems; ++i) {
            if (gpuYq[i] != hostYqRef[i]) {
                if (qMiss == 0) qFirst = i;
                ++qMiss;
            }
        }
        std::printf("  Q   mismatches: %zu / %zu", qMiss, qElems);
        if (qFirst != SIZE_MAX) {
            std::printf("   first @ %zu: gpu=%.6g cpu=%.6g",
                        qFirst,
                        static_cast<double>(gpuYq[qFirst]),
                        static_cast<double>(hostYqRef[qFirst]));
        }
        std::printf("\n");

        // K/V: fp16 bit-exact
        auto checkHalfExact = [&](const char* label,
                                  const std::vector<__half>& gpu,
                                  const std::vector<__half>& ref) {
            std::size_t miss = 0;
            std::size_t first = SIZE_MAX;
            for (std::size_t i = 0; i < gpu.size(); ++i) {
                if (!halfBitsEqual(gpu[i], ref[i])) {
                    if (miss == 0) first = i;
                    ++miss;
                }
            }
            std::printf("  %-3s mismatches: %zu / %zu",
                        label, miss, gpu.size());
            if (first != SIZE_MAX) {
                std::printf("   first @ %zu: gpu=%.6g cpu=%.6g",
                            first,
                            static_cast<double>(__half2float(gpu[first])),
                            static_cast<double>(__half2float(ref[first])));
            }
            std::printf("\n");
            return miss == 0;
        };

        const bool okQ = (qMiss == 0);
        const bool okK = checkHalfExact("K", gpuYk, hostYkRef);
        const bool okV = checkHalfExact("V", gpuYv, hostYvRef);

        // Cache-stomp check: fp16 bit-identical to seed in pre-curLen
        // region. checkHalfExact above catches these mixed with the
        // parity check; this call-out reports the property explicitly.
        const std::size_t kvBase = static_cast<std::size_t>(kCurLen) * kNkv;
        bool untouched = true;
        for (std::size_t i = 0; i < kvBase; ++i) {
            if (!halfBitsEqual(gpuYk[i], hostYkSeed[i]) ||
                !halfBitsEqual(gpuYv[i], hostYvSeed[i])) {
                untouched = false;
                break;
            }
        }
        std::printf("  pre-curLen cache slots untouched: %s\n",
                    untouched ? "yes" : "NO");

        const bool ok = okQ && okK && okV && untouched;
        std::printf("\nhip_qkv_split_fp16_probe: %s\n", ok ? "OK" : "FAIL");
        return ok ? 0 : 1;

    } catch (const std::exception& e) {
        std::fprintf(stderr, "hip_qkv_split_fp16_probe: threw: %s\n", e.what());
        return 2;
    }
}