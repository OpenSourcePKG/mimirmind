// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// hip_moe_down_fused_k_q8_0_probe — parity check for the fused MoE
// down-projection over k active experts.
//
// The accum buffer is read-modify-write: the probe pre-seeds it with
// random content and verifies that the GPU output equals seed +
// computed_delta (not the delta alone). Any bug that overwrites accum
// instead of adding to it would show up as a mismatch on the seed
// component.

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

constexpr int          kK              = 4;      // kActive experts per token
constexpr int          kNExperts       = 8;      // total experts in bank
constexpr int          kFfPer          = 1024;   // FFN inner dim
constexpr int          kDModel         = 128;    // model dim (matmul output)
constexpr int          kBlockElems     = 32;
constexpr int          kBlockBytes     = 34;
constexpr int          kNBlocksPerRow  = kFfPer / kBlockElems;    // 32
constexpr int          kRowBytes       = kNBlocksPerRow * kBlockBytes;  // 1088
constexpr int          kExpertBytes    = kDModel * kRowBytes;          // ~139 KB
constexpr int          kOutputsPerWG   = 4;      // == MOE_DOWN_LOCAL / SG
constexpr std::uint32_t kBlock         = 64;     // MOE_DOWN_LOCAL

std::string defaultHsacoPath(const char* argv0) {
    std::filesystem::path exe = std::filesystem::canonical(argv0);
    return (exe.parent_path() / "hsaco" / "moe_down_fused_k_q8_0.hsaco").string();
}

void fillRandomFloat(std::vector<float>& v, std::uint32_t seed, float scale) {
    std::uint32_t x = seed;
    for (auto& f : v) {
        x = x * 1664525u + 1013904223u;
        const std::int32_t s = static_cast<std::int32_t>(x >> 8) - (1 << 23);
        f = scale * static_cast<float>(s) / static_cast<float>(1 << 23);
    }
}

// Native Q8_0 for a stack of `nExperts` weight matrices, each with
// dModel rows of nBlocksPerRow blocks.
void fillRandomQ8_0Bank(std::vector<std::uint8_t>& bytes,
                        std::uint32_t seed, int nExperts) {
    std::uint32_t x = seed;
    for (int e = 0; e < nExperts; ++e) {
        std::uint8_t* expBase = bytes.data()
                              + static_cast<std::size_t>(e) * kExpertBytes;
        for (int r = 0; r < kDModel; ++r) {
            std::uint8_t* row = expBase + static_cast<std::size_t>(r) * kRowBytes;
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
}

inline float dequantElemNative(const std::uint8_t* rowBase, int col) {
    const int b  = col / kBlockElems;
    const int in = col % kBlockElems;
    const std::uint8_t* blk = rowBase + static_cast<std::size_t>(b) * kBlockBytes;
    __half hScale;
    std::memcpy(&hScale, blk, sizeof(__half));
    const float d = __half2float(hScale);
    const signed char qi = reinterpret_cast<const signed char*>(blk)[2 + in];
    return d * static_cast<float>(qi);
}

// CPU reference: accum[n] = accumSeed[n] + sum_k kw[k] * sum_l
//                              dequant(W[expIdx[k]], n, l) * X[k, l]
void moeDownFusedCpuRef(const std::vector<float>&        X,
                        const std::vector<std::uint8_t>& W,
                        const std::vector<int>&          expIdx,
                        const std::vector<float>&        kw,
                        const std::vector<float>&        accumSeed,
                        std::vector<float>&              accumRef,
                        int K, int dModel, int ffPer)
{
    for (int n = 0; n < dModel; ++n) {
        double outerAcc = 0.0;
        for (int k = 0; k < K; ++k) {
            const int e = expIdx[k];
            const std::uint8_t* row = W.data()
                + static_cast<std::size_t>(e) * kExpertBytes
                + static_cast<std::size_t>(n) * kRowBytes;
            const float* Xk = X.data()
                + static_cast<std::size_t>(k) * ffPer;

            double innerAcc = 0.0;
            for (int l = 0; l < ffPer; ++l) {
                innerAcc += static_cast<double>(Xk[l])
                          * static_cast<double>(dequantElemNative(row, l));
            }
            outerAcc += static_cast<double>(kw[k]) * innerAcc;
        }
        accumRef[n] = accumSeed[n] + static_cast<float>(outerAcc);
    }
}

} // namespace

int main(int argc, char** argv) {
    const std::string hsacoPath =
        (argc > 1) ? std::string{argv[1]} : defaultHsacoPath(argv[0]);

    std::printf("hip_moe_down_fused_k_q8_0_probe:\n  hsaco: %s\n"
                "  K=%d nExperts=%d ffPer=%d dModel=%d block=%u outputs_per_wg=%d\n"
                "  expertBytes=%d\n",
                hsacoPath.c_str(),
                kK, kNExperts, kFfPer, kDModel, kBlock, kOutputsPerWG,
                kExpertBytes);

    try {
        HipContext         ctx{};
        HipMemoryAllocator alloc{ctx};
        HipStream          stream{ctx, HipStreamKind::BlockingDefault};
        HipEvent           evStart{ctx};
        HipEvent           evEnd  {ctx};

        HipModule mod    = HipModule::fromFile(ctx, hsacoPath);
        HipKernel kernel = mod.getKernel("moe_down_fused_k_q8_0");

        const std::size_t xElems     = static_cast<std::size_t>(kK) * kFfPer;
        const std::size_t accumElems = kDModel;
        const std::size_t wBytes     = static_cast<std::size_t>(kNExperts)
                                     * kExpertBytes;

        std::vector<float>        hostX(xElems);
        std::vector<std::uint8_t> hostW(wBytes);
        std::vector<int>          hostExpIdx(kK);
        std::vector<float>        hostKw(kK);
        std::vector<float>        hostAccumSeed(accumElems);

        fillRandomFloat   (hostX,          /*seed=*/0x30E0001u, /*scale=*/0.5f);
        fillRandomQ8_0Bank(hostW,          /*seed=*/0x30E0002u, kNExperts);
        fillRandomFloat   (hostKw,         /*seed=*/0x30E0003u, /*scale=*/1.0f);
        fillRandomFloat   (hostAccumSeed,  /*seed=*/0x30E0004u, /*scale=*/1.0f);

        // Random expert indices in [0, nExperts).
        {
            std::uint32_t x = 0x30E0005u;
            for (int i = 0; i < kK; ++i) {
                x = x * 1664525u + 1013904223u;
                hostExpIdx[i] = static_cast<int>((x >> 16) % kNExperts);
            }
        }

        std::vector<float> hostAccumRef(accumElems);
        moeDownFusedCpuRef(hostX, hostW, hostExpIdx, hostKw,
                           hostAccumSeed, hostAccumRef,
                           kK, kDModel, kFfPer);

        const std::size_t xBytes      = xElems * sizeof(float);
        const std::size_t accumBytes  = accumElems * sizeof(float);
        const std::size_t idxBytes    = kK * sizeof(int);
        const std::size_t kwBytes     = kK * sizeof(float);

        HipBuffer devX     {alloc, xBytes};
        HipBuffer devW     {alloc, wBytes};
        HipBuffer devExpIdx{alloc, idxBytes};
        HipBuffer devKw    {alloc, kwBytes};
        HipBuffer devAccum {alloc, accumBytes};

        alloc.copyH2D(devX     .data(), hostX.data(),         xBytes);
        alloc.copyH2D(devW     .data(), hostW.data(),         wBytes);
        alloc.copyH2D(devExpIdx.data(), hostExpIdx.data(),    idxBytes);
        alloc.copyH2D(devKw    .data(), hostKw.data(),        kwBytes);
        alloc.copyH2D(devAccum .data(), hostAccumSeed.data(), accumBytes);

        kernel.setPtr  (0, devX     .data());
        kernel.setPtr  (1, devW     .data());
        kernel.setPtr  (2, devExpIdx.data());
        kernel.setPtr  (3, devKw    .data());
        kernel.setPtr  (4, devAccum .data());
        kernel.setValue(5, kFfPer);
        kernel.setValue(6, kDModel);
        kernel.setValue(7, kK);
        kernel.setValue(8, kExpertBytes);

        const std::uint32_t gridX =
            (static_cast<std::uint32_t>(kDModel) + kOutputsPerWG - 1) / kOutputsPerWG;

        evStart.record(stream);
        kernel.launch(stream, gridX, 1, 1, kBlock, 1, 1, /*shared=*/0);
        evEnd.record(stream);
        stream.synchronize();
        const float kernelMs = evEnd.elapsedMs(evStart);

        std::vector<float> hostAccumGpu(accumElems);
        alloc.copyD2H(hostAccumGpu.data(), devAccum.data(), accumBytes);

        constexpr float kAbsTol = 1e-3f;
        constexpr float kRelTol = 1e-3f;

        float       maxAbs   = 0.0f;
        float       maxRatio = 0.0f;
        std::size_t badIdx   = SIZE_MAX;
        for (std::size_t i = 0; i < accumElems; ++i) {
            const float d         = std::fabs(hostAccumGpu[i] - hostAccumRef[i]);
            const float threshold = kAbsTol + kRelTol * std::fabs(hostAccumRef[i]);
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
            std::printf("  worst @ n=%zu: gpu=%.6g cpu=%.6g (seed was %.6g)\n",
                        badIdx,
                        static_cast<double>(hostAccumGpu[badIdx]),
                        static_cast<double>(hostAccumRef[badIdx]),
                        static_cast<double>(hostAccumSeed[badIdx]));
        }

        // Explicit RMW check: if the kernel accidentally OVERWROTE
        // accum instead of adding, the diff to (ref - seed) would be
        // large (`ref - seed` is the pure delta, `gpu` should be the
        // seed + delta). This is subsumed by the parity check above
        // but easier to eyeball in the output.
        std::printf("  RMW semantics: seed(mean)=%.4f delta(mean)=%.4f\n",
                    hostAccumSeed[0], hostAccumRef[0] - hostAccumSeed[0]);

        const bool ok = (maxRatio <= 1.0f);
        std::printf("\nhip_moe_down_fused_k_q8_0_probe: %s\n", ok ? "OK" : "FAIL");
        return ok ? 0 : 1;

    } catch (const std::exception& e) {
        std::fprintf(stderr, "hip_moe_down_fused_k_q8_0_probe: threw: %s\n", e.what());
        return 2;
    }
}