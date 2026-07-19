// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// hip_attention_flash_partial_fp16_probe — parity check for the
// fp16-KV variant of the decode-mode FlashAttention partial-tile
// kernel.
//
// K and V are __half; the kernel promotes each load to fp32 in
// registers before the multiply, so the (m, l, o) partials written
// to the fp32 scratch match the algorithm of the f32 variant with
// K/V rounded through fp16 storage.
//
// Two cases share the same random q / K / V inputs:
//   A: slidingWindow=0    — both tiles valid.
//   B: slidingWindow=32   — tile 0 hits the -INFINITY neutral path
//                           for all 8 heads.
//
// Combined-tolerance 1e-4/1e-4. The CPU reference promotes fp16 K/V
// to fp32 via __half2float and does the same math the kernel does —
// so ULP-level fp32 reduction-order drift is the only expected diff.

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
#include <exception>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

namespace {

using namespace mimirmind::core::hip;

constexpr int          kNumHeads   = 8;
constexpr int          kNumKvHeads = 2;
constexpr int          kHeadDim    = 128;
constexpr int          kCurLen     = 100;
constexpr int          kKMax       = kCurLen + 1;
constexpr int          kKTile      = 64;
constexpr int          kNKTiles    = (kKMax + kKTile - 1) / kKTile;  // 2
constexpr int          kCacheSlots = 128;
constexpr int          kKvStride   = kNumKvHeads * kHeadDim;         // 256
constexpr int          kSlotWidth  = 2 + kHeadDim;                   // 130
constexpr std::uint32_t kBlock     = 16;                             // ATTN_FLASH_LOCAL

const float kScale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));

std::string defaultHsacoPath(const char* argv0) {
    std::filesystem::path exe = std::filesystem::canonical(argv0);
    return (exe.parent_path() / "hsaco" / "attention_flash_partial_fp16.hsaco").string();
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

// CPU reference — reads fp16 K/V, promotes each element to fp32 via
// __half2float exactly the way the kernel does.
void attentionFlashPartialFp16CpuRef(
    const std::vector<float>&  q,
    const std::vector<__half>& k,
    const std::vector<__half>& v,
    std::vector<float>&        partialMlo,
    int nHeads, int nKvHeads, int headDim,
    int curLen, float scale, int slidingWindow)
{
    const int kvStride = nKvHeads * headDim;
    const int kMax     = curLen + 1;
    const int kMin     = (slidingWindow > 0 && kMax > slidingWindow)
                           ? (kMax - slidingWindow) : 0;
    const int nKTiles  = (kMax + kKTile - 1) / kKTile;

    for (int hq = 0; hq < nHeads; ++hq) {
        const int hkv = (hq * nKvHeads) / nHeads;
        for (int kt = 0; kt < nKTiles; ++kt) {
            const int kStartRaw = kt * kKTile;
            const int kStart    = (kStartRaw > kMin) ? kStartRaw : kMin;
            const int kEndRaw   = kStartRaw + kKTile;
            const int kEnd      = (kEndRaw < kMax) ? kEndRaw : kMax;

            float* mloPtr = partialMlo.data()
                          + static_cast<std::size_t>(hq * nKTiles + kt)
                          * static_cast<std::size_t>(2 + headDim);

            if (kStart >= kEnd) {
                mloPtr[0] = -std::numeric_limits<float>::infinity();
                mloPtr[1] = 0.0f;
                for (int d = 0; d < headDim; ++d) mloPtr[2 + d] = 0.0f;
                continue;
            }

            const int tileLen = kEnd - kStart;
            const float* qVec = q.data()
                              + static_cast<std::size_t>(hq)
                              * static_cast<std::size_t>(headDim);

            std::vector<double> scores(tileLen);
            for (int kk = 0; kk < tileLen; ++kk) {
                const int absKk = kStart + kk;
                const __half* kVec = k.data()
                                   + static_cast<std::size_t>(absKk) * kvStride
                                   + static_cast<std::size_t>(hkv)   * headDim;
                double acc = 0.0;
                for (int d = 0; d < headDim; ++d) {
                    acc += static_cast<double>(qVec[d]) *
                           static_cast<double>(__half2float(kVec[d]));
                }
                scores[kk] = acc * static_cast<double>(scale);
            }

            double mLocal = -std::numeric_limits<double>::infinity();
            for (double s : scores) if (s > mLocal) mLocal = s;

            double lLocal = 0.0;
            for (double& s : scores) {
                s = std::exp(s - mLocal);
                lLocal += s;
            }

            mloPtr[0] = static_cast<float>(mLocal);
            mloPtr[1] = static_cast<float>(lLocal);

            for (int d = 0; d < headDim; ++d) {
                double acc = 0.0;
                for (int kk = 0; kk < tileLen; ++kk) {
                    const __half* vVec = v.data()
                                       + static_cast<std::size_t>(kStart + kk) * kvStride
                                       + static_cast<std::size_t>(hkv)         * headDim;
                    acc += scores[kk] *
                           static_cast<double>(__half2float(vVec[d]));
                }
                mloPtr[2 + d] = static_cast<float>(acc);
            }
        }
    }
}

struct RunResult { bool passed; float maxAbs; float maxRatio; };

RunResult runOnce(const char* label, int slidingWindow,
                  HipContext& ctx, HipMemoryAllocator& alloc,
                  HipStream& stream, HipKernel& kernel,
                  const std::vector<float>&  hostQ,
                  const std::vector<__half>& hostK,
                  const std::vector<__half>& hostV,
                  void* devQ, void* devK, void* devV, void* devCurLen)
{
    const std::size_t sloElems = static_cast<std::size_t>(kNumHeads)
                               * static_cast<std::size_t>(kNKTiles)
                               * static_cast<std::size_t>(kSlotWidth);
    const std::size_t sloBytes = sloElems * sizeof(float);

    HipBuffer devPartial{alloc, sloBytes};

    std::vector<float> sentinel(sloElems, 12345.0f);
    alloc.copyH2D(devPartial.data(), sentinel.data(), sloBytes);

    kernel.setPtr  (0, devQ);
    kernel.setPtr  (1, devK);
    kernel.setPtr  (2, devV);
    kernel.setPtr  (3, devPartial.data());
    kernel.setValue(4, kNumHeads);
    kernel.setValue(5, kNumKvHeads);
    kernel.setValue(6, kHeadDim);
    kernel.setPtr  (7, devCurLen);
    kernel.setValue(8, kScale);
    kernel.setValue(9, slidingWindow);

    HipEvent evStart{ctx};
    HipEvent evEnd  {ctx};
    evStart.record(stream);
    kernel.launch(stream,
                  /*grid=*/  static_cast<std::uint32_t>(kNumHeads),
                             static_cast<std::uint32_t>(kNKTiles), 1,
                  /*block=*/ kBlock, 1, 1,
                  /*shared=*/0);
    evEnd.record(stream);
    stream.synchronize();
    const float kernelMs = evEnd.elapsedMs(evStart);

    std::vector<float> ref(sloElems, 0.0f);
    attentionFlashPartialFp16CpuRef(hostQ, hostK, hostV, ref,
                                    kNumHeads, kNumKvHeads, kHeadDim,
                                    kCurLen, kScale, slidingWindow);

    std::vector<float> gpu(sloElems, 0.0f);
    alloc.copyD2H(gpu.data(), devPartial.data(), sloBytes);

    constexpr float kAbsTol = 1e-4f;
    constexpr float kRelTol = 1e-4f;

    float       maxAbs   = 0.0f;
    float       maxRatio = 0.0f;
    std::size_t badIdx   = SIZE_MAX;
    std::size_t neutralSlots = 0;
    std::size_t neutralOK    = 0;
    for (int hq = 0; hq < kNumHeads; ++hq) {
        for (int kt = 0; kt < kNKTiles; ++kt) {
            const std::size_t base =
                (static_cast<std::size_t>(hq) * kNKTiles + kt) * kSlotWidth;
            const bool neutralRef = std::isinf(ref[base + 0]) && ref[base + 0] < 0.0f;
            if (neutralRef) {
                ++neutralSlots;
                const bool mOk = std::isinf(gpu[base + 0]) && gpu[base + 0] < 0.0f;
                const bool lOk = gpu[base + 1] == 0.0f;
                bool oOk = true;
                for (int d = 0; d < kHeadDim; ++d) {
                    if (gpu[base + 2 + d] != 0.0f) { oOk = false; break; }
                }
                if (mOk && lOk && oOk) ++neutralOK;
                continue;
            }
            for (int j = 0; j < kSlotWidth; ++j) {
                const float g = gpu[base + j];
                const float r = ref[base + j];
                const float d = std::fabs(g - r);
                const float t = kAbsTol + kRelTol * std::fabs(r);
                const float ratio = d / t;
                if (ratio > maxRatio) { maxRatio = ratio; badIdx = base + j; }
                if (d > maxAbs) maxAbs = d;
            }
        }
    }

    const bool neutralOk = (neutralSlots == neutralOK);
    const bool tolOk     = (maxRatio <= 1.0f);
    const bool ok        = neutralOk && tolOk;

    std::printf("  [%s] slidingWindow=%d  kernel=%.3f ms\n"
                "    non-neutral slots: max abs %.3e   max err/tol %.3f\n"
                "    neutral slots: %zu / %zu match (-inf,0,0…)\n",
                label, slidingWindow, static_cast<double>(kernelMs),
                static_cast<double>(maxAbs), static_cast<double>(maxRatio),
                neutralOK, neutralSlots);
    if (badIdx != SIZE_MAX && maxRatio > 1e-3f) {
        std::printf("    worst @ %zu: gpu=%.6g cpu=%.6g\n",
                    badIdx,
                    static_cast<double>(gpu[badIdx]),
                    static_cast<double>(ref[badIdx]));
    }

    return { ok, maxAbs, maxRatio };
}

} // namespace

int main(int argc, char** argv) {
    const std::string hsacoPath =
        (argc > 1) ? std::string{argv[1]} : defaultHsacoPath(argv[0]);

    std::printf("hip_attention_flash_partial_fp16_probe:\n  hsaco: %s\n"
                "  nHeads=%d nKvHeads=%d headDim=%d curLen=%d\n"
                "  kMax=%d KTile=%d nKTiles=%d slotWidth=%d block=%u\n",
                hsacoPath.c_str(),
                kNumHeads, kNumKvHeads, kHeadDim, kCurLen,
                kKMax, kKTile, kNKTiles, kSlotWidth, kBlock);

    try {
        HipContext         ctx{};
        HipMemoryAllocator alloc{ctx};
        HipStream          stream{ctx, HipStreamKind::BlockingDefault};

        HipModule mod    = HipModule::fromFile(ctx, hsacoPath);
        HipKernel kernel = mod.getKernel("attention_flash_partial_fp16");

        const std::size_t qElems  = static_cast<std::size_t>(kNumHeads)
                                  * static_cast<std::size_t>(kHeadDim);
        const std::size_t kvElems = static_cast<std::size_t>(kCacheSlots)
                                  * static_cast<std::size_t>(kKvStride);

        std::vector<float>  hostQ(qElems);
        std::vector<__half> hostK(kvElems);
        std::vector<__half> hostV(kvElems);
        fillRandomFloat(hostQ, /*seed=*/0xA77E1001u, /*scale=*/1.0f);
        fillRandomHalf (hostK, /*seed=*/0xA77E1002u, /*scale=*/1.0f);
        fillRandomHalf (hostV, /*seed=*/0xA77E1003u, /*scale=*/1.0f);

        HipBuffer devQ      {alloc, qElems  * sizeof(float)};
        HipBuffer devK      {alloc, kvElems * sizeof(__half)};
        HipBuffer devV      {alloc, kvElems * sizeof(__half)};
        HipBuffer devCurLen {alloc, sizeof(int), HipAllocKind::Device};

        alloc.copyH2D(devQ.data(),      hostQ.data(),      qElems  * sizeof(float));
        alloc.copyH2D(devK.data(),      hostK.data(),      kvElems * sizeof(__half));
        alloc.copyH2D(devV.data(),      hostV.data(),      kvElems * sizeof(__half));
        alloc.copyH2D(devCurLen.data(), &kCurLen,          sizeof(int));

        std::printf("\n");
        RunResult a = runOnce("A", /*slidingWindow=*/0,
                              ctx, alloc, stream, kernel,
                              hostQ, hostK, hostV,
                              devQ.data(), devK.data(), devV.data(),
                              devCurLen.data());
        RunResult b = runOnce("B", /*slidingWindow=*/32,
                              ctx, alloc, stream, kernel,
                              hostQ, hostK, hostV,
                              devQ.data(), devK.data(), devV.data(),
                              devCurLen.data());

        const bool ok = a.passed && b.passed;
        std::printf("\n  tol formula:   abs 1.0e-04 + rel 1.0e-04 * |ref|\n");
        std::printf("\nhip_attention_flash_partial_fp16_probe: %s\n", ok ? "OK" : "FAIL");
        return ok ? 0 : 1;

    } catch (const std::exception& e) {
        std::fprintf(stderr, "hip_attention_flash_partial_fp16_probe: threw: %s\n", e.what());
        return 2;
    }
}