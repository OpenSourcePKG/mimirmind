// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// hip_attention_flash_partial_q8_0_probe — parity check for the
// Q8_0-KV variant of the decode-mode FlashAttention partial kernel.
//
// Same 2-case pattern as the f32 / fp16 flash-partial probes. K and V
// are Q8_0 blocks; the kernel dequantises on-the-fly in registers.
// CPU reference reads the exact same bytes and dequants the same way.

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
#include <limits>
#include <string>
#include <vector>

namespace {

using namespace mimirmind::core::hip;

constexpr int          kNumHeads       = 8;
constexpr int          kNumKvHeads     = 2;
constexpr int          kHeadDim        = 128;
constexpr int          kCurLen         = 100;
constexpr int          kKMax           = kCurLen + 1;
constexpr int          kKTile          = 64;
constexpr int          kNKTiles        = (kKMax + kKTile - 1) / kKTile;  // 2
constexpr int          kCacheSlots     = 128;
constexpr int          kKvDim          = kNumKvHeads * kHeadDim;         // 256
constexpr int          kBlockElems     = 32;
constexpr int          kBlockBytes     = 34;
constexpr int          kNBlocksPerRow  = kKvDim / kBlockElems;           // 8
constexpr int          kSlotWidth      = 2 + kHeadDim;                   // 130
constexpr std::uint32_t kBlock         = 16;   // ATTN_FLASH_LOCAL

const float kScale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));

std::string defaultHsacoPath(const char* argv0) {
    std::filesystem::path exe = std::filesystem::canonical(argv0);
    return (exe.parent_path() / "hsaco" / "attention_flash_partial_q8_0.hsaco").string();
}

void fillRandomFloat(std::vector<float>& v, std::uint32_t seed, float scale) {
    std::uint32_t x = seed;
    for (auto& f : v) {
        x = x * 1664525u + 1013904223u;
        const std::int32_t s = static_cast<std::int32_t>(x >> 8) - (1 << 23);
        f = scale * static_cast<float>(s) / static_cast<float>(1 << 23);
    }
}

void fillRandomQ8_0(std::vector<std::uint8_t>& bytes, std::uint32_t seed) {
    std::uint32_t x = seed;
    const std::size_t nBlocks = bytes.size() / kBlockBytes;
    for (std::size_t b = 0; b < nBlocks; ++b) {
        std::uint8_t* blk = bytes.data() + b * kBlockBytes;
        x = x * 1664525u + 1013904223u;
        const float dScalar =
            0.001f + 0.049f * (static_cast<float>(x & 0xFFFF) / 65535.0f);
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

inline float dequantElem(const std::uint8_t* row, int elem) {
    const int blk = elem / kBlockElems;
    const int in  = elem % kBlockElems;
    const std::uint8_t* blkPtr = row + static_cast<std::size_t>(blk) * kBlockBytes;
    __half hScale;
    std::memcpy(&hScale, blkPtr, sizeof(__half));
    const float bscale = __half2float(hScale);
    const signed char qi = reinterpret_cast<const signed char*>(blkPtr)[2 + in];
    return bscale * static_cast<float>(qi);
}

void attentionFlashPartialQ8_0CpuRef(
    const std::vector<float>&        q,
    const std::vector<std::uint8_t>& k,
    const std::vector<std::uint8_t>& v,
    std::vector<float>&              partialMlo,
    int nHeads, int nKvHeads, int headDim,
    int curLen, float scale, int slidingWindow)
{
    const int kvDim         = nKvHeads * headDim;
    const int nBlocksPerRow = kvDim / kBlockElems;
    const int rowBytes      = nBlocksPerRow * kBlockBytes;
    const int kMax          = curLen + 1;
    const int kMin          = (slidingWindow > 0 && kMax > slidingWindow)
                                ? (kMax - slidingWindow) : 0;
    const int nKTiles       = (kMax + kKTile - 1) / kKTile;

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
                const std::uint8_t* kRow = k.data()
                                         + static_cast<std::size_t>(absKk) * rowBytes;
                double acc = 0.0;
                for (int d = 0; d < headDim; ++d) {
                    const float kv = dequantElem(kRow, hkv * headDim + d);
                    acc += static_cast<double>(qVec[d]) * static_cast<double>(kv);
                }
                scores[kk] = acc * static_cast<double>(scale);
            }

            double mLocal = -std::numeric_limits<double>::infinity();
            for (double s : scores) if (s > mLocal) mLocal = s;

            double lLocal = 0.0;
            for (double& s : scores) { s = std::exp(s - mLocal); lLocal += s; }

            mloPtr[0] = static_cast<float>(mLocal);
            mloPtr[1] = static_cast<float>(lLocal);

            for (int d = 0; d < headDim; ++d) {
                double acc = 0.0;
                for (int kk = 0; kk < tileLen; ++kk) {
                    const int absKk = kStart + kk;
                    const std::uint8_t* vRow = v.data()
                                             + static_cast<std::size_t>(absKk) * rowBytes;
                    const float vv = dequantElem(vRow, hkv * headDim + d);
                    acc += scores[kk] * static_cast<double>(vv);
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
                  const std::vector<float>&        hostQ,
                  const std::vector<std::uint8_t>& hostK,
                  const std::vector<std::uint8_t>& hostV,
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
    attentionFlashPartialQ8_0CpuRef(hostQ, hostK, hostV, ref,
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

    std::printf("hip_attention_flash_partial_q8_0_probe:\n  hsaco: %s\n"
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
        HipKernel kernel = mod.getKernel("attention_flash_partial_q8_0");

        const std::size_t qElems  = static_cast<std::size_t>(kNumHeads)
                                  * static_cast<std::size_t>(kHeadDim);
        const std::size_t kvBytes = static_cast<std::size_t>(kCacheSlots)
                                  * static_cast<std::size_t>(kNBlocksPerRow)
                                  * static_cast<std::size_t>(kBlockBytes);

        std::vector<float>        hostQ(qElems);
        std::vector<std::uint8_t> hostK(kvBytes);
        std::vector<std::uint8_t> hostV(kvBytes);
        fillRandomFloat(hostQ, /*seed=*/0xF88D0001u, /*scale=*/1.0f);
        fillRandomQ8_0 (hostK, /*seed=*/0xF88D0002u);
        fillRandomQ8_0 (hostV, /*seed=*/0xF88D0003u);

        HipBuffer devQ      {alloc, qElems * sizeof(float)};
        HipBuffer devK      {alloc, kvBytes};
        HipBuffer devV      {alloc, kvBytes};
        HipBuffer devCurLen {alloc, sizeof(int), HipAllocKind::Device};

        alloc.copyH2D(devQ.data(),      hostQ.data(),      qElems * sizeof(float));
        alloc.copyH2D(devK.data(),      hostK.data(),      kvBytes);
        alloc.copyH2D(devV.data(),      hostV.data(),      kvBytes);
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
        std::printf("\nhip_attention_flash_partial_q8_0_probe: %s\n", ok ? "OK" : "FAIL");
        return ok ? 0 : 1;

    } catch (const std::exception& e) {
        std::fprintf(stderr, "hip_attention_flash_partial_q8_0_probe: threw: %s\n", e.what());
        return 2;
    }
}
