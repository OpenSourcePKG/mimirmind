// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// hip_attention_flash_merge_probe — end-to-end parity check for the
// FlashAttention merge kernel via the online-softmax identity.
//
// The probe generates random q / k / v, computes the partial tensor
// on the CPU (same math as attention_flash_partial's reference),
// uploads it, launches the merge kernel, and compares the merged
// output against a MONOLITHIC (no-tiling) attention fp64 reference.
//
// This exercises TWO invariants at once:
//   1. The merge kernel correctly applies the online-softmax identity.
//   2. The partial→merge round-trip reproduces monolithic attention
//      (the mathematical guarantee that makes flash worthwhile).
//
// Two cases share the same q/k/v inputs and dispatch geometry
// (nHeads=8, nKvHeads=2, headDim=128, curLen=100 → nKTiles=2):
//
//   Case A: slidingWindow=0    — both tiles valid.
//   Case B: slidingWindow=32   — tile 0 hits the neutral-emission
//                                path; merge must handle beta=0 for
//                                (-INFINITY, 0, 0…) slots.
//
// Combined-tolerance 1e-4/1e-4. Merged output is one softmax + one
// K-term V-weighted sum → ULP-level drift on the order of 1e-6.

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
    return (exe.parent_path() / "hsaco" / "attention_flash_merge.hsaco").string();
}

void fillRandom(std::vector<float>& v, std::uint32_t seed, float scale) {
    std::uint32_t x = seed;
    for (auto& f : v) {
        x = x * 1664525u + 1013904223u;
        const std::int32_t s = static_cast<std::int32_t>(x >> 8) - (1 << 23);
        f = scale * static_cast<float>(s) / static_cast<float>(1 << 23);
    }
}

// Build the partial tensor on the CPU exactly the way
// attention_flash_partial does. Used to feed the merge kernel — we
// deliberately DON'T run the partial GPU kernel here, so a merge
// failure can't be blamed on the partial's numerics.
void buildPartialCpuRef(
    const std::vector<float>& q,
    const std::vector<float>& k,
    const std::vector<float>& v,
    std::vector<float>&       partialMlo,
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
                const float* kVec = k.data()
                                  + static_cast<std::size_t>(absKk) * kvStride
                                  + static_cast<std::size_t>(hkv)   * headDim;
                double acc = 0.0;
                for (int d = 0; d < headDim; ++d) {
                    acc += static_cast<double>(qVec[d]) *
                           static_cast<double>(kVec[d]);
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
                    const float* vVec = v.data()
                                      + static_cast<std::size_t>(kStart + kk) * kvStride
                                      + static_cast<std::size_t>(hkv)         * headDim;
                    acc += scores[kk] * static_cast<double>(vVec[d]);
                }
                mloPtr[2 + d] = static_cast<float>(acc);
            }
        }
    }
}

// Monolithic (no-tiling) attention on the CPU — the ground truth the
// merge kernel must reproduce.
void monolithicAttentionCpuRef(
    const std::vector<float>& q,
    const std::vector<float>& k,
    const std::vector<float>& v,
    std::vector<float>&       out,        // [nHeads, headDim]
    int nHeads, int nKvHeads, int headDim,
    int curLen, float scale, int slidingWindow)
{
    const int kvStride = nKvHeads * headDim;
    const int kMax     = curLen + 1;
    const int kMin     = (slidingWindow > 0 && kMax > slidingWindow)
                           ? (kMax - slidingWindow) : 0;

    for (int hq = 0; hq < nHeads; ++hq) {
        const int hkv = (hq * nKvHeads) / nHeads;
        const float* qVec = q.data()
                          + static_cast<std::size_t>(hq)
                          * static_cast<std::size_t>(headDim);
        float* oOut = out.data()
                    + static_cast<std::size_t>(hq)
                    * static_cast<std::size_t>(headDim);

        if (kMin >= kMax) {
            for (int d = 0; d < headDim; ++d) oOut[d] = 0.0f;
            continue;
        }

        const int n = kMax - kMin;
        std::vector<double> scores(n);
        for (int i = 0; i < n; ++i) {
            const int absKk = kMin + i;
            const float* kVec = k.data()
                              + static_cast<std::size_t>(absKk) * kvStride
                              + static_cast<std::size_t>(hkv)   * headDim;
            double acc = 0.0;
            for (int d = 0; d < headDim; ++d) {
                acc += static_cast<double>(qVec[d]) *
                       static_cast<double>(kVec[d]);
            }
            scores[i] = acc * static_cast<double>(scale);
        }

        double m = -std::numeric_limits<double>::infinity();
        for (double s : scores) if (s > m) m = s;

        double l = 0.0;
        for (double& s : scores) { s = std::exp(s - m); l += s; }

        for (int d = 0; d < headDim; ++d) {
            double acc = 0.0;
            for (int i = 0; i < n; ++i) {
                const int absKk = kMin + i;
                const float* vVec = v.data()
                                  + static_cast<std::size_t>(absKk) * kvStride
                                  + static_cast<std::size_t>(hkv)   * headDim;
                acc += scores[i] * static_cast<double>(vVec[d]);
            }
            oOut[d] = static_cast<float>(acc / l);
        }
    }
}

struct RunResult { bool passed; float maxAbs; float maxRatio; };

RunResult runOnce(const char* label, int slidingWindow,
                  HipContext& ctx, HipMemoryAllocator& alloc,
                  HipStream& stream, HipKernel& kernel,
                  const std::vector<float>& hostQ,
                  const std::vector<float>& hostK,
                  const std::vector<float>& hostV,
                  void* devCurLen)
{
    const std::size_t sloElems = static_cast<std::size_t>(kNumHeads)
                               * static_cast<std::size_t>(kNKTiles)
                               * static_cast<std::size_t>(kSlotWidth);
    const std::size_t sloBytes = sloElems * sizeof(float);
    const std::size_t outElems = static_cast<std::size_t>(kNumHeads)
                               * static_cast<std::size_t>(kHeadDim);
    const std::size_t outBytes = outElems * sizeof(float);

    // ---- CPU-computed partial tensor (input to merge) ---------------
    std::vector<float> hostPartial(sloElems, 0.0f);
    buildPartialCpuRef(hostQ, hostK, hostV, hostPartial,
                       kNumHeads, kNumKvHeads, kHeadDim,
                       kCurLen, kScale, slidingWindow);

    // ---- Monolithic CPU reference (ground truth for merge output) ---
    std::vector<float> hostRef(outElems, 0.0f);
    monolithicAttentionCpuRef(hostQ, hostK, hostV, hostRef,
                              kNumHeads, kNumKvHeads, kHeadDim,
                              kCurLen, kScale, slidingWindow);

    // ---- device tensors ---------------------------------------------
    HipBuffer devPartial{alloc, sloBytes};
    HipBuffer devOut    {alloc, outBytes};

    alloc.copyH2D(devPartial.data(), hostPartial.data(), sloBytes);
    // Pre-seed output with a sentinel to catch un-written slots.
    std::vector<float> sentinel(outElems, 12345.0f);
    alloc.copyH2D(devOut.data(), sentinel.data(), outBytes);

    kernel.setPtr  (0, devPartial.data());
    kernel.setPtr  (1, devOut    .data());
    kernel.setValue(2, kNumHeads);
    kernel.setValue(3, kHeadDim);
    kernel.setPtr  (4, devCurLen);

    HipEvent evStart{ctx};
    HipEvent evEnd  {ctx};
    evStart.record(stream);
    kernel.launch(stream,
                  /*grid=*/  static_cast<std::uint32_t>(kNumHeads), 1, 1,
                  /*block=*/ kBlock, 1, 1,
                  /*shared=*/0);
    evEnd.record(stream);
    stream.synchronize();
    const float kernelMs = evEnd.elapsedMs(evStart);

    std::vector<float> gpuOut(outElems, 0.0f);
    alloc.copyD2H(gpuOut.data(), devOut.data(), outBytes);

    constexpr float kAbsTol = 1e-4f;
    constexpr float kRelTol = 1e-4f;

    float       maxAbs   = 0.0f;
    float       maxRatio = 0.0f;
    std::size_t badIdx   = SIZE_MAX;
    for (std::size_t i = 0; i < outElems; ++i) {
        const float d         = std::fabs(gpuOut[i] - hostRef[i]);
        const float threshold = kAbsTol + kRelTol * std::fabs(hostRef[i]);
        const float ratio     = d / threshold;
        if (ratio > maxRatio) { maxRatio = ratio; badIdx = i; }
        if (d > maxAbs) maxAbs = d;
    }
    const bool ok = (maxRatio <= 1.0f);

    std::printf("  [%s] slidingWindow=%d  kernel=%.3f ms\n"
                "    max abs err %.3e   max err/tol %.3f\n",
                label, slidingWindow, static_cast<double>(kernelMs),
                static_cast<double>(maxAbs),
                static_cast<double>(maxRatio));
    if (badIdx != SIZE_MAX && maxRatio > 1e-3f) {
        const std::size_t hq = badIdx / static_cast<std::size_t>(kHeadDim);
        const std::size_t d  = badIdx % static_cast<std::size_t>(kHeadDim);
        std::printf("    worst @ hq=%zu d=%zu: gpu=%.6g cpu=%.6g\n",
                    hq, d,
                    static_cast<double>(gpuOut[badIdx]),
                    static_cast<double>(hostRef[badIdx]));
    }
    return { ok, maxAbs, maxRatio };
}

} // namespace

int main(int argc, char** argv) {
    const std::string hsacoPath =
        (argc > 1) ? std::string{argv[1]} : defaultHsacoPath(argv[0]);

    std::printf("hip_attention_flash_merge_probe:\n  hsaco: %s\n"
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
        HipKernel kernel = mod.getKernel("attention_flash_merge");

        // ---- host inputs (same seeds as partial probe) ----------------
        const std::size_t qElems  = static_cast<std::size_t>(kNumHeads)
                                  * static_cast<std::size_t>(kHeadDim);
        const std::size_t kvElems = static_cast<std::size_t>(kCacheSlots)
                                  * static_cast<std::size_t>(kKvStride);

        std::vector<float> hostQ(qElems);
        std::vector<float> hostK(kvElems);
        std::vector<float> hostV(kvElems);
        fillRandom(hostQ, /*seed=*/0xA77E1001u, /*scale=*/1.0f);
        fillRandom(hostK, /*seed=*/0xA77E1002u, /*scale=*/1.0f);
        fillRandom(hostV, /*seed=*/0xA77E1003u, /*scale=*/1.0f);

        HipBuffer devCurLen{alloc, sizeof(int), HipAllocKind::Device};
        alloc.copyH2D(devCurLen.data(), &kCurLen, sizeof(int));

        std::printf("\n");
        RunResult a = runOnce("A", /*slidingWindow=*/0,
                              ctx, alloc, stream, kernel,
                              hostQ, hostK, hostV, devCurLen.data());
        RunResult b = runOnce("B", /*slidingWindow=*/32,
                              ctx, alloc, stream, kernel,
                              hostQ, hostK, hostV, devCurLen.data());

        const bool ok = a.passed && b.passed;
        std::printf("\n  tol formula:   abs 1.0e-04 + rel 1.0e-04 * |ref|\n");
        std::printf("\nhip_attention_flash_merge_probe: %s\n", ok ? "OK" : "FAIL");
        return ok ? 0 : 1;

    } catch (const std::exception& e) {
        std::fprintf(stderr, "hip_attention_flash_merge_probe: threw: %s\n", e.what());
        return 2;
    }
}