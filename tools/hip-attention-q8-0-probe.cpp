// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// hip_attention_q8_0_probe — parity check for the Q8_0-KV variant of
// the untiled multi-head attention kernel.
//
// K and V live as ggml-style Q8_0 blocks (2 B fp16 scale + 32 B int8
// quants = 34 B / block). Q and OUT stay fp32. The kernel dequantises
// each block on-the-fly in registers.
//
// Probe strategy (same as hip_matmul_q8_probe): we DON'T round-trip
// real floats through Q8_0 packing. Random fp16 scales + random int8
// quants are generated directly and given to both sides — so any diff
// is a bug in the dequant-and-attend path, not packing noise.
//
// Two cases share the same random Q/K/V-Q8_0:
//   A: slidingWindow=0    — full causal.
//   B: slidingWindow=16   — SWA-truncated to 16 keys per query.

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

constexpr int          kT_q            = 4;
constexpr int          kNumHeads       = 8;
constexpr int          kNumKvHeads     = 2;
constexpr int          kHeadDim        = 128;
constexpr int          kCurLen         = 32;
constexpr int          kCacheSlots     = 48;
constexpr int          kQStride        = kNumHeads   * kHeadDim;
constexpr int          kKvDim          = kNumKvHeads * kHeadDim;
constexpr int          kBlockElems     = 32;
constexpr int          kBlockBytes     = 34;
constexpr int          kNBlocksPerRow  = kKvDim / kBlockElems;   // 16
constexpr std::uint32_t kBlock         = 16;                     // ATTN_LOCAL

const float kScale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));

std::string defaultHsacoPath(const char* argv0) {
    std::filesystem::path exe = std::filesystem::canonical(argv0);
    return (exe.parent_path() / "hsaco" / "attention_q8_0.hsaco").string();
}

void fillRandomFloat(std::vector<float>& v, std::uint32_t seed, float scale) {
    std::uint32_t x = seed;
    for (auto& f : v) {
        x = x * 1664525u + 1013904223u;
        const std::int32_t s = static_cast<std::int32_t>(x >> 8) - (1 << 23);
        f = scale * static_cast<float>(s) / static_cast<float>(1 << 23);
    }
}

// Fill a Q8_0-formatted byte buffer with random block scales + random
// int8 quants. Both GPU and CPU read from these exact bytes — no
// fp32-through-quantize round-trip involved, so any diff is a
// dequant-and-attend bug.
void fillRandomQ8_0(std::vector<std::uint8_t>& bytes, std::uint32_t seed) {
    std::uint32_t x = seed;
    const std::size_t nBlocks = bytes.size() / kBlockBytes;
    for (std::size_t b = 0; b < nBlocks; ++b) {
        std::uint8_t* blk = bytes.data() + b * kBlockBytes;

        // Scale in [0.001, 0.05] so dequant values live in [-1, 1] approx.
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

// Read one fp32 value from the Q8_0-formatted cache at absolute
// element index `elem` inside the packed row.
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

void attentionQ8_0CpuRef(
    const std::vector<float>&        q,
    const std::vector<std::uint8_t>& k,
    const std::vector<std::uint8_t>& v,
    std::vector<float>&              out,
    int T_q, int nHeads, int nKvHeads, int headDim,
    int curLen, float scale, int slidingWindow)
{
    const int qStride       = nHeads   * headDim;
    const int kvDim         = nKvHeads * headDim;
    const int nBlocksPerRow = kvDim / kBlockElems;
    const int rowBytes      = nBlocksPerRow * kBlockBytes;

    for (int pq = 0; pq < T_q; ++pq) {
        const int absPos = curLen + pq;
        const int kMax   = absPos + 1;
        const int kMin   = (slidingWindow > 0 && kMax > slidingWindow)
                             ? (kMax - slidingWindow) : 0;

        for (int hq = 0; hq < nHeads; ++hq) {
            const int hkv = (hq * nKvHeads) / nHeads;
            const float* qVec = q.data()
                              + static_cast<std::size_t>(pq) * qStride
                              + static_cast<std::size_t>(hq) * headDim;
            float* oVec = out.data()
                        + static_cast<std::size_t>(pq) * qStride
                        + static_cast<std::size_t>(hq) * headDim;

            const int n = kMax - kMin;
            std::vector<double> scores(n);
            for (int i = 0; i < n; ++i) {
                const int kk = kMin + i;
                const std::uint8_t* kRow = k.data()
                                         + static_cast<std::size_t>(kk) * rowBytes;
                double acc = 0.0;
                for (int d = 0; d < headDim; ++d) {
                    const float kv = dequantElem(kRow, hkv * headDim + d);
                    acc += static_cast<double>(qVec[d]) *
                           static_cast<double>(kv);
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
                    const int kk = kMin + i;
                    const std::uint8_t* vRow = v.data()
                                             + static_cast<std::size_t>(kk) * rowBytes;
                    const float vv = dequantElem(vRow, hkv * headDim + d);
                    acc += scores[i] * static_cast<double>(vv);
                }
                oVec[d] = static_cast<float>(acc / l);
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
    const std::size_t outElems = static_cast<std::size_t>(kT_q)
                               * static_cast<std::size_t>(kQStride);
    const std::size_t outBytes = outElems * sizeof(float);

    HipBuffer devOut{alloc, outBytes};
    std::vector<float> sentinel(outElems, 12345.0f);
    alloc.copyH2D(devOut.data(), sentinel.data(), outBytes);

    kernel.setPtr  (0, devQ);
    kernel.setPtr  (1, devK);
    kernel.setPtr  (2, devV);
    kernel.setPtr  (3, devOut.data());
    kernel.setValue(4, kT_q);
    kernel.setValue(5, kNumHeads);
    kernel.setValue(6, kNumKvHeads);
    kernel.setValue(7, kHeadDim);
    kernel.setPtr  (8, devCurLen);
    kernel.setValue(9, kScale);
    kernel.setValue(10, slidingWindow);

    HipEvent evStart{ctx};
    HipEvent evEnd  {ctx};
    evStart.record(stream);
    kernel.launch(stream,
                  /*grid=*/  static_cast<std::uint32_t>(kNumHeads),
                             static_cast<std::uint32_t>(kT_q), 1,
                  /*block=*/ kBlock, 1, 1,
                  /*shared=*/0);
    evEnd.record(stream);
    stream.synchronize();
    const float kernelMs = evEnd.elapsedMs(evStart);

    std::vector<float> ref(outElems, 0.0f);
    attentionQ8_0CpuRef(hostQ, hostK, hostV, ref,
                        kT_q, kNumHeads, kNumKvHeads, kHeadDim,
                        kCurLen, kScale, slidingWindow);

    std::vector<float> gpu(outElems, 0.0f);
    alloc.copyD2H(gpu.data(), devOut.data(), outBytes);

    constexpr float kAbsTol = 1e-4f;
    constexpr float kRelTol = 1e-4f;

    float       maxAbs   = 0.0f;
    float       maxRatio = 0.0f;
    std::size_t badIdx   = SIZE_MAX;
    for (std::size_t i = 0; i < outElems; ++i) {
        const float d         = std::fabs(gpu[i] - ref[i]);
        const float threshold = kAbsTol + kRelTol * std::fabs(ref[i]);
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
        std::printf("    worst @ idx=%zu: gpu=%.6g cpu=%.6g\n",
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

    std::printf("hip_attention_q8_0_probe:\n  hsaco: %s\n"
                "  T_q=%d nHeads=%d nKvHeads=%d headDim=%d curLen=%d block=%u\n",
                hsacoPath.c_str(),
                kT_q, kNumHeads, kNumKvHeads, kHeadDim, kCurLen, kBlock);

    try {
        HipContext         ctx{};
        HipMemoryAllocator alloc{ctx};
        HipStream          stream{ctx, HipStreamKind::BlockingDefault};

        HipModule mod    = HipModule::fromFile(ctx, hsacoPath);
        HipKernel kernel = mod.getKernel("attention_q8_0");

        const std::size_t qElems  = static_cast<std::size_t>(kT_q) * kQStride;
        const std::size_t kvBytes = static_cast<std::size_t>(kCacheSlots)
                                  * static_cast<std::size_t>(kNBlocksPerRow)
                                  * static_cast<std::size_t>(kBlockBytes);

        std::vector<float>        hostQ(qElems);
        std::vector<std::uint8_t> hostK(kvBytes);
        std::vector<std::uint8_t> hostV(kvBytes);
        fillRandomFloat(hostQ, /*seed=*/0xA88D0001u, /*scale=*/1.0f);
        fillRandomQ8_0 (hostK, /*seed=*/0xA88D0002u);
        fillRandomQ8_0 (hostV, /*seed=*/0xA88D0003u);

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
        RunResult b = runOnce("B", /*slidingWindow=*/16,
                              ctx, alloc, stream, kernel,
                              hostQ, hostK, hostV,
                              devQ.data(), devK.data(), devV.data(),
                              devCurLen.data());

        const bool ok = a.passed && b.passed;
        std::printf("\n  tol formula:   abs 1.0e-04 + rel 1.0e-04 * |ref|\n");
        std::printf("\nhip_attention_q8_0_probe: %s\n", ok ? "OK" : "FAIL");
        return ok ? 0 : 1;

    } catch (const std::exception& e) {
        std::fprintf(stderr, "hip_attention_q8_0_probe: threw: %s\n", e.what());
        return 2;
    }
}