// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// hip_attention_prefill_flash_fp16_probe — fp16-KV variant of the
// prefill-flash probe. K/V are __half; CPU reference promotes via
// __half2float exactly the way the kernel does.

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

constexpr int          kT_q        = 6;
constexpr int          kNumHeads   = 8;
constexpr int          kNumKvHeads = 2;
constexpr int          kHeadDim    = 128;
constexpr int          kCurLen     = 64;
constexpr int          kCacheSlots = 80;
constexpr int          kQStride    = kNumHeads   * kHeadDim;
constexpr int          kKvStride   = kNumKvHeads * kHeadDim;
constexpr std::uint32_t kBlock     = 16;

const float kScale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));

std::string defaultHsacoPath(const char* argv0) {
    std::filesystem::path exe = std::filesystem::canonical(argv0);
    return (exe.parent_path() / "hsaco" / "attention_prefill_flash_fp16.hsaco").string();
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

void monolithicAttentionFp16CpuRef(
    const std::vector<float>&  q,
    const std::vector<__half>& k,
    const std::vector<__half>& v,
    std::vector<float>&        out,
    int T_q, int nHeads, int nKvHeads, int headDim,
    int curLen, float scale, int slidingWindow)
{
    const int qStride  = nHeads   * headDim;
    const int kvStride = nKvHeads * headDim;

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
                const __half* kVec = k.data()
                                   + static_cast<std::size_t>(kk)  * kvStride
                                   + static_cast<std::size_t>(hkv) * headDim;
                double acc = 0.0;
                for (int d = 0; d < headDim; ++d) {
                    acc += static_cast<double>(qVec[d]) *
                           static_cast<double>(__half2float(kVec[d]));
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
                    const __half* vVec = v.data()
                                       + static_cast<std::size_t>(kk)  * kvStride
                                       + static_cast<std::size_t>(hkv) * headDim;
                    acc += scores[i] *
                           static_cast<double>(__half2float(vVec[d]));
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
                  const std::vector<float>&  hostQ,
                  const std::vector<__half>& hostK,
                  const std::vector<__half>& hostV,
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
    monolithicAttentionFp16CpuRef(hostQ, hostK, hostV, ref,
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

    std::printf("hip_attention_prefill_flash_fp16_probe:\n  hsaco: %s\n"
                "  T_q=%d nHeads=%d nKvHeads=%d headDim=%d curLen=%d block=%u\n",
                hsacoPath.c_str(),
                kT_q, kNumHeads, kNumKvHeads, kHeadDim, kCurLen, kBlock);

    try {
        HipContext         ctx{};
        HipMemoryAllocator alloc{ctx};
        HipStream          stream{ctx, HipStreamKind::BlockingDefault};

        HipModule mod    = HipModule::fromFile(ctx, hsacoPath);
        HipKernel kernel = mod.getKernel("attention_prefill_flash_fp16");

        const std::size_t qElems  = static_cast<std::size_t>(kT_q) * kQStride;
        const std::size_t kvElems = static_cast<std::size_t>(kCacheSlots) * kKvStride;

        std::vector<float>  hostQ(qElems);
        std::vector<__half> hostK(kvElems);
        std::vector<__half> hostV(kvElems);
        fillRandomFloat(hostQ, /*seed=*/0xF16E1001u, /*scale=*/1.0f);
        fillRandomHalf (hostK, /*seed=*/0xF16E1002u, /*scale=*/1.0f);
        fillRandomHalf (hostV, /*seed=*/0xF16E1003u, /*scale=*/1.0f);

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
        std::printf("\nhip_attention_prefill_flash_fp16_probe: %s\n", ok ? "OK" : "FAIL");
        return ok ? 0 : 1;

    } catch (const std::exception& e) {
        std::fprintf(stderr, "hip_attention_prefill_flash_fp16_probe: threw: %s\n", e.what());
        return 2;
    }
}