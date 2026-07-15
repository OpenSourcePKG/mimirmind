// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// hip_rope_ff_probe — parity check for the RoPE-with-frequency-
// factors kernel. Same shape as hip_rope_probe but with an added
// [halfDim] `freq_factors` buffer that divides the per-pair theta
// (mirrors ggml_rope_ext's YaRN-style scaling on Gemma-family
// global-attention layers).

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

constexpr int   kSeqLen   = 8;
constexpr int   kNumHeads = 4;
constexpr int   kHeadDim  = 64;
constexpr int   kStartPos = 5;
constexpr float kBase     = 10000.0f;
constexpr int   kWriteOff = 0;

constexpr std::uint32_t kBlock = 256;   // == ROPE_LOCAL

std::string defaultHsacoPath(const char* argv0) {
    std::filesystem::path exe = std::filesystem::canonical(argv0);
    return (exe.parent_path() / "hsaco" / "rope_inplace_ff.hsaco").string();
}

void fillRandom(std::vector<float>& v, std::uint32_t seed, float scale) {
    std::uint32_t x = seed;
    for (auto& f : v) {
        x = x * 1664525u + 1013904223u;
        const std::int32_t s = static_cast<std::int32_t>(x >> 8) - (1 << 23);
        f = scale * static_cast<float>(s) / static_cast<float>(1 << 23);
    }
}

void ropeFfCpuRef(std::vector<float>&       y,   // in+out
                  const std::vector<float>& freqFactors,   // [halfDim]
                  int seqLen, int numHeads, int headDim,
                  int startPos, float base) {
    const int halfDim = headDim / 2;
    for (int p = 0; p < seqLen; ++p) {
        const double pos = static_cast<double>(startPos + p);
        for (int h = 0; h < numHeads; ++h) {
            const int headBase = (p * numHeads + h) * headDim;
            for (int i = 0; i < halfDim; ++i) {
                const double invDim   = 1.0 / static_cast<double>(headDim);
                const double baseFreq = std::pow(static_cast<double>(base),
                                                 -static_cast<double>(2 * i) * invDim);
                const double ff       = static_cast<double>(freqFactors[i]);
                const double freq     = baseFreq / ff;
                const double theta    = pos * freq;
                const double c = std::cos(theta);
                const double s = std::sin(theta);
                const double a = static_cast<double>(y[headBase + i]);
                const double b = static_cast<double>(y[headBase + i + halfDim]);
                y[headBase + i]           = static_cast<float>(a * c - b * s);
                y[headBase + i + halfDim] = static_cast<float>(a * s + b * c);
            }
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    const std::string hsacoPath =
        (argc > 1) ? std::string{argv[1]} : defaultHsacoPath(argv[0]);

    const int total     = kSeqLen * kNumHeads * kHeadDim;
    const int totalWork = kSeqLen * kNumHeads * (kHeadDim / 2);
    const int halfDim   = kHeadDim / 2;

    std::printf("hip_rope_ff_probe:\n  hsaco: %s\n"
                "  seqLen=%d numHeads=%d headDim=%d startPos=%d base=%.0f\n",
                hsacoPath.c_str(), kSeqLen, kNumHeads, kHeadDim, kStartPos,
                static_cast<double>(kBase));

    try {
        HipContext         ctx{};
        HipMemoryAllocator alloc{ctx};
        HipStream          stream{ctx, HipStreamKind::BlockingDefault};
        HipEvent           evStart{ctx};
        HipEvent           evEnd  {ctx};

        HipModule mod    = HipModule::fromFile(ctx, hsacoPath);
        HipKernel kernel = mod.getKernel("rope_inplace_ff");

        std::vector<float> input(static_cast<std::size_t>(total));
        fillRandom(input, /*seed=*/0xDEADBEEFu, /*scale=*/1.0f);

        // freq_factors ∈ [0.5, 1.5] — non-trivial scaling that isn't
        // uniformly 1.0 (which would degenerate to plain RoPE). Range
        // is representative of Gemma 4 global-attention layers.
        std::vector<float> freqFactors(static_cast<std::size_t>(halfDim));
        fillRandom(freqFactors, /*seed=*/0xFACADE01u, /*scale=*/0.5f);
        for (auto& f : freqFactors) f = f + 1.0f;   // shift to [0.5, 1.5]

        std::vector<float> hostRef = input;
        ropeFfCpuRef(hostRef, freqFactors,
                     kSeqLen, kNumHeads, kHeadDim, kStartPos, kBase);

        std::vector<float> hostGpu(static_cast<std::size_t>(total), 0.0f);

        const std::size_t xBytes = static_cast<std::size_t>(total) * sizeof(float);
        const std::size_t fBytes = static_cast<std::size_t>(halfDim) * sizeof(float);
        HipBuffer devX    {alloc, xBytes};
        HipBuffer devFF   {alloc, fBytes};
        HipBuffer devStart{alloc, sizeof(int), HipAllocKind::Device};

        alloc.copyH2D(devX    .data(), input      .data(), xBytes);
        alloc.copyH2D(devFF   .data(), freqFactors.data(), fBytes);
        alloc.copyH2D(devStart.data(), &kStartPos,         sizeof(int));

        kernel.setPtr  (0, devX    .data());
        kernel.setPtr  (1, devFF   .data());
        kernel.setValue(2, kSeqLen);
        kernel.setValue(3, kNumHeads);
        kernel.setValue(4, kHeadDim);
        kernel.setPtr  (5, devStart.data());
        kernel.setValue(6, kBase);
        kernel.setValue(7, kWriteOff);

        const std::uint32_t grid =
            (static_cast<std::uint32_t>(totalWork) + kBlock - 1) / kBlock;

        evStart.record(stream);
        kernel.launch(stream, grid, 1, 1, kBlock, 1, 1, /*shared=*/0);
        evEnd.record(stream);
        stream.synchronize();

        const float kernelMs = evEnd.elapsedMs(evStart);

        alloc.copyD2H(hostGpu.data(), devX.data(), xBytes);

        constexpr float kAbsTol = 1e-5f;
        constexpr float kRelTol = 1e-4f;

        float       maxAbs   = 0.0f;
        float       maxRatio = 0.0f;
        std::size_t badIdx   = SIZE_MAX;
        for (std::size_t i = 0; i < static_cast<std::size_t>(total); ++i) {
            const float d         = std::fabs(hostGpu[i] - hostRef[i]);
            const float threshold = kAbsTol + kRelTol * std::fabs(hostRef[i]);
            const float ratio     = d / threshold;
            if (ratio > maxRatio) { maxRatio = ratio; badIdx = i; }
            if (d > maxAbs) maxAbs = d;
        }

        std::printf("\n  kernel:              %.3f ms\n",
                    static_cast<double>(kernelMs));
        std::printf("  max abs err:         %.3e\n",
                    static_cast<double>(maxAbs));
        std::printf("  max err / tol:       %.3f (fails if > 1.0)\n",
                    static_cast<double>(maxRatio));
        if (badIdx != SIZE_MAX) {
            std::printf("  worst @ idx=%zu: gpu=%.6g cpu=%.6g\n",
                        badIdx,
                        static_cast<double>(hostGpu[badIdx]),
                        static_cast<double>(hostRef[badIdx]));
        }

        const bool ok = (maxRatio <= 1.0f);
        std::printf("\nhip_rope_ff_probe: %s\n", ok ? "OK" : "FAIL");
        return ok ? 0 : 1;

    } catch (const std::exception& e) {
        std::fprintf(stderr, "hip_rope_ff_probe: threw: %s\n", e.what());
        return 2;
    }
}