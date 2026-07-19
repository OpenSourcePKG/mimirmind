// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// hip_rope_probe — parity check for the RoPE kernel port. Loads
// rope_inplace.hsaco, applies rotary embedding to a synthetic
// [seqLen, numHeads, headDim] tensor, compares against an inline
// double-precision CPU reference.
//
// Uses writeOffsetStride=0 (Q-rope shape) — the K-rope shape with
// per-token stride is exercised implicitly at integration time in
// the eventual runtime port, not here.

#include "core/gpu/hip/HipContext.hpp"
#include "core/gpu/hip/HipEvent.hpp"
#include "core/gpu/hip/HipKernel.hpp"
#include "core/gpu/hip/HipMemoryAllocator.hpp"
#include "core/gpu/hip/HipModule.hpp"
#include "core/gpu/hip/HipStream.hpp"

#include <algorithm>
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
constexpr int   kHeadDim  = 64;      // halfDim = 32
constexpr int   kStartPos = 5;       // arbitrary non-zero to exercise the offset path
constexpr float kBase     = 10000.0f;
constexpr int   kWriteOff = 0;       // Q-rope: no cache-slot shift

constexpr std::uint32_t kBlock = 256;   // == ROPE_LOCAL

std::string defaultHsacoPath(const char* argv0) {
    std::filesystem::path exe = std::filesystem::canonical(argv0);
    return (exe.parent_path() / "hsaco" / "rope_inplace.hsaco").string();
}

void fillRandom(std::vector<float>& v, std::uint32_t seed) {
    std::uint32_t x = seed;
    for (auto& f : v) {
        x = x * 1664525u + 1013904223u;
        const std::int32_t s = static_cast<std::int32_t>(x >> 8) - (1 << 23);
        f = static_cast<float>(s) / static_cast<float>(1 << 23);
    }
}

// CPU reference: same formula as the kernel, double precision. In-place
// on `y` (which starts as a copy of the input).
void ropeCpuRef(std::vector<float>& y,
                int seqLen, int numHeads, int headDim,
                int startPos, float base) {
    const int halfDim = headDim / 2;
    for (int p = 0; p < seqLen; ++p) {
        const double pos = static_cast<double>(startPos + p);
        for (int h = 0; h < numHeads; ++h) {
            const int headBase = (p * numHeads + h) * headDim;
            for (int i = 0; i < halfDim; ++i) {
                const double invDim = 1.0 / static_cast<double>(headDim);
                const double freq   = std::pow(static_cast<double>(base),
                                               -static_cast<double>(2 * i) * invDim);
                const double theta  = pos * freq;
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

    const int total = kSeqLen * kNumHeads * kHeadDim;
    const int totalWork = kSeqLen * kNumHeads * (kHeadDim / 2);

    std::printf("hip_rope_probe:\n  hsaco: %s\n"
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
        HipKernel kernel = mod.getKernel("rope_inplace");

        // ---- host tensors -----------------------------------------------
        std::vector<float> input(static_cast<std::size_t>(total));
        fillRandom(input, /*seed=*/0xDEADBEEFu);

        std::vector<float> hostYRef = input;   // copy — CPU rotates in place
        ropeCpuRef(hostYRef, kSeqLen, kNumHeads, kHeadDim, kStartPos, kBase);

        std::vector<float> hostYGpu(static_cast<std::size_t>(total), 0.0f);

        // ---- device tensors ---------------------------------------------
        const std::size_t xBytes = static_cast<std::size_t>(total) * sizeof(float);
        HipBuffer devX{alloc, xBytes};
        HipBuffer devStart{alloc, sizeof(int), HipAllocKind::Device};

        alloc.copyH2D(devX.data(),     input.data(),      xBytes);
        alloc.copyH2D(devStart.data(), &kStartPos,        sizeof(int));

        // ---- launch -----------------------------------------------------
        kernel.setPtr  (0, devX.data());
        kernel.setValue(1, kSeqLen);
        kernel.setValue(2, kNumHeads);
        kernel.setValue(3, kHeadDim);
        kernel.setPtr  (4, devStart.data());
        kernel.setValue(5, kBase);
        kernel.setValue(6, kWriteOff);

        const std::uint32_t grid =
            (static_cast<std::uint32_t>(totalWork) + kBlock - 1) / kBlock;

        evStart.record(stream);
        kernel.launch(stream, grid, 1, 1, kBlock, 1, 1, /*shared=*/0);
        evEnd.record(stream);
        stream.synchronize();

        const float kernelMs = evEnd.elapsedMs(evStart);

        // ---- readback + compare -----------------------------------------
        alloc.copyD2H(hostYGpu.data(), devX.data(), xBytes);

        // Combined tolerance: `|diff| <= abs_tol + rel_tol * |ref|`.
        // Standard numerical-test pattern — handles values close to
        // zero (where a tiny absolute drift becomes a huge relative
        // one) without either false-positive-failing or false-
        // negative-passing kernels with real bugs.
        constexpr float kAbsTol = 1e-5f;
        constexpr float kRelTol = 1e-4f;

        float       maxAbs   = 0.0f;
        float       maxRatio = 0.0f;  // abs_diff / (abs_tol + rel_tol * |ref|)
        std::size_t badIdx   = SIZE_MAX;
        for (std::size_t i = 0; i < static_cast<std::size_t>(total); ++i) {
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
            std::printf("  worst @ idx=%zu: gpu=%.6g cpu=%.6g\n",
                        badIdx,
                        static_cast<double>(hostYGpu[badIdx]),
                        static_cast<double>(hostYRef[badIdx]));
        }

        const bool ok = (maxRatio <= 1.0f);
        std::printf("\nhip_rope_probe: %s\n", ok ? "OK" : "FAIL");
        return ok ? 0 : 1;

    } catch (const std::exception& e) {
        std::fprintf(stderr, "hip_rope_probe: threw: %s\n", e.what());
        return 2;
    }
}