// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// hip_mrope_probe — parity check for the IMRoPE (rope_mrope) kernel port,
// exercising the PARTIAL-ROTARY path (Qwen3-Next / Qwen3.5-MoE,
// partial_rotary_factor = 0.25). Loads rope_mrope.hsaco, applies IMRoPE to
// a synthetic [seqLen, numHeads, headDim] tensor with dimension sections
// summing to rotary_dim/2, and compares against an inline double-precision
// CPU reference.
//
// head_dim = 256, so only rotary_dim = 64 head dims (= 32 rotated pairs)
// rotate and the remaining 192 head dims pass through untouched; the freq
// denominator is rotary_dim (64), NOT head_dim (256). This guards the
// norm-blind directional bug that shipped incoherent long-gen on
// Qwen3.6-NVFP4 (the kernel used to rotate all 128 pairs with denom 512).
// Mirrors the L0 (gpu_tests) and CUDA (cuda_parity_tests) partial-rotary
// cases. Beyond CPU-vs-GPU parity it asserts the pass-through invariant
// directly, so BOTH paths regressing to full-head rotation together is
// still caught.
//
// Uses writeOffsetStride=0 (contiguous Q-rope shape), same as
// hip_rope_probe.

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

constexpr int   kSeqLen   = 3;
constexpr int   kNumHeads = 2;
constexpr int   kHeadDim  = 256;     // Qwen3-Next attention head_dim; halfDim = 128
constexpr int   kStartPos = 7;       // arbitrary non-zero to exercise the offset path
constexpr float kBase     = 10000.0f;
constexpr int   kWriteOff = 0;       // Q-rope: no cache-slot shift

// GGUF <arch>.rope.dimension_sections (time/height/width/extra). The sum is
// rotary_dim/2, so 16+8+8+0 = 32 rotated pairs -> rotary_dim = 64 head dims.
constexpr int   kSec0 = 16;
constexpr int   kSec1 = 8;
constexpr int   kSec2 = 8;
constexpr int   kSec3 = 0;
constexpr int   kSectDims  = kSec0 + kSec1 + kSec2 + kSec3;   // 32
constexpr int   kRotPairs  = kSectDims;                       // rotated pairs
constexpr int   kRotaryDim = 2 * kRotPairs;                   // 64 head dims

constexpr std::uint32_t kBlock = 256;   // == ROPE_LOCAL

std::string defaultHsacoPath(const char* argv0) {
    std::filesystem::path exe = std::filesystem::canonical(argv0);
    return (exe.parent_path() / "hsaco" / "rope_mrope.hsaco").string();
}

void fillRandom(std::vector<float>& v, std::uint32_t seed) {
    std::uint32_t x = seed;
    for (auto& f : v) {
        x = x * 1664525u + 1013904223u;
        const std::int32_t s = static_cast<std::int32_t>(x >> 8) - (1 << 23);
        f = static_cast<float>(s) / static_cast<float>(1 << 23);
    }
}

// CPU reference: same partial-rotary formula as the kernel, double
// precision. Text-only positions -> all four IMRoPE axes equal the sequence
// position, so the sector axis-selection collapses to a single position and
// only the pair count / partner offset / freq denominator matter. In-place
// on `y` (which starts as a copy of the input); head dims [rotaryDim,
// headDim) are never written (pass-through).
void mropeCpuRef(std::vector<float>& y,
                 int seqLen, int numHeads, int headDim,
                 int startPos, float base, int sectDims) {
    const int rotPairs  = (sectDims > 0) ? sectDims : headDim / 2;
    const int rotaryDim = 2 * rotPairs;
    for (int p = 0; p < seqLen; ++p) {
        const double pos = static_cast<double>(startPos + p);
        for (int h = 0; h < numHeads; ++h) {
            const int headBase = (p * numHeads + h) * headDim;
            for (int i = 0; i < rotPairs; ++i) {
                const double invDim = 1.0 / static_cast<double>(rotaryDim);
                const double freq   = std::pow(static_cast<double>(base),
                                               -static_cast<double>(2 * i) * invDim);
                const double theta  = pos * freq;
                const double c = std::cos(theta);
                const double s = std::sin(theta);
                const double a = static_cast<double>(y[headBase + i]);
                const double b = static_cast<double>(y[headBase + i + rotPairs]);
                y[headBase + i]            = static_cast<float>(a * c - b * s);
                y[headBase + i + rotPairs] = static_cast<float>(a * s + b * c);
            }
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    const std::string hsacoPath =
        (argc > 1) ? std::string{argv[1]} : defaultHsacoPath(argv[0]);

    const int total     = kSeqLen * kNumHeads * kHeadDim;
    const int totalWork = kSeqLen * kNumHeads * (kHeadDim / 2);  // one thread per pair

    std::printf("hip_mrope_probe:\n  hsaco: %s\n"
                "  seqLen=%d numHeads=%d headDim=%d startPos=%d base=%.0f\n"
                "  sections={%d,%d,%d,%d} sectDims=%d rotaryDim=%d (partial_rotary=0.25)\n",
                hsacoPath.c_str(), kSeqLen, kNumHeads, kHeadDim, kStartPos,
                static_cast<double>(kBase),
                kSec0, kSec1, kSec2, kSec3, kSectDims, kRotaryDim);

    try {
        HipContext         ctx{};
        HipMemoryAllocator alloc{ctx};
        HipStream          stream{ctx, HipStreamKind::BlockingDefault};
        HipEvent           evStart{ctx};
        HipEvent           evEnd  {ctx};

        HipModule mod    = HipModule::fromFile(ctx, hsacoPath);
        HipKernel kernel = mod.getKernel("rope_mrope");

        // ---- host tensors -----------------------------------------------
        std::vector<float> input(static_cast<std::size_t>(total));
        fillRandom(input, /*seed=*/0x63C0FFEEu);

        std::vector<float> hostYRef = input;   // copy — CPU rotates in place
        mropeCpuRef(hostYRef, kSeqLen, kNumHeads, kHeadDim, kStartPos, kBase,
                    kSectDims);

        std::vector<float> hostYGpu(static_cast<std::size_t>(total), 0.0f);

        // ---- device tensors ---------------------------------------------
        const std::size_t xBytes = static_cast<std::size_t>(total) * sizeof(float);
        HipBuffer devX{alloc, xBytes};
        HipBuffer devStart{alloc, sizeof(int), HipAllocKind::Device};

        alloc.copyH2D(devX.data(),     input.data(), xBytes);
        alloc.copyH2D(devStart.data(), &kStartPos,   sizeof(int));

        // ---- launch (arg order matches kernels/hip/llm/rope_mrope.hip) --
        kernel.setPtr  (0,  devX.data());
        kernel.setValue(1,  kSeqLen);
        kernel.setValue(2,  kNumHeads);
        kernel.setValue(3,  kHeadDim);
        kernel.setPtr  (4,  devStart.data());
        kernel.setValue(5,  kBase);
        kernel.setValue(6,  kWriteOff);
        kernel.setValue(7,  kSec0);
        kernel.setValue(8,  kSec1);
        kernel.setValue(9,  kSec2);
        kernel.setValue(10, kSec3);

        const std::uint32_t grid =
            (static_cast<std::uint32_t>(totalWork) + kBlock - 1) / kBlock;

        evStart.record(stream);
        kernel.launch(stream, grid, 1, 1, kBlock, 1, 1, /*shared=*/0);
        evEnd.record(stream);
        stream.synchronize();

        const float kernelMs = evEnd.elapsedMs(evStart);

        // ---- readback + compare -----------------------------------------
        alloc.copyD2H(hostYGpu.data(), devX.data(), xBytes);

        // Combined tolerance: `|diff| <= abs_tol + rel_tol * |ref|` — the
        // same near-zero-safe pattern as hip_rope_probe.
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

        // Reference-independent invariants: head dims [rotaryDim, headDim)
        // must be byte-identical to the input (pass-through), and the rotary
        // block [0, rotaryDim) must actually have changed.
        float passMaxDiff = 0.0f;
        bool  rotaryMoved = false;
        for (int p = 0; p < kSeqLen; ++p) {
            for (int h = 0; h < kNumHeads; ++h) {
                const int headBase = (p * kNumHeads + h) * kHeadDim;
                for (int d = kRotaryDim; d < kHeadDim; ++d) {
                    passMaxDiff = std::max(passMaxDiff,
                        std::fabs(hostYGpu[headBase + d] - input[headBase + d]));
                }
                for (int d = 0; d < kRotaryDim; ++d) {
                    if (std::fabs(hostYGpu[headBase + d] - input[headBase + d]) > 1e-4f) {
                        rotaryMoved = true;
                    }
                }
            }
        }

        std::printf("\n  kernel:        %.3f ms\n", static_cast<double>(kernelMs));
        std::printf("  max abs err:   %.3e\n", static_cast<double>(maxAbs));
        std::printf("  max err / tol: %.3f (fails if > 1.0)\n",
                    static_cast<double>(maxRatio));
        std::printf("  pass-through max diff: %.3e (must be 0)\n",
                    static_cast<double>(passMaxDiff));
        std::printf("  rotary block moved:    %s\n", rotaryMoved ? "yes" : "NO");
        if (badIdx != SIZE_MAX) {
            std::printf("  worst @ idx=%zu: gpu=%.6g cpu=%.6g\n",
                        badIdx,
                        static_cast<double>(hostYGpu[badIdx]),
                        static_cast<double>(hostYRef[badIdx]));
        }

        const bool ok = (maxRatio <= 1.0f) && (passMaxDiff == 0.0f) && rotaryMoved;
        std::printf("\nhip_mrope_probe: %s\n", ok ? "OK" : "FAIL");
        return ok ? 0 : 1;

    } catch (const std::exception& e) {
        std::fprintf(stderr, "hip_mrope_probe: threw: %s\n", e.what());
        return 2;
    }
}
