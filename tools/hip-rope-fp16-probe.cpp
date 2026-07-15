// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// hip_rope_fp16_probe — parity check for the fp16-KV RoPE variant.
// Loads rope_inplace_fp16.hsaco, applies rotary embedding to a
// synthetic [seqLen, numHeads, headDim] fp16 tensor, compares
// against an inline double-precision CPU reference (rotation done
// in fp64, result quantised through the fp16 round-trip that the
// kernel produces on store).
//
// Combined-tolerance envelope is looser than the fp32 rope probe
// because the fp16 store on the two writes introduces ~2^-11
// quantisation on top of the sin/cos + multiply-subtract drift.

#include "core/gpu/hip/HipContext.hpp"
#include "core/gpu/hip/HipEvent.hpp"
#include "core/gpu/hip/HipKernel.hpp"
#include "core/gpu/hip/HipMemoryAllocator.hpp"
#include "core/gpu/hip/HipModule.hpp"
#include "core/gpu/hip/HipStream.hpp"

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

constexpr int   kSeqLen   = 8;
constexpr int   kNumHeads = 4;
constexpr int   kHeadDim  = 64;
constexpr int   kStartPos = 5;
constexpr float kBase     = 10000.0f;
constexpr int   kWriteOff = 0;

constexpr std::uint32_t kBlock = 256;   // == ROPE_LOCAL

std::string defaultHsacoPath(const char* argv0) {
    std::filesystem::path exe = std::filesystem::canonical(argv0);
    return (exe.parent_path() / "hsaco" / "rope_inplace_fp16.hsaco").string();
}

void fillRandom(std::vector<float>& v, std::uint32_t seed) {
    std::uint32_t x = seed;
    for (auto& f : v) {
        x = x * 1664525u + 1013904223u;
        const std::int32_t s = static_cast<std::int32_t>(x >> 8) - (1 << 23);
        f = static_cast<float>(s) / static_cast<float>(1 << 23);
    }
}

// Host-side fp32↔fp16 round-trip. Uses IEEE 754 half-precision
// conversion — same rounding mode (round-to-nearest even) that the
// GPU __float2half intrinsic uses. Avoids pulling in <hip/hip_fp16.h>
// on the host build (which would tie the probe to HIP for the CPU
// reference too).
std::uint16_t fp32ToFp16(float f) {
    std::uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const std::uint32_t sign  = (x >> 16) & 0x8000u;
    std::int32_t        exp   = static_cast<std::int32_t>((x >> 23) & 0xFF) - 127 + 15;
    std::uint32_t       mant  = x & 0x7FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return static_cast<std::uint16_t>(sign);
        mant = (mant | 0x800000u) >> static_cast<std::uint32_t>(1 - exp);
        // Round to nearest even.
        if (mant & 0x1000u) mant += 0x2000u;
        return static_cast<std::uint16_t>(sign | (mant >> 13));
    } else if (exp >= 31) {
        return static_cast<std::uint16_t>(sign | 0x7C00u |
                                          ((x & 0x7FFFFFu) ? 0x200u : 0));
    }
    if (mant & 0x1000u) {
        mant += 0x2000u;
        if (mant & 0x800000u) { mant = 0; ++exp; if (exp >= 31)
            return static_cast<std::uint16_t>(sign | 0x7C00u); }
    }
    return static_cast<std::uint16_t>(sign |
                                      (static_cast<std::uint32_t>(exp) << 10) |
                                      (mant >> 13));
}

float fp16ToFp32(std::uint16_t h) {
    const std::uint32_t sign = (h & 0x8000u) << 16;
    std::uint32_t       exp  = (h & 0x7C00u) >> 10;
    std::uint32_t       mant = (h & 0x03FFu);
    if (exp == 0) {
        if (mant == 0) {
            std::uint32_t r = sign;
            float out;
            std::memcpy(&out, &r, sizeof(out));
            return out;
        }
        while ((mant & 0x0400u) == 0) { mant <<= 1; --exp; }
        ++exp; mant &= 0x03FFu;
    } else if (exp == 31) {
        std::uint32_t r = sign | 0x7F800000u | (mant << 13);
        float out;
        std::memcpy(&out, &r, sizeof(out));
        return out;
    }
    exp += (127 - 15);
    std::uint32_t r = sign | (exp << 23) | (mant << 13);
    float out;
    std::memcpy(&out, &r, sizeof(out));
    return out;
}

// CPU reference: rotation in fp64, each write round-tripped through
// fp16 to match what the kernel stores.
void ropeFp16CpuRef(std::vector<std::uint16_t>& y,   // in+out fp16
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
                const double a = static_cast<double>(
                    fp16ToFp32(y[headBase + i]));
                const double b = static_cast<double>(
                    fp16ToFp32(y[headBase + i + halfDim]));
                y[headBase + i]           =
                    fp32ToFp16(static_cast<float>(a * c - b * s));
                y[headBase + i + halfDim] =
                    fp32ToFp16(static_cast<float>(a * s + b * c));
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

    std::printf("hip_rope_fp16_probe:\n  hsaco: %s\n"
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
        HipKernel kernel = mod.getKernel("rope_inplace_fp16");

        // Deterministic fp32 input, quantised through fp16 so the
        // kernel and the CPU reference start from the exact same fp16
        // bits.
        std::vector<float>         inputF32(static_cast<std::size_t>(total));
        std::vector<std::uint16_t> inputFp16(static_cast<std::size_t>(total));
        fillRandom(inputF32, /*seed=*/0xDEADBEEFu);
        for (std::size_t i = 0; i < inputF32.size(); ++i) {
            inputFp16[i] = fp32ToFp16(inputF32[i]);
        }

        std::vector<std::uint16_t> hostRef = inputFp16;
        ropeFp16CpuRef(hostRef, kSeqLen, kNumHeads, kHeadDim,
                       kStartPos, kBase);

        std::vector<std::uint16_t> hostGpu(static_cast<std::size_t>(total), 0);

        const std::size_t xBytes =
            static_cast<std::size_t>(total) * sizeof(std::uint16_t);
        HipBuffer devX{alloc, xBytes};
        HipBuffer devStart{alloc, sizeof(int), HipAllocKind::Device};

        alloc.copyH2D(devX.data(),     inputFp16.data(),  xBytes);
        alloc.copyH2D(devStart.data(), &kStartPos,        sizeof(int));

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

        alloc.copyD2H(hostGpu.data(), devX.data(), xBytes);

        // Compare in fp32 space (both sides go through the same fp16
        // round-trip so the drift is entirely from sin/cos + fp32
        // multiply-subtract rounding before the fp16 store).
        constexpr float kAbsTol = 5e-4f;
        constexpr float kRelTol = 5e-3f;

        float       maxAbs   = 0.0f;
        float       maxRatio = 0.0f;
        std::size_t badIdx   = SIZE_MAX;
        for (std::size_t i = 0; i < static_cast<std::size_t>(total); ++i) {
            const float g = fp16ToFp32(hostGpu[i]);
            const float r = fp16ToFp32(hostRef[i]);
            const float d = std::fabs(g - r);
            const float threshold = kAbsTol + kRelTol * std::fabs(r);
            const float ratio = d / threshold;
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
                        static_cast<double>(fp16ToFp32(hostGpu[badIdx])),
                        static_cast<double>(fp16ToFp32(hostRef[badIdx])));
        }

        const bool ok = (maxRatio <= 1.0f);
        std::printf("\nhip_rope_fp16_probe: %s\n", ok ? "OK" : "FAIL");
        return ok ? 0 : 1;

    } catch (const std::exception& e) {
        std::fprintf(stderr, "hip_rope_fp16_probe: threw: %s\n", e.what());
        return 2;
    }
}