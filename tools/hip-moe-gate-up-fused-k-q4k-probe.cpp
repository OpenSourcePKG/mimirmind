// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// hip_moe_gate_up_fused_k_q4k_probe — parity check for the fused MoE gate+up
// projection over K routed experts (Q4_K) on gfx1101. Mirrors the CUDA
// cuda_moe_gate_up_fused_k_q4k_parity test but goes through the HIP
// compute::hip::GpuMatmul::moeGateUpFusedKAsync entry point, so it verifies
// BOTH the ported kernel and the new HIP GpuMatmul wiring (kernel selection,
// argument order, launch geometry) in one shot.
//
// The weight banks are random Q4_K-shaped bytes; the CPU reference dequantises
// the SAME bytes with the exact super-block unpack the kernel uses
// (getScaleMinK4 + the pair/lane mapping), so parity isolates a device-side
// indexing or warp-reduce (wave32) divergence — it does not depend on any
// other matmul kernel being correct.

#include "compute/hip/GpuMatmul.hpp"
#include "compute/hip/GpuOps.hpp"
#include "core/gpu/hip/HipComputeContext.hpp"
#include "core/gguf/GgufTypes.hpp"

#include <hip/hip_fp16.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <vector>

namespace {

using ::mimirmind::compute::hip::GpuMatmul;
using ::mimirmind::compute::hip::GpuOps;
using ::mimirmind::core::hip::HipComputeContext;
using GgmlType = ::mimirmind::core::gguf::GgmlType;

constexpr int kBlockElems = 256;
constexpr int kBlockBytes = 144;

// LCG matching the other HIP probes — deterministic, no rand/Date.
struct Lcg {
    std::uint32_t s;
    float nextSigned(float scale) {
        s = s * 1664525u + 1013904223u;
        const std::int32_t v = static_cast<std::int32_t>(s >> 8) - (1 << 23);
        return scale * static_cast<float>(v) / static_cast<float>(1 << 23);
    }
    std::uint8_t nextByte() {
        s = s * 1664525u + 1013904223u;
        return static_cast<std::uint8_t>(s >> 24);
    }
};

// Fill a stack of Q4_K super-blocks with plausible random content:
// fp16 d/dmin (small positive) + random 6-bit-packed scales + random qs.
void fillRandomQ4KBank(std::vector<std::uint8_t>& bytes, std::uint32_t seed) {
    Lcg g{seed};
    const std::size_t nBlocks = bytes.size() / kBlockBytes;
    for (std::size_t b = 0; b < nBlocks; ++b) {
        std::uint8_t* blk = bytes.data() + b * kBlockBytes;
        const float dS    = 0.001f + 0.02f * (static_cast<float>(g.nextByte()) / 255.0f);
        const float dminS = 0.001f + 0.02f * (static_cast<float>(g.nextByte()) / 255.0f);
        const __half dH    = __float2half(dS);
        const __half dminH = __float2half(dminS);
        std::memcpy(blk,     &dH,    sizeof(__half));
        std::memcpy(blk + 2, &dminH, sizeof(__half));
        for (int i = 4; i < kBlockBytes; ++i) {
            blk[i] = g.nextByte();
        }
    }
}

// CPU mirror of the kernel's getScaleMinK4 — 6-bit scale/min unpack.
void getScaleMinK4(int j, const std::uint8_t* q,
                   unsigned& scale, unsigned& min) {
    if (j < 4) {
        scale = static_cast<unsigned>(q[j]     & 0x3Fu);
        min   = static_cast<unsigned>(q[j + 4] & 0x3Fu);
    } else {
        scale = static_cast<unsigned>((q[j + 4] & 0x0Fu) | ((q[j - 4] >> 6) << 4));
        min   = static_cast<unsigned>((q[j + 4] >> 4)    | ((q[j    ] >> 6) << 4));
    }
}

// Dequant-and-dot one Q4_K weight row (nBlocks super-blocks) against x.
// Exact mirror of the device q4kBlockDot summed over the 32 lanes.
double q4kRowDot(const std::uint8_t* row, const float* x, int dModel) {
    const int nBlocks = dModel / kBlockElems;
    double acc = 0.0;
    for (int b = 0; b < nBlocks; ++b) {
        const std::uint8_t* blk = row + static_cast<std::size_t>(b) * kBlockBytes;
        __half dH, dminH;
        std::memcpy(&dH,    blk,     sizeof(__half));
        std::memcpy(&dminH, blk + 2, sizeof(__half));
        const float d    = __half2float(dH);
        const float dmin = __half2float(dminH);
        const std::uint8_t* scales = blk + 4;
        const std::uint8_t* qs     = blk + 16;
        const float* xBase = x + static_cast<std::size_t>(b) * kBlockElems;

        for (int pair = 0; pair < 4; ++pair) {
            const int jLo = 2 * pair;
            const int jHi = jLo + 1;
            unsigned sLo, mLo, sHi, mHi;
            getScaleMinK4(jLo, scales, sLo, mLo);
            getScaleMinK4(jHi, scales, sHi, mHi);
            const float dLo = d * static_cast<float>(sLo);
            const float mLoF = dmin * static_cast<float>(mLo);
            const float dHi = d * static_cast<float>(sHi);
            const float mHiF = dmin * static_cast<float>(mHi);
            for (int l = 0; l < 32; ++l) {
                const std::uint8_t qb = qs[pair * 32 + l];
                const float wLo = dLo * static_cast<float>(qb & 0x0Fu) - mLoF;
                const float wHi = dHi * static_cast<float>(qb >> 4)    - mHiF;
                acc += static_cast<double>(xBase[jLo * 32 + l]) * wLo;
                acc += static_cast<double>(xBase[jHi * 32 + l]) * wHi;
            }
        }
    }
    return acc;
}

struct Config {
    int           dModel;
    int           nFf;
    int           kActive;
    int           nExperts;
    std::uint32_t seed;
};

bool runConfig(GpuOps& ops, GpuMatmul& gmm, const Config& c) {
    const std::size_t rowBytes   = static_cast<std::size_t>(c.dModel / kBlockElems) * kBlockBytes;
    const std::size_t bytesGate  = static_cast<std::size_t>(c.nFf) * rowBytes;
    const std::size_t bankBytes  = static_cast<std::size_t>(c.nExperts) * bytesGate;

    Lcg g{c.seed};
    std::vector<float> x(c.dModel);
    for (auto& v : x) v = g.nextSigned(0.5f);

    std::vector<std::uint8_t> gateBank(bankBytes), upBank(bankBytes);
    fillRandomQ4KBank(gateBank, c.seed ^ 0xA1A1u);
    fillRandomQ4KBank(upBank,   c.seed ^ 0xB2B2u);

    std::vector<std::int32_t> expIdx(c.kActive);
    for (int k = 0; k < c.kActive; ++k) {
        expIdx[k] = static_cast<std::int32_t>(g.nextByte() % c.nExperts);
    }

    // CPU reference: silu(Wg[e].x) * (Wu[e].x) into K-strided [K, nFf].
    std::vector<float> ref(static_cast<std::size_t>(c.kActive) * c.nFf);
    for (int k = 0; k < c.kActive; ++k) {
        const std::size_t e = static_cast<std::size_t>(expIdx[k]);
        for (int f = 0; f < c.nFf; ++f) {
            const std::uint8_t* gRow = gateBank.data() + e * bytesGate + static_cast<std::size_t>(f) * rowBytes;
            const std::uint8_t* uRow = upBank.data()   + e * bytesGate + static_cast<std::size_t>(f) * rowBytes;
            const float gate = static_cast<float>(q4kRowDot(gRow, x.data(), c.dModel));
            const float up   = static_cast<float>(q4kRowDot(uRow, x.data(), c.dModel));
            const float silu = gate / (1.0f + std::exp(-gate));
            ref[static_cast<std::size_t>(k) * c.nFf + f] = silu * up;
        }
    }

    // Device: upload + one fused launch through the GpuMatmul wiring.
    auto dX  = ops.allocate(x.size() * sizeof(float));
    auto dWg = ops.allocate(bankBytes);
    auto dWu = ops.allocate(bankBytes);
    auto dE  = ops.allocate(expIdx.size() * sizeof(std::int32_t));
    auto dY  = ops.allocate(ref.size() * sizeof(float));
    ops.uploadHostBytes(dX.get(),  x.data(),       x.size() * sizeof(float));
    ops.uploadHostBytes(dWg.get(), gateBank.data(), bankBytes);
    ops.uploadHostBytes(dWu.get(), upBank.data(),   bankBytes);
    ops.uploadHostBytes(dE.get(),  expIdx.data(),   expIdx.size() * sizeof(std::int32_t));

    gmm.moeGateUpFusedKAsync(GgmlType::Q4_K,
                             static_cast<const float*>(dX.get()),
                             dWg.get(), dWu.get(),
                             static_cast<const std::int32_t*>(dE.get()),
                             static_cast<float*>(dY.get()),
                             static_cast<std::size_t>(c.dModel),
                             static_cast<std::size_t>(c.nFf),
                             static_cast<std::size_t>(c.kActive),
                             bytesGate, bytesGate);
    gmm.sync();

    std::vector<float> got(ref.size());
    ops.readbackToHost(got.data(), dY.get(), got.size() * sizeof(float));

    constexpr float kAbsTol = 2e-3f;
    constexpr float kRelTol = 2e-3f;
    float maxRatio = 0.0f;
    int   printed  = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const float diff = std::fabs(got[i] - ref[i]);
        const float thr  = kAbsTol + kRelTol * std::fabs(ref[i]);
        const float r    = diff / thr;
        if (r > maxRatio) maxRatio = r;
        if (r > 1.0f && printed < 5) {
            std::printf("    Y[%zu] gpu=%.6f cpu=%.6f\n", i, got[i], ref[i]);
            ++printed;
        }
    }
    const bool ok = (maxRatio <= 1.0f);
    std::printf("  dModel=%d nFf=%d K=%d nExp=%d seed=%#x : maxErr/tol=%.3f  %s\n",
                c.dModel, c.nFf, c.kActive, c.nExperts, c.seed,
                static_cast<double>(maxRatio), ok ? "OK" : "FAIL");
    return ok;
}

} // namespace

int main() {
    std::printf("hip_moe_gate_up_fused_k_q4k_probe:\n");
    try {
        HipComputeContext ctx{};
        GpuOps    ops{ctx};
        GpuMatmul gmm{ctx, ops};

        if (!gmm.moeGateUpFusedKAvailable(GgmlType::Q4_K)) {
            std::fprintf(stderr, "  moeGateUpFusedKAvailable(Q4_K) == false — "
                                 "kernel not loaded\n");
            return 1;
        }

        const Config configs[] = {
            {/*dModel=*/256, /*nFf=*/256, /*K=*/2, /*nExp=*/4, 0x0C0Fu},
            {/*dModel=*/512, /*nFf=*/128, /*K=*/3, /*nExp=*/6, 0xBEE5u},  // 2 super-blocks/row
            {/*dModel=*/256, /*nFf=*/ 96, /*K=*/1, /*nExp=*/8, 0x51EDu},  // K=1, nFf not mult of 4
        };

        bool allOk = true;
        for (const auto& c : configs) {
            allOk &= runConfig(ops, gmm, c);
        }

        std::printf("\nhip_moe_gate_up_fused_k_q4k_probe: %s\n", allOk ? "OK" : "FAIL");
        return allOk ? 0 : 1;

    } catch (const std::exception& e) {
        std::fprintf(stderr, "hip_moe_gate_up_fused_k_q4k_probe: threw: %s\n", e.what());
        return 2;
    }
}
