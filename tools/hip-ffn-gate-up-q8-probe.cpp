// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// hip_ffn_gate_up_q8_probe — parity check for the fused shared-expert gate+up
// kernel (kernels_hip/ffn_gate_up_fused_q8_0.hip) on gfx1101. Computes
//
//   Y[n] = silu( sum_k Wg[n,k]*x[k] ) * ( sum_k Wu[n,k]*x[k] )
//
// with native Q8_0 gate/up weights and compares against a CPU reference that
// dequantises the SAME bytes. Self-contained: it builds the Q8_0 rows inline
// (fp16 scale via __float2half + 32 int8 quants) and dequantises them the same
// way the kernel does (__half2float * q), so the only thing under test is the
// device dot-product + warp reduce + SiLU, not the quantiser. The 32-lane warp
// reduce is the wave32-sensitive part — a CDNA-style width assumption would
// surface here as a systematic mismatch.
//
// Combined-tolerance gate (|diff| <= abs + rel*|ref|); the CPU reference sums
// in double so it does not track the device's float accumulation order exactly,
// hence a loose-ish 1e-2 gate (Q8_0 dynamic range is ~1e2 per output).

#include "core/gpu/hip/HipContext.hpp"
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
#include <string>
#include <vector>

namespace {

using namespace mimirmind::core::hip;

constexpr int kBlockElems = 32;
constexpr int kBlockBytes = 34;

std::string defaultHsacoPath(const char* argv0) {
    std::filesystem::path exe = std::filesystem::canonical(argv0);
    return (exe.parent_path() / "hsaco" / "ffn_gate_up_fused_q8_0.hsaco").string();
}

// Deterministic pseudo-random in [-1, 1) — reproducible, no rand/Date.
struct Lcg {
    std::uint32_t s;
    float next() {
        s = s * 1664525u + 1013904223u;
        return static_cast<float>(s >> 8) / 16777216.0f * 2.0f - 1.0f;
    }
};

// Quantize K f32 values into native Q8_0 (fp16 d + 32 int8) at `dst`, and write
// the dequantised values back into `deq` so the CPU reference reads the exact
// lossy f32 the kernel reconstructs via __half2float(d) * q.
void quantizeQ8(const float* src, int K, unsigned char* dst, float* deq) {
    const int nBlocks = K / kBlockElems;
    for (int b = 0; b < nBlocks; ++b) {
        const float* w = src + b * kBlockElems;
        float amax = 0.0f;
        for (int i = 0; i < kBlockElems; ++i) {
            amax = std::fmax(amax, std::fabs(w[i]));
        }
        const float d  = amax / 127.0f;
        const float id = d > 0.0f ? 1.0f / d : 0.0f;

        const __half hd = __float2half(d);
        const float  dq = __half2float(hd);

        unsigned char* blk = dst + b * kBlockBytes;
        std::memcpy(blk, &hd, 2);
        signed char* qs = reinterpret_cast<signed char*>(blk + 2);
        for (int i = 0; i < kBlockElems; ++i) {
            int q = static_cast<int>(std::lround(w[i] * id));
            if (q >  127) q =  127;
            if (q < -127) q = -127;
            qs[i]                    = static_cast<signed char>(q);
            deq[b * kBlockElems + i] = dq * static_cast<float>(q);
        }
    }
}

struct Config {
    int           K;   // dModel
    int           N;   // nFf
    std::uint32_t seed;
};

bool runConfig(HipMemoryAllocator& alloc, HipStream& stream, HipKernel& kernel,
               const Config& c) {
    const int K        = c.K;
    const int N        = c.N;
    const int nBlocks  = K / kBlockElems;
    const int rowBytes = nBlocks * kBlockBytes;

    Lcg g{c.seed};
    std::vector<float> x(K);
    for (auto& v : x) v = g.next();
    std::vector<float> wg(static_cast<std::size_t>(N) * K);
    std::vector<float> wu(static_cast<std::size_t>(N) * K);
    for (auto& v : wg) v = g.next() * 0.5f;
    for (auto& v : wu) v = g.next() * 0.5f;

    std::vector<unsigned char> qg(static_cast<std::size_t>(N) * rowBytes);
    std::vector<unsigned char> qu(static_cast<std::size_t>(N) * rowBytes);
    std::vector<float>         dg(static_cast<std::size_t>(N) * K);
    std::vector<float>         du(static_cast<std::size_t>(N) * K);
    for (int n = 0; n < N; ++n) {
        quantizeQ8(wg.data() + static_cast<std::size_t>(n) * K, K,
                   qg.data() + static_cast<std::size_t>(n) * rowBytes,
                   dg.data() + static_cast<std::size_t>(n) * K);
        quantizeQ8(wu.data() + static_cast<std::size_t>(n) * K, K,
                   qu.data() + static_cast<std::size_t>(n) * rowBytes,
                   du.data() + static_cast<std::size_t>(n) * K);
    }

    // CPU golden reference — accumulate in double, mirror the kernel's math.
    std::vector<float> ref(N);
    for (int n = 0; n < N; ++n) {
        double gsum = 0.0, usum = 0.0;
        for (int k = 0; k < K; ++k) {
            gsum += static_cast<double>(dg[static_cast<std::size_t>(n) * K + k]) * x[k];
            usum += static_cast<double>(du[static_cast<std::size_t>(n) * K + k]) * x[k];
        }
        const float gf   = static_cast<float>(gsum);
        const float silu = gf / (1.0f + std::exp(-gf));
        ref[n] = silu * static_cast<float>(usum);
    }

    // Device buffers.
    HipBuffer dX {alloc, static_cast<std::size_t>(K) * sizeof(float)};
    HipBuffer dWg{alloc, static_cast<std::size_t>(N) * rowBytes};
    HipBuffer dWu{alloc, static_cast<std::size_t>(N) * rowBytes};
    HipBuffer dY {alloc, static_cast<std::size_t>(N) * sizeof(float)};
    alloc.copyH2D(dX.data(),  x.data(),  static_cast<std::size_t>(K) * sizeof(float));
    alloc.copyH2D(dWg.data(), qg.data(), static_cast<std::size_t>(N) * rowBytes);
    alloc.copyH2D(dWu.data(), qu.data(), static_cast<std::size_t>(N) * rowBytes);

    kernel.setPtr  (0, dX.data());
    kernel.setPtr  (1, dWg.data());
    kernel.setPtr  (2, dWu.data());
    kernel.setPtr  (3, dY.data());
    kernel.setValue(4, static_cast<std::int32_t>(K));
    kernel.setValue(5, static_cast<std::int32_t>(N));

    constexpr std::uint32_t kOutputsPerGroup = 4;   // FFN_GU_Q8_LOCAL/32
    constexpr std::uint32_t kBlock           = 128;
    const std::uint32_t grid = (static_cast<std::uint32_t>(N) + kOutputsPerGroup - 1)
                               / kOutputsPerGroup;
    kernel.launch(stream, grid, 1, 1, kBlock, 1, 1, /*shared=*/0);
    stream.synchronize();

    std::vector<float> gpu(N);
    alloc.copyD2H(gpu.data(), dY.data(), static_cast<std::size_t>(N) * sizeof(float));

    constexpr float kAbsTol = 1e-2f;
    constexpr float kRelTol = 1e-2f;
    float maxRatio = 0.0f;
    int   printed  = 0;
    for (int n = 0; n < N; ++n) {
        const float diff = std::fabs(gpu[n] - ref[n]);
        const float thr  = kAbsTol + kRelTol * std::fabs(ref[n]);
        const float r    = diff / thr;
        if (r > maxRatio) maxRatio = r;
        if (r > 1.0f && printed < 5) {
            std::printf("    Y[%d] gpu=%.5f cpu=%.5f\n", n, gpu[n], ref[n]);
            ++printed;
        }
    }
    const bool ok = (maxRatio <= 1.0f);
    std::printf("  K=%d N=%d seed=%#x : maxErr/tol=%.3f  %s\n",
                K, N, c.seed, static_cast<double>(maxRatio), ok ? "OK" : "FAIL");
    return ok;
}

} // namespace

int main(int argc, char** argv) {
    const std::string hsacoPath =
        (argc > 1) ? std::string{argv[1]} : defaultHsacoPath(argv[0]);

    std::printf("hip_ffn_gate_up_q8_probe:\n  hsaco: %s\n", hsacoPath.c_str());

    try {
        HipContext         ctx{};
        HipMemoryAllocator alloc{ctx};
        HipStream          stream{ctx, HipStreamKind::BlockingDefault};

        HipModule mod    = HipModule::fromFile(ctx, hsacoPath);
        HipKernel kernel = mod.getKernel("ffn_gate_up_fused_q8_0");

        const Config configs[] = {
            {/*K=*/ 256, /*N=*/10, 0xF00Du},   // N not a multiple of 4 -> active guard
            {/*K=*/2048, /*N=*/64, 0xBEEFu},   // two X tiles, decode shared-expert shape
            {/*K=*/1024, /*N=*/ 1, 0x5EEDu},   // single output row
        };

        bool allOk = true;
        for (const auto& c : configs) {
            allOk &= runConfig(alloc, stream, kernel, c);
        }

        std::printf("\nhip_ffn_gate_up_q8_probe: %s\n", allOk ? "OK" : "FAIL");
        return allOk ? 0 : 1;

    } catch (const std::exception& e) {
        std::fprintf(stderr, "hip_ffn_gate_up_q8_probe: threw: %s\n", e.what());
        return 2;
    }
}
