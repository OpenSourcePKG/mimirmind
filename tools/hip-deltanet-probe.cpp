// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// hip_deltanet_probe — GPU-vs-CPU parity for the Qwen3-Next GatedDeltaNet HIP
// kernels (deltanet_gate + gated_deltanet_ar) on gfx1101. Ports the CUDA
// cuda_deltanet_gate_parity / cuda_gated_deltanet_ar_parity checks to HIP so
// the linear-attention decode path is verified on RDNA3 (wave32) — where a
// CDNA-derived kernel can silently mis-index. Runs each op against the
// compute::GatedDeltaNet golden reference on identical random inputs and
// returns 0 iff both agree within tolerance.

#include "compute/GatedDeltaNet.hpp"
#include "compute/hip/GpuOps.hpp"
#include "core/gpu/hip/HipComputeContext.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <vector>

namespace {

using ::mimirmind::compute::hip::GpuOps;
using ::mimirmind::core::hip::HipComputeContext;

// Same LCG family the CUDA parity harness uses — any deterministic stream
// works since both the CPU reference and the GPU op read the same values.
struct Lcg {
    std::uint32_t s;
    float next() {
        s = s * 1664525u + 1013904223u;
        return static_cast<float>(s >> 8) / 16777216.0f * 2.0f - 1.0f;
    }
};

std::vector<float> randVec(std::size_t n, std::uint32_t seed) {
    Lcg g{seed};
    std::vector<float> v(n);
    for (auto& x : v) x = g.next();
    return v;
}

::mimirmind::compute::ComputeBuffer toDevice(GpuOps& ops, const std::vector<float>& h) {
    auto buf = ops.allocate(h.size() * sizeof(float));
    ops.uploadHostBytes(buf.get(), h.data(), h.size() * sizeof(float));
    return buf;
}

std::vector<float> fromDevice(GpuOps& ops, const void* dev, std::size_t n) {
    std::vector<float> h(n);
    ops.readbackToHost(h.data(), dev, n * sizeof(float));
    return h;
}

int countMismatch(const std::vector<float>& got, const std::vector<float>& ref,
                  float tol, const char* tag) {
    int fails = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (std::fabs(got[i] - ref[i]) > tol) {
            if (fails < 5) {
                std::fprintf(stderr, "  %s[%zu] gpu=%.6f cpu=%.6f\n",
                             tag, i, got[i], ref[i]);
            }
            ++fails;
        }
    }
    std::fprintf(stderr, "%s: %d / %zu mismatch (tol=%g)\n", tag, fails,
                 got.size(), tol);
    return fails;
}

bool testDeltanetGate(GpuOps& ops) {
    const std::size_t T = 5, H = 8;
    auto alpha = randVec(T * H, 0x51u);
    auto ssmA  = randVec(H,     0x62u);
    auto ssmDt = randVec(H,     0x73u);

    std::vector<float> ref(T * H);
    ::mimirmind::compute::deltanetGate(alpha.data(), ssmA.data(), ssmDt.data(),
                                       ref.data(), T, H);

    auto da = toDevice(ops, alpha);
    auto dA = toDevice(ops, ssmA);
    auto dD = toDevice(ops, ssmDt);
    auto dg = ops.allocate(T * H * sizeof(float));
    ops.deltanetGateAsync(static_cast<const float*>(da.get()),
                          static_cast<const float*>(dA.get()),
                          static_cast<const float*>(dD.get()),
                          static_cast<float*>(dg.get()), T, H);
    ops.flush();
    auto got = fromDevice(ops, dg.get(), T * H);
    return countMismatch(got, ref, 1e-3f, "deltanet_gate") == 0;
}

bool testGatedDeltaNetAr(GpuOps& ops) {
    const std::size_t T = 4, H = 3, S = 16;
    auto q     = randVec(T * H * S, 0x0a1u);
    auto k     = randVec(T * H * S, 0x0b2u);
    auto v     = randVec(T * H * S, 0x0c3u);
    auto gLog  = randVec(T * H,     0x0d4u);
    auto beta  = randVec(T * H,     0x0e5u);
    auto state = randVec(H * S * S, 0x0f6u);

    std::vector<float> refState = state;
    std::vector<float> refOut(T * H * S);
    ::mimirmind::compute::gatedDeltaNetRecurrent(
        q.data(), k.data(), v.data(), gLog.data(), beta.data(),
        refState.data(), refOut.data(), T, H, S);

    auto dq = toDevice(ops, q);
    auto dk = toDevice(ops, k);
    auto dv = toDevice(ops, v);
    auto dg = toDevice(ops, gLog);
    auto db = toDevice(ops, beta);
    auto ds = toDevice(ops, state);
    auto dout = ops.allocate(T * H * S * sizeof(float));
    ops.gatedDeltaNetRecurrentAsync(
        static_cast<const float*>(dq.get()),
        static_cast<const float*>(dk.get()),
        static_cast<const float*>(dv.get()),
        static_cast<const float*>(dg.get()),
        static_cast<const float*>(db.get()),
        static_cast<float*>(ds.get()),
        static_cast<float*>(dout.get()),
        T, H, S);
    ops.flush();
    auto gotOut   = fromDevice(ops, dout.get(), T * H * S);
    auto gotState = fromDevice(ops, ds.get(),   H * S * S);

    int fails = countMismatch(gotOut, refOut, 2e-3f, "gdn_ar.out");
    fails += countMismatch(gotState, refState, 2e-3f, "gdn_ar.state");
    return fails == 0;
}

} // namespace

int main() {
    try {
        HipComputeContext ctx{};
        GpuOps ops{ctx};

        bool ok = true;
        ok = testDeltanetGate(ops) && ok;
        ok = testGatedDeltaNetAr(ops) && ok;

        if (ok) {
            std::fprintf(stderr, "hip_deltanet_probe: PASS\n");
            return 0;
        }
        std::fprintf(stderr, "hip_deltanet_probe: FAIL\n");
        return 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "hip_deltanet_probe: EXCEPTION %s\n", e.what());
        return 2;
    }
}
