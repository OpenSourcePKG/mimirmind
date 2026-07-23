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

#include "compute/ComputeOps.hpp"
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

bool testL2Norm(GpuOps& ops) {
    const std::size_t rows = 7, dim = 40;
    const float eps = 1e-6f;
    auto host = randVec(rows * dim, 0x1234u);
    std::vector<float> ref = host;
    ::mimirmind::compute::l2NormInPlace(ref.data(), rows, dim, eps);

    auto buf = toDevice(ops, host);
    ops.l2NormInPlaceAsync(static_cast<float*>(buf.get()), rows, dim, eps);
    ops.flush();
    auto got = fromDevice(ops, buf.get(), rows * dim);
    return countMismatch(got, ref, 1e-3f, "l2_norm") == 0;
}

bool testSsmConv1d(GpuOps& ops) {
    const std::size_t T = 5, channels = 12, K = 4;
    auto convInput = randVec((K - 1 + T) * channels, 0x2222u);
    auto kernel    = randVec(K * channels,           0x3333u);
    std::vector<float> ref(T * channels);
    ::mimirmind::compute::causalConv1dSilu(convInput.data(), kernel.data(),
                                           ref.data(), T, channels, K);

    auto dIn  = toDevice(ops, convInput);
    auto dKer = toDevice(ops, kernel);
    auto dOut = ops.allocate(T * channels * sizeof(float));
    ops.causalConv1dSiluAsync(static_cast<const float*>(dIn.get()),
                              static_cast<const float*>(dKer.get()),
                              static_cast<float*>(dOut.get()), T, channels, K);
    ops.flush();
    auto got = fromDevice(ops, dOut.get(), T * channels);
    return countMismatch(got, ref, 1e-3f, "ssm_conv1d") == 0;
}

bool testSigmoidInplace(GpuOps& ops) {
    const std::size_t n = 50;
    auto host = randVec(n, 0x84u);
    std::vector<float> ref = host;
    ::mimirmind::compute::sigmoidInPlace(ref.data(), n);

    auto buf = toDevice(ops, host);
    ops.sigmoidInPlaceAsync(static_cast<float*>(buf.get()), n);
    ops.flush();
    auto got = fromDevice(ops, buf.get(), n);
    return countMismatch(got, ref, 1e-4f, "sigmoid_inplace") == 0;
}

bool testGatherHeads(GpuOps& ops) {
    const std::size_t T = 2, convTotalWidth = 20, offset = 5;
    const std::size_t srcHeads = 2, dstHeads = 4, S = 3;
    auto src = randVec(T * convTotalWidth, 0x95u);
    std::vector<float> ref(T * dstHeads * S);
    ::mimirmind::compute::gatherHeadsFromChannels(
        src.data(), ref.data(), T, offset, srcHeads, dstHeads, S, convTotalWidth);

    auto dsrc = toDevice(ops, src);
    auto ddst = ops.allocate(T * dstHeads * S * sizeof(float));
    ops.gatherHeadsFromChannelsAsync(static_cast<const float*>(dsrc.get()),
                                     static_cast<float*>(ddst.get()),
                                     T, offset, srcHeads, dstHeads, S,
                                     convTotalWidth);
    ops.flush();
    auto got = fromDevice(ops, ddst.get(), T * dstHeads * S);
    return countMismatch(got, ref, 1e-6f, "gather_heads") == 0;
}

} // namespace

int main() {
    try {
        HipComputeContext ctx{};
        GpuOps ops{ctx};

        bool ok = true;
        ok = testDeltanetGate(ops) && ok;
        ok = testGatedDeltaNetAr(ops) && ok;
        ok = testL2Norm(ops) && ok;
        ok = testSsmConv1d(ops) && ok;
        ok = testSigmoidInplace(ops) && ok;
        ok = testGatherHeads(ops) && ok;

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
