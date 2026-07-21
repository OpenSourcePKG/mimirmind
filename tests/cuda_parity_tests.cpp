// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// CUDA GPU-vs-CPU parity tests for the M-Q3N.3 GatedDeltaNet primitives.
// Runs each GPU op (compute::cuda::GpuOps) against the CPU golden reference
// (compute/GatedDeltaNet) on random inputs and asserts they agree. Built
// and run only on a CUDA host (MIMIRMIND_ENABLE_CUDA); this is the
// correctness gate the box build exercises before the kernels get wired
// into the Qwen35MoeBackend recurrent path.

#include "TestFramework.hpp"

#include "compute/GatedDeltaNet.hpp"
#include "compute/cuda/GpuOps.hpp"
#include "core/gpu/cuda/CudaComputeContext.hpp"

#include <cstdint>
#include <vector>

namespace {

using ::mimirmind::compute::cuda::GpuOps;
using ::mimirmind::core::cuda::CudaComputeContext;

// Deterministic pseudo-random in [-1, 1) — no Date/rand, reproducible.
struct Lcg {
    std::uint32_t s;
    float next() {
        s = s * 1664525u + 1013904223u;
        return static_cast<float>((s >> 8) & 0xFFFFu) / 32768.0f - 1.0f;
    }
};

std::vector<float> randVec(std::size_t n, std::uint32_t seed) {
    Lcg g{seed};
    std::vector<float> v(n);
    for (auto& x : v) x = g.next();
    return v;
}

// Upload a host vector into a fresh device buffer.
::mimirmind::compute::ComputeBuffer
toDevice(GpuOps& ops, const std::vector<float>& h) {
    auto buf = ops.allocate(h.size() * sizeof(float));
    ops.uploadHostBytes(buf.get(), h.data(), h.size() * sizeof(float));
    return buf;
}

std::vector<float> fromDevice(GpuOps& ops, const void* dev, std::size_t n) {
    std::vector<float> h(n);
    ops.readbackToHost(h.data(), dev, n * sizeof(float));
    return h;
}

} // namespace

TEST(cuda_l2norm_parity) {
    CudaComputeContext ctx{};
    GpuOps ops{ctx};

    const std::size_t rows = 7, dim = 40;
    const float eps = 1e-6f;
    auto host = randVec(rows * dim, 0x1234u);

    std::vector<float> ref = host;
    ::mimirmind::compute::l2NormInPlace(ref.data(), rows, dim, eps);

    auto buf = toDevice(ops, host);
    ops.l2NormInPlaceAsync(static_cast<float*>(buf.get()), rows, dim, eps);
    ops.flush();
    auto got = fromDevice(ops, buf.get(), rows * dim);

    for (std::size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], ref[i], 1e-3f);
    }
}

TEST(cuda_ssm_conv1d_parity) {
    CudaComputeContext ctx{};
    GpuOps ops{ctx};

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
                              static_cast<float*>(dOut.get()),
                              T, channels, K);
    ops.flush();
    auto got = fromDevice(ops, dOut.get(), T * channels);

    for (std::size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], ref[i], 1e-3f);
    }
}

TEST(cuda_gated_deltanet_ar_parity) {
    CudaComputeContext ctx{};
    GpuOps ops{ctx};

    const std::size_t T = 4, H = 3, S = 16;
    auto q     = randVec(T * H * S, 0x0a1u);
    auto k     = randVec(T * H * S, 0x0b2u);
    auto v     = randVec(T * H * S, 0x0c3u);
    auto gLog  = randVec(T * H,     0x0d4u);
    auto beta  = randVec(T * H,     0x0e5u);
    auto state = randVec(H * S * S, 0x0f6u);

    // CPU reference (own state copy).
    std::vector<float> refState = state;
    std::vector<float> refOut(T * H * S);
    ::mimirmind::compute::gatedDeltaNetRecurrent(
        q.data(), k.data(), v.data(), gLog.data(), beta.data(),
        refState.data(), refOut.data(), T, H, S);

    // GPU (fresh state copy uploaded).
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

    for (std::size_t i = 0; i < gotOut.size(); ++i) {
        EXPECT_NEAR(gotOut[i], refOut[i], 2e-3f);
    }
    for (std::size_t i = 0; i < gotState.size(); ++i) {
        EXPECT_NEAR(gotState[i], refState[i], 2e-3f);
    }
}

TEST(cuda_deltanet_gate_parity) {
    CudaComputeContext ctx{};
    GpuOps ops{ctx};

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

    for (std::size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], ref[i], 1e-3f);
    }
}

TEST(cuda_sigmoid_inplace_parity) {
    CudaComputeContext ctx{};
    GpuOps ops{ctx};

    const std::size_t n = 50;
    auto host = randVec(n, 0x84u);
    std::vector<float> ref = host;
    ::mimirmind::compute::sigmoidInPlace(ref.data(), n);

    auto buf = toDevice(ops, host);
    ops.sigmoidInPlaceAsync(static_cast<float*>(buf.get()), n);
    ops.flush();
    auto got = fromDevice(ops, buf.get(), n);

    for (std::size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], ref[i], 1e-4f);
    }
}

int main() {
    return mm::test::run();
}