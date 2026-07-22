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

TEST(cuda_gather_heads_from_channels_parity) {
    CudaComputeContext ctx{};
    GpuOps ops{ctx};

    // Wide conv buffer [T, convTotalWidth]; extract a head block at `offset`
    // and repeat srcHeads=2 -> dstHeads=4 (GQA), S=3.
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

    for (std::size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], ref[i], 1e-6f);
    }
}

// ===========================================================================
// M-Q3N.4b — chunked-prefill GPU kernels (K0/K1/K2) vs the CPU stages.
//
// These fixtures are the correctness gate for the CUDA port of the chunked
// GatedDeltaNet prefill (M-Q3N.4a.2 CPU stages: deltanetChunkCumGate /
// deltanetKktSolveInverse / deltanetChunkForward). They validate each GPU
// kernel against its CPU stage AND the intermediate hand-off tensors G and
// A0 directly — so a divergence is localised to a single kernel rather than
// only showing up in the fused output.
//
// The GPU ops do not exist yet; 4b must add these methods to
// compute::cuda::GpuOps (mirroring the existing ...Async signatures), then
// define MIMIRMIND_GDN_CHUNK_GPU_READY (or drop the guard) to activate:
//
//   void deltanetChunkCumGateAsync(const float* gLog, float* gCum,
//                                  std::size_t T, std::size_t H,
//                                  std::size_t chunkSize);
//   void deltanetKktSolveInverseAsync(const float* k, const float* beta,
//                                     float* a0, std::size_t T, std::size_t H,
//                                     std::size_t S, std::size_t chunkSize);
//   void deltanetChunkForwardAsync(const float* q, const float* k,
//                                  const float* v, const float* gCum,
//                                  const float* beta, const float* a0,
//                                  float* state, float* out, std::size_t T,
//                                  std::size_t H, std::size_t S,
//                                  std::size_t chunkSize);
//
// a0 layout is [nChunks, H, C, C] row-major, nChunks = ceil(T/chunkSize),
// C = chunkSize (see GatedDeltaNet.hpp deltanetKktSolveInverse).
// ===========================================================================

// K0 — cumulative decay gate G. The prefix-sum hand-off tensor to K1/K2.
// K0's GPU kernel has landed (deltanetChunkCumGateAsync); K1/K2/pipeline
// stay gated behind MIMIRMIND_GDN_CHUNK_GPU_READY below until their kernels land.
TEST(cuda_deltanet_chunk_cumgate_parity) {
    CudaComputeContext ctx{};
    GpuOps ops{ctx};

    const std::size_t T = 10, H = 3, C = 4;   // chunks 4/4/2 (partial tail)
    auto gLog = randVec(T * H, 0x1c0u);

    std::vector<float> ref(T * H);
    ::mimirmind::compute::deltanetChunkCumGate(gLog.data(), ref.data(), T, H, C);

    auto dg  = toDevice(ops, gLog);
    auto dgc = ops.allocate(T * H * sizeof(float));
    ops.deltanetChunkCumGateAsync(static_cast<const float*>(dg.get()),
                                  static_cast<float*>(dgc.get()), T, H, C);
    ops.flush();
    auto got = fromDevice(ops, dgc.get(), T * H);

    for (std::size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], ref[i], 1e-3f);
    }
}

namespace {
inline std::size_t nChunksOf(std::size_t T, std::size_t C) {
    return (T + C - 1) / C;
}
} // namespace

// K1 — per-chunk ungated inverse A0 = (I + strictLower(diag(beta) K K^T))^-1.
// The most error-prone kernel (triangular solve); checked directly on A0.
// Realistic head_dim S=128, chunk C=64 to exercise the real inverse width.
TEST(cuda_deltanet_kkt_solve_parity) {
    CudaComputeContext ctx{};
    GpuOps ops{ctx};

    const std::size_t T = 130, H = 4, S = 128, C = 64; // chunks 64/64/2
    const std::size_t nc = nChunksOf(T, C);
    auto k    = randVec(T * H * S, 0x2c1u);
    auto beta = randVec(T * H,     0x2c2u);

    std::vector<float> ref(nc * H * C * C);
    ::mimirmind::compute::deltanetKktSolveInverse(k.data(), beta.data(),
                                                  ref.data(), T, H, S, C);

    auto dk = toDevice(ops, k);
    auto db = toDevice(ops, beta);
    auto da0 = ops.allocate(nc * H * C * C * sizeof(float));
    ops.deltanetKktSolveInverseAsync(static_cast<const float*>(dk.get()),
                                     static_cast<const float*>(db.get()),
                                     static_cast<float*>(da0.get()),
                                     T, H, S, C);
    ops.flush();
    auto got = fromDevice(ops, da0.get(), nc * H * C * C);

    // A0 is a triangular inverse: with random k the strict-lower Gram is
    // ill-conditioned, so the inverse entries grow to ~1e4 where an absolute
    // 2e-3 tolerance sits below the fp32 accumulation floor (a few ULP at that
    // magnitude already exceed it). The GPU matches the CPU reference to
    // ~5e-7 relative, so use an absolute floor plus a relative term.
    for (std::size_t i = 0; i < got.size(); ++i) {
        const float ar  = ref[i] < 0.0f ? -ref[i] : ref[i];
        const float tol = 2e-3f + 1e-4f * ar;
        EXPECT_NEAR(got[i], ref[i], tol);
    }
}

// K2 — chunk forward: consumes G (K0) and A0 (K1), carries state, writes out.
// Fed the CPU-reference G/A0 so a failure here is isolated to K2's own math.
// K2's GPU kernel has landed (deltanetChunkForwardAsync) — un-gated.
TEST(cuda_deltanet_chunk_forward_parity) {
    CudaComputeContext ctx{};
    GpuOps ops{ctx};

    const std::size_t T = 10, H = 3, S = 16, C = 4;
    const std::size_t nc = nChunksOf(T, C);
    auto q     = randVec(T * H * S, 0x3c1u);
    auto k     = randVec(T * H * S, 0x3c2u);
    auto v     = randVec(T * H * S, 0x3c3u);
    auto gLog  = randVec(T * H,     0x3c4u);
    auto beta  = randVec(T * H,     0x3c5u);
    auto state = randVec(H * S * S, 0x3c6u);

    // CPU-reference hand-off tensors (G, A0) and golden out/state.
    std::vector<float> gCum(T * H);
    ::mimirmind::compute::deltanetChunkCumGate(gLog.data(), gCum.data(), T, H, C);
    std::vector<float> a0(nc * H * C * C);
    ::mimirmind::compute::deltanetKktSolveInverse(k.data(), beta.data(),
                                                  a0.data(), T, H, S, C);
    std::vector<float> refState = state;
    std::vector<float> refOut(T * H * S);
    ::mimirmind::compute::deltanetChunkForward(q.data(), k.data(), v.data(),
                                               gCum.data(), beta.data(),
                                               a0.data(), refState.data(),
                                               refOut.data(), T, H, S, C);

    auto dq  = toDevice(ops, q);
    auto dk  = toDevice(ops, k);
    auto dv  = toDevice(ops, v);
    auto dgc = toDevice(ops, gCum);
    auto db  = toDevice(ops, beta);
    auto da0 = toDevice(ops, a0);
    auto ds  = toDevice(ops, state);
    auto dout = ops.allocate(T * H * S * sizeof(float));
    ops.deltanetChunkForwardAsync(static_cast<const float*>(dq.get()),
                                  static_cast<const float*>(dk.get()),
                                  static_cast<const float*>(dv.get()),
                                  static_cast<const float*>(dgc.get()),
                                  static_cast<const float*>(db.get()),
                                  static_cast<const float*>(da0.get()),
                                  static_cast<float*>(ds.get()),
                                  static_cast<float*>(dout.get()),
                                  T, H, S, C);
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

#if defined(MIMIRMIND_GDN_CHUNK_GPU_READY)

// End-to-end: run all three GPU kernels chained (G, A0 stay on device) and
// close the loop against the autoregressive golden — the same output the
// prefill path must produce before it can replace the AR loop.
TEST(cuda_deltanet_chunk_pipeline_vs_recurrent) {
    CudaComputeContext ctx{};
    GpuOps ops{ctx};

    const std::size_t T = 10, H = 3, S = 16, C = 4;
    const std::size_t nc = nChunksOf(T, C);
    auto q     = randVec(T * H * S, 0x4c1u);
    auto k     = randVec(T * H * S, 0x4c2u);
    auto v     = randVec(T * H * S, 0x4c3u);
    auto gLog  = randVec(T * H,     0x4c4u);
    auto beta  = randVec(T * H,     0x4c5u);
    auto state = randVec(H * S * S, 0x4c6u);

    // Golden: the autoregressive recurrence (chunk-size invariant).
    std::vector<float> refState = state;
    std::vector<float> refOut(T * H * S);
    ::mimirmind::compute::gatedDeltaNetRecurrent(
        q.data(), k.data(), v.data(), gLog.data(), beta.data(),
        refState.data(), refOut.data(), T, H, S);

    auto dq  = toDevice(ops, q);
    auto dk  = toDevice(ops, k);
    auto dv  = toDevice(ops, v);
    auto dg  = toDevice(ops, gLog);
    auto db  = toDevice(ops, beta);
    auto ds  = toDevice(ops, state);
    auto dgc = ops.allocate(T * H * sizeof(float));
    auto da0 = ops.allocate(nc * H * C * C * sizeof(float));
    auto dout = ops.allocate(T * H * S * sizeof(float));

    ops.deltanetChunkCumGateAsync(static_cast<const float*>(dg.get()),
                                  static_cast<float*>(dgc.get()), T, H, C);
    ops.deltanetKktSolveInverseAsync(static_cast<const float*>(dk.get()),
                                     static_cast<const float*>(db.get()),
                                     static_cast<float*>(da0.get()),
                                     T, H, S, C);
    ops.deltanetChunkForwardAsync(static_cast<const float*>(dq.get()),
                                  static_cast<const float*>(dk.get()),
                                  static_cast<const float*>(dv.get()),
                                  static_cast<const float*>(dgc.get()),
                                  static_cast<const float*>(db.get()),
                                  static_cast<const float*>(da0.get()),
                                  static_cast<float*>(ds.get()),
                                  static_cast<float*>(dout.get()),
                                  T, H, S, C);
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

#endif // MIMIRMIND_GDN_CHUNK_GPU_READY

int main() {
    return mm::test::run();
}