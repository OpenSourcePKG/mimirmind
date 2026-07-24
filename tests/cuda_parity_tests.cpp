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
#include "compute/cuda/GpuMatmul.hpp"
#include "compute/cuda/GpuOps.hpp"
#include "core/gpu/cuda/CudaComputeContext.hpp"
#include "core/gguf/GgufTypes.hpp"
#include "compute/quant/Q8_0.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <chrono>
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

// Build a device-resident K-quant weight bank of `blockCount` blocks. Bytes
// are pseudo-random EXCEPT each super-block's leading fp16 d/dmin scales,
// which are forced to a finite value (0.03125). Both the fused kernel's
// inline dequant and the sequential matmul path read these identical bytes,
// so the parity check only needs the weights to be finite and identical --
// not "correctly quantized" floats (there is no K-quant quantizer; K-quants
// are load-only from GGUF). Forcing the scales finite avoids inf/nan from a
// random fp16 that would poison both paths.
::mimirmind::compute::ComputeBuffer
buildQuantBank(GpuOps& ops, std::size_t blockBytes, std::size_t blockCount,
               std::uint32_t seed) {
    Lcg g{seed};
    std::vector<std::uint8_t> bytes(blockBytes * blockCount);
    for (auto& b : bytes) {
        g.s = g.s * 1664525u + 1013904223u;
        b = static_cast<std::uint8_t>(g.s >> 24);
    }
    for (std::size_t blk = 0; blk < blockCount; ++blk) {
        std::uint8_t* p = bytes.data() + blk * blockBytes;
        p[0] = 0x00; p[1] = 0x28;   // d    = fp16 0.03125
        p[2] = 0x00; p[3] = 0x28;   // dmin = fp16 0.03125
    }
    auto buf = ops.allocate(bytes.size());
    ops.uploadHostBytes(buf.get(), bytes.data(), bytes.size());
    return buf;
}

// Upload a small host array of trivially-copyable Ts into a fresh buffer.
template <typename T>
::mimirmind::compute::ComputeBuffer
uploadRaw(GpuOps& ops, const std::vector<T>& h) {
    auto buf = ops.allocate(h.size() * sizeof(T));
    ops.uploadHostBytes(buf.get(), h.data(), h.size() * sizeof(T));
    return buf;
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

// ---------------------------------------------------------------------------
// Fused-K MoE decode kernels (M-Q3N.4c/.4d). The end-to-end run verified these
// byte-identical to the sequential per-expert path; these tests isolate that
// same claim as a unit. The reference IS the sequential ComputeMatmul path
// (matmulAsync per expert + silu/scaledAdd) that runMoeFfn falls back to, run
// on the SAME device against the SAME quantised banks -- a GPU-vs-GPU parity
// that sidesteps the fp16-scale CPU-reference round-trip trap.
// ---------------------------------------------------------------------------

using ::mimirmind::compute::cuda::GpuMatmul;
using ::mimirmind::core::gguf::GgmlType;

// .4d: moeGateUpFusedKAsync == per-expert (matmul Wg, matmul Wu, silu*up).
TEST(cuda_moe_gate_up_fused_k_q4k_parity) {
    CudaComputeContext ctx{};
    GpuOps    ops{ctx};
    GpuMatmul gmm{ctx, ops};

    const GgmlType Q4K = GgmlType::Q4_K;
    if (!gmm.moeGateUpFusedKAvailable(Q4K)) {
        EXPECT_TRUE(false && "moe_gate_up_fused_k_q4k kernel not loaded");
        return;
    }

    const std::size_t dModel = 256, nFf = 256, K = 2, nExp = 4;
    // Separate Q4_K gate/up banks: per expert [nFf, dModel] = nFf blocks of 144B
    // (dModel/256 == 1 block per row). Matches runMoeFfn's `bytesGate`.
    const std::size_t blkBytes  = 144;
    const std::size_t bytesGate = nFf * (dModel / 256) * blkBytes;
    const std::size_t blkCount  = nExp * nFf * (dModel / 256);

    auto gateBank = buildQuantBank(ops, blkBytes, blkCount, 0xA1A1u);
    auto upBank   = buildQuantBank(ops, blkBytes, blkCount, 0xB2B2u);
    auto x        = toDevice(ops, randVec(dModel, 0x1111u));

    const std::vector<std::int32_t> expIdx{1, 3};   // K routed experts
    auto dExpIdx = uploadRaw(ops, expIdx);

    // Reference: sequential silu(Wg·x)*(Wu·x) into K-strided [K, nFf].
    auto refBuf  = ops.allocate(K * nFf * sizeof(float));
    auto gateTmp = ops.allocate(nFf * sizeof(float));
    auto upTmp   = ops.allocate(nFf * sizeof(float));
    auto scratch = ops.allocate(std::max(dModel, nFf) * sizeof(float));
    const auto* gateBase = static_cast<const std::uint8_t*>(gateBank.get());
    const auto* upBase   = static_cast<const std::uint8_t*>(upBank.get());
    for (std::size_t k = 0; k < K; ++k) {
        const std::size_t e = static_cast<std::size_t>(expIdx[k]);
        gmm.matmulAsync(Q4K, gateBase + e * bytesGate, nFf, dModel,
                        static_cast<const float*>(x.get()), 1,
                        static_cast<float*>(gateTmp.get()),
                        static_cast<float*>(scratch.get()));
        gmm.matmulAsync(Q4K, upBase + e * bytesGate, nFf, dModel,
                        static_cast<const float*>(x.get()), 1,
                        static_cast<float*>(upTmp.get()),
                        static_cast<float*>(scratch.get()));
        ops.siluMulAsync(static_cast<float*>(gateTmp.get()),
                         static_cast<const float*>(upTmp.get()), nFf);
        ops.appendMemoryCopy(static_cast<float*>(refBuf.get()) + k * nFf,
                             gateTmp.get(), nFf * sizeof(float));
    }
    ops.flush();
    auto ref = fromDevice(ops, refBuf.get(), K * nFf);

    // Fused: one launch for all K×2 GEMVs + silu.
    auto gotBuf = ops.allocate(K * nFf * sizeof(float));
    gmm.moeGateUpFusedKAsync(Q4K, static_cast<const float*>(x.get()),
                             gateBank.get(), upBank.get(),
                             static_cast<const std::int32_t*>(dExpIdx.get()),
                             static_cast<float*>(gotBuf.get()),
                             dModel, nFf, K, bytesGate, bytesGate);
    ops.flush();
    auto got = fromDevice(ops, gotBuf.get(), K * nFf);

    // Magnitude-relative: identical dequant + fp32 sum agree to ~eps*|v|;
    // a real kernel divergence is order-1 relative and still trips this.
    for (std::size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], ref[i], 2e-3f * (1.0f + std::fabs(ref[i])));
    }
}

// .4c: moeDownFusedKAsync == per-expert (matmul Wd, scaledAdd kw*·).
TEST(cuda_moe_down_fused_k_q5k_parity) {
    CudaComputeContext ctx{};
    GpuOps    ops{ctx};
    GpuMatmul gmm{ctx, ops};

    const GgmlType Q5K = GgmlType::Q5_K;
    if (!gmm.moeDownFusedKAvailable(Q5K)) {
        EXPECT_TRUE(false && "moe_down_fused_k_q5k kernel not loaded");
        return;
    }

    const std::size_t dModel = 256, nFf = 256, K = 2, nExp = 4;
    // down bank: per expert [dModel, nFf] = dModel blocks of 176B (nFf/256 == 1
    // block per row). Matches runMoeFfn's `bytesDown`.
    const std::size_t blkBytes  = 176;
    const std::size_t bytesDown = dModel * (nFf / 256) * blkBytes;
    const std::size_t blkCount  = nExp * dModel * (nFf / 256);

    auto downBank = buildQuantBank(ops, blkBytes, blkCount, 0xD0D0u);
    auto gateAct  = toDevice(ops, randVec(K * nFf, 0x2222u));

    const std::vector<std::int32_t> expIdx{0, 2};
    const std::vector<float>        kw{0.6f, 0.4f};   // router weights (× wScale)
    auto dExpIdx = uploadRaw(ops, expIdx);
    auto dKw     = uploadRaw(ops, kw);

    // Reference: accum += kw[k] * (Wd[e_k] @ gateAct[k]).
    auto refAcc  = ops.allocate(dModel * sizeof(float));
    auto expOut  = ops.allocate(dModel * sizeof(float));
    auto scratch = ops.allocate(std::max(dModel, nFf) * sizeof(float));
    ops.mulScalarAsync(static_cast<float*>(refAcc.get()), 0.0f, dModel);
    const auto* downBase = static_cast<const std::uint8_t*>(downBank.get());
    for (std::size_t k = 0; k < K; ++k) {
        const std::size_t e = static_cast<std::size_t>(expIdx[k]);
        gmm.matmulAsync(Q5K, downBase + e * bytesDown, dModel, nFf,
                        static_cast<const float*>(gateAct.get()) + k * nFf, 1,
                        static_cast<float*>(expOut.get()),
                        static_cast<float*>(scratch.get()));
        ops.scaledAddResidualAsync(static_cast<float*>(refAcc.get()),
                                   static_cast<const float*>(expOut.get()),
                                   kw[k], dModel);
    }
    ops.flush();
    auto ref = fromDevice(ops, refAcc.get(), dModel);

    // Fused: one launch for all K down-GEMVs + router-weighted residual add.
    auto gotAcc = ops.allocate(dModel * sizeof(float));
    ops.mulScalarAsync(static_cast<float*>(gotAcc.get()), 0.0f, dModel);
    gmm.moeDownFusedKAsync(Q5K, static_cast<const float*>(gateAct.get()),
                           downBank.get(),
                           static_cast<const std::int32_t*>(dExpIdx.get()),
                           static_cast<const float*>(dKw.get()),
                           static_cast<float*>(gotAcc.get()),
                           nFf, dModel, K, bytesDown);
    ops.flush();
    auto got = fromDevice(ops, gotAcc.get(), dModel);

    for (std::size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], ref[i], 2e-3f * (1.0f + std::fabs(ref[i])));
    }
}

// M-Cuda.MMQ B1: matmul_q8_0_mmq (int8 dp4a GEMM) vs the fp32 reference.
// Unlike the exact fused-MoE parity above, MMQ int8-quantises the activations
// -> lossy vs fp32. Assert a relative-L2 bound (structural bugs blow it up;
// int8 quant noise stays well under it), not bit-exactness. Reference is the
// untuned matmulAsync (matvec-loop, exact fp32 dequant).
TEST(cuda_matmul_q8_0_mmq_vs_fp32) {
    CudaComputeContext ctx{};
    GpuOps    ops{ctx};
    GpuMatmul gmm{ctx, ops};

    const std::size_t M = 10, N = 14, K = 512;   // partial M-tile + partial N
    const std::size_t blkBytes = 34, nBlocks = K / 32;
    auto W = buildQuantBank(ops, blkBytes, N * nBlocks, 0x8080u);
    auto X = toDevice(ops, randVec(M * K, 0x3131u));

    auto Yref    = ops.allocate(M * N * sizeof(float));
    auto Ymmq    = ops.allocate(M * N * sizeof(float));
    auto scratch = ops.allocate(std::max(N, K) * sizeof(float));

    gmm.matmulAsync(GgmlType::Q8_0, W.get(), N, K,
                    static_cast<const float*>(X.get()), M,
                    static_cast<float*>(Yref.get()),
                    static_cast<float*>(scratch.get()));
    gmm.matmulQ8_0MmqAsync(W.get(), N, K,
                           static_cast<const float*>(X.get()), M,
                           static_cast<float*>(Ymmq.get()));
    ops.flush();

    auto ref = fromDevice(ops, Yref.get(), M * N);
    auto got = fromDevice(ops, Ymmq.get(), M * N);

    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double d = static_cast<double>(got[i]) - static_cast<double>(ref[i]);
        num += d * d;
        den += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
    }
    const double relL2 = (den > 0.0) ? std::sqrt(num / den) : std::sqrt(num);
    EXPECT_TRUE(den > 0.0);            // reference is non-trivial
    EXPECT_NEAR(relL2, 0.0, 0.05);     // int8-activation quant is lossy
}

// M-Cuda.MMQ B2: matmul_q4k_mmq (int8 dp4a GEMM) vs fp32 q4k_vec reference.
// Q4_K affine dequant (a_j*nibble - b_j) folded into the int8 decomposition;
// lossy (int8 activations) -> relative-L2 bound, not bit-exact.
TEST(cuda_matmul_q4k_mmq_vs_fp32) {
    CudaComputeContext ctx{};
    GpuOps    ops{ctx};
    GpuMatmul gmm{ctx, ops};

    const std::size_t M = 10, N = 14, K = 512;      // K multiple of 256 (Q4_K)
    const std::size_t blkBytes = 144, nSuper = K / 256;
    auto W = buildQuantBank(ops, blkBytes, N * nSuper, 0x4B4Bu);
    auto X = toDevice(ops, randVec(M * K, 0x5151u));

    auto Yref    = ops.allocate(M * N * sizeof(float));
    auto Ymmq    = ops.allocate(M * N * sizeof(float));
    auto scratch = ops.allocate(std::max(N, K) * sizeof(float));

    gmm.matmulAsync(GgmlType::Q4_K, W.get(), N, K,
                    static_cast<const float*>(X.get()), M,
                    static_cast<float*>(Yref.get()),
                    static_cast<float*>(scratch.get()));
    gmm.matmulQ4KMmqAsync(W.get(), N, K,
                          static_cast<const float*>(X.get()), M,
                          static_cast<float*>(Ymmq.get()));
    ops.flush();

    auto ref = fromDevice(ops, Yref.get(), M * N);
    auto got = fromDevice(ops, Ymmq.get(), M * N);

    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double d = static_cast<double>(got[i]) - static_cast<double>(ref[i]);
        num += d * d;
        den += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
    }
    const double relL2 = (den > 0.0) ? std::sqrt(num / den) : std::sqrt(num);
    EXPECT_TRUE(den > 0.0);
    EXPECT_NEAR(relL2, 0.0, 0.05);
}

// M-Cuda.MMQ B1b: matmul_q8_0_mmq_tc (int8 tensor-core wmma) vs fp32 reference.
TEST(cuda_matmul_q8_0_mmq_tc_vs_fp32) {
    CudaComputeContext ctx{};
    GpuOps    ops{ctx};
    GpuMatmul gmm{ctx, ops};

    const std::size_t M = 10, N = 14, K = 512;
    const std::size_t blkBytes = 34, nBlocks = K / 32;
    auto W = buildQuantBank(ops, blkBytes, N * nBlocks, 0x7C7Cu);
    auto X = toDevice(ops, randVec(M * K, 0x6262u));

    auto Yref = ops.allocate(M * N * sizeof(float));
    auto Ytc  = ops.allocate(M * N * sizeof(float));
    auto scr  = ops.allocate(std::max(N, K) * sizeof(float));

    gmm.matmulAsync(GgmlType::Q8_0, W.get(), N, K,
                    static_cast<const float*>(X.get()), M,
                    static_cast<float*>(Yref.get()),
                    static_cast<float*>(scr.get()));
    gmm.matmulQ8_0MmqTcAsync(W.get(), N, K,
                             static_cast<const float*>(X.get()), M,
                             static_cast<float*>(Ytc.get()));
    ops.flush();

    auto ref = fromDevice(ops, Yref.get(), M * N);
    auto got = fromDevice(ops, Ytc.get(), M * N);

    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double d = static_cast<double>(got[i]) - static_cast<double>(ref[i]);
        num += d * d;
        den += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
    }
    const double relL2 = (den > 0.0) ? std::sqrt(num / den) : std::sqrt(num);
    EXPECT_TRUE(den > 0.0);
    EXPECT_NEAR(relL2, 0.0, 0.05);
}

// M-Cuda.MMQ B2: matmul_q5k_mmq (int8 dp4a GEMM) vs fp32 q5k_vec reference.
TEST(cuda_matmul_q5k_mmq_vs_fp32) {
    CudaComputeContext ctx{};
    GpuOps    ops{ctx};
    GpuMatmul gmm{ctx, ops};

    const std::size_t M = 10, N = 14, K = 512;      // K multiple of 256 (Q5_K)
    const std::size_t blkBytes = 176, nSuper = K / 256;
    auto W = buildQuantBank(ops, blkBytes, N * nSuper, 0x5A5Au);
    auto X = toDevice(ops, randVec(M * K, 0x7373u));

    auto Yref = ops.allocate(M * N * sizeof(float));
    auto Ymmq = ops.allocate(M * N * sizeof(float));
    auto scr  = ops.allocate(std::max(N, K) * sizeof(float));

    gmm.matmulAsync(GgmlType::Q5_K, W.get(), N, K,
                    static_cast<const float*>(X.get()), M,
                    static_cast<float*>(Yref.get()),
                    static_cast<float*>(scr.get()));
    gmm.matmulQ5KMmqAsync(W.get(), N, K,
                          static_cast<const float*>(X.get()), M,
                          static_cast<float*>(Ymmq.get()));
    ops.flush();

    auto ref = fromDevice(ops, Yref.get(), M * N);
    auto got = fromDevice(ops, Ymmq.get(), M * N);

    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double d = static_cast<double>(got[i]) - static_cast<double>(ref[i]);
        num += d * d;
        den += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
    }
    const double relL2 = (den > 0.0) ? std::sqrt(num / den) : std::sqrt(num);
    EXPECT_TRUE(den > 0.0);
    EXPECT_NEAR(relL2, 0.0, 0.05);
}

// ---------------------------------------------------------------------------
// M-Cuda.MMQ large-SHAPE parity — distinguishes (a) int8-lossiness from
// (b) a large-M kernel bug behind the long-prefill quality collapse.
// The earlier MMQ parity used M=10 (< the TC M_TILE=16, so the mGroups>1 path
// was NEVER exercised); prod prefill is M~2000. These run M=2048 (many M-tiles)
// and PRINT the relative-L2. A gross M-tiling bug blows relative-L2 up (>>0.1);
// mere int8-lossiness stays modest. Loose EXPECT so a bug fails loudly.
// ---------------------------------------------------------------------------

namespace {
double relL2LargeM(GpuOps& ops, GpuMatmul& gmm, bool tc,
                   std::size_t M, std::size_t N, std::size_t K,
                   float outlierMag = 0.0f) {
    const std::size_t nBlocks = K / 32;
    auto W   = buildQuantBank(ops, 34, N * nBlocks, 0xABCDu);
    auto xh  = randVec(M * K, 0x1234u);
    if (outlierMag > 0.0f) {
        // Inject a couple of large outliers per row (real LLM activations have
        // per-channel outliers; per-32-block-absmax quant then rounds the other
        // 31 block values toward 0). 2 outliers / row at fixed-ish channels.
        Lcg g{0xF00Du};
        for (std::size_t r = 0; r < M; ++r) {
            for (int o = 0; o < 2; ++o) {
                g.s = g.s * 1664525u + 1013904223u;
                const std::size_t c = (g.s >> 8) % K;
                xh[r * K + c] = (o & 1 ? -outlierMag : outlierMag);
            }
        }
    }
    auto X   = toDevice(ops, xh);
    auto Yr  = ops.allocate(M * N * sizeof(float));
    auto Ym  = ops.allocate(M * N * sizeof(float));
    auto scr = ops.allocate(std::max(N, K) * sizeof(float));

    gmm.matmulAsync(GgmlType::Q8_0, W.get(), N, K,
                    static_cast<const float*>(X.get()), M,
                    static_cast<float*>(Yr.get()),
                    static_cast<float*>(scr.get()));
    if (tc) {
        gmm.matmulQ8_0MmqTcAsync(W.get(), N, K,
                                 static_cast<const float*>(X.get()), M,
                                 static_cast<float*>(Ym.get()));
    } else {
        gmm.matmulQ8_0MmqAsync(W.get(), N, K,
                               static_cast<const float*>(X.get()), M,
                               static_cast<float*>(Ym.get()));
    }
    ops.flush();

    auto r = fromDevice(ops, Yr.get(), M * N);
    auto m = fromDevice(ops, Ym.get(), M * N);
    double num = 0.0, den = 0.0, maxAbs = 0.0, maxRel = 0.0;
    for (std::size_t i = 0; i < r.size(); ++i) {
        const double d = static_cast<double>(m[i]) - static_cast<double>(r[i]);
        num += d * d;
        den += static_cast<double>(r[i]) * static_cast<double>(r[i]);
        const double ad = std::fabs(d);
        if (ad > maxAbs) maxAbs = ad;
        const double rel = ad / (1e-6 + std::fabs(static_cast<double>(r[i])));
        if (rel > maxRel) maxRel = rel;
    }
    // Print max-element error too: a low aggregate L2 can hide a few
    // catastrophically-wrong outputs that flip the critical argmax logit.
    std::printf("    (maxAbsErr=%.4f  maxRelErr=%.4f)\n", maxAbs, maxRel);
    return (den > 0.0) ? std::sqrt(num / den) : std::sqrt(num);
}
} // namespace

TEST(cuda_matmul_q8_0_mmq_largeM_dp4a) {
    CudaComputeContext ctx{};
    GpuOps ops{ctx}; GpuMatmul gmm{ctx, ops};
    const double rl = relL2LargeM(ops, gmm, /*tc=*/false, 2048, 256, 2048);
    std::printf("[largeM dp4a M=2048 K=2048] relative-L2 = %.6f\n", rl);
    EXPECT_TRUE(rl < 0.15);   // gross kernel bug >> 0.1; pure int8-loss stays low
}

// Outlier-injected: confirms whether real-activation outliers (absent in the
// uniform-random test above) are what drives the long-prefill quality collapse.
// If relative-L2 blows up here vs the clean 0.0038, the per-32-block-absmax
// activation quant is the culprit (fix at the quant scheme).
TEST(cuda_matmul_q8_0_mmq_largeM_dp4a_outlier) {
    CudaComputeContext ctx{};
    GpuOps ops{ctx}; GpuMatmul gmm{ctx, ops};
    const double rl = relL2LargeM(ops, gmm, /*tc=*/false, 2048, 256, 2048, 300.0f);
    std::printf("[largeM dp4a OUTLIER=300] relative-L2 = %.6f\n", rl);
    EXPECT_TRUE(rl >= 0.0);   // diagnostic: read the printed value
}

TEST(cuda_matmul_q8_0_mmq_largeM_tc_outlier) {
    CudaComputeContext ctx{};
    GpuOps ops{ctx}; GpuMatmul gmm{ctx, ops};
    const double rl = relL2LargeM(ops, gmm, /*tc=*/true, 2048, 256, 2048, 300.0f);
    std::printf("[largeM tc   OUTLIER=300] relative-L2 = %.6f\n", rl);
    EXPECT_TRUE(rl >= 0.0);   // diagnostic
}

TEST(cuda_matmul_q8_0_mmq_largeM_tc) {
    CudaComputeContext ctx{};
    GpuOps ops{ctx}; GpuMatmul gmm{ctx, ops};
    const double rl = relL2LargeM(ops, gmm, /*tc=*/true, 2048, 256, 2048);
    std::printf("[largeM tc   M=2048 K=2048] relative-L2 = %.6f\n", rl);
    EXPECT_TRUE(rl < 0.15);
}

// Localise the per-element defect: dump the worst (m,n) outputs + histograms
// over m%16 / n%16 (TC tile = 16x16) to reveal a tile-boundary structure.
// Clean inputs (no outliers) since the defect is present there too.
TEST(cuda_matmul_q8_0_mmq_tc_localize) {
    CudaComputeContext ctx{};
    GpuOps ops{ctx}; GpuMatmul gmm{ctx, ops};
    const std::size_t M = 2048, N = 256, K = 2048, nBlocks = K / 32;
    auto W  = buildQuantBank(ops, 34, N * nBlocks, 0xABCDu);
    auto X  = toDevice(ops, randVec(M * K, 0x1234u));
    auto Yr = ops.allocate(M * N * sizeof(float));
    auto Ym = ops.allocate(M * N * sizeof(float));
    auto sc = ops.allocate(std::max(N, K) * sizeof(float));
    gmm.matmulAsync(GgmlType::Q8_0, W.get(), N, K,
                    static_cast<const float*>(X.get()), M,
                    static_cast<float*>(Yr.get()), static_cast<float*>(sc.get()));
    gmm.matmulQ8_0MmqTcAsync(W.get(), N, K,
                             static_cast<const float*>(X.get()), M,
                             static_cast<float*>(Ym.get()));
    ops.flush();
    auto r = fromDevice(ops, Yr.get(), M * N);
    auto m = fromDevice(ops, Ym.get(), M * N);

    struct E { float err; int mi; int ni; float ref; float got; };
    std::vector<E> bad;
    for (std::size_t i = 0; i < r.size(); ++i) {
        const float e = std::fabs(m[i] - r[i]);
        if (e > 0.3f) bad.push_back({e, int(i / N), int(i % N), r[i], m[i]});
    }
    std::sort(bad.begin(), bad.end(),
              [](const E& a, const E& b) { return a.err > b.err; });
    int hm[16] = {0}, hn[16] = {0};
    for (const auto& e : bad) { hm[e.mi % 16]++; hn[e.ni % 16]++; }
    std::printf("=== LOCALIZE clean M=2048 N=256 K=2048: %zu elems err>0.3 ===\n",
                bad.size());
    std::printf("m%%16: "); for (int i = 0; i < 16; ++i) std::printf("%d ", hm[i]);
    std::printf("\nn%%16: "); for (int i = 0; i < 16; ++i) std::printf("%d ", hn[i]);
    std::printf("\n");
    for (std::size_t i = 0; i < bad.size() && i < 15; ++i)
        std::printf("  [m=%d n=%d] ref=%.4f got=%.4f err=%.4f\n",
                    bad[i].mi, bad[i].ni, bad[i].ref, bad[i].got, bad[i].err);
    EXPECT_TRUE(true);
}

// M-Q3N.4 microbench (2026-07-24, GB10) — Q8_0 decode GEMV: native
// interleaved 34-byte layout vs reordered [scales|quants] layout.
// Answers whether the reorder layout lifts the ~16-27% DRAM throughput
// that ncu measured on matmul_q8_0_vec. Parity-checks reorderRow + the
// reorder kernel addressing, then times both on the big decode shape.
TEST(cuda_matmul_q8_0_vec_reorder_bench) {
    CudaComputeContext ctx{};
    GpuOps    ops{ctx};
    GpuMatmul gmm{ctx, ops};
    const GgmlType Q8 = GgmlType::Q8_0;

    const std::size_t N = 8192, K = 4096;
    const std::size_t blockBytes   = 34;
    const std::size_t blocksPerRow = K / 32;
    const std::size_t bytesPerRow  = blocksPerRow * blockBytes;
    const std::size_t totalBytes   = N * bytesPerRow;

    // Deterministic native Q8_0 weights: pseudo-random quants, every
    // block scale pinned to a benign finite fp16 (0x2C00 = 0.0625).
    std::vector<std::uint8_t> wNative(totalBytes);
    for (std::size_t i = 0; i < totalBytes; ++i) {
        wNative[i] = static_cast<std::uint8_t>((i * 2654435761u) >> 24);
    }
    for (std::size_t n = 0; n < N; ++n) {
        for (std::size_t b = 0; b < blocksPerRow; ++b) {
            std::uint8_t* s = wNative.data() + n * bytesPerRow + b * blockBytes;
            s[0] = 0x00; s[1] = 0x2C;
        }
    }

    std::vector<std::uint8_t> wReorder(totalBytes);
    for (std::size_t n = 0; n < N; ++n) {
        mimirmind::compute::quant::Q8_0::reorderRow(
            wNative.data()  + n * bytesPerRow, K,
            wReorder.data() + n * bytesPerRow);
    }

    auto x         = toDevice(ops, randVec(K, 0x2222u));
    auto dWNative  = uploadRaw(ops, wNative);
    auto dWReorder = uploadRaw(ops, wReorder);
    auto yNative   = ops.allocate(N * sizeof(float));
    auto yReorder  = ops.allocate(N * sizeof(float));
    auto scratch   = ops.allocate(K * sizeof(float));
    const float* xp = static_cast<const float*>(x.get());

    // Parity — same math, different memory layout.
    gmm.matmulAsync(Q8, dWNative.get(), N, K, xp, 1,
                    static_cast<float*>(yNative.get()),
                    static_cast<float*>(scratch.get()));
    ops.matmulQ8_0VecReorderAsync(dWReorder.get(), N, K, xp,
                                  static_cast<float*>(yReorder.get()));
    ops.flush();

    auto hN = fromDevice(ops, yNative.get(), N);
    auto hR = fromDevice(ops, yReorder.get(), N);
    double maxErr = 0.0, num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        const double d = static_cast<double>(hN[i]) - static_cast<double>(hR[i]);
        maxErr = std::max(maxErr, std::fabs(d));
        num += d * d;
        den += static_cast<double>(hN[i]) * static_cast<double>(hN[i]);
    }
    const double relL2 = den > 0.0 ? std::sqrt(num / den) : 0.0;
    std::printf("[reorder-bench] parity maxErr=%.5f relL2=%.6f\n", maxErr, relL2);
    EXPECT_NEAR(relL2, 0.0, 0.02);

    const int iters = 300;
    for (int w = 0; w < 20; ++w) {
        gmm.matmulAsync(Q8, dWNative.get(), N, K, xp, 1,
                        static_cast<float*>(yNative.get()),
                        static_cast<float*>(scratch.get()));
    }
    ops.flush();

    auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; ++it) {
        gmm.matmulAsync(Q8, dWNative.get(), N, K, xp, 1,
                        static_cast<float*>(yNative.get()),
                        static_cast<float*>(scratch.get()));
    }
    ops.flush();
    auto t1 = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; ++it) {
        ops.matmulQ8_0VecReorderAsync(dWReorder.get(), N, K, xp,
                                      static_cast<float*>(yReorder.get()));
    }
    ops.flush();
    auto t2 = std::chrono::steady_clock::now();

    const double nativeUs  =
        std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
    const double reorderUs =
        std::chrono::duration<double, std::micro>(t2 - t1).count() / iters;
    std::printf("[reorder-bench] N=%zu K=%zu native=%.2f us reorder=%.2f us speedup=%.2fx\n",
                N, K, nativeUs, reorderUs,
                reorderUs > 0.0 ? nativeUs / reorderUs : 0.0);
    EXPECT_TRUE(true);
}

// M-Cuda.Batch Cat C-P0 — batched gated_deltanet_ar vs N single-sequence
// runs. Same kernel math with a per-sequence offset, so batched output and
// state must be byte-identical to running each sequence on its own.
TEST(cuda_gated_deltanet_ar_batched_parity) {
    CudaComputeContext ctx{};
    GpuOps ops{ctx};

    const std::size_t nSeq = 3, T = 5, H = 4, S = 16;
    const std::size_t act = T * H * S;   // per-seq q/k/v/out
    const std::size_t gat = T * H;       // per-seq gLog/beta
    const std::size_t stt = H * S * S;   // per-seq state

    std::vector<float> Q(nSeq*act), K(nSeq*act), V(nSeq*act);
    std::vector<float> G(nSeq*gat), B(nSeq*gat), ST(nSeq*stt);
    for (std::size_t s = 0; s < nSeq; ++s) {
        const std::uint32_t o = static_cast<std::uint32_t>(s) * 7u;
        auto q  = randVec(act, 0x0a1u + o);
        auto k  = randVec(act, 0x0b2u + o);
        auto v  = randVec(act, 0x0c3u + o);
        auto g  = randVec(gat, 0x0d4u + o);
        auto b  = randVec(gat, 0x0e5u + o);
        auto st = randVec(stt, 0x0f6u + o);
        std::copy(q.begin(),  q.end(),  Q.begin()  + s*act);
        std::copy(k.begin(),  k.end(),  K.begin()  + s*act);
        std::copy(v.begin(),  v.end(),  V.begin()  + s*act);
        std::copy(g.begin(),  g.end(),  G.begin()  + s*gat);
        std::copy(b.begin(),  b.end(),  B.begin()  + s*gat);
        std::copy(st.begin(), st.end(), ST.begin() + s*stt);
    }

    // Batched: one launch over all nSeq.
    auto dQ = toDevice(ops, Q); auto dK = toDevice(ops, K); auto dV = toDevice(ops, V);
    auto dG = toDevice(ops, G); auto dB = toDevice(ops, B); auto dS = toDevice(ops, ST);
    auto dOut = ops.allocate(nSeq*act*sizeof(float));
    ops.gatedDeltaNetRecurrentBatchedAsync(
        static_cast<const float*>(dQ.get()), static_cast<const float*>(dK.get()),
        static_cast<const float*>(dV.get()), static_cast<const float*>(dG.get()),
        static_cast<const float*>(dB.get()), static_cast<float*>(dS.get()),
        static_cast<float*>(dOut.get()), nSeq, T, H, S);
    ops.flush();
    auto outB   = fromDevice(ops, dOut.get(), nSeq*act);
    auto stateB = fromDevice(ops, dS.get(),   nSeq*stt);

    // Reference: each sequence through the single-seq kernel.
    double maxErr = 0.0;
    for (std::size_t s = 0; s < nSeq; ++s) {
        std::vector<float> q(Q.begin()+s*act, Q.begin()+(s+1)*act);
        std::vector<float> k(K.begin()+s*act, K.begin()+(s+1)*act);
        std::vector<float> v(V.begin()+s*act, V.begin()+(s+1)*act);
        std::vector<float> g(G.begin()+s*gat, G.begin()+(s+1)*gat);
        std::vector<float> b(B.begin()+s*gat, B.begin()+(s+1)*gat);
        std::vector<float> st(ST.begin()+s*stt, ST.begin()+(s+1)*stt);
        auto dq=toDevice(ops,q); auto dk=toDevice(ops,k); auto dv=toDevice(ops,v);
        auto dg=toDevice(ops,g); auto db=toDevice(ops,b); auto dst=toDevice(ops,st);
        auto dSingle = ops.allocate(act*sizeof(float));
        ops.gatedDeltaNetRecurrentAsync(
            static_cast<const float*>(dq.get()), static_cast<const float*>(dk.get()),
            static_cast<const float*>(dv.get()), static_cast<const float*>(dg.get()),
            static_cast<const float*>(db.get()), static_cast<float*>(dst.get()),
            static_cast<float*>(dSingle.get()), T, H, S);
        ops.flush();
        auto outS   = fromDevice(ops, dSingle.get(), act);
        auto stateS = fromDevice(ops, dst.get(),     stt);
        for (std::size_t i = 0; i < act; ++i) {
            maxErr = std::max(maxErr, std::fabs((double)outB[s*act+i] - (double)outS[i]));
            EXPECT_NEAR(outB[s*act+i], outS[i], 1e-5f);
        }
        for (std::size_t i = 0; i < stt; ++i) {
            maxErr = std::max(maxErr, std::fabs((double)stateB[s*stt+i] - (double)stateS[i]));
            EXPECT_NEAR(stateB[s*stt+i], stateS[i], 1e-5f);
        }
    }
    std::printf("[gdn-batched-parity] nSeq=%zu T=%zu H=%zu S=%zu maxErr=%.2e\n",
                nSeq, T, H, S, maxErr);
}

// M-Cuda.Batch Cat C-P0 — batched ssm_conv1d vs N single-sequence runs.
// Each sequence has its own conv input (its rolling conv-tail prepended);
// batched output must be byte-identical to running each sequence alone.
TEST(cuda_ssm_conv1d_batched_parity) {
    CudaComputeContext ctx{};
    GpuOps ops{ctx};

    const std::size_t nSeq = 3, T = 5, channels = 12, K = 4;
    const std::size_t inPer  = (K - 1 + T) * channels;   // per-seq conv input
    const std::size_t outPer = T * channels;             // per-seq output

    auto kernel = randVec(K * channels, 0x3333u);        // shared across seqs
    std::vector<float> IN(nSeq * inPer);
    for (std::size_t s = 0; s < nSeq; ++s) {
        auto in = randVec(inPer, 0x2200u + static_cast<std::uint32_t>(s) * 13u);
        std::copy(in.begin(), in.end(), IN.begin() + s * inPer);
    }

    // Batched: one launch over all nSeq.
    auto dIn  = toDevice(ops, IN);
    auto dKer = toDevice(ops, kernel);
    auto dOut = ops.allocate(nSeq * outPer * sizeof(float));
    ops.causalConv1dSiluBatchedAsync(
        static_cast<const float*>(dIn.get()),
        static_cast<const float*>(dKer.get()),
        static_cast<float*>(dOut.get()), nSeq, T, channels, K);
    ops.flush();
    auto outB = fromDevice(ops, dOut.get(), nSeq * outPer);

    // Reference: each sequence through the single-seq kernel.
    double maxErr = 0.0;
    for (std::size_t s = 0; s < nSeq; ++s) {
        std::vector<float> in(IN.begin() + s * inPer, IN.begin() + (s + 1) * inPer);
        auto di  = toDevice(ops, in);
        auto dk  = toDevice(ops, kernel);
        auto doS = ops.allocate(outPer * sizeof(float));
        ops.causalConv1dSiluAsync(static_cast<const float*>(di.get()),
                                  static_cast<const float*>(dk.get()),
                                  static_cast<float*>(doS.get()), T, channels, K);
        ops.flush();
        auto outS = fromDevice(ops, doS.get(), outPer);
        for (std::size_t i = 0; i < outPer; ++i) {
            maxErr = std::max(maxErr,
                std::fabs((double)outB[s*outPer+i] - (double)outS[i]));
            EXPECT_NEAR(outB[s*outPer+i], outS[i], 1e-5f);
        }
    }
    std::printf("[conv1d-batched-parity] nSeq=%zu T=%zu channels=%zu K=%zu maxErr=%.2e\n",
                nSeq, T, channels, K, maxErr);
}

// M-Cuda.Batch Cat B — batched moe_gate_up_fused_k_q4k vs N single-token
// runs. Each token has its own x and routed experts; batched output must
// be byte-identical to running each token alone.
TEST(cuda_moe_gate_up_fused_k_q4k_batched_parity) {
    CudaComputeContext ctx{};
    GpuOps    ops{ctx};
    GpuMatmul gmm{ctx, ops};
    const GgmlType Q4K = GgmlType::Q4_K;
    if (!gmm.moeGateUpFusedKAvailable(Q4K)) {
        EXPECT_TRUE(false && "moe_gate_up_fused_k_q4k kernel not loaded");
        return;
    }

    const std::size_t nSeq = 3, dModel = 256, nFf = 256, K = 2, nExp = 4;
    const std::size_t blkBytes  = 144;
    const std::size_t bytesGate = nFf * (dModel / 256) * blkBytes;
    const std::size_t blkCount  = nExp * nFf * (dModel / 256);

    auto gateBank = buildQuantBank(ops, blkBytes, blkCount, 0xA1A1u);
    auto upBank   = buildQuantBank(ops, blkBytes, blkCount, 0xB2B2u);

    // Per-token x and routed experts (distinct experts per token).
    std::vector<float> Xh(nSeq * dModel);
    std::vector<std::int32_t> Eh(nSeq * K);
    const std::int32_t experts[3][2] = {{1, 3}, {0, 2}, {3, 1}};
    for (std::size_t s = 0; s < nSeq; ++s) {
        auto xs = randVec(dModel, 0x1100u + static_cast<std::uint32_t>(s) * 17u);
        std::copy(xs.begin(), xs.end(), Xh.begin() + s * dModel);
        Eh[s * K + 0] = experts[s][0];
        Eh[s * K + 1] = experts[s][1];
    }
    auto dX = toDevice(ops, Xh);
    auto dE = uploadRaw(ops, Eh);

    // Batched: one launch over all nSeq tokens.
    auto gotBuf = ops.allocate(nSeq * K * nFf * sizeof(float));
    gmm.moeGateUpFusedKBatchedAsync(Q4K, static_cast<const float*>(dX.get()),
        gateBank.get(), upBank.get(),
        static_cast<const std::int32_t*>(dE.get()),
        static_cast<float*>(gotBuf.get()),
        nSeq, dModel, nFf, K, bytesGate, bytesGate);
    ops.flush();
    auto got = fromDevice(ops, gotBuf.get(), nSeq * K * nFf);

    // Reference: each token through the single-token fused kernel.
    double maxErr = 0.0;
    for (std::size_t s = 0; s < nSeq; ++s) {
        std::vector<float> xs(Xh.begin() + s * dModel, Xh.begin() + (s + 1) * dModel);
        std::vector<std::int32_t> es(Eh.begin() + s * K, Eh.begin() + (s + 1) * K);
        auto dxs = toDevice(ops, xs);
        auto des = uploadRaw(ops, es);
        auto refBuf = ops.allocate(K * nFf * sizeof(float));
        gmm.moeGateUpFusedKAsync(Q4K, static_cast<const float*>(dxs.get()),
            gateBank.get(), upBank.get(),
            static_cast<const std::int32_t*>(des.get()),
            static_cast<float*>(refBuf.get()),
            dModel, nFf, K, bytesGate, bytesGate);
        ops.flush();
        auto ref = fromDevice(ops, refBuf.get(), K * nFf);
        for (std::size_t i = 0; i < K * nFf; ++i) {
            maxErr = std::max(maxErr,
                std::fabs((double)got[s * K * nFf + i] - (double)ref[i]));
            EXPECT_NEAR(got[s * K * nFf + i], ref[i], 1e-5f);
        }
    }
    std::printf("[moe-gu-batched-parity] nSeq=%zu dModel=%zu nFf=%zu K=%zu maxErr=%.2e\n",
                nSeq, dModel, nFf, K, maxErr);
}

// M-Cuda.Batch Cat B — batched moe_down_fused_k_q5k vs N single-token runs.
// Each token has its own gate activations, routed experts, router weights
// and RMW accumulator; batched output must be byte-identical to per-token.
TEST(cuda_moe_down_fused_k_q5k_batched_parity) {
    CudaComputeContext ctx{};
    GpuOps    ops{ctx};
    GpuMatmul gmm{ctx, ops};
    const GgmlType Q5K = GgmlType::Q5_K;
    if (!gmm.moeDownFusedKAvailable(Q5K)) {
        EXPECT_TRUE(false && "moe_down_fused_k_q5k kernel not loaded");
        return;
    }

    const std::size_t nSeq = 3, dModel = 256, nFf = 256, K = 2, nExp = 4;
    const std::size_t blkBytes  = 176;
    const std::size_t bytesDown = dModel * (nFf / 256) * blkBytes;
    const std::size_t blkCount  = nExp * dModel * (nFf / 256);
    auto downBank = buildQuantBank(ops, blkBytes, blkCount, 0xD0D0u);

    std::vector<float>        GA(nSeq * K * nFf);
    std::vector<std::int32_t> EI(nSeq * K);
    std::vector<float>        KW(nSeq * K);
    const std::int32_t experts[3][2] = {{0, 2}, {1, 3}, {2, 0}};
    const float        kws[3][2]     = {{0.6f, 0.4f}, {0.7f, 0.3f}, {0.5f, 0.5f}};
    for (std::size_t s = 0; s < nSeq; ++s) {
        auto ga = randVec(K * nFf, 0x2200u + static_cast<std::uint32_t>(s) * 19u);
        std::copy(ga.begin(), ga.end(), GA.begin() + s * K * nFf);
        for (std::size_t k = 0; k < K; ++k) {
            EI[s * K + k] = experts[s][k];
            KW[s * K + k] = kws[s][k];
        }
    }
    auto dGA = toDevice(ops, GA);
    auto dEI = uploadRaw(ops, EI);
    auto dKW = uploadRaw(ops, KW);

    // Batched: accum[nSeq, dModel] seeded to 0, one launch.
    auto gotAcc = ops.allocate(nSeq * dModel * sizeof(float));
    ops.mulScalarAsync(static_cast<float*>(gotAcc.get()), 0.0f, nSeq * dModel);
    gmm.moeDownFusedKBatchedAsync(Q5K, static_cast<const float*>(dGA.get()),
        downBank.get(), static_cast<const std::int32_t*>(dEI.get()),
        static_cast<const float*>(dKW.get()),
        static_cast<float*>(gotAcc.get()),
        nSeq, nFf, dModel, K, bytesDown);
    ops.flush();
    auto got = fromDevice(ops, gotAcc.get(), nSeq * dModel);

    // Reference: each token through the single-token fused kernel.
    double maxErr = 0.0;
    for (std::size_t s = 0; s < nSeq; ++s) {
        std::vector<float> ga(GA.begin() + s * K * nFf, GA.begin() + (s + 1) * K * nFf);
        std::vector<std::int32_t> ei(EI.begin() + s * K, EI.begin() + (s + 1) * K);
        std::vector<float> kw(KW.begin() + s * K, KW.begin() + (s + 1) * K);
        auto dga = toDevice(ops, ga);
        auto dei = uploadRaw(ops, ei);
        auto dkw = uploadRaw(ops, kw);
        auto refAcc = ops.allocate(dModel * sizeof(float));
        ops.mulScalarAsync(static_cast<float*>(refAcc.get()), 0.0f, dModel);
        gmm.moeDownFusedKAsync(Q5K, static_cast<const float*>(dga.get()),
            downBank.get(), static_cast<const std::int32_t*>(dei.get()),
            static_cast<const float*>(dkw.get()),
            static_cast<float*>(refAcc.get()),
            nFf, dModel, K, bytesDown);
        ops.flush();
        auto ref = fromDevice(ops, refAcc.get(), dModel);
        for (std::size_t i = 0; i < dModel; ++i) {
            maxErr = std::max(maxErr,
                std::fabs((double)got[s * dModel + i] - (double)ref[i]));
            EXPECT_NEAR(got[s * dModel + i], ref[i], 1e-5f);
        }
    }
    std::printf("[moe-down-batched-parity] nSeq=%zu dModel=%zu nFf=%zu K=%zu maxErr=%.2e\n",
                nSeq, dModel, nFf, K, maxErr);
}

// M-Cuda.Batch Cat B — batched rope_mrope vs N single-sequence runs. Each
// sequence has its own x region and start position; batched result must be
// byte-identical to running each sequence alone. (Provisional Cat-B x
// layout: seq s at xBase + s*xSeqStride, settled in Phase D.)
TEST(cuda_rope_mrope_batched_parity) {
    CudaComputeContext ctx{};
    GpuOps ops{ctx};

    const std::size_t nSeq = 3, seqLen = 1, numHeads = 4, headDim = 16;
    const std::size_t writeStride = numHeads * headDim;   // one token slot
    const std::size_t xSeqStride  = 8 * writeStride;      // holds startPos < 8
    const float base = 1000000.0f;
    const std::int32_t sections[4] = {2, 2, 2, 2};
    const std::int32_t startPos[3] = {2, 5, 0};

    std::vector<float> X(nSeq * xSeqStride);
    for (std::size_t s = 0; s < nSeq; ++s) {
        auto xs = randVec(xSeqStride, 0x7700u + static_cast<std::uint32_t>(s) * 23u);
        std::copy(xs.begin(), xs.end(), X.begin() + s * xSeqStride);
    }

    // Batched: one launch over all nSeq (per-seq startPos device array).
    auto dX = toDevice(ops, X);
    std::vector<std::int32_t> sp(startPos, startPos + nSeq);
    auto dSp = uploadRaw(ops, sp);
    ops.mropeInPlaceBatchedAsync(dX.get(), nSeq, xSeqStride, seqLen, numHeads,
                                 headDim,
                                 static_cast<const std::int32_t*>(dSp.get()),
                                 base, sections, writeStride);
    ops.flush();
    auto gotAll = fromDevice(ops, dX.get(), nSeq * xSeqStride);

    // Reference: each sequence through the single-seq kernel on its own copy.
    double maxErr = 0.0;
    for (std::size_t s = 0; s < nSeq; ++s) {
        std::vector<float> xs(X.begin() + s * xSeqStride, X.begin() + (s + 1) * xSeqStride);
        auto dxs = toDevice(ops, xs);
        ops.mropeInPlaceAsync(dxs.get(), seqLen, numHeads, headDim,
                              static_cast<std::size_t>(startPos[s]), base,
                              sections, writeStride);
        ops.flush();
        auto refSeg = fromDevice(ops, dxs.get(), xSeqStride);
        for (std::size_t i = 0; i < xSeqStride; ++i) {
            maxErr = std::max(maxErr,
                std::fabs((double)gotAll[s * xSeqStride + i] - (double)refSeg[i]));
            EXPECT_NEAR(gotAll[s * xSeqStride + i], refSeg[i], 1e-5f);
        }
    }
    std::printf("[rope-mrope-batched-parity] nSeq=%zu numHeads=%zu headDim=%zu maxErr=%.2e\n",
                nSeq, numHeads, headDim, maxErr);
}

// M-Cuda.Batch (attention) — batched decode flash-attention vs N single
// runs. Each sequence has its own query, KV cache and length; batched
// output must be byte-identical to running each sequence alone. Reference
// is the public attentionAsync (T_q==1 routes to the decode-flash path).
TEST(cuda_attention_flash_decode_batched_parity) {
    CudaComputeContext ctx{};
    GpuOps ops{ctx};

    const std::size_t nSeq = 3, nHeads = 4, nKvHeads = 2, headDim = 16;
    const std::size_t T_k = 128;                    // KV cache capacity per seq
    const std::int32_t curLen[3] = {30, 70, 10};    // positionOffset per seq
    const std::size_t maxKTiles = 2;                // ceil((70+1)/64)
    const std::size_t qSeqStride      = nHeads * headDim;
    const std::size_t kvSeqStride     = T_k * nKvHeads * headDim;
    const std::size_t outSeqStride    = nHeads * headDim;
    const std::size_t partialSeqStride = nHeads * maxKTiles * (2 + headDim);
    const float scale = 1.0f / std::sqrt((float)headDim);

    std::vector<float> Q(nSeq * qSeqStride), Kc(nSeq * kvSeqStride), Vc(nSeq * kvSeqStride);
    for (std::size_t s = 0; s < nSeq; ++s) {
        const std::uint32_t o = static_cast<std::uint32_t>(s) * 31u;
        auto qs = randVec(qSeqStride,  0x9000u + o);
        auto ks = randVec(kvSeqStride, 0xA000u + o);
        auto vs = randVec(kvSeqStride, 0xB000u + o);
        std::copy(qs.begin(), qs.end(), Q.begin()  + s * qSeqStride);
        std::copy(ks.begin(), ks.end(), Kc.begin() + s * kvSeqStride);
        std::copy(vs.begin(), vs.end(), Vc.begin() + s * kvSeqStride);
    }

    // Batched: one launch pair (partial + merge) over all nSeq.
    auto dQ = toDevice(ops, Q);
    auto dK = toDevice(ops, Kc);
    auto dV = toDevice(ops, Vc);
    std::vector<std::int32_t> cl(curLen, curLen + nSeq);
    auto dCl = uploadRaw(ops, cl);
    auto dPartial = ops.allocate(nSeq * partialSeqStride * sizeof(float));
    auto dOut = ops.allocate(nSeq * outSeqStride * sizeof(float));
    ops.attentionDecodeFlashBatchedAsync(
        static_cast<const float*>(dQ.get()), static_cast<const float*>(dK.get()),
        static_cast<const float*>(dV.get()), static_cast<float*>(dPartial.get()),
        static_cast<float*>(dOut.get()), nSeq, maxKTiles, qSeqStride, kvSeqStride,
        partialSeqStride, outSeqStride, nHeads, nKvHeads, headDim,
        static_cast<const std::int32_t*>(dCl.get()), scale, 0,
        ::mimirmind::runtime::KvDtype::F32);
    ops.flush();
    auto gotAll = fromDevice(ops, dOut.get(), nSeq * outSeqStride);

    // Reference: each sequence via the single-seq decode-flash path.
    double maxErr = 0.0;
    for (std::size_t s = 0; s < nSeq; ++s) {
        std::vector<float> qs(Q.begin() + s * qSeqStride,  Q.begin() + (s + 1) * qSeqStride);
        std::vector<float> ks(Kc.begin() + s * kvSeqStride, Kc.begin() + (s + 1) * kvSeqStride);
        std::vector<float> vs(Vc.begin() + s * kvSeqStride, Vc.begin() + (s + 1) * kvSeqStride);
        auto dqs = toDevice(ops, qs);
        auto dks = toDevice(ops, ks);
        auto dvs = toDevice(ops, vs);
        auto dos = ops.allocate(outSeqStride * sizeof(float));
        ops.attentionAsync(static_cast<const float*>(dqs.get()),
                           static_cast<const float*>(dks.get()),
                           static_cast<const float*>(dvs.get()),
                           1, T_k, nHeads, nKvHeads, headDim,
                           static_cast<std::size_t>(curLen[s]), scale,
                           static_cast<float*>(dos.get()), 0,
                           ::mimirmind::runtime::KvDtype::F32);
        ops.flush();
        auto ref = fromDevice(ops, dos.get(), outSeqStride);
        for (std::size_t i = 0; i < outSeqStride; ++i) {
            maxErr = std::max(maxErr,
                std::fabs((double)gotAll[s * outSeqStride + i] - (double)ref[i]));
            EXPECT_NEAR(gotAll[s * outSeqStride + i], ref[i], 1e-5f);
        }
    }
    std::printf("[attn-flash-decode-batched-parity] nSeq=%zu nHeads=%zu headDim=%zu maxErr=%.2e\n",
                nSeq, nHeads, headDim, maxErr);
}

// M-Cuda.Batch Cat C-P1 — batched chunked-prefill (cumgate + forward) vs N
// single-sequence pipelines. kkt-solve (K1) is run per sequence to build the
// shared a0; the batched cumgate (K0) and forward (K2) must be byte-identical
// to running each sequence alone.
TEST(cuda_deltanet_chunk_batched_parity) {
    CudaComputeContext ctx{};
    GpuOps ops{ctx};

    const std::size_t nSeq = 3, T = 10, H = 3, S = 16, C = 4;
    const std::size_t nc   = nChunksOf(T, C);
    const std::size_t actP = T * H * S, gateP = T * H, stP = H * S * S, a0P = nc * H * C * C;

    std::vector<float> Q(nSeq*actP), K(nSeq*actP), V(nSeq*actP);
    std::vector<float> G(nSeq*gateP), B(nSeq*gateP), ST(nSeq*stP);
    for (std::size_t s = 0; s < nSeq; ++s) {
        const std::uint32_t o = static_cast<std::uint32_t>(s) * 37u;
        auto q = randVec(actP, 0x4c1u + o); auto k = randVec(actP, 0x4c2u + o);
        auto v = randVec(actP, 0x4c3u + o); auto g = randVec(gateP, 0x4c4u + o);
        auto b = randVec(gateP, 0x4c5u + o); auto st = randVec(stP, 0x4c6u + o);
        std::copy(q.begin(),q.end(),Q.begin()+s*actP);
        std::copy(k.begin(),k.end(),K.begin()+s*actP);
        std::copy(v.begin(),v.end(),V.begin()+s*actP);
        std::copy(g.begin(),g.end(),G.begin()+s*gateP);
        std::copy(b.begin(),b.end(),B.begin()+s*gateP);
        std::copy(st.begin(),st.end(),ST.begin()+s*stP);
    }

    // Batched pipeline.
    auto dQ=toDevice(ops,Q); auto dK=toDevice(ops,K); auto dV=toDevice(ops,V);
    auto dG=toDevice(ops,G); auto dB=toDevice(ops,B); auto dS=toDevice(ops,ST);
    auto dGc=ops.allocate(nSeq*gateP*sizeof(float));
    auto dA0=ops.allocate(nSeq*a0P*sizeof(float));
    auto dOut=ops.allocate(nSeq*actP*sizeof(float));
    ops.deltanetChunkCumGateBatchedAsync(static_cast<const float*>(dG.get()),
        static_cast<float*>(dGc.get()), nSeq, T, H, C);
    for (std::size_t s = 0; s < nSeq; ++s) {
        ops.deltanetKktSolveInverseAsync(
            static_cast<const float*>(dK.get()) + s*actP,
            static_cast<const float*>(dB.get()) + s*gateP,
            static_cast<float*>(dA0.get()) + s*a0P, T, H, S, C);
    }
    ops.deltanetChunkForwardBatchedAsync(
        static_cast<const float*>(dQ.get()), static_cast<const float*>(dK.get()),
        static_cast<const float*>(dV.get()), static_cast<const float*>(dGc.get()),
        static_cast<const float*>(dB.get()), static_cast<const float*>(dA0.get()),
        static_cast<float*>(dS.get()), static_cast<float*>(dOut.get()),
        nSeq, T, H, S, C);
    ops.flush();
    auto outB = fromDevice(ops, dOut.get(), nSeq*actP);
    auto stateB = fromDevice(ops, dS.get(), nSeq*stP);

    // Reference: each sequence through the single-seq pipeline.
    double maxErr = 0.0;
    for (std::size_t s = 0; s < nSeq; ++s) {
        std::vector<float> q(Q.begin()+s*actP,Q.begin()+(s+1)*actP);
        std::vector<float> k(K.begin()+s*actP,K.begin()+(s+1)*actP);
        std::vector<float> v(V.begin()+s*actP,V.begin()+(s+1)*actP);
        std::vector<float> g(G.begin()+s*gateP,G.begin()+(s+1)*gateP);
        std::vector<float> b(B.begin()+s*gateP,B.begin()+(s+1)*gateP);
        std::vector<float> st(ST.begin()+s*stP,ST.begin()+(s+1)*stP);
        auto dq=toDevice(ops,q); auto dk=toDevice(ops,k); auto dv=toDevice(ops,v);
        auto dg=toDevice(ops,g); auto db=toDevice(ops,b); auto ds=toDevice(ops,st);
        auto dgc=ops.allocate(gateP*sizeof(float));
        auto da0=ops.allocate(a0P*sizeof(float));
        auto dout=ops.allocate(actP*sizeof(float));
        ops.deltanetChunkCumGateAsync(static_cast<const float*>(dg.get()),
            static_cast<float*>(dgc.get()), T, H, C);
        ops.deltanetKktSolveInverseAsync(static_cast<const float*>(dk.get()),
            static_cast<const float*>(db.get()),
            static_cast<float*>(da0.get()), T, H, S, C);
        ops.deltanetChunkForwardAsync(static_cast<const float*>(dq.get()),
            static_cast<const float*>(dk.get()), static_cast<const float*>(dv.get()),
            static_cast<const float*>(dgc.get()), static_cast<const float*>(db.get()),
            static_cast<const float*>(da0.get()), static_cast<float*>(ds.get()),
            static_cast<float*>(dout.get()), T, H, S, C);
        ops.flush();
        auto outS = fromDevice(ops, dout.get(), actP);
        auto stateS = fromDevice(ops, ds.get(), stP);
        for (std::size_t i = 0; i < actP; ++i) {
            maxErr = std::max(maxErr, std::fabs((double)outB[s*actP+i]-(double)outS[i]));
            EXPECT_NEAR(outB[s*actP+i], outS[i], 1e-5f);
        }
        for (std::size_t i = 0; i < stP; ++i) {
            maxErr = std::max(maxErr, std::fabs((double)stateB[s*stP+i]-(double)stateS[i]));
            EXPECT_NEAR(stateB[s*stP+i], stateS[i], 1e-5f);
        }
    }
    std::printf("[chunk-batched-parity] nSeq=%zu T=%zu H=%zu S=%zu C=%zu maxErr=%.2e\n",
                nSeq, T, H, S, C, maxErr);
}

int main() {
    return mm::test::run();
}