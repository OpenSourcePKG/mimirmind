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

#include <algorithm>
#include <cmath>
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

int main() {
    return mm::test::run();
}