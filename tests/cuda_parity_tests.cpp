// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// CUDA GPU-vs-CPU parity tests for the M-Q3N.3 GatedDeltaNet primitives.
// Runs each GPU op (compute::cuda::GpuOps) against the CPU golden reference
// (compute/GatedDeltaNet) on random inputs and asserts they agree. Built
// and run only on a CUDA host (MIMIRMIND_ENABLE_CUDA); this is the
// correctness gate the box build exercises before the kernels get wired
// into the Qwen3_5MoeBackend recurrent path.

#include "TestFramework.hpp"

#include "compute/Activations.hpp"
#include "compute/Attention.hpp"
#include "compute/Embedding.hpp"
#include "compute/GatedDeltaNet.hpp"
#include "compute/Norm.hpp"
#include "compute/Rope.hpp"
#include "compute/cuda/GpuMatmul.hpp"
#include "compute/cuda/GpuOps.hpp"
#include "core/gpu/cuda/CudaComputeContext.hpp"
#include "core/gpu/cuda/CudaKernel.hpp"
#include "core/gpu/cuda/CudaModule.hpp"
#include "core/gguf/GgufTypes.hpp"
#include "compute/quant/Q8_0.hpp"
#include "runtime/encoder/EncoderModel.hpp"
#include "runtime/encoder/EncoderRunner.hpp"
#include "model/XlmRobertaTokenizer.hpp"
#include "runtime/encoder/RerankEngine.hpp"

#include "core/modelopt/BlockScaleSwizzle.hpp" // swizzledBlockScaleBytes / swizzleBlockScale
#ifdef MIMIRMIND_HAVE_CUTLASS_MOE
#include "MoeGroupedGemmNvfp4Tc.hpp"          // E-d.3 CUTLASS grouped NVFP4-TC GEMM
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <string>
#include <string_view>
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

// Read a device int32 array back to host.
std::vector<std::int32_t>
fromDeviceI32(GpuOps& ops, const void* dev, std::size_t n) {
    std::vector<std::int32_t> h(n);
    ops.readbackToHost(h.data(), dev, n * sizeof(std::int32_t));
    return h;
}

// Locate a "<name>.ptx" the same way GpuMatmul/MoeTopKRouteDevice do, so a
// raw kernel with no GpuOps/GpuMatmul entry point yet can be launched through
// the driver API (CudaModule / CudaKernel) directly in a parity test.
std::string resolvePtx(std::string_view name) {
    const std::string filename = std::string{name} + ".ptx";
    const char* env = std::getenv("MIMIRMIND_HSACO_DIR");
    for (const auto& d : std::array<std::string, 6>{
             env ? std::string{env} : std::string{},
             "build/ptx", "build-cuda/ptx", "../build/ptx",
             "../build-cuda/ptx", "ptx"}) {
        if (d.empty()) {
            continue;
        }
        std::filesystem::path p = std::filesystem::path{d} / filename;
        if (std::filesystem::exists(p)) {
            return p.string();
        }
    }
    throw std::runtime_error(
        "cuda_parity_tests: cannot find " + filename +
        " — build the CUDA kernels or set MIMIRMIND_HSACO_DIR");
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

TEST(cuda_layernorm_parity) {
    CudaComputeContext ctx{};
    GpuOps ops{ctx};

    // XLM-R hidden = 1024; the EncoderRunner / cross-encoder reranker path.
    const std::size_t rows = 5, dim = 1024;
    const float eps = 1e-5f;
    auto host   = randVec(rows * dim, 0x1a1u);
    auto weight = randVec(dim, 0x2b2u);   // gamma
    auto bias   = randVec(dim, 0x3c3u);   // beta

    std::vector<float> ref(rows * dim);
    ::mimirmind::compute::layerNorm(host.data(), rows, dim,
                                    weight.data(), bias.data(), eps, ref.data());

    auto dx = toDevice(ops, host);
    auto dw = toDevice(ops, weight);
    auto db = toDevice(ops, bias);
    auto dy = ops.allocate(rows * dim * sizeof(float));
    ops.layerNormAsync(static_cast<const float*>(dx.get()), rows, dim,
                       static_cast<const float*>(dw.get()),
                       static_cast<const float*>(db.get()), eps,
                       static_cast<float*>(dy.get()));
    ops.flush();
    auto got = fromDevice(ops, dy.get(), rows * dim);

    for (std::size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], ref[i], 1e-3f);
    }
}

TEST(cuda_gelu_erf_parity) {
    CudaComputeContext ctx{};
    GpuOps ops{ctx};

    // XLM-R FFN width = 4096; a few rows. erf-GELU (EncoderRunner FFN).
    const std::size_t n = 4096 * 3;
    auto host = randVec(n, 0x9e3u);

    std::vector<float> ref = host;
    ::mimirmind::compute::geluErfInPlace(ref.data(), n);

    auto buf = toDevice(ops, host);
    ops.geluErfAsync(static_cast<float*>(buf.get()), n);
    ops.flush();
    auto got = fromDevice(ops, buf.get(), n);

    for (std::size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], ref[i], 1e-4f);
    }
}

TEST(cuda_attention_encoder_parity) {
    CudaComputeContext ctx{};
    GpuOps ops{ctx};

    // Non-causal (bidirectional) encoder MHA. nHeads == nKvHeads (no GQA).
    const std::size_t T = 12, H = 4, S = 16;
    const float scale = 0.25f;   // 1/sqrt(16)
    auto q = randVec(T * H * S, 0x51a1u);
    auto k = randVec(T * H * S, 0x52b2u);
    auto v = randVec(T * H * S, 0x53c3u);

    // positionOffset == T makes multiHeadAttention's min(posOff+p+1, T) clamp
    // let every query attend to ALL keys [0, T) — the full/bidirectional ref.
    std::vector<float> ref(T * H * S);
    std::vector<float> scratch(T);
    ::mimirmind::compute::multiHeadAttention(
        q.data(), k.data(), v.data(), T, T, H, H, S, /*positionOffset=*/T,
        scratch.data(), ref.data(), /*slidingWindow=*/0, scale);

    auto dq = toDevice(ops, q);
    auto dk = toDevice(ops, k);
    auto dv = toDevice(ops, v);
    auto dOut = ops.allocate(T * H * S * sizeof(float));
    ops.attentionEncoderAsync(static_cast<const float*>(dq.get()),
                              static_cast<const float*>(dk.get()),
                              static_cast<const float*>(dv.get()),
                              T, H, H, S, scale,
                              static_cast<float*>(dOut.get()));
    ops.flush();
    auto got = fromDevice(ops, dOut.get(), T * H * S);

    for (std::size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], ref[i], 1e-3f);
    }
}

TEST(cuda_attention_encoder_batched_parity) {
    CudaComputeContext ctx{};
    GpuOps ops{ctx};

    // Three packed sequences of different lengths; each row r = b*Tmax + t.
    const std::size_t H = 4, S = 16, Tmax = 16;
    const std::array<std::size_t, 3> lens{12, 7, 16};
    const std::size_t B = lens.size();
    const float scale = 1.0f / std::sqrt(static_cast<float>(S));
    const std::size_t rowElems = H * S;

    std::vector<float> q(B * Tmax * rowElems, 0.0f);
    std::vector<float> k(B * Tmax * rowElems, 0.0f);
    std::vector<float> vv(B * Tmax * rowElems, 0.0f);
    for (std::size_t b = 0; b < B; ++b) {
        auto qb = randVec(lens[b] * rowElems, 0x9100u + static_cast<std::uint32_t>(b));
        auto kb = randVec(lens[b] * rowElems, 0x9200u + static_cast<std::uint32_t>(b));
        auto vb = randVec(lens[b] * rowElems, 0x9300u + static_cast<std::uint32_t>(b));
        const std::size_t off = b * Tmax * rowElems;
        std::copy(qb.begin(), qb.end(), q.begin() + static_cast<std::ptrdiff_t>(off));
        std::copy(kb.begin(), kb.end(), k.begin() + static_cast<std::ptrdiff_t>(off));
        std::copy(vb.begin(), vb.end(), vv.begin() + static_cast<std::ptrdiff_t>(off));
    }

    std::vector<std::int32_t> seqLens(B);
    for (std::size_t b = 0; b < B; ++b) seqLens[b] = static_cast<std::int32_t>(lens[b]);

    auto dq = toDevice(ops, q);
    auto dk = toDevice(ops, k);
    auto dv = toDevice(ops, vv);
    auto dOut = ops.allocate(B * Tmax * rowElems * sizeof(float));
    auto dLens = ops.allocate(B * sizeof(std::int32_t));
    ops.uploadHostBytes(dLens.get(), seqLens.data(), B * sizeof(std::int32_t));

    ops.attentionEncoderBatchedAsync(
        static_cast<const float*>(dq.get()), static_cast<const float*>(dk.get()),
        static_cast<const float*>(dv.get()), static_cast<float*>(dOut.get()),
        static_cast<const std::int32_t*>(dLens.get()), B, Tmax, H, H, S, scale);
    ops.flush();
    auto got = fromDevice(ops, dOut.get(), B * Tmax * rowElems);

    // Each sequence must match the single-sequence reference over its length.
    for (std::size_t b = 0; b < B; ++b) {
        const std::size_t off = b * Tmax * rowElems;
        std::vector<float> ref(lens[b] * rowElems);
        std::vector<float> scratch(lens[b]);
        ::mimirmind::compute::multiHeadAttention(
            q.data() + off, k.data() + off, vv.data() + off,
            lens[b], lens[b], H, H, S, /*positionOffset=*/lens[b],
            scratch.data(), ref.data(), /*slidingWindow=*/0, scale);
        for (std::size_t i = 0; i < ref.size(); ++i) {
            EXPECT_NEAR(got[off + i], ref[i], 1e-3f);
        }
    }
}

TEST(cuda_encoder_embed_add_parity) {
    CudaComputeContext ctx{};
    GpuOps ops{ctx};

    // XLM-R hidden = 1024; posOffset 2 (pad_token_id 1). Additive pos+type.
    const std::size_t T = 12, hidden = 1024, posOffset = 2, maxPos = 64;
    auto x   = randVec(T * hidden, 0x71a1u);
    auto pos = randVec(maxPos * hidden, 0x72b2u);
    auto typ = randVec(hidden, 0x73c3u);

    std::vector<float> ref = x;
    ::mimirmind::compute::encoderEmbedAdd(ref.data(), pos.data(), typ.data(),
                                          T, hidden, posOffset);

    auto dx  = toDevice(ops, x);
    auto dp  = toDevice(ops, pos);
    auto dt  = toDevice(ops, typ);
    ops.encoderEmbedAddAsync(static_cast<float*>(dx.get()),
                             static_cast<const float*>(dp.get()),
                             static_cast<const float*>(dt.get()),
                             T, hidden, posOffset);
    ops.flush();
    auto got = fromDevice(ops, dx.get(), T * hidden);

    for (std::size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], ref[i], 1e-4f);
    }
}

TEST(cuda_tanh_parity) {
    CudaComputeContext ctx{};
    GpuOps ops{ctx};

    // RoBERTa/XLM-R classifier-head activation (dense -> tanh -> out_proj).
    const std::size_t n = 1024 * 3;
    auto host = randVec(n, 0x7a11u);

    std::vector<float> ref = host;
    ::mimirmind::compute::tanhInPlace(ref.data(), n);

    auto buf = toDevice(ops, host);
    ops.tanhInPlaceAsync(static_cast<float*>(buf.get()), n);
    ops.flush();
    auto got = fromDevice(ops, buf.get(), n);

    for (std::size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], ref[i], 1e-5f);
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

// ---- Encoder layer-0 parity (Phase 0b integration) ----------------------
// Composes the five encoder primitives (encoderEmbedAdd, layerNorm,
// non-causal attention, erf-GELU) plus the trusted matmul/addBias/residual
// path into the real XLM-R (bge-reranker-v2-m3) embeddings block + first
// transformer layer, and checks it bit-close against a HF reference dumped
// to a flat fixture (scratchpad/encoder_oracle.bin). This validates the
// KERNEL COMPOSITION with real weights — the last de-risk before wiring the
// full 24-layer EncoderRunner (which the GGUF loader + tokenizer feed).
struct OracleRec {
    std::vector<int>            dims;
    std::vector<float>          f;   // dtype 0
    std::vector<std::int32_t>   i;   // dtype 1
    std::size_t count() const {
        std::size_t n = 1;
        for (int d : dims) n *= static_cast<std::size_t>(d);
        return n;
    }
};
using OracleMap = std::map<std::string, OracleRec>;

bool readOracle(const std::string& path, OracleMap& out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    auto rd = [&](std::int32_t& v) {
        in.read(reinterpret_cast<char*>(&v), sizeof(v));
    };
    std::int32_t count = 0;
    rd(count);
    for (std::int32_t r = 0; r < count; ++r) {
        std::int32_t nameLen = 0;
        rd(nameLen);
        std::string name(static_cast<std::size_t>(nameLen), '\0');
        in.read(name.data(), nameLen);
        std::int32_t dtype = 0, ndim = 0;
        rd(dtype);
        rd(ndim);
        OracleRec rec;
        std::size_t n = 1;
        for (int d = 0; d < ndim; ++d) {
            std::int32_t s = 0;
            rd(s);
            rec.dims.push_back(s);
            n *= static_cast<std::size_t>(s);
        }
        if (dtype == 1) {
            rec.i.resize(n);
            in.read(reinterpret_cast<char*>(rec.i.data()),
                    static_cast<std::streamsize>(n * sizeof(std::int32_t)));
        } else {
            rec.f.resize(n);
            in.read(reinterpret_cast<char*>(rec.f.data()),
                    static_cast<std::streamsize>(n * sizeof(float)));
        }
        out.emplace(std::move(name), std::move(rec));
    }
    return static_cast<bool>(in);
}
} // namespace

TEST(cuda_encoder_layer0_parity) {
    const char* envp = std::getenv("MIMIRMIND_ENCODER_ORACLE");
    const std::string path = envp ? envp : "scratchpad/encoder_oracle.bin";
    OracleMap O;
    if (!readOracle(path, O)) {
        std::printf("[SKIP] encoder oracle fixture not found at %s\n",
                    path.c_str());
        return;
    }

    CudaComputeContext ctx{};
    GpuOps    ops{ctx};
    GpuMatmul gmm{ctx, ops};
    const auto F32 = GgmlType::F32;

    // XLM-R-large (bge-reranker-v2-m3) shape.
    const std::size_t T   = static_cast<std::size_t>(O.at("word_emb").dims[0]);
    const std::size_t H   = 1024;
    const std::size_t hds = 16;    // heads
    const std::size_t hd  = 64;    // head dim
    const std::size_t ffn = 4096;
    const float eps       = 1e-5f;
    const float scale     = 1.0f / std::sqrt(static_cast<float>(hd));
    const std::size_t posOffset = 2;   // pad_token_id(1) + 1

    auto W = [&](const char* k) { return toDevice(ops, O.at(k).f); };
    auto dPos    = W("pos_table");
    auto dType   = W("type_vec");
    auto dEmbLnW = W("emb_ln_w");
    auto dEmbLnB = W("emb_ln_b");
    auto dQw = W("q_w"),  dQb = W("q_b");
    auto dKw = W("k_w"),  dKb = W("k_b");
    auto dVw = W("v_w"),  dVb = W("v_b");
    auto dAoW = W("ao_w"), dAoB = W("ao_b");
    auto dAoLnW = W("ao_ln_w"), dAoLnB = W("ao_ln_b");
    auto dFiW = W("fi_w"), dFiB = W("fi_b");
    auto dFoW = W("fo_w"), dFoB = W("fo_b");
    auto dOutLnW = W("out_ln_w"), dOutLnB = W("out_ln_b");

    auto ptr = [](auto& b) { return static_cast<float*>(b.get()); };
    auto cptr = [](auto& b) { return static_cast<const float*>(b.get()); };

    auto scratch = ops.allocate(T * ffn * sizeof(float));

    // ---- embeddings block: word + pos + type, then LayerNorm ----
    auto x = toDevice(ops, O.at("word_emb").f);   // [T,H] starts at word emb
    ops.encoderEmbedAddAsync(ptr(x), cptr(dPos), cptr(dType), T, H, posOffset);
    auto embOut = ops.allocate(T * H * sizeof(float));
    ops.layerNormAsync(ptr(x), T, H, cptr(dEmbLnW), cptr(dEmbLnB), eps,
                       ptr(embOut));
    ops.flush();

    auto cmp = [&](const char* tag, const std::vector<float>& got,
                   const std::vector<float>& ref, float tol) {
        double mx = 0.0;
        std::size_t at = 0;
        for (std::size_t k = 0; k < got.size(); ++k) {
            const double e = std::fabs(static_cast<double>(got[k]) -
                                       static_cast<double>(ref[k]));
            if (e > mx) { mx = e; at = k; }
        }
        std::printf("    [%s] max_abs_err=%.6g at %zu (got %.6g ref %.6g)\n",
                    tag, mx, at, got[at], ref[at]);
        EXPECT_NEAR(static_cast<float>(mx), 0.0f, tol);
    };

    cmp("emb_out", fromDevice(ops, embOut.get(), T * H),
        O.at("emb_out").f, 2e-3f);

    // ---- encoder layer 0 ----
    auto qBuf   = ops.allocate(T * H * sizeof(float));
    auto kBuf   = ops.allocate(T * H * sizeof(float));
    auto vBuf   = ops.allocate(T * H * sizeof(float));
    auto attn   = ops.allocate(T * H * sizeof(float));
    auto aoBuf  = ops.allocate(T * H * sizeof(float));
    auto ln1    = ops.allocate(T * H * sizeof(float));
    auto inter  = ops.allocate(T * ffn * sizeof(float));
    auto ffnBuf = ops.allocate(T * H * sizeof(float));
    auto l0     = ops.allocate(T * H * sizeof(float));

    // Q/K/V = h * Wqkv^T + b   (h = embOut)
    gmm.matmulAsync(F32, cptr(dQw), H, H, cptr(embOut), T, ptr(qBuf), ptr(scratch));
    ops.addBiasAsync(ptr(qBuf), T, H, cptr(dQb));
    gmm.matmulAsync(F32, cptr(dKw), H, H, cptr(embOut), T, ptr(kBuf), ptr(scratch));
    ops.addBiasAsync(ptr(kBuf), T, H, cptr(dKb));
    gmm.matmulAsync(F32, cptr(dVw), H, H, cptr(embOut), T, ptr(vBuf), ptr(scratch));
    ops.addBiasAsync(ptr(vBuf), T, H, cptr(dVb));

    // non-causal (bidirectional) MHA
    ops.attentionEncoderAsync(cptr(qBuf), cptr(kBuf), cptr(vBuf), T, hds, hds,
                              hd, scale, ptr(attn));

    // attention output dense + residual(h) + LayerNorm
    gmm.matmulAsync(F32, cptr(dAoW), H, H, cptr(attn), T, ptr(aoBuf), ptr(scratch));
    ops.addBiasAsync(ptr(aoBuf), T, H, cptr(dAoB));
    ops.addResidualAsync(ptr(aoBuf), cptr(embOut), T * H);
    ops.layerNormAsync(ptr(aoBuf), T, H, cptr(dAoLnW), cptr(dAoLnB), eps, ptr(ln1));

    // FFN: intermediate (erf-GELU) -> output dense + residual(ln1) + LayerNorm
    gmm.matmulAsync(F32, cptr(dFiW), ffn, H, cptr(ln1), T, ptr(inter), ptr(scratch));
    ops.addBiasAsync(ptr(inter), T, ffn, cptr(dFiB));
    ops.geluErfAsync(ptr(inter), T * ffn);
    gmm.matmulAsync(F32, cptr(dFoW), H, ffn, cptr(inter), T, ptr(ffnBuf), ptr(scratch));
    ops.addBiasAsync(ptr(ffnBuf), T, H, cptr(dFoB));
    ops.addResidualAsync(ptr(ffnBuf), cptr(ln1), T * H);
    ops.layerNormAsync(ptr(ffnBuf), T, H, cptr(dOutLnW), cptr(dOutLnB), eps, ptr(l0));
    ops.flush();

    cmp("layer0_out", fromDevice(ops, l0.get(), T * H),
        O.at("layer0_out").f, 5e-3f);
}

// Full 24-layer forward through the production EncoderModel + EncoderRunner
// against the real bge-reranker-v2-m3 safetensors on disk: loads the dense
// F32 checkpoint natively (no GGUF), feeds the oracle's tokenized input_ids,
// and checks the classifier logit against the HF reference score. Validates
// the loader + the whole runner end-to-end (tokenizer excluded — ids come
// from the fixture). Self-skips if the fixture or the model dir is absent.
TEST(cuda_encoder_full_forward_parity) {
    const char* envp = std::getenv("MIMIRMIND_ENCODER_ORACLE");
    const std::string fx = envp ? envp : "scratchpad/encoder_oracle.bin";
    OracleMap O;
    if (!readOracle(fx, O)) {
        std::printf("[SKIP] encoder oracle fixture not found at %s\n", fx.c_str());
        return;
    }
    const char* mdl = std::getenv("MIMIRMIND_BGE_DIR");
    const std::string dir = mdl ? mdl : "models/bge-reranker-v2-m3";
    if (!std::filesystem::exists(std::filesystem::path(dir) / "config.json")) {
        std::printf("[SKIP] bge model dir not found at %s\n", dir.c_str());
        return;
    }

    CudaComputeContext ctx{};
    GpuOps    ops{ctx};
    GpuMatmul gmm{ctx, ops};

    ::mimirmind::runtime::encoder::EncoderModel model;
    model.load(dir, ops);
    ::mimirmind::runtime::encoder::EncoderRunner runner{model, ops, gmm};

    const std::vector<std::int32_t>& ids = O.at("input_ids").i;
    const std::vector<float> logits = runner.forwardLogits(ids);
    const float ref = O.at("final_score").f.at(0);

    std::printf("    [full_forward] score=%.6f ref=%.6f (T=%zu layers=%zu labels=%zu)\n",
                logits.at(0), ref, ids.size(), model.config().numLayers,
                model.config().numLabels);
    EXPECT_NEAR(logits.at(0), ref, 5e-2f);
}

// Batched forward parity: running B sequences of DIFFERENT lengths through
// forwardLogitsBatch (padded + per-sequence attention masking) must match
// scoring each one individually with forwardLogits. Validates the batched
// EncoderRunner path (padding never leaks across sequences).
TEST(cuda_encoder_batched_forward_parity) {
    const char* envp = std::getenv("MIMIRMIND_ENCODER_ORACLE");
    const std::string fx = envp ? envp : "scratchpad/encoder_oracle.bin";
    OracleMap O;
    if (!readOracle(fx, O)) {
        std::printf("[SKIP] encoder oracle fixture not found at %s\n", fx.c_str());
        return;
    }
    const char* mdl = std::getenv("MIMIRMIND_BGE_DIR");
    const std::string dir = mdl ? mdl : "models/bge-reranker-v2-m3";
    if (!std::filesystem::exists(std::filesystem::path(dir) / "config.json")) {
        std::printf("[SKIP] bge model dir not found at %s\n", dir.c_str());
        return;
    }

    CudaComputeContext ctx{};
    GpuOps    ops{ctx};
    GpuMatmul gmm{ctx, ops};
    ::mimirmind::runtime::encoder::EncoderModel model;
    model.load(dir, ops);
    ::mimirmind::runtime::encoder::EncoderRunner runner{model, ops, gmm};

    const std::vector<std::int32_t>& ids = O.at("input_ids").i;
    // Three sequences of differing lengths (full, and two truncations that
    // keep the leading <s> so the [CLS] head still has a valid row 0).
    std::vector<std::vector<std::int32_t>> seqs = {
        ids,
        std::vector<std::int32_t>(ids.begin(), ids.begin() + 40),
        std::vector<std::int32_t>(ids.begin(), ids.begin() + 60),
    };

    const auto batched = runner.forwardLogitsBatch(
        std::span<const std::vector<std::int32_t>>{seqs});
    EXPECT_EQ(batched.size(), seqs.size());

    for (std::size_t b = 0; b < seqs.size(); ++b) {
        const auto single = runner.forwardLogits(seqs[b]);
        std::printf("    [batch b=%zu len=%zu] batched=%.6f single=%.6f\n",
                    b, seqs[b].size(), batched[b].at(0), single.at(0));
        EXPECT_NEAR(batched[b].at(0), single.at(0), 2e-3f);
    }
}

// XLM-R Unigram tokenizer parity: the oracle's input_ids came from the HF
// tokenizer on this exact (query, passage) pair, so reproducing them bit-for-
// bit validates normalize + Viterbi + fairseq id remap + sentence-pair framing.
// The strings are reference test data (the oracle pair), not prose. No GPU.
TEST(cuda_xlmr_tokenizer_parity) {
    const char* envp = std::getenv("MIMIRMIND_ENCODER_ORACLE");
    const std::string fx = envp ? envp : "scratchpad/encoder_oracle.bin";
    OracleMap O;
    if (!readOracle(fx, O)) {
        std::printf("[SKIP] encoder oracle fixture not found at %s\n", fx.c_str());
        return;
    }
    const char* mdl = std::getenv("MIMIRMIND_BGE_DIR");
    const std::string dir = mdl ? mdl : "models/bge-reranker-v2-m3";
    const std::string spm = dir + "/sentencepiece.bpe.model";
    if (!std::filesystem::exists(spm)) {
        std::printf("[SKIP] sentencepiece model not found at %s\n", spm.c_str());
        return;
    }

    // The oracle reference pair (gen_oracle.py) — real DE text with umlauts.
    // UTF-8 literals directly (this file is UTF-8); \x escapes would swallow a
    // following hex-digit char (e.g. "\xBC" + 'c' in "Rück").
    const std::string query =
        "Welche Anschlüsse hat der NUC10 an der Vorder- und Rückseite?";
    const std::string passage =
        "Der Intel NUC10 verfügt an der Vorderseite über zwei "
        "USB-3.1-Ports und einen Kopfhörer-Ausgang; an der Rückseite "
        "HDMI 2.0, USB-C/Thunderbolt, zwei USB-3.1, Gigabit-Ethernet (RJ-45) und "
        "den Stromanschluss.";

    ::mimirmind::model::XlmRobertaTokenizer tok;
    tok.load(spm);
    const std::vector<std::int32_t> got = tok.encodePair(query, passage);
    const std::vector<std::int32_t>& ref = O.at("input_ids").i;

    std::printf("    [xlmr_tok] vocab=%zu got %zu ids, ref %zu ids\n",
                tok.vocabSize(), got.size(), ref.size());
    EXPECT_EQ(got.size(), ref.size());
    const std::size_t nmin = std::min(got.size(), ref.size());
    std::size_t mism = 0;
    for (std::size_t i = 0; i < nmin; ++i) {
        if (got[i] != ref[i]) {
            if (mism < 5) {
                std::printf("    diff [%zu] got=%d ref=%d\n", i, got[i], ref[i]);
            }
            ++mism;
        }
    }
    EXPECT_EQ(mism, std::size_t{0});
}

// End-to-end RerankEngine smoke: real bge-reranker-v2-m3 weights + tokenizer,
// rank a relevant vs an irrelevant document against a query. Validates the
// full serving wrapper (tokenize pair -> forward -> score -> sort). No fixture
// needed for the assertion; skips if the model dir is absent.
TEST(cuda_rerank_engine_smoke) {
    const char* mdl = std::getenv("MIMIRMIND_BGE_DIR");
    const std::string dir = mdl ? mdl : "models/bge-reranker-v2-m3";
    if (!std::filesystem::exists(std::filesystem::path(dir) / "config.json")) {
        std::printf("[SKIP] bge model dir not found at %s\n", dir.c_str());
        return;
    }

    CudaComputeContext ctx{};
    GpuOps    ops{ctx};
    GpuMatmul gmm{ctx, ops};
    ::mimirmind::runtime::encoder::RerankEngine re{dir, ops, gmm};

    const std::string query = "Welche Anschlüsse hat der NUC10?";
    const std::vector<std::string> docs = {
        "Der Intel NUC10 hat an der Rückseite HDMI, USB-C/Thunderbolt und "
        "Gigabit-Ethernet.",                                  // relevant
        "Die Hauptstadt von Frankreich ist Paris, durchflossen von der Seine.",
    };
    const auto results = re.rerank(query, std::span<const std::string>{docs});

    EXPECT_EQ(results.size(), docs.size());
    std::printf("    [rerank] top index=%zu score=%.4f  (other score=%.4f)\n",
                results[0].index, results[0].score, results[1].score);
    EXPECT_EQ(results[0].index, std::size_t{0});         // relevant ranks first
    EXPECT_TRUE(results[0].score > results[1].score);
}

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
        static_cast<float*>(dOut.get()),
        mimirmind::compute::GdnBatchedShape{nSeq, T, H, S});
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
        // Batched vs single-seq differ only in FMA/reduction ordering, so hold
        // each element to a magnitude-relative bound (atol + rtol*|ref|). A flat
        // 1e-5 atol demands ~4e-7 relative on large-magnitude state entries
        // (|state| ~= 25), which f32 cannot guarantee across reassociation and
        // which the CUDA 13.3 toolchain's FMA contraction exceeds by ~14%.
        for (std::size_t i = 0; i < act; ++i) {
            maxErr = std::max(maxErr, std::fabs((double)outB[s*act+i] - (double)outS[i]));
            EXPECT_NEAR(outB[s*act+i], outS[i], 1e-5f + 1e-6f * std::fabs(outS[i]));
        }
        for (std::size_t i = 0; i < stt; ++i) {
            maxErr = std::max(maxErr, std::fabs((double)stateB[s*stt+i] - (double)stateS[i]));
            EXPECT_NEAR(stateB[s*stt+i], stateS[i], 1e-5f + 1e-6f * std::fabs(stateS[i]));
        }
    }
    std::printf("[gdn-batched-parity] nSeq=%zu T=%zu H=%zu S=%zu maxErr=%.2e\n",
                nSeq, T, H, S, maxErr);
}

// 5.21 Increment II — VARLEN batched GDN recurrence: ragged per-slot seqT
// (mixed token counts) vs N single-seq runs each with its own T. Proves the
// ragged seqT/seqOff indexing produces the same math as per-slot decode/prefill.
TEST(cuda_gated_deltanet_ar_batched_varlen_parity) {
    CudaComputeContext ctx{};
    GpuOps ops{ctx};

    const std::size_t H = 4, S = 16;
    const std::vector<std::int32_t> seqT = {2, 5, 3};   // ragged per-slot tokens
    const std::size_t nSeq = seqT.size();
    std::vector<std::int32_t> seqOff(nSeq, 0);
    std::size_t totalTok = 0;
    for (std::size_t s = 0; s < nSeq; ++s) {
        seqOff[s] = static_cast<std::int32_t>(totalTok);
        totalTok += static_cast<std::size_t>(seqT[s]);
    }
    const std::size_t hs = H * S, stt = H * S * S;

    std::vector<float> Q(totalTok*hs), K(totalTok*hs), V(totalTok*hs);
    std::vector<float> G(totalTok*H),  B(totalTok*H),  ST(nSeq*stt);
    for (std::size_t s = 0; s < nSeq; ++s) {
        const std::uint32_t o = static_cast<std::uint32_t>(s) * 7u;
        const std::size_t tt = static_cast<std::size_t>(seqT[s]);
        const std::size_t to = static_cast<std::size_t>(seqOff[s]);
        auto q=randVec(tt*hs,0x1a1u+o); auto k=randVec(tt*hs,0x1b2u+o);
        auto v=randVec(tt*hs,0x1c3u+o); auto g=randVec(tt*H,0x1d4u+o);
        auto b=randVec(tt*H,0x1e5u+o);  auto st=randVec(stt,0x1f6u+o);
        std::copy(q.begin(),q.end(),Q.begin()+to*hs);
        std::copy(k.begin(),k.end(),K.begin()+to*hs);
        std::copy(v.begin(),v.end(),V.begin()+to*hs);
        std::copy(g.begin(),g.end(),G.begin()+to*H);
        std::copy(b.begin(),b.end(),B.begin()+to*H);
        std::copy(st.begin(),st.end(),ST.begin()+s*stt);
    }

    auto dQ=toDevice(ops,Q); auto dK=toDevice(ops,K); auto dV=toDevice(ops,V);
    auto dG=toDevice(ops,G); auto dB=toDevice(ops,B); auto dSt=toDevice(ops,ST);
    auto dOut = ops.allocate(totalTok*hs*sizeof(float));
    auto dSeqT = ops.allocate(nSeq*sizeof(std::int32_t));
    auto dSeqOff = ops.allocate(nSeq*sizeof(std::int32_t));
    ops.uploadHostBytes(dSeqT.get(),   seqT.data(),   nSeq*sizeof(std::int32_t));
    ops.uploadHostBytes(dSeqOff.get(), seqOff.data(), nSeq*sizeof(std::int32_t));
    std::size_t maxT = 1; for (auto t : seqT) maxT = std::max(maxT, (std::size_t)t);
    mimirmind::compute::GdnBatchedShape shp{
        nSeq, maxT, H, S, /*activeMask=*/nullptr,
        static_cast<const std::int32_t*>(dSeqT.get()),
        static_cast<const std::int32_t*>(dSeqOff.get())};
    ops.gatedDeltaNetRecurrentBatchedAsync(
        static_cast<const float*>(dQ.get()), static_cast<const float*>(dK.get()),
        static_cast<const float*>(dV.get()), static_cast<const float*>(dG.get()),
        static_cast<const float*>(dB.get()), static_cast<float*>(dSt.get()),
        static_cast<float*>(dOut.get()), shp);
    ops.flush();
    auto outB   = fromDevice(ops, dOut.get(), totalTok*hs);
    auto stateB = fromDevice(ops, dSt.get(),  nSeq*stt);

    double maxErr = 0.0;
    for (std::size_t s = 0; s < nSeq; ++s) {
        const std::size_t tt = static_cast<std::size_t>(seqT[s]);
        const std::size_t to = static_cast<std::size_t>(seqOff[s]);
        std::vector<float> q(Q.begin()+to*hs, Q.begin()+(to+tt)*hs);
        std::vector<float> k(K.begin()+to*hs, K.begin()+(to+tt)*hs);
        std::vector<float> v(V.begin()+to*hs, V.begin()+(to+tt)*hs);
        std::vector<float> g(G.begin()+to*H,  G.begin()+(to+tt)*H);
        std::vector<float> b(B.begin()+to*H,  B.begin()+(to+tt)*H);
        std::vector<float> st(ST.begin()+s*stt, ST.begin()+(s+1)*stt);
        auto dq=toDevice(ops,q); auto dk=toDevice(ops,k); auto dv=toDevice(ops,v);
        auto dg=toDevice(ops,g); auto db=toDevice(ops,b); auto dst=toDevice(ops,st);
        auto dSingle = ops.allocate(tt*hs*sizeof(float));
        ops.gatedDeltaNetRecurrentAsync(
            static_cast<const float*>(dq.get()), static_cast<const float*>(dk.get()),
            static_cast<const float*>(dv.get()), static_cast<const float*>(dg.get()),
            static_cast<const float*>(db.get()), static_cast<float*>(dst.get()),
            static_cast<float*>(dSingle.get()), tt, H, S);
        ops.flush();
        auto outS   = fromDevice(ops, dSingle.get(), tt*hs);
        auto stateS = fromDevice(ops, dst.get(),     stt);
        for (std::size_t i = 0; i < tt*hs; ++i) {
            maxErr = std::max(maxErr, std::fabs((double)outB[to*hs+i]-(double)outS[i]));
            EXPECT_NEAR(outB[to*hs+i], outS[i], 1e-5f + 1e-6f*std::fabs(outS[i]));
        }
        for (std::size_t i = 0; i < stt; ++i) {
            maxErr = std::max(maxErr, std::fabs((double)stateB[s*stt+i]-(double)stateS[i]));
            EXPECT_NEAR(stateB[s*stt+i], stateS[i], 1e-5f + 1e-6f*std::fabs(stateS[i]));
        }
    }
    std::printf("[gdn-varlen-parity] nSeq=%zu seqT={2,5,3} H=%zu S=%zu maxErr=%.2e\n",
                nSeq, H, S, maxErr);
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

// 5.21 Increment II — VARLEN batched ssm_conv1d: ragged per-slot seqT (each slot
// carries its own K-1 conv-tail) vs N single-seq convs each with its own T.
TEST(cuda_ssm_conv1d_batched_varlen_parity) {
    CudaComputeContext ctx{};
    GpuOps ops{ctx};

    const std::size_t channels = 12, K = 4;
    const std::vector<std::int32_t> seqT = {2, 5, 3};
    const std::size_t nSeq = seqT.size();
    std::vector<std::int32_t> inOff(nSeq, 0), outOff(nSeq, 0);
    std::size_t inTot = 0, outTot = 0;
    for (std::size_t s = 0; s < nSeq; ++s) {
        inOff[s]  = static_cast<std::int32_t>(inTot);
        outOff[s] = static_cast<std::int32_t>(outTot);
        inTot  += static_cast<std::size_t>(seqT[s]) + K - 1;   // tail + tokens
        outTot += static_cast<std::size_t>(seqT[s]);
    }
    auto kernel = randVec(K * channels, 0x3733u);
    std::vector<float> IN(inTot * channels);
    for (std::size_t s = 0; s < nSeq; ++s) {
        const std::size_t rows = static_cast<std::size_t>(seqT[s]) + K - 1;
        auto in = randVec(rows * channels, 0x2700u + static_cast<std::uint32_t>(s)*13u);
        std::copy(in.begin(), in.end(),
                  IN.begin() + static_cast<std::size_t>(inOff[s]) * channels);
    }

    auto dIn  = toDevice(ops, IN);
    auto dKer = toDevice(ops, kernel);
    auto dOut = ops.allocate(outTot * channels * sizeof(float));
    auto dSeqT = ops.allocate(nSeq*sizeof(std::int32_t));
    auto dInOff = ops.allocate(nSeq*sizeof(std::int32_t));
    auto dOutOff = ops.allocate(nSeq*sizeof(std::int32_t));
    ops.uploadHostBytes(dSeqT.get(),   seqT.data(),   nSeq*sizeof(std::int32_t));
    ops.uploadHostBytes(dInOff.get(),  inOff.data(),  nSeq*sizeof(std::int32_t));
    ops.uploadHostBytes(dOutOff.get(), outOff.data(), nSeq*sizeof(std::int32_t));
    std::size_t maxT = 1; for (auto t : seqT) maxT = std::max(maxT, (std::size_t)t);
    ops.causalConv1dSiluBatchedAsync(
        static_cast<const float*>(dIn.get()), static_cast<const float*>(dKer.get()),
        static_cast<float*>(dOut.get()), nSeq, maxT, channels, K,
        static_cast<const std::int32_t*>(dSeqT.get()),
        static_cast<const std::int32_t*>(dInOff.get()),
        static_cast<const std::int32_t*>(dOutOff.get()));
    ops.flush();
    auto outB = fromDevice(ops, dOut.get(), outTot * channels);

    double maxErr = 0.0;
    for (std::size_t s = 0; s < nSeq; ++s) {
        const std::size_t tt = static_cast<std::size_t>(seqT[s]);
        const std::size_t rows = tt + K - 1;
        std::vector<float> in(IN.begin() + (std::size_t)inOff[s]*channels,
                              IN.begin() + ((std::size_t)inOff[s]+rows)*channels);
        auto di = toDevice(ops, in); auto dk = toDevice(ops, kernel);
        auto doS = ops.allocate(tt*channels*sizeof(float));
        ops.causalConv1dSiluAsync(static_cast<const float*>(di.get()),
                                  static_cast<const float*>(dk.get()),
                                  static_cast<float*>(doS.get()), tt, channels, K);
        ops.flush();
        auto outS = fromDevice(ops, doS.get(), tt*channels);
        for (std::size_t i = 0; i < tt*channels; ++i) {
            const std::size_t o = (std::size_t)outOff[s]*channels + i;
            maxErr = std::max(maxErr, std::fabs((double)outB[o] - (double)outS[i]));
            EXPECT_NEAR(outB[o], outS[i], 1e-5f);
        }
    }
    std::printf("[conv1d-varlen-parity] nSeq=%zu seqT={2,5,3} channels=%zu K=%zu maxErr=%.2e\n",
                nSeq, channels, K, maxErr);
}

// 5.21 Increment II — VARLEN KV write: the batched KV scatter is per-ROW, so a
// prefill slot's chunk = several per-token rows with per-token block/slot targets.
// This locks in that guarantee (per-token placement) + the activeMask freeze that
// Increment III relies on; the kernel needs no varlen-specific change.
TEST(cuda_kv_write_tokens_batched_varlen_mask) {
    CudaComputeContext ctx{};
    GpuOps ops{ctx};

    const std::size_t nKvHeads = 2, headSize = 8, width = nKvHeads * headSize;
    const std::size_t blockSize = 4, numBlocks = 8;
    // 5 rows: slot A prefill tokens @pos 0,1,2 (block 0 slots 0,1,2), slot B decode
    // @pos 5 (block 1 slot 1), slot C FROZEN (must not be written).
    const std::size_t nRow = 5;
    std::vector<std::uint32_t> wbid = {0, 0, 0, 1, 2};
    std::vector<std::int32_t>  wslot= {0, 1, 2, 1, 0};
    std::vector<std::uint8_t>  mask = {1, 1, 1, 1, 0};   // row 4 (slot C) frozen

    std::vector<float> kProj(nRow*width), vProj(nRow*width);
    for (std::size_t i = 0; i < nRow*width; ++i) { kProj[i] = 1.f + 0.01f*i; vProj[i] = 2.f + 0.01f*i; }

    const std::size_t poolElems = numBlocks*blockSize*width;
    std::vector<float> kInit(poolElems, -7.f), vInit(poolElems, -9.f);  // sentinel
    auto dK = toDevice(ops, kInit); auto dV = toDevice(ops, vInit);
    auto dKp = toDevice(ops, kProj); auto dVp = toDevice(ops, vProj);
    auto dWb = ops.allocate(nRow*sizeof(std::uint32_t));
    auto dWs = ops.allocate(nRow*sizeof(std::int32_t));
    auto dMask = ops.allocate(nRow*sizeof(std::uint8_t));
    ops.uploadHostBytes(dWb.get(),   wbid.data(),  nRow*sizeof(std::uint32_t));
    ops.uploadHostBytes(dWs.get(),   wslot.data(), nRow*sizeof(std::int32_t));
    ops.uploadHostBytes(dMask.get(), mask.data(),  nRow*sizeof(std::uint8_t));

    ops.writeKvTokensBatchedAsync(
        static_cast<const float*>(dKp.get()), static_cast<const float*>(dVp.get()),
        static_cast<const std::uint32_t*>(dWb.get()),
        static_cast<const std::int32_t*>(dWs.get()),
        dK.get(), dV.get(), nRow, blockSize, width,
        mimirmind::runtime::KvDtype::F32,
        static_cast<const std::uint8_t*>(dMask.get()));
    ops.flush();
    auto kOut = fromDevice(ops, dK.get(), poolElems);
    auto vOut = fromDevice(ops, dV.get(), poolElems);

    double maxErr = 0.0;
    for (std::size_t r = 0; r < nRow; ++r) {
        const std::size_t off = (static_cast<std::size_t>(wbid[r])*blockSize
                                 + static_cast<std::size_t>(wslot[r])) * width;
        for (std::size_t j = 0; j < width; ++j) {
            if (mask[r] == 0) {   // frozen row: destination must stay sentinel
                EXPECT_NEAR(kOut[off+j], -7.f, 0.f);
                EXPECT_NEAR(vOut[off+j], -9.f, 0.f);
            } else {              // active row: destination == this row's proj
                maxErr = std::max(maxErr, std::fabs((double)kOut[off+j]-(double)kProj[r*width+j]));
                EXPECT_NEAR(kOut[off+j], kProj[r*width+j], 0.f);
                EXPECT_NEAR(vOut[off+j], vProj[r*width+j], 0.f);
            }
        }
    }
    std::printf("[kv-write-varlen-mask] nRow=%zu width=%zu (row4 frozen) maxErr=%.2e\n",
                nRow, width, maxErr);
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

// IMRoPE partial rotary (Qwen3-Next / Qwen3.5-MoE, partial_rotary_factor =
// 0.25). head_dim = 256, so only rotary_dim = 64 head dims (= 32 pairs) rotate
// and the remaining 192 pass through untouched; the freq denominator is
// rotary_dim (64), NOT head_dim (256). Guards the norm-blind directional bug
// that shipped incoherent long-gen on Qwen3.6-NVFP4 (the kernel used to rotate
// all 128 pairs with denom 512). Mirrors the L0 gpu_tests case: the CUDA kernel
// is held to the compute:: partial-rotary reference, plus a reference-
// independent pass-through invariant so BOTH paths regressing to full-head
// rotation together is still caught.
TEST(cuda_mrope_partial_rotary_parity) {
    CudaComputeContext ctx{};
    GpuOps ops{ctx};

    const std::size_t seqLen = 3, numHeads = 2, headDim = 256;
    const std::size_t startPos = 7;                   // non-zero base position
    const float base = 10000.0f;
    // GGUF <arch>.rope.dimension_sections; sum is rotary_dim/2 -> 32 pairs.
    const std::int32_t sections[4] = {16, 8, 8, 0};
    const std::size_t rotaryDim = 64;                 // 2 * sum(sections)

    const std::size_t n = seqLen * numHeads * headDim;
    auto x = randVec(n, 0x63u);

    auto dX = toDevice(ops, x);
    ops.mropeInPlaceAsync(dX.get(), seqLen, numHeads, headDim, startPos, base,
                          sections);
    ops.flush();
    auto got = fromDevice(ops, dX.get(), n);

    std::vector<float> ref = x;
    ::mimirmind::compute::applyMropeInPlace(ref.data(), seqLen, numHeads,
                                            headDim, startPos, base, sections);

    // (1) CUDA kernel matches the compute:: partial-rotary reference.
    double maxErr = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        maxErr = std::max(maxErr, std::fabs((double)got[i] - (double)ref[i]));
        EXPECT_NEAR(got[i], ref[i], 1e-4f);
    }

    // (2) Reference-independent invariants: head dims [rotaryDim, headDim) are
    // byte-identical to the input (pass-through), and the rotary block
    // [0, rotaryDim) actually changed.
    float passMaxDiff = 0.0f;
    bool  rotaryMoved = false;
    for (std::size_t p = 0; p < seqLen; ++p) {
        for (std::size_t h = 0; h < numHeads; ++h) {
            const std::size_t off = (p * numHeads + h) * headDim;
            for (std::size_t d = rotaryDim; d < headDim; ++d) {
                passMaxDiff = std::max(passMaxDiff,
                                       std::fabs(got[off + d] - x[off + d]));
            }
            for (std::size_t d = 0; d < rotaryDim; ++d) {
                if (std::fabs(got[off + d] - x[off + d]) > 1e-4f) {
                    rotaryMoved = true;
                }
            }
        }
    }
    EXPECT_EQ(passMaxDiff, 0.0f);   // pass-through dims untouched
    EXPECT_TRUE(rotaryMoved);       // rotary block was actually rotated
    std::printf("[mrope-partial-rotary-parity] headDim=%zu rotaryDim=%zu maxErr=%.2e\n",
                headDim, rotaryDim, maxErr);
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

// 5.21 Increment II — VARLEN paged CAUSAL prefill attention: 2 ragged slots
// (different startPos + seqT) in one launch vs per-(slot,pq) paged_attention_v1
// decode with seq_len = startPos+pq+1. Same streaming-softmax => byte-exact.
TEST(cuda_paged_attention_prefill_causal_parity) {
    CudaComputeContext ctx{};
    GpuOps ops{ctx};

    const int nHeads = 4, nKvHeads = 2, headSize = 32, blockSize = 4;
    const std::vector<int> startPos = {3, 1};
    const std::vector<int> seqT     = {2, 3};
    const int nSeq  = static_cast<int>(seqT.size());
    const int kvDim = nKvHeads * headSize;
    const float scale = 1.0f / std::sqrt(static_cast<float>(headSize));

    std::vector<int> nBlk(nSeq); int maxBlk = 0, numBlocks = 0;
    for (int s = 0; s < nSeq; ++s) {
        const int len = startPos[s] + seqT[s];
        nBlk[s] = (len + blockSize - 1) / blockSize;
        maxBlk = std::max(maxBlk, nBlk[s]); numBlocks += nBlk[s];
    }
    std::vector<int> blockTables(static_cast<std::size_t>(nSeq)*maxBlk, -1);
    { int nx = 0; for (int s=0;s<nSeq;++s) for (int b=0;b<nBlk[s];++b)
        blockTables[static_cast<std::size_t>(s)*maxBlk + b] = nx++; }

    std::vector<int> queryOff(nSeq, 0); int totalTok = 0, maxT = 1;
    for (int s=0;s<nSeq;++s){ queryOff[s]=totalTok; totalTok+=seqT[s]; maxT=std::max(maxT,seqT[s]); }

    auto kPool = randVec(static_cast<std::size_t>(numBlocks)*blockSize*kvDim, 0x51u);
    auto vPool = randVec(static_cast<std::size_t>(numBlocks)*blockSize*kvDim, 0x52u);
    auto query = randVec(static_cast<std::size_t>(totalTok)*nHeads*headSize, 0x53u);

    auto dK = toDevice(ops, kPool); auto dV = toDevice(ops, vPool);
    auto dQ = toDevice(ops, query);
    auto dBT = uploadRaw(ops, blockTables);
    auto dSeqT = uploadRaw(ops, seqT);
    auto dQOff = uploadRaw(ops, queryOff);
    auto dSP  = uploadRaw(ops, startPos);
    auto dOut = ops.allocate(static_cast<std::size_t>(totalTok)*nHeads*headSize*sizeof(float));

    ops.pagedAttentionPrefillCausalAsync(
        static_cast<float*>(dOut.get()), static_cast<const float*>(dQ.get()),
        static_cast<const float*>(dK.get()), static_cast<const float*>(dV.get()),
        static_cast<const std::int32_t*>(dBT.get()),
        static_cast<const std::int32_t*>(dSeqT.get()),
        static_cast<const std::int32_t*>(dQOff.get()),
        static_cast<const std::int32_t*>(dSP.get()),
        nSeq, nHeads, nKvHeads, headSize, blockSize, maxBlk, maxT, scale, 0.0f);
    ops.flush();
    auto outB = fromDevice(ops, dOut.get(), static_cast<std::size_t>(totalTok)*nHeads*headSize);

    double maxErr = 0.0;
    const std::size_t rowElems = static_cast<std::size_t>(nHeads)*headSize;
    for (int s = 0; s < nSeq; ++s) {
        std::vector<int> btRow(blockTables.begin()+static_cast<std::size_t>(s)*maxBlk,
                               blockTables.begin()+static_cast<std::size_t>(s+1)*maxBlk);
        auto dBTs = uploadRaw(ops, btRow);
        for (int pq = 0; pq < seqT[s]; ++pq) {
            const std::size_t tok = static_cast<std::size_t>(queryOff[s]) + pq;
            std::vector<float> q1(query.begin()+tok*rowElems, query.begin()+(tok+1)*rowElems);
            std::vector<int> sl = {startPos[s] + pq + 1};
            auto dq1 = toDevice(ops, q1); auto dsl = uploadRaw(ops, sl);
            auto do1 = ops.allocate(rowElems*sizeof(float));
            ops.pagedAttentionDecodeV1Async(
                static_cast<float*>(do1.get()), static_cast<const float*>(dq1.get()),
                static_cast<const float*>(dK.get()), static_cast<const float*>(dV.get()),
                static_cast<const std::int32_t*>(dBTs.get()),
                static_cast<const std::int32_t*>(dsl.get()),
                1, nHeads, nKvHeads, headSize, blockSize, maxBlk, scale, 0.0f);
            ops.flush();
            auto o1 = fromDevice(ops, do1.get(), rowElems);
            for (std::size_t i = 0; i < rowElems; ++i) {
                maxErr = std::max(maxErr, std::fabs((double)outB[tok*rowElems+i]-(double)o1[i]));
                EXPECT_NEAR(outB[tok*rowElems+i], o1[i], 0.f);
            }
        }
    }
    std::printf("[paged-prefill-causal-parity] nSeq=%d seqT={2,3} startPos={3,1} maxErr=%.2e\n",
                nSeq, maxErr);
}

// M-Cuda.Batch B2 — paged_attention_v1 baseline decode kernel vs a CPU
// softmax-attention reference. Builds a synthetic paged KV pool (block
// tables + per-block slots), one query token per sequence, ragged
// seq_lens, GQA (nKvHeads < nHeads). The kernel walks each sequence's
// block table with streaming softmax; the reference recomputes the same
// attention on the host from the identical pooled K/V.
TEST(cuda_paged_attention_v1_decode_parity) {
    CudaComputeContext ctx{};
    GpuOps ops{ctx};

    const int   nSeq = 3, nHeads = 8, nKvHeads = 2, headSize = 256, blockSize = 16;
    const int   seqLens[nSeq] = {1, 2, 7};
    const float scale   = 1.0f / std::sqrt(static_cast<float>(headSize));
    const float softcap = 0.0f;

    int maxBlocks = 0;
    for (int s = 0; s < nSeq; ++s) {
        maxBlocks = std::max(maxBlocks, (seqLens[s] + blockSize - 1) / blockSize);
    }
    std::vector<int> blockTables(static_cast<std::size_t>(nSeq) * maxBlocks, -1);
    int nextBlock = 0;
    for (int s = 0; s < nSeq; ++s) {
        const int nb = (seqLens[s] + blockSize - 1) / blockSize;
        for (int b = 0; b < nb; ++b) {
            blockTables[static_cast<std::size_t>(s) * maxBlocks + b] = nextBlock++;
        }
    }
    const int numBlocks = nextBlock;
    const int kvDim     = nKvHeads * headSize;

    // Paged pools; scatter random K/V into each sequence's real slots.
    std::vector<float> keyPool(static_cast<std::size_t>(numBlocks) * blockSize * kvDim, 0.0f);
    std::vector<float> valPool(static_cast<std::size_t>(numBlocks) * blockSize * kvDim, 0.0f);
    Lcg gk{0xA1u}, gv{0xB2u}, gq{0xC3u};
    std::vector<float> query(static_cast<std::size_t>(nSeq) * nHeads * headSize);
    for (auto& x : query) x = gq.next();
    for (int s = 0; s < nSeq; ++s) {
        for (int p = 0; p < seqLens[s]; ++p) {
            const int blk  = blockTables[static_cast<std::size_t>(s) * maxBlocks + p / blockSize];
            const int slot = p % blockSize;
            for (int h = 0; h < nKvHeads; ++h) {
                for (int d = 0; d < headSize; ++d) {
                    const long off =
                        ((static_cast<long>(blk) * blockSize + slot) * nKvHeads + h) * headSize + d;
                    keyPool[off] = gk.next();
                    valPool[off] = gv.next();
                }
            }
        }
    }

    auto dOut = ops.allocate(static_cast<std::size_t>(nSeq) * nHeads * headSize * sizeof(float));
    auto dQ   = toDevice(ops, query);
    auto dK   = toDevice(ops, keyPool);
    auto dV   = toDevice(ops, valPool);
    auto dBT  = uploadRaw(ops, blockTables);
    std::vector<int> seqLensV(seqLens, seqLens + nSeq);
    auto dSL  = uploadRaw(ops, seqLensV);

    ops.pagedAttentionDecodeV1Async(
        static_cast<float*>(dOut.get()),
        static_cast<const float*>(dQ.get()),
        static_cast<const float*>(dK.get()),
        static_cast<const float*>(dV.get()),
        static_cast<const std::int32_t*>(dBT.get()),
        static_cast<const std::int32_t*>(dSL.get()),
        nSeq, nHeads, nKvHeads, headSize, blockSize, maxBlocks, scale, softcap);
    ops.flush();
    auto got = fromDevice(ops, dOut.get(),
                          static_cast<std::size_t>(nSeq) * nHeads * headSize);

    double maxErr = 0.0;
    for (int s = 0; s < nSeq; ++s) {
        for (int hq = 0; hq < nHeads; ++hq) {
            const int hkv = (hq * nKvHeads) / nHeads;
            std::vector<float> logits(seqLens[s]);
            float mx = -1.0e30f;
            for (int p = 0; p < seqLens[s]; ++p) {
                const int  blk  = blockTables[static_cast<std::size_t>(s) * maxBlocks + p / blockSize];
                const int  slot = p % blockSize;
                const long koff = (static_cast<long>(blk) * blockSize + slot) * nKvHeads * headSize
                                  + static_cast<long>(hkv) * headSize;
                float dot = 0.0f;
                for (int d = 0; d < headSize; ++d) {
                    dot += query[(static_cast<std::size_t>(s) * nHeads + hq) * headSize + d]
                           * keyPool[koff + d];
                }
                float lg = dot * scale;
                if (softcap > 0.0f) lg = softcap * std::tanh(lg / softcap);
                logits[p] = lg;
                mx = std::max(mx, lg);
            }
            float sum = 0.0f;
            for (float& lg : logits) { lg = std::exp(lg - mx); sum += lg; }
            std::vector<float> o(headSize, 0.0f);
            for (int p = 0; p < seqLens[s]; ++p) {
                const int  blk  = blockTables[static_cast<std::size_t>(s) * maxBlocks + p / blockSize];
                const int  slot = p % blockSize;
                const long voff = (static_cast<long>(blk) * blockSize + slot) * nKvHeads * headSize
                                  + static_cast<long>(hkv) * headSize;
                const float wgt = logits[p] / sum;
                for (int d = 0; d < headSize; ++d) o[d] += wgt * valPool[voff + d];
            }
            for (int d = 0; d < headSize; ++d) {
                const float g = got[(static_cast<std::size_t>(s) * nHeads + hq) * headSize + d];
                maxErr = std::max(maxErr, std::fabs(static_cast<double>(g) - static_cast<double>(o[d])));
                EXPECT_NEAR(g, o[d], 1e-3f);
            }
        }
    }
    std::printf("[paged-attn-v1-parity] nSeq=%d nHeads=%d nKv=%d hd=%d blk=%d maxErr=%.2e\n",
                nSeq, nHeads, nKvHeads, headSize, blockSize, maxErr);
}

// M-Cuda.Batch B3 — paged_attention_v2 (split-K FlashDecoding) vs the SAME CPU
// softmax reference. seq_lens > PARTITION_SIZE (512) so V2 actually splits the
// KV into partitions + runs the reduce merge. Ragged lengths cover full,
// partial and single-partition sequences. Must match V1 / the CPU ref.
TEST(cuda_paged_attention_v2_decode_parity) {
    CudaComputeContext ctx{};
    GpuOps ops{ctx};

    const int   nSeq = 3, nHeads = 8, nKvHeads = 2, headSize = 256, blockSize = 16;
    const int   seqLens[nSeq] = {600, 1300, 100};   // 2 / 3 / 1 partitions
    const float scale   = 1.0f / std::sqrt(static_cast<float>(headSize));
    const float softcap = 0.0f;
    int maxSeqLen = 0;
    for (int s = 0; s < nSeq; ++s) maxSeqLen = std::max(maxSeqLen, seqLens[s]);

    int maxBlocks = 0;
    for (int s = 0; s < nSeq; ++s) {
        maxBlocks = std::max(maxBlocks, (seqLens[s] + blockSize - 1) / blockSize);
    }
    std::vector<int> blockTables(static_cast<std::size_t>(nSeq) * maxBlocks, -1);
    int nextBlock = 0;
    for (int s = 0; s < nSeq; ++s) {
        const int nb = (seqLens[s] + blockSize - 1) / blockSize;
        for (int b = 0; b < nb; ++b) {
            blockTables[static_cast<std::size_t>(s) * maxBlocks + b] = nextBlock++;
        }
    }
    const int numBlocks = nextBlock;
    const int kvDim     = nKvHeads * headSize;

    std::vector<float> keyPool(static_cast<std::size_t>(numBlocks) * blockSize * kvDim, 0.0f);
    std::vector<float> valPool(static_cast<std::size_t>(numBlocks) * blockSize * kvDim, 0.0f);
    Lcg gk{0x51u}, gv{0x62u}, gq{0x73u};
    std::vector<float> query(static_cast<std::size_t>(nSeq) * nHeads * headSize);
    for (auto& x : query) x = gq.next();
    for (int s = 0; s < nSeq; ++s) {
        for (int p = 0; p < seqLens[s]; ++p) {
            const int blk  = blockTables[static_cast<std::size_t>(s) * maxBlocks + p / blockSize];
            const int slot = p % blockSize;
            for (int h = 0; h < nKvHeads; ++h) {
                for (int d = 0; d < headSize; ++d) {
                    const long off =
                        ((static_cast<long>(blk) * blockSize + slot) * nKvHeads + h) * headSize + d;
                    keyPool[off] = gk.next();
                    valPool[off] = gv.next();
                }
            }
        }
    }

    auto dOut = ops.allocate(static_cast<std::size_t>(nSeq) * nHeads * headSize * sizeof(float));
    auto dQ   = toDevice(ops, query);
    auto dK   = toDevice(ops, keyPool);
    auto dV   = toDevice(ops, valPool);
    auto dBT  = uploadRaw(ops, blockTables);
    std::vector<int> seqLensV(seqLens, seqLens + nSeq);
    auto dSL  = uploadRaw(ops, seqLensV);

    ops.pagedAttentionDecodeV2Async(
        static_cast<float*>(dOut.get()),
        static_cast<const float*>(dQ.get()),
        static_cast<const float*>(dK.get()),
        static_cast<const float*>(dV.get()),
        static_cast<const std::int32_t*>(dBT.get()),
        static_cast<const std::int32_t*>(dSL.get()),
        nSeq, nHeads, nKvHeads, headSize, blockSize, maxBlocks,
        static_cast<std::size_t>(maxSeqLen), scale, softcap);
    ops.flush();
    auto got = fromDevice(ops, dOut.get(),
                          static_cast<std::size_t>(nSeq) * nHeads * headSize);

    double maxErr = 0.0;
    for (int s = 0; s < nSeq; ++s) {
        for (int hq = 0; hq < nHeads; ++hq) {
            const int hkv = (hq * nKvHeads) / nHeads;
            std::vector<float> logits(seqLens[s]);
            float mx = -1.0e30f;
            for (int p = 0; p < seqLens[s]; ++p) {
                const int  blk  = blockTables[static_cast<std::size_t>(s) * maxBlocks + p / blockSize];
                const int  slot = p % blockSize;
                const long koff = (static_cast<long>(blk) * blockSize + slot) * nKvHeads * headSize
                                  + static_cast<long>(hkv) * headSize;
                float dot = 0.0f;
                for (int d = 0; d < headSize; ++d) {
                    dot += query[(static_cast<std::size_t>(s) * nHeads + hq) * headSize + d]
                           * keyPool[koff + d];
                }
                float lg = dot * scale;
                if (softcap > 0.0f) lg = softcap * std::tanh(lg / softcap);
                logits[p] = lg;
                mx = std::max(mx, lg);
            }
            float sum = 0.0f;
            for (float& lg : logits) { lg = std::exp(lg - mx); sum += lg; }
            std::vector<float> o(headSize, 0.0f);
            for (int p = 0; p < seqLens[s]; ++p) {
                const int  blk  = blockTables[static_cast<std::size_t>(s) * maxBlocks + p / blockSize];
                const int  slot = p % blockSize;
                const long voff = (static_cast<long>(blk) * blockSize + slot) * nKvHeads * headSize
                                  + static_cast<long>(hkv) * headSize;
                const float wgt = logits[p] / sum;
                for (int d = 0; d < headSize; ++d) o[d] += wgt * valPool[voff + d];
            }
            for (int d = 0; d < headSize; ++d) {
                const float g = got[(static_cast<std::size_t>(s) * nHeads + hq) * headSize + d];
                maxErr = std::max(maxErr, std::fabs(static_cast<double>(g) - static_cast<double>(o[d])));
                EXPECT_NEAR(g, o[d], 1e-3f);
            }
        }
    }
    std::printf("[paged-attn-v2-parity] nSeq=%d maxSeqLen=%d parts=%d maxErr=%.2e\n",
                nSeq, maxSeqLen, (maxSeqLen + 511) / 512, maxErr);
}

// M-Cuda.MoeGroup Sub-Step A — device token-grouping build vs CPU golden.
//
// moe_group_build turns flat per-assignment routing (expIdx[R], kw[R],
// R = T*K) into the offset table + stable permutation a grouped-by-expert
// GEMM consumes: expOffset[nE+1] (exclusive prefix sum), rowSrcTok[R]
// (source token per compacted row), rowKw[R] (router weight per compacted
// row). The CPU reference is a plain stable counting sort — the kernel's v1
// visits assignments in ascending index order, so the two match exactly
// (offsets, permutation and gathered weights, not just within fp tolerance).
namespace {

void checkMoeGroupParity(std::size_t T, std::size_t nExperts, std::size_t K,
                         std::uint32_t seed) {
    CudaComputeContext ctx{};
    GpuOps ops{ctx};
    ::mimirmind::core::cuda::CudaModule mod =
        ::mimirmind::core::cuda::CudaModule::fromFile(ctx.cudaContext(),
                                                      resolvePtx("moe_group_build"));
    ::mimirmind::core::cuda::CudaKernel kern = mod.getFunction("moe_group_build");

    const std::size_t R = T * K;

    // Random routing: each of the R assignments picks an expert in [0,nE).
    Lcg g{seed};
    std::vector<std::int32_t> expIdx(R);
    std::vector<float>        kw(R);
    for (std::size_t i = 0; i < R; ++i) {
        // Map the LCG output into [0, nExperts); +1 keeps it in-range at 1.0.
        float u = (g.next() + 1.0f) * 0.5f;                 // [0,1)
        std::int32_t e = static_cast<std::int32_t>(u * static_cast<float>(nExperts));
        if (e >= static_cast<std::int32_t>(nExperts)) {
            e = static_cast<std::int32_t>(nExperts) - 1;
        }
        expIdx[i] = e;
        kw[i]     = g.next();                                // any finite weight
    }

    // CPU golden: stable counting sort.
    std::vector<std::int32_t> refOffset(nExperts + 1, 0);
    for (std::size_t i = 0; i < R; ++i) {
        refOffset[expIdx[i] + 1] += 1;
    }
    for (std::size_t e = 1; e <= nExperts; ++e) {
        refOffset[e] += refOffset[e - 1];
    }
    std::vector<std::int32_t> refRowTok(R, 0);
    std::vector<float>        refRowKw(R, 0.0f);
    std::vector<std::int32_t> refAsnToRow(R, -1);
    {
        std::vector<std::int32_t> cursor(refOffset.begin(),
                                         refOffset.begin() + nExperts);
        for (std::size_t i = 0; i < R; ++i) {
            const std::int32_t e = expIdx[i];
            const std::int32_t pos = cursor[e]++;
            refRowTok[pos] = static_cast<std::int32_t>(i / K);
            refRowKw[pos]  = kw[i];
            refAsnToRow[i] = pos;
        }
    }

    // Device run.
    auto dExpIdx = uploadRaw(ops, expIdx);
    auto dKw     = uploadRaw(ops, kw);
    auto dOffset = ops.allocate((nExperts + 1) * sizeof(std::int32_t));
    auto dRowTok = ops.allocate(R * sizeof(std::int32_t));
    auto dRowKw  = ops.allocate(R * sizeof(float));
    auto dAsnRow = ops.allocate(R * sizeof(std::int32_t));

    kern.setPtr  (0, dExpIdx.get());
    kern.setPtr  (1, dKw.get());
    kern.setPtr  (2, dOffset.get());
    kern.setPtr  (3, dRowTok.get());
    kern.setPtr  (4, dRowKw.get());
    kern.setPtr  (5, dAsnRow.get());
    kern.setValue(6, static_cast<std::int32_t>(R));
    kern.setValue(7, static_cast<std::int32_t>(nExperts));
    kern.setValue(8, static_cast<std::int32_t>(K));
    kern.launch(ctx.stream(), 1, 1, 1, 1, 1, 1);
    ops.flush();

    auto gotOffset = fromDeviceI32(ops, dOffset.get(), nExperts + 1);
    auto gotRowTok = fromDeviceI32(ops, dRowTok.get(), R);
    auto gotRowKw  = fromDevice   (ops, dRowKw.get(),  R);
    auto gotAsnRow = fromDeviceI32(ops, dAsnRow.get(), R);

    for (std::size_t e = 0; e <= nExperts; ++e) {
        EXPECT_EQ(gotOffset[e], refOffset[e]);
    }
    EXPECT_EQ(gotOffset[nExperts], static_cast<std::int32_t>(R));
    for (std::size_t r = 0; r < R; ++r) {
        EXPECT_EQ(gotRowTok[r], refRowTok[r]);
        EXPECT_NEAR(gotRowKw[r], refRowKw[r], 0.0f);
    }
    // Inverse-permutation consistency: asnToRow is a bijection [0,R) and
    // rowSrcTok[asnToRow[i]] == i/K (the assignment's source token).
    for (std::size_t i = 0; i < R; ++i) {
        EXPECT_EQ(gotAsnRow[i], refAsnToRow[i]);
        EXPECT_EQ(gotRowTok[gotAsnRow[i]], static_cast<std::int32_t>(i / K));
    }
    std::printf("[moe-group-build] T=%zu nExp=%zu K=%zu R=%zu OK\n",
                T, nExperts, K, R);
}

} // namespace

TEST(cuda_moe_group_build_parity_k8) {
    // Serving prefill shape: 128 experts, top-8, a full chunk of tokens.
    checkMoeGroupParity(/*T=*/512, /*nExperts=*/128, /*K=*/8, 0x6A08u);
}

TEST(cuda_moe_group_build_parity_small) {
    checkMoeGroupParity(/*T=*/6, /*nExperts=*/8, /*K=*/2, 0xC0FFEEu);
}

TEST(cuda_moe_group_build_parity_single_token) {
    checkMoeGroupParity(/*T=*/1, /*nExperts=*/128, /*K=*/8, 0x5EEDu);
}

// M-Cuda.MoeGroup Sub-Step E-a — moe_group_tiles turns the per-expert row
// ranges (expOffset, exclusive prefix sum) into a compact per-tile schedule a
// single device-driven grouped GEMM consumes in one launch (no expOffset D2H,
// no per-expert host loop). Expert e's count is split into ceil(count/tileM)
// tiles of <= tileM rows; unused tail tiles carry the -1 sentinel. The CPU
// golden walks experts in the same order, so offsets/rows/sentinels match
// exactly. Also asserts the static host-side upper bound is never exceeded.
namespace {

void checkMoeGroupTilesParity(const std::vector<std::int32_t>& counts,
                              int tileM) {
    CudaComputeContext ctx{};
    GpuOps ops{ctx};
    ::mimirmind::core::cuda::CudaModule mod =
        ::mimirmind::core::cuda::CudaModule::fromFile(ctx.cudaContext(),
                                                      resolvePtx("moe_group_tiles"));
    ::mimirmind::core::cuda::CudaKernel kern = mod.getFunction("moe_group_tiles");

    const int nExperts = static_cast<int>(counts.size());

    // expOffset = exclusive prefix sum of the per-expert counts.
    std::vector<std::int32_t> expOffset(nExperts + 1, 0);
    for (int e = 0; e < nExperts; ++e) {
        expOffset[e + 1] = expOffset[e] + counts[e];
    }
    const int R = expOffset[nExperts];

    // Static upper bound (host-computable, no D2H): ceil(R/tileM) + nExperts.
    const int maxTiles = (R + tileM - 1) / tileM + nExperts;

    // CPU golden: sequential per-expert tile walk (matches the kernel order).
    std::vector<std::int32_t> refExpert(maxTiles, -1);
    std::vector<std::int32_t> refRow0(maxTiles, 0);
    std::vector<std::int32_t> refRows(maxTiles, 0);
    int refN = 0;
    for (int e = 0; e < nExperts; ++e) {
        int row = expOffset[e];
        const int end = expOffset[e + 1];
        while (row < end) {
            int rows = end - row;
            if (rows > tileM) {
                rows = tileM;
            }
            refExpert[refN] = e;
            refRow0[refN]   = row;
            refRows[refN]   = rows;
            row += rows;
            ++refN;
        }
    }
    // The upper bound must never be exceeded (the whole no-D2H design relies
    // on this being tight enough to size the grid on the host).
    EXPECT_EQ(static_cast<int>(refN <= maxTiles), 1);

    // Device run.
    auto dOffset = uploadRaw(ops, expOffset);
    auto dExpert = ops.allocate(maxTiles * sizeof(std::int32_t));
    auto dRow0   = ops.allocate(maxTiles * sizeof(std::int32_t));
    auto dRows   = ops.allocate(maxTiles * sizeof(std::int32_t));
    auto dN      = ops.allocate(sizeof(std::int32_t));

    kern.setPtr  (0, dOffset.get());
    kern.setPtr  (1, dExpert.get());
    kern.setPtr  (2, dRow0.get());
    kern.setPtr  (3, dRows.get());
    kern.setPtr  (4, dN.get());
    kern.setValue(5, nExperts);
    kern.setValue(6, maxTiles);
    kern.setValue(7, tileM);
    kern.launch(ctx.stream(), 1, 1, 1, 1, 1, 1);
    ops.flush();

    auto gotExpert = fromDeviceI32(ops, dExpert.get(), maxTiles);
    auto gotRow0   = fromDeviceI32(ops, dRow0.get(),   maxTiles);
    auto gotRows   = fromDeviceI32(ops, dRows.get(),   maxTiles);
    auto gotN      = fromDeviceI32(ops, dN.get(),      1);

    EXPECT_EQ(gotN[0], refN);
    for (int t = 0; t < maxTiles; ++t) {
        EXPECT_EQ(gotExpert[t], refExpert[t]);
        EXPECT_EQ(gotRow0[t],   refRow0[t]);
        EXPECT_EQ(gotRows[t],   refRows[t]);
    }
    // Coverage invariant: the emitted tiles partition every expert's row range
    // exactly (contiguous, no gaps/overlap), so the grouped GEMM touches each
    // compacted row once.
    for (int e = 0; e < nExperts; ++e) {
        int expect = expOffset[e];
        for (int t = 0; t < gotN[0]; ++t) {
            if (gotExpert[t] != e) {
                continue;
            }
            EXPECT_EQ(gotRow0[t], expect);
            expect += gotRows[t];
        }
        EXPECT_EQ(expect, expOffset[e + 1]);
    }
    std::printf("[moe-group-tiles] nExp=%d R=%d tileM=%d nTiles=%d/%d OK\n",
                nExperts, R, tileM, gotN[0], maxTiles);
}

} // namespace

TEST(cuda_moe_group_tiles_parity_k8) {
    // Serving prefill shape: 128 experts, ~full chunk of top-8 assignments.
    // Random-ish skewed counts around the R=512*8/128 = 32 average.
    Lcg g{0x71235u};
    std::vector<std::int32_t> counts(128);
    for (auto& c : counts) {
        const float u = (g.next() + 1.0f) * 0.5f;            // [0,1)
        c = static_cast<std::int32_t>(u * 64.0f);            // 0..63, avg ~32
    }
    checkMoeGroupTilesParity(counts, /*tileM=*/16);
}

TEST(cuda_moe_group_tiles_parity_edges) {
    // Empty experts, exact-multiple, and one-over-multiple counts all present.
    checkMoeGroupTilesParity({0, 16, 1, 17, 32, 0, 15, 48}, /*tileM=*/16);
}

TEST(cuda_moe_group_tiles_parity_single_expert) {
    // One expert takes every row (degenerate routing) -> ceil(R/tileM) tiles.
    checkMoeGroupTilesParity({100, 0, 0, 0}, /*tileM=*/16);
}

// M-Cuda.MoeGroup Sub-Step E-b — moe_grouped_gemm_nvfp4blk: ONE device-driven
// launch reproduces the per-expert sequential NVFP4 GEMM by reading each tile's
// (expert, row-range) from the moe_group_tiles schedule on the device — no
// expOffset D2H, no per-expert host loop. Random blocked-NVFP4 weight bytes
// (finite fp16 scales, any nibbles) are shared byte-for-byte between the
// reference (one matmul_nvfp4blk_gemm launch per tile) and the grouped kernel;
// both iterate K in identical order, so the outputs are BIT-identical.
namespace {

constexpr int kNvSuperElems = 32;
constexpr int kNvSuperBytes = 20;

void checkMoeGroupedGemmParity(const std::vector<std::int32_t>& counts,
                               int N, int K, std::uint32_t seed) {
    namespace cc = ::mimirmind::core::cuda;
    CudaComputeContext ctx{};
    GpuOps ops{ctx};
    cc::CudaModule tilesMod = cc::CudaModule::fromFile(ctx.cudaContext(),
                                                       resolvePtx("moe_group_tiles"));
    cc::CudaKernel tilesKern = tilesMod.getFunction("moe_group_tiles");
    cc::CudaModule refMod = cc::CudaModule::fromFile(ctx.cudaContext(),
                                                     resolvePtx("matmul_nvfp4blk_gemm"));
    cc::CudaKernel refKern = refMod.getFunction("matmul_nvfp4blk_gemm");
    cc::CudaModule grpMod = cc::CudaModule::fromFile(ctx.cudaContext(),
                                                     resolvePtx("moe_grouped_gemm_nvfp4blk"));
    cc::CudaKernel grpKern = grpMod.getFunction("moe_grouped_gemm_nvfp4blk");

    const int nExperts = static_cast<int>(counts.size());
    const int tileM    = 16;                          // == GEMM_MAX_M

    std::vector<std::int32_t> expOffset(nExperts + 1, 0);
    for (int e = 0; e < nExperts; ++e) {
        expOffset[e + 1] = expOffset[e] + counts[e];
    }
    const int R        = expOffset[nExperts];
    const int maxTiles = (R + tileM - 1) / tileM + nExperts;
    const int nSuper   = K / kNvSuperElems;
    const std::size_t expertBytes =
        static_cast<std::size_t>(N) * nSuper * kNvSuperBytes;

    Lcg g{seed};
    auto nextU32 = [&g]() -> std::uint32_t {
        g.s = g.s * 1664525u + 1013904223u;
        return g.s;
    };

    // Random activations in [-1, 1].
    std::vector<float> X(static_cast<std::size_t>(R) * K);
    for (auto& v : X) v = g.next();

    // Random blocked-NVFP4 weight bank [nExperts][N][K]. Each 20-byte super =
    // two fp16 scales (bytes 0..3) + 16 nibble-bytes (4..19). We force the fp16
    // scale exponent off 0x1F so no scale decodes to NaN/Inf; any other bits
    // (incl. nibbles) are fair game — the kernels interpret them identically.
    std::vector<unsigned char> W(static_cast<std::size_t>(nExperts) * expertBytes);
    for (std::size_t off = 0; off + kNvSuperBytes <= W.size(); off += kNvSuperBytes) {
        for (int h = 0; h < 2; ++h) {                 // two fp16 scales
            std::uint32_t bits = nextU32() & 0xFFFFu;
            if (((bits >> 10) & 0x1Fu) == 0x1Fu) {
                bits &= ~(1u << 14);                  // clear exp MSB -> not Inf/NaN
            }
            W[off + h * 2 + 0] = static_cast<unsigned char>(bits & 0xFFu);
            W[off + h * 2 + 1] = static_cast<unsigned char>((bits >> 8) & 0xFFu);
        }
        for (int b = 4; b < kNvSuperBytes; ++b) {
            W[off + b] = static_cast<unsigned char>(nextU32() & 0xFFu);
        }
    }

    auto dX   = uploadRaw(ops, X);
    auto dW   = uploadRaw(ops, W);
    std::vector<float> zeros(static_cast<std::size_t>(R) * N, 0.0f);
    auto dYref = uploadRaw(ops, zeros);
    auto dYgrp = uploadRaw(ops, zeros);

    const auto*  Wbase = static_cast<const unsigned char*>(dW.get());
    const float* Xbase = static_cast<const float*>(dX.get());
    const std::uint32_t gx = static_cast<std::uint32_t>((N + 3) / 4);   // ceil(N/4)

    // --- reference: one dense per-expert GEMM per tile (CPU-walked schedule) ---
    float* Yref = static_cast<float*>(dYref.get());
    for (int e = 0; e < nExperts; ++e) {
        int row = expOffset[e];
        const int end = expOffset[e + 1];
        while (row < end) {
            int rows = end - row;
            if (rows > tileM) rows = tileM;
            refKern.setPtr  (0, Xbase + static_cast<std::size_t>(row) * K);
            refKern.setPtr  (1, Wbase + static_cast<std::size_t>(e) * expertBytes);
            refKern.setPtr  (2, Yref  + static_cast<std::size_t>(row) * N);
            refKern.setValue(3, K);
            refKern.setValue(4, N);
            refKern.setValue(5, rows);
            refKern.launch(ctx.stream(), gx, 1, 1, 128, 1, 1);
            row += rows;
        }
    }

    // --- grouped: device schedule build + ONE grouped launch ------------------
    auto dExpert = ops.allocate(maxTiles * sizeof(std::int32_t));
    auto dRow0   = ops.allocate(maxTiles * sizeof(std::int32_t));
    auto dRows   = ops.allocate(maxTiles * sizeof(std::int32_t));
    auto dN      = ops.allocate(sizeof(std::int32_t));
    auto dOffset = uploadRaw(ops, expOffset);

    tilesKern.setPtr  (0, dOffset.get());
    tilesKern.setPtr  (1, dExpert.get());
    tilesKern.setPtr  (2, dRow0.get());
    tilesKern.setPtr  (3, dRows.get());
    tilesKern.setPtr  (4, dN.get());
    tilesKern.setValue(5, nExperts);
    tilesKern.setValue(6, maxTiles);
    tilesKern.setValue(7, tileM);
    tilesKern.launch(ctx.stream(), 1, 1, 1, 1, 1, 1);

    grpKern.setPtr  (0, dX.get());
    grpKern.setPtr  (1, dW.get());
    grpKern.setPtr  (2, dYgrp.get());
    grpKern.setPtr  (3, dExpert.get());
    grpKern.setPtr  (4, dRow0.get());
    grpKern.setPtr  (5, dRows.get());
    grpKern.setValue(6, K);
    grpKern.setValue(7, N);
    grpKern.launch(ctx.stream(), gx, static_cast<std::uint32_t>(maxTiles), 1,
                   128, 1, 1);
    ops.flush();

    auto gotRef = fromDevice(ops, dYref.get(), static_cast<std::size_t>(R) * N);
    auto gotGrp = fromDevice(ops, dYgrp.get(), static_cast<std::size_t>(R) * N);
    for (std::size_t i = 0; i < gotRef.size(); ++i) {
        EXPECT_NEAR(gotGrp[i], gotRef[i], 0.0f);      // bit-identical FMA order
    }
    std::printf("[moe-grouped-gemm] nExp=%d R=%d N=%d K=%d OK\n",
                nExperts, R, N, K);
}

} // namespace

TEST(cuda_moe_grouped_gemm_parity_k8) {
    // Serving-ish shape: many experts, skewed counts, multi-tile experts.
    Lcg g{0x9A11u};
    std::vector<std::int32_t> counts(16);
    for (auto& c : counts) {
        const float u = (g.next() + 1.0f) * 0.5f;
        c = static_cast<std::int32_t>(u * 40.0f);     // 0..39, some > tileM
    }
    checkMoeGroupedGemmParity(counts, /*N=*/64, /*K=*/64, 0x3131u);
}

TEST(cuda_moe_grouped_gemm_parity_nonmult_n) {
    // N not a multiple of the 4-output group -> exercises the active-column
    // guard; down-projection-ish (K > N).
    checkMoeGroupedGemmParity({20, 0, 33, 5, 16, 17}, /*N=*/70, /*K=*/128, 0x77u);
}

TEST(cuda_moe_grouped_gemm_parity_single_expert) {
    // One expert takes every row -> ceil(R/tileM) tiles, single weight base.
    checkMoeGroupedGemmParity({50, 0, 0}, /*N=*/32, /*K=*/96, 0xB0Bu);
}

// NVFP4 de-interleaved vectorised matvec (M=1) — parity vs the 20-byte blocked
// matmul_nvfp4blk_gemm(rows=1) baseline + a bandwidth A/B. Proves whether the
// uint4 coalesced load on a de-interleaved layout lifts the ~42%-of-peak DRAM
// throughput of the interleaved byte/half path toward vLLM's ~80%.
TEST(cuda_nvfp4blk_deint_vec_bw) {
    CudaComputeContext ctx{};
    GpuOps ops{ctx};

    constexpr int N = 8192, K = 2048;
    constexpr int kNvSuperBytes = 20;
    const int nSuper = K / 32;

    Lcg g{0xDEEEu};
    auto nextU32 = [&]() -> std::uint32_t {
        const float f = g.next();
        return static_cast<std::uint32_t>((f + 1.0f) * 2.1e9f);
    };

    std::vector<float> X(K);
    for (auto& v : X) v = g.next();

    // 20-byte blocked weight [N][nSuper][20] (fp16 s0|s1 | 16 nibble-bytes).
    std::vector<unsigned char> W(static_cast<std::size_t>(N) * nSuper * kNvSuperBytes);
    for (std::size_t off = 0; off + kNvSuperBytes <= W.size(); off += kNvSuperBytes) {
        for (int h = 0; h < 2; ++h) {
            std::uint32_t bits = nextU32() & 0xFFFFu;
            if (((bits >> 10) & 0x1Fu) == 0x1Fu) bits &= ~(1u << 14);
            W[off + h * 2 + 0] = static_cast<unsigned char>(bits & 0xFFu);
            W[off + h * 2 + 1] = static_cast<unsigned char>((bits >> 8) & 0xFFu);
        }
        for (int b = 4; b < kNvSuperBytes; ++b)
            W[off + b] = static_cast<unsigned char>(nextU32() & 0xFFu);
    }

    // De-interleave: nib[N][nSuper][16], scale-bytes[N][nSuper][4] (=2 fp16).
    std::vector<unsigned char> nibD(static_cast<std::size_t>(N) * nSuper * 16);
    std::vector<unsigned char> scD (static_cast<std::size_t>(N) * nSuper * 4);
    for (std::size_t s = 0; s < static_cast<std::size_t>(N) * nSuper; ++s) {
        const unsigned char* src = &W[s * kNvSuperBytes];
        for (int b = 0; b < 4;  ++b) scD [s * 4  + b] = src[b];
        for (int b = 0; b < 16; ++b) nibD[s * 16 + b] = src[4 + b];
    }

    namespace cc = ::mimirmind::core::cuda;
    cc::CudaModule oldMod = cc::CudaModule::fromFile(
        ctx.cudaContext(), resolvePtx("matmul_nvfp4blk_gemm"));
    cc::CudaKernel oldK = oldMod.getFunction("matmul_nvfp4blk_gemm");
    cc::CudaModule newMod = cc::CudaModule::fromFile(
        ctx.cudaContext(), resolvePtx("matmul_nvfp4blk_deint_vec"));
    cc::CudaKernel newK = newMod.getFunction("matmul_nvfp4blk_deint_vec");

    auto dX   = toDevice(ops, X);
    auto dW   = uploadRaw(ops, W);
    auto dNib = uploadRaw(ops, nibD);
    auto dSc  = uploadRaw(ops, scD);
    std::vector<float> zeros(N, 0.0f);
    auto dYo = uploadRaw(ops, zeros);
    auto dYn = uploadRaw(ops, zeros);
    const std::uint32_t gx = static_cast<std::uint32_t>((N + 3) / 4);

    auto launchOld = [&]() {
        oldK.setPtr(0, dX.get()); oldK.setPtr(1, dW.get()); oldK.setPtr(2, dYo.get());
        oldK.setValue(3, K); oldK.setValue(4, N); oldK.setValue(5, 1);
        oldK.launch(ctx.stream(), gx, 1, 1, 128, 1, 1);
    };
    auto launchNew = [&]() {
        newK.setPtr(0, dX.get()); newK.setPtr(1, dNib.get()); newK.setPtr(2, dSc.get());
        newK.setPtr(3, dYn.get()); newK.setValue(4, N); newK.setValue(5, K);
        newK.launch(ctx.stream(), gx, 1, 1, 128, 1, 1, K * sizeof(float));
    };

    launchOld(); launchNew(); ops.flush();
    auto yo = fromDevice(ops, dYo.get(), N);
    auto yn = fromDevice(ops, dYn.get(), N);
    double maxRel = 0.0;
    for (int i = 0; i < N; ++i) {
        const double denom = std::max(1e-3, std::fabs(static_cast<double>(yo[i])));
        maxRel = std::max(maxRel, std::fabs(static_cast<double>(yn[i] - yo[i])) / denom);
        EXPECT_NEAR(yn[i], yo[i], 1e-2f * (1.0f + std::fabs(yo[i])));
    }

    constexpr int ITERS = 2000;
    const double bytes = static_cast<double>(N) * K * 0.625;   // NVFP4 weight bytes
    auto bench = [&](auto fn) {
        fn(); ops.flush();
        const auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; ++i) fn();
        ops.flush();
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double sec = std::chrono::duration<double>(t1 - t0).count();
        return (bytes * ITERS) / sec / 1e9;                   // GB/s
    };
    const double gbOld = bench(launchOld);
    const double gbNew = bench(launchNew);
    std::printf("[nvfp4-deint-vec] N=%d K=%d maxRel=%.2e | OLD %.1f GB/s | NEW %.1f GB/s "
                "(%.2fx, peak 273)\n", N, K, maxRel, gbOld, gbNew, gbNew / gbOld);
}

// M-Cuda.MoeGroup Sub-Step B — moe_gather_rows: xCompact[r]=x[rowSrcTok[r]].
TEST(cuda_moe_gather_rows_parity) {
    CudaComputeContext ctx{};
    GpuOps ops{ctx};
    ::mimirmind::core::cuda::CudaModule mod =
        ::mimirmind::core::cuda::CudaModule::fromFile(ctx.cudaContext(),
                                                      resolvePtx("moe_gather_rows"));
    ::mimirmind::core::cuda::CudaKernel kern = mod.getFunction("moe_gather_rows");

    const std::size_t T = 40, dModel = 384, R = 300;
    auto Xh = randVec(T * dModel, 0x0A11u);
    Lcg g{0xBADu};
    std::vector<std::int32_t> rowSrcTok(R);
    for (auto& v : rowSrcTok) {
        v = static_cast<std::int32_t>(((g.s = g.s * 1664525u + 1013904223u) >> 9) % T);
    }

    auto dX   = toDevice(ops, Xh);
    auto dTok = uploadRaw(ops, rowSrcTok);
    auto dOut = ops.allocate(R * dModel * sizeof(float));
    kern.setPtr  (0, dX.get());
    kern.setPtr  (1, dTok.get());
    kern.setPtr  (2, dOut.get());
    kern.setValue(3, static_cast<std::int32_t>(dModel));
    kern.setValue(4, static_cast<std::int32_t>(R));
    kern.launch(ctx.stream(), static_cast<std::uint32_t>(R), 1, 1, 256, 1, 1);
    ops.flush();
    auto got = fromDevice(ops, dOut.get(), R * dModel);

    for (std::size_t r = 0; r < R; ++r) {
        const std::size_t t = static_cast<std::size_t>(rowSrcTok[r]);
        for (std::size_t j = 0; j < dModel; ++j) {
            EXPECT_EQ(got[r * dModel + j], Xh[t * dModel + j]);
        }
    }
    std::printf("[moe-gather] T=%zu d=%zu R=%zu OK\n", T, dModel, R);
}

// M-Cuda.MoeGroup Sub-Step C — moe_scatter_expert_out deterministic fold.
//   accum[t] = sum_k kw[t*K+k] * y[asnToRow[t*K+k]]
// asnToRow is a bijection [0,R); the CPU golden folds in the same fixed k
// order, so the match is exact (not just within fp tolerance).
TEST(cuda_moe_scatter_expert_out_parity) {
    CudaComputeContext ctx{};
    GpuOps ops{ctx};
    ::mimirmind::core::cuda::CudaModule mod =
        ::mimirmind::core::cuda::CudaModule::fromFile(ctx.cudaContext(),
                                                      resolvePtx("moe_scatter_expert_out"));
    ::mimirmind::core::cuda::CudaKernel kern =
        mod.getFunction("moe_scatter_expert_out");

    const std::size_t T = 32, K = 8, dModel = 256, R = T * K;
    auto Yh  = randVec(R * dModel, 0x5CA7u);
    auto kwh = randVec(R, 0x42u);
    // Deterministic Fisher-Yates permutation of [0,R): asnToRow[i] = perm[i].
    std::vector<std::int32_t> asnToRow(R);
    for (std::size_t i = 0; i < R; ++i) {
        asnToRow[i] = static_cast<std::int32_t>(i);
    }
    Lcg g{0xF15Eu};
    for (std::size_t i = R; i > 1; --i) {
        g.s = g.s * 1664525u + 1013904223u;
        const std::size_t j = (g.s >> 8) % i;
        std::swap(asnToRow[i - 1], asnToRow[j]);
    }

    // CPU golden (same fixed k order as the kernel).
    std::vector<float> ref(T * dModel, 0.0f);
    for (std::size_t t = 0; t < T; ++t) {
        for (std::size_t d = 0; d < dModel; ++d) {
            float acc = 0.0f;
            for (std::size_t k = 0; k < K; ++k) {
                const std::int32_t row = asnToRow[t * K + k];
                acc += kwh[t * K + k] * Yh[static_cast<std::size_t>(row) * dModel + d];
            }
            ref[t * dModel + d] = acc;
        }
    }

    auto dY   = toDevice(ops, Yh);
    auto dRow = uploadRaw(ops, asnToRow);
    auto dKw  = toDevice(ops, kwh);
    auto dAcc = ops.allocate(T * dModel * sizeof(float));
    kern.setPtr  (0, dY.get());
    kern.setPtr  (1, dRow.get());
    kern.setPtr  (2, dKw.get());
    kern.setPtr  (3, dAcc.get());
    kern.setValue(4, static_cast<std::int32_t>(dModel));
    kern.setValue(5, static_cast<std::int32_t>(T));
    kern.setValue(6, static_cast<std::int32_t>(K));
    kern.launch(ctx.stream(), static_cast<std::uint32_t>(T), 1, 1, 256, 1, 1);
    ops.flush();
    auto got = fromDevice(ops, dAcc.get(), T * dModel);

    double maxErr = 0.0;
    for (std::size_t i = 0; i < T * dModel; ++i) {
        maxErr = std::max(maxErr,
            std::fabs(static_cast<double>(got[i]) - static_cast<double>(ref[i])));
        // Tolerance, not exact: nvcc contracts kw*y+acc into FMA; the host
        // reference may not, so the two can differ by ~1 ULP per fold step.
        EXPECT_NEAR(got[i], ref[i], 1e-4f);
    }
    std::printf("[moe-scatter] T=%zu K=%zu d=%zu R=%zu maxErr=%.2e OK\n",
                T, K, dModel, R, maxErr);
}

// M-Cuda.MoeGroup Sub-Step E-d — moe_act_quant_nvfp4: F32 activations ->
// NVFP4 (E2M1 nibbles) + per-16 UE4M3 block scales in the CUTLASS swizzled SF
// layout [numMTiles, numKTiles, 32, 4, 4]. There is no host-side dependency on
// the exact E4M3/E2M1 rounding: we read the device output back and reconstruct
// value = e2m1(nibble) * e4m3(SF) / gscale, then check it lands within NVFP4
// quant error of the original activation. Crucially the SF is fetched through
// the SAME swizzle-offset formula the kernel writes with — a wrong swizzle
// pulls a different block's scale (or a zeroed pad slot) and blows the bound.
namespace {

// E4M3 decode matching CUDA __nv_fp8_e4m3 / repackage_nvfp4_to_blk.cu (bias 7).
double aq_dec_e4m3(unsigned b) {
    const unsigned s = (b >> 7) & 0x1u;
    const unsigned e = (b >> 3) & 0xFu;
    const unsigned m = b & 0x7u;
    const double sign = s ? -1.0 : 1.0;
    if (e == 0u) return sign * std::ldexp(static_cast<double>(m) / 8.0, -6);
    return sign * std::ldexp(1.0 + static_cast<double>(m) / 8.0,
                             static_cast<int>(e) - 7);
}

// E2M1 decode: 3-bit magnitude LUT + sign bit.
double aq_dec_e2m1(unsigned nib) {
    static const double mag[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
    const double v = mag[nib & 0x7u];
    return (nib & 0x8u) ? -v : v;
}

long aq_test_sf_offset(int mIdx, int kIdx, int numKTiles) {
    const int mTileIdx  = mIdx >> 7;
    const int outerMIdx = mIdx & 31;
    const int innerMIdx = (mIdx >> 5) & 3;
    const int kTileIdx  = kIdx >> 2;
    const int innerKIdx = kIdx & 3;
    return ((static_cast<long>(mTileIdx) * numKTiles + kTileIdx) << 9)
         | (static_cast<long>(outerMIdx) << 4)
         | (static_cast<long>(innerMIdx) << 2)
         | static_cast<long>(innerKIdx);
}

void checkActQuantNvfp4Parity(int M, int K, float gscale, std::uint32_t seed) {
    namespace cc = ::mimirmind::core::cuda;
    CudaComputeContext ctx{};
    GpuOps ops{ctx};
    cc::CudaModule mod = cc::CudaModule::fromFile(ctx.cudaContext(),
                                                  resolvePtx("moe_act_quant_nvfp4"));
    cc::CudaKernel kern = mod.getFunction("moe_act_quant_nvfp4");

    const int nBlocks   = K / 16;
    const int numKTiles = (nBlocks + 3) / 4;
    const int numMTiles = (M + 127) / 128;
    const std::size_t sfBytes =
        static_cast<std::size_t>(numMTiles) * numKTiles * 512;   // 32*4*4

    Lcg g{seed};
    std::vector<float> X(static_cast<std::size_t>(M) * K);
    for (auto& v : X) v = g.next() * 3.0f;                        // ~[-3, 3]

    auto dX  = uploadRaw(ops, X);
    auto dOut = ops.allocate(static_cast<std::size_t>(M) * K / 2);
    std::vector<unsigned char> sfZero(sfBytes, 0u);
    auto dSF  = uploadRaw(ops, sfZero);                           // pre-zeroed pad

    kern.setPtr  (0, dX.get());
    kern.setPtr  (1, dOut.get());
    kern.setPtr  (2, dSF.get());
    kern.setValue(3, gscale);
    kern.setValue(4, M);
    kern.setValue(5, K);
    const std::uint32_t gy = static_cast<std::uint32_t>((nBlocks + 255) / 256);
    kern.launch(ctx.stream(), static_cast<std::uint32_t>(M), gy, 1, 256, 1, 1);
    ops.flush();

    std::vector<unsigned char> out(static_cast<std::size_t>(M) * K / 2);
    std::vector<unsigned char> sf(sfBytes);
    ops.readbackToHost(out.data(), dOut.get(), out.size());
    ops.readbackToHost(sf.data(),  dSF.get(),  sf.size());

    double maxErr = 0.0;
    for (int m = 0; m < M; ++m) {
        for (int b = 0; b < nBlocks; ++b) {
            const int k0 = b * 16;
            double absmax = 0.0;
            for (int i = 0; i < 16; ++i) {
                absmax = std::max(absmax,
                    std::fabs(static_cast<double>(X[static_cast<std::size_t>(m) * K + k0 + i])));
            }
            const double sfVal = aq_dec_e4m3(sf[aq_test_sf_offset(m, b, numKTiles)]);
            // Bound: worst NVFP4 gap is 2*step (between codes 6 and 7), so half
            // is 1*step == sfVal; ×2 margin covers the E4M3 scale rounding too.
            const double bound = 0.34 * absmax + 1e-4;
            const std::size_t pbase = (static_cast<std::size_t>(m) * K + k0) / 2;
            for (int j = 0; j < 8; ++j) {
                const unsigned byte = out[pbase + j];
                const unsigned lo = byte & 0x0Fu;
                const unsigned hi = (byte >> 4) & 0x0Fu;
                const double recLo = aq_dec_e2m1(lo) * sfVal / gscale;
                const double recHi = aq_dec_e2m1(hi) * sfVal / gscale;
                const double xLo = X[static_cast<std::size_t>(m) * K + k0 + 2 * j];
                const double xHi = X[static_cast<std::size_t>(m) * K + k0 + 2 * j + 1];
                const double eLo = std::fabs(recLo - xLo);
                const double eHi = std::fabs(recHi - xHi);
                maxErr = std::max(maxErr, std::max(eLo, eHi));
                EXPECT_TRUE(eLo <= bound);
                EXPECT_TRUE(eHi <= bound);
            }
        }
    }
    std::printf("[act-quant-nvfp4] M=%d K=%d gscale=%.3f maxErr=%.3e OK\n",
                M, K, gscale, maxErr);
}

} // namespace

TEST(cuda_moe_act_quant_nvfp4_parity_basic) {
    checkActQuantNvfp4Parity(/*M=*/20, /*K=*/64, /*gscale=*/1.0f, 0x51A1u);
}

TEST(cuda_moe_act_quant_nvfp4_parity_hidden) {
    // Realistic Qwen3.6 hidden K=2048, a full 128-row tile plus change.
    checkActQuantNvfp4Parity(/*M=*/130, /*K=*/2048, /*gscale=*/1.0f, 0x7C3Du);
}

TEST(cuda_moe_act_quant_nvfp4_parity_kpad) {
    // nBlocks=3 (K/16) is not a multiple of 4 -> exercises kTile padding.
    checkActQuantNvfp4Parity(/*M=*/17, /*K=*/48, /*gscale=*/1.0f, 0x2B9Fu);
}

TEST(cuda_moe_act_quant_nvfp4_parity_gscale) {
    // Non-unit global scale (K=512 = down-proj intermediate).
    checkActQuantNvfp4Parity(/*M=*/96, /*K=*/512, /*gscale=*/0.5f, 0x1DE7u);
}

// M-Cuda.MoeGroup Sub-Step E-d.2b — the device weight-SF swizzle kernel
// (moe_weight_sf_swizzle) must be byte-identical to the host swizzleBlockScale
// (verified vs cute's tile_atom_to_shape_SFA on GB10), so the loader can build
// the SFB bank on-device without a D2H/H2D round-trip.
TEST(cuda_moe_weight_sf_swizzle_parity) {
    namespace cc = ::mimirmind::core::cuda;
    namespace mo = ::mimirmind::core::modelopt;
    CudaComputeContext ctx{};
    GpuOps ops{ctx};
    cc::CudaModule mod = cc::CudaModule::fromFile(
        ctx.cudaContext(), resolvePtx("moe_weight_sf_swizzle"));
    cc::CudaKernel kern = mod.getFunction("moe_weight_sf_swizzle");

    const std::array<std::array<int, 2>, 3> shapes{{{40, 128}, {130, 2048}, {17, 48}}};
    for (const auto& sh : shapes) {
        const int rows = sh[0], in = sh[1], ksf = in / 16;
        const std::size_t bytes = mo::swizzledBlockScaleBytes(
            static_cast<std::uint64_t>(rows), static_cast<std::uint64_t>(ksf));

        Lcg g{static_cast<std::uint32_t>(0x1234u + rows)};
        std::vector<unsigned char> src(static_cast<std::size_t>(rows) * ksf);
        for (auto& b : src) { g.s = g.s * 1664525u + 1013904223u; b = static_cast<unsigned char>(g.s >> 24); }

        std::vector<unsigned char> ref(bytes);
        mo::swizzleBlockScale(src.data(), rows, ksf, ref.data());

        auto dSrc = uploadRaw(ops, src);
        std::vector<unsigned char> zero(bytes, 0u);
        auto dDst = uploadRaw(ops, zero);
        kern.setPtr  (0, dSrc.get());
        kern.setPtr  (1, dDst.get());
        kern.setValue(2, rows);
        kern.setValue(3, ksf);
        kern.launch(ctx.stream(), static_cast<std::uint32_t>(rows),
                    static_cast<std::uint32_t>((ksf + 127) / 128), 1, 128, 1, 1);
        ops.flush();

        std::vector<unsigned char> got(bytes);
        ops.readbackToHost(got.data(), dDst.get(), bytes);
        for (std::size_t i = 0; i < bytes; ++i) EXPECT_EQ(got[i], ref[i]);
        std::printf("[weight-sf-swizzle] rows=%d in=%d bytes=%zu OK\n", rows, in, bytes);
    }
}

// M-Cuda.MoeGroup Sub-Step E-d.4b — padding infrastructure kernels. Validate
// padOffset (prefix of round_up(count,128)), contigToPad (contiguous row ->
// padded row, binary search over expOffset), the row spread, and the index
// remap. Edge cases: an empty expert (count 0) and a small tail expert.
TEST(cuda_moe_pad_infra) {
    namespace cc = ::mimirmind::core::cuda;
    CudaComputeContext ctx{};
    GpuOps ops{ctx};
    auto load = [&](const char* n) {
        return cc::CudaModule::fromFile(ctx.cudaContext(), resolvePtx(n));
    };
    cc::CudaModule mOff = load("moe_pad");
    cc::CudaKernel kOff  = mOff.getFunction("moe_pad_offsets");
    cc::CudaKernel kC2P  = mOff.getFunction("moe_contig_to_pad");
    cc::CudaKernel kSpr  = mOff.getFunction("moe_rows_scatter_f32");
    cc::CudaKernel kGat  = mOff.getFunction("moe_index_gather_i32");

    const std::vector<int> counts{32, 16, 80, 0, 5};
    const int nE = static_cast<int>(counts.size());
    std::vector<std::int32_t> expOff(nE + 1, 0);
    for (int e = 0; e < nE; ++e) expOff[e + 1] = expOff[e] + counts[e];
    const int R = expOff[nE];

    std::vector<std::int32_t> padRef(nE + 1, 0);
    for (int e = 0; e < nE; ++e) padRef[e + 1] = padRef[e] + ((counts[e] + 127) / 128) * 128;
    const int totalPad = padRef[nE];

    auto dExp = uploadRaw(ops, expOff);
    auto dPad = ops.allocate((nE + 1) * sizeof(std::int32_t));
    kOff.setPtr(0, dExp.get()); kOff.setPtr(1, dPad.get()); kOff.setValue(2, nE);
    kOff.launch(ctx.stream(), 1, 1, 1, 1, 1, 1);
    ops.flush();
    std::vector<std::int32_t> padGot(nE + 1);
    ops.readbackToHost(padGot.data(), dPad.get(), padGot.size() * sizeof(std::int32_t));
    for (int e = 0; e <= nE; ++e) EXPECT_EQ(padGot[e], padRef[e]);

    // contigToPad
    auto dC2P = ops.allocate(static_cast<std::size_t>(R) * sizeof(std::int32_t));
    kC2P.setPtr(0, dExp.get()); kC2P.setPtr(1, dPad.get()); kC2P.setPtr(2, dC2P.get());
    kC2P.setValue(3, nE); kC2P.setValue(4, R);
    kC2P.launch(ctx.stream(), static_cast<std::uint32_t>((R + 127) / 128), 1, 1, 128, 1, 1);
    ops.flush();
    std::vector<std::int32_t> c2p(R), c2pRef(R);
    for (int e = 0; e < nE; ++e)
        for (int i = 0; i < counts[e]; ++i) c2pRef[expOff[e] + i] = padRef[e] + i;
    ops.readbackToHost(c2p.data(), dC2P.get(), static_cast<std::size_t>(R) * sizeof(std::int32_t));
    for (int r = 0; r < R; ++r) EXPECT_EQ(c2p[r], c2pRef[r]);

    // rows spread
    const int dim = 4;
    std::vector<float> src(static_cast<std::size_t>(R) * dim);
    for (int r = 0; r < R; ++r) for (int c = 0; c < dim; ++c) src[r * dim + c] = r * 10.0f + c;
    auto dSrc = uploadRaw(ops, src);
    std::vector<float> zero(static_cast<std::size_t>(totalPad) * dim, -1.0f);
    auto dDst = uploadRaw(ops, zero);
    kSpr.setPtr(0, dSrc.get()); kSpr.setPtr(1, dC2P.get()); kSpr.setPtr(2, dDst.get());
    kSpr.setValue(3, R); kSpr.setValue(4, dim);
    kSpr.launch(ctx.stream(), static_cast<std::uint32_t>(R), 1, 1, 32, 1, 1);
    ops.flush();
    std::vector<float> dst(static_cast<std::size_t>(totalPad) * dim);
    ops.readbackToHost(dst.data(), dDst.get(), dst.size() * sizeof(float));
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < dim; ++c)
            EXPECT_TRUE(dst[static_cast<std::size_t>(c2pRef[r]) * dim + c] == src[r * dim + c]);

    // index remap
    std::vector<std::int32_t> asn(R);
    for (int i = 0; i < R; ++i) asn[i] = R - 1 - i;
    auto dAsn = uploadRaw(ops, asn);
    auto dOut = ops.allocate(static_cast<std::size_t>(R) * sizeof(std::int32_t));
    kGat.setPtr(0, dAsn.get()); kGat.setPtr(1, dC2P.get()); kGat.setPtr(2, dOut.get());
    kGat.setValue(3, R);
    kGat.launch(ctx.stream(), static_cast<std::uint32_t>((R + 127) / 128), 1, 1, 128, 1, 1);
    ops.flush();
    std::vector<std::int32_t> out(R);
    ops.readbackToHost(out.data(), dOut.get(), static_cast<std::size_t>(R) * sizeof(std::int32_t));
    for (int a = 0; a < R; ++a) EXPECT_EQ(out[a], c2pRef[asn[a]]);
    std::printf("[moe-pad-infra] nE=%d R=%d totalPad=%d OK\n", nE, R, totalPad);
}

// M-Cuda.MoeGroup Sub-Step E-d.5 — the FP4-TC DECODE kernels
// (moe_gate_up_fused_k_nvfp4tc_batched + moe_down_fused_k_nvfp4tc_batched) read
// the SAME plain-nibbles + swizzled-SFB + F32-global banks the prefill grouped
// GEMM uses, so the blocked-NVFP4 bank can be dropped. Validate both against a
// CPU dequant-and-matmul reference that reads the identical device banks
// (nib * ue4m3(SFB[swizzle]) * global), through the same swizzle-offset formula.
namespace {

// Quantise a host [rows, K] F32 matrix (gscale=1) into a device nibble-bank
// slot + swizzled-SFB slot via moe_act_quant_nvfp4; keep host copies for the
// reference. Reuses the E-d.1 decoders aq_dec_e2m1 / aq_dec_e4m3 (== dq_ue4m3).
struct TcWeightSlot {
    std::vector<unsigned char> nibHost;   // [rows * K/2]
    std::vector<unsigned char> sfHost;    // swizzled slot
    int rows{0}, K{0}, numKTiles{0};
};

TcWeightSlot tcQuantSlot(GpuOps& ops, ::mimirmind::core::cuda::CudaKernel& kern,
                         CudaComputeContext& ctx,
                         const std::vector<float>& W, int rows, int K,
                         void* dNibSlot, void* dSfSlot) {
    namespace mo = ::mimirmind::core::modelopt;
    const int ksf = K / 16;
    TcWeightSlot s; s.rows = rows; s.K = K; s.numKTiles = (ksf + 3) / 4;
    const std::size_t sfBytes = mo::swizzledBlockScaleBytes(rows, ksf);
    // pre-zero the swizzled slot (padding), then quantise into it.
    std::vector<unsigned char> zero(sfBytes, 0u);
    ops.uploadHostBytes(dSfSlot, zero.data(), sfBytes);
    auto dW = uploadRaw(ops, W);
    kern.setPtr(0, dW.get()); kern.setPtr(1, dNibSlot); kern.setPtr(2, dSfSlot);
    kern.setValue(3, 1.0f); kern.setValue(4, rows); kern.setValue(5, K);
    const std::uint32_t gy = static_cast<std::uint32_t>((ksf + 255) / 256);
    kern.launch(ctx.stream(), static_cast<std::uint32_t>(rows), gy, 1, 256, 1, 1);
    ops.flush();
    s.nibHost.resize(static_cast<std::size_t>(rows) * (K / 2));
    s.sfHost.resize(sfBytes);
    ops.readbackToHost(s.nibHost.data(), dNibSlot, s.nibHost.size());
    ops.readbackToHost(s.sfHost.data(),  dSfSlot,  sfBytes);
    return s;
}

double tcDequant(const TcWeightSlot& s, int row, int k) {
    const std::size_t byte = (static_cast<std::size_t>(row) * s.K + k) / 2;
    const unsigned nib = (k & 1) ? (s.nibHost[byte] >> 4) & 0xF : s.nibHost[byte] & 0xF;
    const double sf = aq_dec_e4m3(s.sfHost[aq_test_sf_offset(row, k / 16, s.numKTiles)]);
    return aq_dec_e2m1(nib) * sf;   // global == 1
}

} // namespace

TEST(cuda_moe_nvfp4tc_decode_parity) {
    namespace cc = ::mimirmind::core::cuda;
    namespace mo = ::mimirmind::core::modelopt;
    CudaComputeContext ctx{};
    GpuOps ops{ctx};
    auto ptx = [&](const char* n) { return cc::CudaModule::fromFile(ctx.cudaContext(), resolvePtx(n)); };
    cc::CudaModule mAq = ptx("moe_act_quant_nvfp4");
    cc::CudaKernel kAq = mAq.getFunction("moe_act_quant_nvfp4");
    cc::CudaModule mGu = ptx("moe_gate_up_fused_k_nvfp4tc_batched");
    cc::CudaKernel kGu = mGu.getFunction("moe_gate_up_fused_k_nvfp4tc_batched");
    cc::CudaModule mDn = ptx("moe_down_fused_k_nvfp4tc_batched");
    cc::CudaKernel kDn = mDn.getFunction("moe_down_fused_k_nvfp4tc_batched");

    const int dModel = 128, nFf = 256, nExperts = 2, expUsed = 1, nSeq = 1;
    const int E = 1;  // routed expert index (exercises the per-expert stride)
    const std::size_t guNibStride = static_cast<std::size_t>(nFf) * (dModel / 2);
    const std::size_t guSfStride  = mo::swizzledBlockScaleBytes(nFf, dModel / 16);
    const std::size_t dnNibStride = static_cast<std::size_t>(dModel) * (nFf / 2);
    const std::size_t dnSfStride  = mo::swizzledBlockScaleBytes(dModel, nFf / 16);

    auto gNibBank = ops.allocate(nExperts * guNibStride);
    auto uNibBank = ops.allocate(nExperts * guNibStride);
    auto gSfBank  = ops.allocate(nExperts * guSfStride);
    auto uSfBank  = ops.allocate(nExperts * guSfStride);
    auto dNibBank = ops.allocate(nExperts * dnNibStride);
    auto dSfBank  = ops.allocate(nExperts * dnSfStride);
    auto ones     = uploadRaw(ops, std::vector<float>(nExperts, 1.0f)); // globals

    Lcg g{0x2C7Bu};
    auto rnd = [&](std::size_t n) { std::vector<float> v(n); for (auto& x : v) x = g.next(); return v; };
    std::vector<float> Wg = rnd(static_cast<std::size_t>(nFf) * dModel);
    std::vector<float> Wu = rnd(static_cast<std::size_t>(nFf) * dModel);
    std::vector<float> Wd = rnd(static_cast<std::size_t>(dModel) * nFf);
    auto slotN = [](::mimirmind::compute::ComputeBuffer& b, std::size_t stride, int e) {
        return static_cast<unsigned char*>(b.get()) + static_cast<std::size_t>(e) * stride;
    };
    TcWeightSlot gS = tcQuantSlot(ops, kAq, ctx, Wg, nFf, dModel, slotN(gNibBank, guNibStride, E), slotN(gSfBank, guSfStride, E));
    TcWeightSlot uS = tcQuantSlot(ops, kAq, ctx, Wu, nFf, dModel, slotN(uNibBank, guNibStride, E), slotN(uSfBank, guSfStride, E));
    TcWeightSlot dS = tcQuantSlot(ops, kAq, ctx, Wd, dModel, nFf, slotN(dNibBank, dnNibStride, E), slotN(dSfBank, dnSfStride, E));

    std::vector<float> X = rnd(dModel);
    auto dX = uploadRaw(ops, X);
    std::vector<std::int32_t> expIdx{E};
    auto dExpIdx = uploadRaw(ops, expIdx);
    auto dKw     = uploadRaw(ops, std::vector<float>{1.0f});
    auto dGate   = ops.allocate(static_cast<std::size_t>(nFf) * sizeof(float));
    auto dAccum  = ops.allocate(static_cast<std::size_t>(dModel) * sizeof(float));
    ops.uploadHostBytes(dAccum.get(), std::vector<float>(dModel, 0.0f).data(), dModel * sizeof(float));

    // gate+up: grid (ceil(expUsed*nFf/4), nSeq), block 128.
    kGu.setPtr(0, dX.get()); kGu.setPtr(1, gNibBank.get()); kGu.setPtr(2, uNibBank.get());
    kGu.setPtr(3, gSfBank.get()); kGu.setPtr(4, uSfBank.get());
    kGu.setPtr(5, ones.get()); kGu.setPtr(6, ones.get());
    kGu.setPtr(7, dExpIdx.get()); kGu.setPtr(8, dGate.get());
    kGu.setValue(9, dModel); kGu.setValue(10, nFf); kGu.setValue(11, expUsed);
    kGu.setValue(12, static_cast<std::int32_t>(guSfStride));
    kGu.launch(ctx.stream(), static_cast<std::uint32_t>((expUsed * nFf + 3) / 4),
               static_cast<std::uint32_t>(nSeq), 1, 128, 1, 1);
    ops.flush();
    std::vector<float> gateAct(nFf);
    ops.readbackToHost(gateAct.data(), dGate.get(), nFf * sizeof(float));

    // reference gate/up
    double maxGuErr = 0.0;
    for (int f = 0; f < nFf; ++f) {
        double gd = 0.0, ud = 0.0;
        for (int k = 0; k < dModel; ++k) { gd += X[k] * tcDequant(gS, f, k); ud += X[k] * tcDequant(uS, f, k); }
        const double silu = gd / (1.0 + std::exp(-gd));
        maxGuErr = std::max(maxGuErr, std::fabs(gateAct[f] - silu * ud));
    }

    // down: grid (ceil(dModel/4), nSeq), block 64. input = gateAct [expUsed, nFf].
    kDn.setPtr(0, dGate.get()); kDn.setPtr(1, dNibBank.get()); kDn.setPtr(2, dSfBank.get());
    kDn.setPtr(3, ones.get()); kDn.setPtr(4, dExpIdx.get()); kDn.setPtr(5, dKw.get());
    kDn.setPtr(6, dAccum.get());
    kDn.setValue(7, nFf); kDn.setValue(8, dModel); kDn.setValue(9, expUsed);
    kDn.setValue(10, static_cast<std::int32_t>(dnSfStride));
    kDn.launch(ctx.stream(), static_cast<std::uint32_t>((dModel + 3) / 4),
               static_cast<std::uint32_t>(nSeq), 1, 64, 1, 1);
    ops.flush();
    std::vector<float> accum(dModel);
    ops.readbackToHost(accum.data(), dAccum.get(), dModel * sizeof(float));

    double maxDnErr = 0.0;
    for (int n = 0; n < dModel; ++n) {
        double d = 0.0;
        for (int k = 0; k < nFf; ++k) d += gateAct[k] * tcDequant(dS, n, k);
        maxDnErr = std::max(maxDnErr, std::fabs(accum[n] - d));
    }
    EXPECT_TRUE(maxGuErr <= 1e-2);
    EXPECT_TRUE(maxDnErr <= 1e-2);
    std::printf("[nvfp4tc-decode] maxGuErr=%.3e maxDnErr=%.3e OK\n", maxGuErr, maxDnErr);
}

#ifdef MIMIRMIND_HAVE_CUTLASS_MOE
// M-Cuda.MoeGroup Sub-Step E-d.3 — end-to-end numeric parity of the CUTLASS
// block-scaled NVFP4 tensor-core grouped GEMM (runGroupedNvfp4TcF32) against a
// CPU dequant-and-matmul reference. Both A and B are quantised on the device by
// the E-d.1 act-quant kernel (F32 -> NVFP4 E2M1 + swizzled UE4M3 SF): A=[M,K]
// gives SFA (rows=M), the same kernel on the weight [N,K] gives B nibbles +
// SFB (rows=N). gscale=1 for both, so per-group alpha=1 and each quantised
// value reconstructs to e2m1(nibble)*e4m3(SF). The reference dequantises the
// SAME device-side nibbles/scales the GEMM reads, so the only slack is bf16
// output rounding + FP accumulation order — a tight tolerance.
namespace {

// Quantise a host [rows, K] F32 matrix to NVFP4 via moe_act_quant_nvfp4.
// Returns the device nibble bank [rows*K/2], the device swizzled SF bank, and
// keeps host copies of both for the CPU reference. gscale=1.
struct QuantOperand {
    ::mimirmind::compute::ComputeBuffer nib;   // [rows*K/2] E2M1
    ::mimirmind::compute::ComputeBuffer sf;    // swizzled UE4M3 (pre-zeroed)
    std::vector<unsigned char>          nibHost;
    std::vector<unsigned char>          sfHost;
    int rows{0};
    int K{0};
    int numKTiles{0};
};

QuantOperand quantizeNvfp4(GpuOps& ops, ::mimirmind::core::cuda::CudaKernel& kern,
                           CudaComputeContext& ctx,
                           const std::vector<float>& X, int rows, int K) {
    namespace mo = ::mimirmind::core::modelopt;
    const int nBlocks   = K / 16;
    const int numKTiles = (nBlocks + 3) / 4;
    const std::size_t sfBytes = mo::swizzledBlockScaleBytes(
        static_cast<std::uint64_t>(rows), static_cast<std::uint64_t>(nBlocks));

    QuantOperand q;
    q.rows = rows; q.K = K; q.numKTiles = numKTiles;
    q.nib = ops.allocate(static_cast<std::size_t>(rows) * K / 2);
    std::vector<unsigned char> sfZero(sfBytes, 0u);
    q.sf = uploadRaw(ops, sfZero);
    auto dX = uploadRaw(ops, X);

    kern.setPtr  (0, dX.get());
    kern.setPtr  (1, q.nib.get());
    kern.setPtr  (2, q.sf.get());
    kern.setValue(3, 1.0f);      // gscale = 1
    kern.setValue(4, rows);
    kern.setValue(5, K);
    const std::uint32_t gy = static_cast<std::uint32_t>((nBlocks + 255) / 256);
    kern.launch(ctx.stream(), static_cast<std::uint32_t>(rows), gy, 1, 256, 1, 1);
    ops.flush();

    q.nibHost.resize(static_cast<std::size_t>(rows) * K / 2);
    q.sfHost.resize(sfBytes);
    ops.readbackToHost(q.nibHost.data(), q.nib.get(), q.nibHost.size());
    ops.readbackToHost(q.sfHost.data(),  q.sf.get(),  q.sfHost.size());
    return q;
}

// Dequantised value of element (r, k) of a quantised operand.
double dequantElem(const QuantOperand& q, int r, int k) {
    const int blk = k / 16;
    const std::size_t byte = (static_cast<std::size_t>(r) * q.K + k) / 2;
    const unsigned nib = (k & 1) ? (q.nibHost[byte] >> 4) & 0xF
                                 :  q.nibHost[byte]       & 0xF;
    const double sf = aq_dec_e4m3(q.sfHost[aq_test_sf_offset(r, blk, q.numKTiles)]);
    return aq_dec_e2m1(nib) * sf;
}

void checkGroupedNvfp4TcParity(const std::vector<std::array<int, 3>>& groups,
                               std::uint32_t seed, bool deviceDriven = false) {
    namespace cc = ::mimirmind::core::cuda;
    CudaComputeContext ctx{};
    GpuOps ops{ctx};
    cc::CudaModule mod = cc::CudaModule::fromFile(ctx.cudaContext(),
                                                  resolvePtx("moe_act_quant_nvfp4"));
    cc::CudaKernel kern = mod.getFunction("moe_act_quant_nvfp4");

    const int G = static_cast<int>(groups.size());
    std::vector<int> mH(G), nH(G), kH(G);
    std::vector<QuantOperand> A, B;
    A.reserve(G); B.reserve(G);
    std::vector<::mimirmind::compute::ComputeBuffer> dOut, dAlphaVal;
    std::vector<const void*>  pA(G), pSFA(G), pB(G), pSFB(G);
    std::vector<void*>        pD(G);
    std::vector<const float*> pAlpha(G);

    Lcg g{seed};
    for (int e = 0; e < G; ++e) {
        const int M = groups[e][0], N = groups[e][1], K = groups[e][2];
        mH[e] = M; nH[e] = N; kH[e] = K;
        std::vector<float> Xa(static_cast<std::size_t>(M) * K);
        std::vector<float> Xb(static_cast<std::size_t>(N) * K);
        for (auto& v : Xa) v = g.next();          // ~[-1,1]
        for (auto& v : Xb) v = g.next();
        A.push_back(quantizeNvfp4(ops, kern, ctx, Xa, M, K));
        B.push_back(quantizeNvfp4(ops, kern, ctx, Xb, N, K));

        auto d = ops.allocate(static_cast<std::size_t>(M) * N * sizeof(float)); // f32 out
        dOut.push_back(std::move(d));
        std::vector<float> one{1.0f};
        dAlphaVal.push_back(uploadRaw(ops, one));

        pA[e]    = A[e].nib.get(); pSFA[e] = A[e].sf.get();
        pB[e]    = B[e].nib.get(); pSFB[e] = B[e].sf.get();
        pD[e]    = dOut[e].get();
        pAlpha[e]= static_cast<const float*>(dAlphaVal[e].get());
    }

    // Device arrays of device pointers (the grouped operand arrays).
    auto dpA = uploadRaw(ops, pA);
    auto dpSFA = uploadRaw(ops, pSFA);
    auto dpB = uploadRaw(ops, pB);
    auto dpSFB = uploadRaw(ops, pSFB);
    auto dpD = uploadRaw(ops, pD);
    auto dpAlpha = uploadRaw(ops, pAlpha);
    ops.flush();

    namespace cm = ::mimirmind::kernels::cutlassmoe;
    int rc;
    // Keep expOffset alive across the device-driven call (it reads M on device).
    ::mimirmind::compute::ComputeBuffer dExp;
    if (deviceDriven) {
        // Device path: M per group comes from a device expOffset (cumulative
        // row starts); N,K shared across groups. Nothing crosses to the host.
        std::vector<std::int32_t> expOff(G + 1, 0);
        for (int e = 0; e < G; ++e) expOff[e + 1] = expOff[e] + mH[e];
        dExp = uploadRaw(ops, expOff);
        ops.flush();
        rc = cm::runGroupedNvfp4TcF32DeviceDriven(
            G, nH[0], kH[0], static_cast<const std::int32_t*>(dExp.get()),
            static_cast<const void* const*>(dpA.get()),
            static_cast<const void* const*>(dpSFA.get()),
            static_cast<const void* const*>(dpB.get()),
            static_cast<const void* const*>(dpSFB.get()),
            reinterpret_cast<const float* const*>(dpAlpha.get()),
            static_cast<void* const*>(dpD.get()),
            ctx.stream().handle());
    } else {
        rc = cm::runGroupedNvfp4TcF32(
            G, mH.data(), nH.data(), kH.data(),
            static_cast<const void* const*>(dpA.get()),
            static_cast<const void* const*>(dpSFA.get()),
            static_cast<const void* const*>(dpB.get()),
            static_cast<const void* const*>(dpSFB.get()),
            reinterpret_cast<const float* const*>(dpAlpha.get()),
            static_cast<void* const*>(dpD.get()),
            ctx.stream().handle());
    }
    EXPECT_EQ(rc, 0);

    double maxRel = 0.0, maxAbs = 0.0;
    for (int e = 0; e < G; ++e) {
        const int M = mH[e], N = nH[e], K = kH[e];
        std::vector<float> D(static_cast<std::size_t>(M) * N);
        ops.readbackToHost(D.data(), dOut[e].get(), D.size() * sizeof(float));
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                double ref = 0.0;
                for (int k = 0; k < K; ++k)
                    ref += dequantElem(A[e], m, k) * dequantElem(B[e], n, k);
                const double got = D[static_cast<std::size_t>(m) * N + n];
                const double aerr = std::fabs(got - ref);
                const double rerr = aerr / (std::fabs(ref) + 1e-3);
                maxAbs = std::max(maxAbs, aerr);
                maxRel = std::max(maxRel, rerr);
                // bf16 out (8-bit mantissa) + FP accumulation order: ~1% rel.
                EXPECT_TRUE(aerr <= 0.02 * std::fabs(ref) + 0.05);
            }
        }
    }
    std::printf("[grouped-nvfp4-tc] G=%d maxAbs=%.3e maxRel=%.3e OK\n",
                G, maxAbs, maxRel);
}

} // namespace

TEST(cuda_moe_grouped_nvfp4_tc_parity_single) {
    checkGroupedNvfp4TcParity({{48, 64, 64}}, 0x3311u);
}

TEST(cuda_moe_grouped_nvfp4_tc_parity_multi) {
    // Variable M per group (the MoE case: tokens/expert differ), shared-ish N/K.
    checkGroupedNvfp4TcParity({{32, 64, 64}, {16, 128, 128}, {80, 64, 128}}, 0x77A2u);
}

// E-d.3b — fully device-driven: per-group M / problem_sizes / strides / SF
// layouts built ON DEVICE from a device expOffset (no host M, no D2H). N,K are
// shared across groups (the real MoE shape: only tokens-per-expert vary).
TEST(cuda_moe_grouped_nvfp4_tc_device_single) {
    checkGroupedNvfp4TcParity({{64, 128, 128}}, 0x51B3u, /*deviceDriven=*/true);
}

TEST(cuda_moe_grouped_nvfp4_tc_device_multi) {
    checkGroupedNvfp4TcParity({{32, 128, 128}, {16, 128, 128}, {80, 128, 128}},
                              0x9E4Cu, /*deviceDriven=*/true);
}

// E-d.4 — the banks runtime op (runGroupedNvfp4TcF32Banks): contiguous device
// banks + device expOffset/padOffset, all per-group pointers built on device.
// Mirrors the runtime: ONE act-quant over the padded [totalPad,K] activation
// buffer (each expert padded to 128 rows so its SFA sub-tensor is tile-aligned),
// contiguous weight nibble+SFB banks, per-expert F32 globals. Compares the real
// (non-padding) output rows vs the CPU dequant-and-matmul reference.
TEST(cuda_moe_grouped_nvfp4_tc_banks_multi) {
    namespace cc = ::mimirmind::core::cuda;
    namespace mo = ::mimirmind::core::modelopt;
    CudaComputeContext ctx{};
    GpuOps ops{ctx};
    cc::CudaModule mod = cc::CudaModule::fromFile(ctx.cudaContext(),
                                                  resolvePtx("moe_act_quant_nvfp4"));
    cc::CudaKernel kern = mod.getFunction("moe_act_quant_nvfp4");

    const std::vector<int> M{32, 16, 80};
    const int G = static_cast<int>(M.size());
    const int N = 128, K = 128, ksf = K / 16;
    const std::size_t sfbStride = mo::moeSwizzledScaleStride(N, ksf);

    std::vector<std::int32_t> expOff(G + 1, 0), padOff(G + 1, 0);
    for (int e = 0; e < G; ++e) {
        expOff[e + 1] = expOff[e] + M[e];
        padOff[e + 1] = padOff[e] + ((M[e] + 127) / 128) * 128;
    }
    const int totalPad = padOff[G];

    Lcg g{0x4B7Fu};
    auto launchQuant = [&](const void* dX, void* dNib, void* dSf, int rows) {
        kern.setPtr(0, dX); kern.setPtr(1, dNib); kern.setPtr(2, dSf);
        kern.setValue(3, 1.0f); kern.setValue(4, rows); kern.setValue(5, K);
        const std::uint32_t gy = static_cast<std::uint32_t>(((K / 16) + 255) / 256);
        kern.launch(ctx.stream(), static_cast<std::uint32_t>(rows), gy, 1, 256, 1, 1);
    };

    // Padded activation buffer: real rows random, padding rows zero.
    std::vector<float> Xa(static_cast<std::size_t>(totalPad) * K, 0.0f);
    for (int e = 0; e < G; ++e)
        for (int m = 0; m < M[e]; ++m)
            for (int k = 0; k < K; ++k)
                Xa[static_cast<std::size_t>(padOff[e] + m) * K + k] = g.next();
    QuantOperand A = quantizeNvfp4(ops, kern, ctx, Xa, totalPad, K);  // aBank + big SFA

    // Weight banks: quantise each expert's [N,K] into the contiguous nib/SFB banks.
    auto bBank = ops.allocate(static_cast<std::size_t>(G) * N * (K / 2));
    std::vector<unsigned char> sfbZero(static_cast<std::size_t>(G) * sfbStride, 0u);
    auto sfbBank = uploadRaw(ops, sfbZero);
    auto globals = uploadRaw(ops, std::vector<float>(static_cast<std::size_t>(G), 1.0f));
    std::vector<QuantOperand> B(G);
    for (int e = 0; e < G; ++e) {
        std::vector<float> Xb(static_cast<std::size_t>(N) * K);
        for (auto& v : Xb) v = g.next();
        auto dXb = uploadRaw(ops, Xb);
        auto* nibDst = static_cast<unsigned char*>(bBank.get()) + static_cast<std::size_t>(e) * N * (K / 2);
        auto* sfDst  = static_cast<unsigned char*>(sfbBank.get()) + static_cast<std::size_t>(e) * sfbStride;
        launchQuant(dXb.get(), nibDst, sfDst, N);
        ops.flush();
        B[e].rows = N; B[e].K = K; B[e].numKTiles = (ksf + 3) / 4;
        B[e].nibHost.resize(static_cast<std::size_t>(N) * (K / 2));
        B[e].sfHost.resize(sfbStride);
        ops.readbackToHost(B[e].nibHost.data(), nibDst, B[e].nibHost.size());
        ops.readbackToHost(B[e].sfHost.data(),  sfDst,  sfbStride);
    }

    auto dOut  = ops.allocate(static_cast<std::size_t>(totalPad) * N * sizeof(float));  // f32 out
    auto dExp  = uploadRaw(ops, expOff);
    auto dPad  = uploadRaw(ops, padOff);
    ops.flush();

    const std::size_t scratchBytes =
        ::mimirmind::kernels::cutlassmoe::groupedNvfp4TcBanksScratchBytes(G);
    auto dScratch = ops.allocate(scratchBytes);
    const int rc = ::mimirmind::kernels::cutlassmoe::runGroupedNvfp4TcF32Banks(
        G, N, K,
        static_cast<const std::int32_t*>(dExp.get()),
        static_cast<const std::int32_t*>(dPad.get()),
        A.nib.get(), A.sf.get(), bBank.get(), sfbBank.get(),
        static_cast<const float*>(globals.get()), dOut.get(),
        dScratch.get(), scratchBytes, ctx.stream().handle());
    ops.flush();  // banks op no longer syncs internally
    EXPECT_EQ(rc, 0);

    std::vector<float> D(static_cast<std::size_t>(totalPad) * N);
    ops.readbackToHost(D.data(), dOut.get(), D.size() * sizeof(float));

    double maxRel = 0.0, maxAbs = 0.0;
    for (int e = 0; e < G; ++e) {
        for (int m = 0; m < M[e]; ++m) {
            const int row = padOff[e] + m;
            for (int n = 0; n < N; ++n) {
                double ref = 0.0;
                for (int k = 0; k < K; ++k)
                    ref += dequantElem(A, row, k) * dequantElem(B[e], n, k);
                const double got = D[static_cast<std::size_t>(row) * N + n];
                const double aerr = std::fabs(got - ref);
                maxAbs = std::max(maxAbs, aerr);
                maxRel = std::max(maxRel, aerr / (std::fabs(ref) + 1e-3));
                EXPECT_TRUE(aerr <= 0.02 * std::fabs(ref) + 0.05);
            }
        }
    }
    std::printf("[grouped-nvfp4-tc-banks] G=%d totalPad=%d maxAbs=%.3e maxRel=%.3e OK\n",
                G, totalPad, maxAbs, maxRel);
}

// Fp4TcDec-b PoC (env-gated, NOT part of the correctness gate): time the two
// grouped-MoE decode GEMM paths — blocked GD-b (`moe_grouped_gemm_nvfp4blk_m4`,
// the current default-on decode kernel) vs the CUTLASS NVFP4 tensor-core banks
// path (`runGroupedNvfp4TcF32Banks`) — across a per-expert decode-M sweep at the
// real gate_up shape (dModel=2048, gate+up width=1024, 256 experts). Answers
// the scoping question: at what per-expert M does TC overtake the scalar-dequant
// blocked kernel, and what is TC's ceiling once its 128-row M-tile is filled.
// Run with MIMIRMIND_MOE_DECODE_BENCH=1; otherwise this returns immediately so
// ctest is unaffected. See research note vllm-anchor-decode-lever-scoping.
TEST(fp4_tc_decode_m_sweep_bench) {
    if (std::getenv("MIMIRMIND_MOE_DECODE_BENCH") == nullptr) return;

    namespace cc = ::mimirmind::core::cuda;
    namespace mo = ::mimirmind::core::modelopt;
    namespace cm = ::mimirmind::kernels::cutlassmoe;
    CudaComputeContext ctx{};
    GpuOps ops{ctx};

    const int G = 256;          // routed experts
    const int K = 2048;         // dModel
    const int N = 1024;         // gate+up output width (2 * moe_ff, moe_ff=512)
    const int ksf = K / 16;
    const int nSuper = K / kNvSuperElems;
    const std::size_t expertBytesBlk =
        static_cast<std::size_t>(N) * nSuper * kNvSuperBytes;

    cc::CudaModule tilesMod = cc::CudaModule::fromFile(
        ctx.cudaContext(), resolvePtx("moe_group_tiles"));
    cc::CudaKernel tilesKern = tilesMod.getFunction("moe_group_tiles");
    cc::CudaModule aqMod = cc::CudaModule::fromFile(
        ctx.cudaContext(), resolvePtx("moe_act_quant_nvfp4"));
    cc::CudaKernel aqKern = aqMod.getFunction("moe_act_quant_nvfp4");

    const bool tcOk = ops.moeGroupedGemmNvfp4TcAvailable();

    // ---- blocked-NVFP4 weight bank [G][N][K] (shared across M sweep) ----------
    Lcg wg{0xD00Du};
    auto nextU32 = [&wg]() -> std::uint32_t {
        wg.s = wg.s * 1664525u + 1013904223u; return wg.s;
    };
    std::vector<unsigned char> Wblk(static_cast<std::size_t>(G) * expertBytesBlk);
    for (std::size_t off = 0; off + kNvSuperBytes <= Wblk.size(); off += kNvSuperBytes) {
        for (int h = 0; h < 2; ++h) {
            std::uint32_t bits = nextU32() & 0xFFFFu;
            if (((bits >> 10) & 0x1Fu) == 0x1Fu) bits &= ~(1u << 14);
            Wblk[off + h * 2 + 0] = static_cast<unsigned char>(bits & 0xFFu);
            Wblk[off + h * 2 + 1] = static_cast<unsigned char>((bits >> 8) & 0xFFu);
        }
        for (int b = 4; b < kNvSuperBytes; ++b)
            Wblk[off + b] = static_cast<unsigned char>(nextU32() & 0xFFu);
    }
    auto dWblk = uploadRaw(ops, Wblk);

    // ---- CUTLASS TC weight banks [G] of [N,K] (nibbles + swizzled SF) ---------
    const std::size_t sfbStride = mo::moeSwizzledScaleStride(N, ksf);
    ::mimirmind::compute::ComputeBuffer bBank, sfbBank, globals;
    if (tcOk) {
        bBank   = ops.allocate(static_cast<std::size_t>(G) * N * (K / 2));
        std::vector<unsigned char> sfbZero(static_cast<std::size_t>(G) * sfbStride, 0u);
        sfbBank = uploadRaw(ops, sfbZero);
        globals = uploadRaw(ops, std::vector<float>(static_cast<std::size_t>(G), 1.0f));
        Lcg bg{0xBEEFu};
        for (int e = 0; e < G; ++e) {
            std::vector<float> Xb(static_cast<std::size_t>(N) * K);
            for (auto& v : Xb) v = bg.next();
            auto dXb = uploadRaw(ops, Xb);
            auto* nibDst = static_cast<unsigned char*>(bBank.get())
                         + static_cast<std::size_t>(e) * N * (K / 2);
            auto* sfDst  = static_cast<unsigned char*>(sfbBank.get())
                         + static_cast<std::size_t>(e) * sfbStride;
            aqKern.setPtr(0, dXb.get()); aqKern.setPtr(1, nibDst); aqKern.setPtr(2, sfDst);
            aqKern.setValue(3, 1.0f); aqKern.setValue(4, N); aqKern.setValue(5, K);
            const std::uint32_t gy = static_cast<std::uint32_t>(((K / 16) + 255) / 256);
            aqKern.launch(ctx.stream(), static_cast<std::uint32_t>(N), gy, 1, 256, 1, 1);
        }
        ops.flush();
    }

    auto timeIt = [&](const std::function<void()>& enqueue) -> double {
        constexpr int kWarm = 5, kIter = 30;
        for (int i = 0; i < kWarm; ++i) enqueue();
        ctx.stream().synchronize();
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < kIter; ++i) enqueue();
        ctx.stream().synchronize();
        const auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count() / kIter;
    };

    std::printf("[fp4-tc-decode-sweep] G=%d K=%d N=%d  (gate_up shape)  tc=%s\n",
                G, K, N, tcOk ? "yes" : "no(CUTLASS not linked)");
    std::printf("  %6s | %10s | %10s | %9s | %7s | %8s\n",
                "M/exp", "blocked_ms", "tc_net_ms", "tc_gemm", "aq_ms", "net_spdup");
    for (int M : {1, 2, 4, 8, 16, 32, 64, 128}) {
        const int R = G * M;
        // Blocked inputs: X[R,K] f32, device tile schedule (tileM=4 -> m4 kernel).
        std::vector<float> X(static_cast<std::size_t>(R) * K);
        { Lcg xg{0x1234u + static_cast<std::uint32_t>(M)}; for (auto& v : X) v = xg.next(); }
        auto dX = uploadRaw(ops, X);
        auto dY = ops.allocate(static_cast<std::size_t>(R) * N * sizeof(float));

        std::vector<std::int32_t> expOff(G + 1, 0);
        for (int e = 0; e < G; ++e) expOff[e + 1] = expOff[e] + M;
        const int tileM = 4;
        const int maxTiles = (R + tileM - 1) / tileM + G;
        auto dTileE = ops.allocate(maxTiles * sizeof(std::int32_t));
        auto dTileR0 = ops.allocate(maxTiles * sizeof(std::int32_t));
        auto dTileRows = ops.allocate(maxTiles * sizeof(std::int32_t));
        auto dNslot = ops.allocate(sizeof(std::int32_t));
        auto dOff = uploadRaw(ops, expOff);
        tilesKern.setPtr(0, dOff.get()); tilesKern.setPtr(1, dTileE.get());
        tilesKern.setPtr(2, dTileR0.get()); tilesKern.setPtr(3, dTileRows.get());
        tilesKern.setPtr(4, dNslot.get());
        tilesKern.setValue(5, G); tilesKern.setValue(6, maxTiles); tilesKern.setValue(7, tileM);
        tilesKern.launch(ctx.stream(), 1, 1, 1, 1, 1, 1);
        ops.flush();

        const double blkMs = timeIt([&] {
            ops.moeGroupedGemmNvfp4Async(
                static_cast<const float*>(dX.get()),
                static_cast<const unsigned char*>(dWblk.get()),
                static_cast<float*>(dY.get()),
                static_cast<const std::int32_t*>(dTileE.get()),
                static_cast<const std::int32_t*>(dTileR0.get()),
                static_cast<const std::int32_t*>(dTileRows.get()),
                K, N, maxTiles, /*decodeSmallM=*/true);
        });

        double tcNetMs = 0.0, tcGemmMs = 0.0, aqMs = 0.0;
        if (tcOk) {
            std::vector<std::int32_t> padOff(G + 1, 0);
            for (int e = 0; e < G; ++e)
                padOff[e + 1] = padOff[e] + ((M + 127) / 128) * 128;
            const int totalPad = padOff[G];
            std::vector<float> Xa(static_cast<std::size_t>(totalPad) * K, 0.0f);
            { Lcg ag{0x9999u + static_cast<std::uint32_t>(M)};
              for (int e = 0; e < G; ++e)
                for (int m = 0; m < M; ++m)
                  for (int k = 0; k < K; ++k)
                    Xa[static_cast<std::size_t>(padOff[e] + m) * K + k] = ag.next(); }
            auto dXa = uploadRaw(ops, Xa);
            QuantOperand A = quantizeNvfp4(ops, aqKern, ctx, Xa, totalPad, K);
            auto dOut = ops.allocate(static_cast<std::size_t>(totalPad) * N * sizeof(float));
            auto dExpO = uploadRaw(ops, expOff);
            auto dPadO = uploadRaw(ops, padOff);
            const std::size_t scratchBytes = cm::groupedNvfp4TcBanksScratchBytes(G);
            auto dScratch = ops.allocate(scratchBytes);
            ops.flush();
            // Real W4A4 decode cost = activation-quantise A each step + TC GEMM.
            const std::uint32_t aqGy = static_cast<std::uint32_t>(((K / 16) + 255) / 256);
            auto actQuant = [&] {
                aqKern.setPtr(0, dXa.get()); aqKern.setPtr(1, A.nib.get());
                aqKern.setPtr(2, A.sf.get()); aqKern.setValue(3, 1.0f);
                aqKern.setValue(4, totalPad); aqKern.setValue(5, K);
                aqKern.launch(ctx.stream(), static_cast<std::uint32_t>(totalPad),
                              aqGy, 1, 256, 1, 1);
            };
            auto gemm = [&] {
                cm::runGroupedNvfp4TcF32Banks(
                    G, N, K,
                    static_cast<const std::int32_t*>(dExpO.get()),
                    static_cast<const std::int32_t*>(dPadO.get()),
                    A.nib.get(), A.sf.get(), bBank.get(), sfbBank.get(),
                    static_cast<const float*>(globals.get()), dOut.get(),
                    dScratch.get(), scratchBytes, ctx.stream().handle());
            };
            aqMs     = timeIt(actQuant);
            tcGemmMs = timeIt(gemm);
            tcNetMs  = timeIt([&] { actQuant(); gemm(); });
        }
        std::printf("  %6d | %10.4f | %10.4f | %9.4f | %7.4f | %8.2f\n",
                    M, blkMs, tcNetMs, tcGemmMs, aqMs,
                    (tcNetMs > 0.0 ? blkMs / tcNetMs : 0.0));
    }
    std::printf("[fp4-tc-decode-sweep] done  (blocked=GD-b m4 scalar; "
                "tc_net=actquant+CUTLASS-TC; W4A4)\n");
}
#endif // MIMIRMIND_HAVE_CUTLASS_MOE

int main() {
    return mm::test::run();
}