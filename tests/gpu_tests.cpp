// GPU-backed integration tests for compute::GpuOps and compute::GpuMatmul.
//
// We're NOT testing the iGPU hardware here — we're checking that our
// SPV kernels produce the same numerical result as the CPU reference
// implementations (compute::rmsNorm, siluInPlace + mulInPlace,
// applyRopeInPlace, compute::matmul) within an appropriate tolerance.
//
// Requires:
//   - A Level Zero device (Intel iGPU) reachable via /dev/dri.
//   - SPV kernels under /usr/local/share/mimirmind/spv (baked into the
//     runtime image by the build).
//
// Each test allocates USM via UsmAllocator (RAII wrapper UsmBuf below),
// fills with deterministic data, dispatches the GPU op, flushes the
// command queue, and compares against the CPU reference.

#include "TestFramework.hpp"

#include "compute/Activations.hpp"
#include "compute/Attention.hpp"
#include "compute/GpuMatmul.hpp"
#include "compute/GpuOps.hpp"
#include "compute/Matmul.hpp"
#include "compute/Norm.hpp"
#include "compute/QuantTypeRegistry.hpp"
#include "compute/Rope.hpp"
#include "model/GgufTypes.hpp"
#include "runtime/CommandQueue.hpp"
#include "runtime/L0Context.hpp"
#include "runtime/UsmAllocator.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace {

// -----------------------------------------------------------------------
// Shared L0 fixture — one init per binary run.
// -----------------------------------------------------------------------

struct GpuFixture {
    mimirmind::runtime::L0Context    ctx;
    mimirmind::runtime::UsmAllocator usm{ctx};
    mimirmind::runtime::CommandQueue queue{ctx};
    mimirmind::compute::GpuOps       ops{ctx, usm, queue};
    mimirmind::compute::GpuMatmul    gmm{ctx, queue};
};

GpuFixture& fx() {
    static GpuFixture f;
    return f;
}

// RAII wrapper around a USM allocation.
class UsmBuf {
public:
    UsmBuf(std::size_t bytes)
        : _bytes{bytes}, _ptr{fx().usm.allocate(bytes)} {}
    ~UsmBuf() { fx().usm.deallocate(_ptr, _bytes); }

    UsmBuf(const UsmBuf&)            = delete;
    UsmBuf& operator=(const UsmBuf&) = delete;
    UsmBuf(UsmBuf&&)                 = delete;
    UsmBuf& operator=(UsmBuf&&)      = delete;

    template <typename T>
    [[nodiscard]] T* as() noexcept { return static_cast<T*>(_ptr); }

    [[nodiscard]] void* raw() noexcept { return _ptr; }
    [[nodiscard]] std::size_t bytes() const noexcept { return _bytes; }

private:
    std::size_t _bytes;
    void*       _ptr;
};

// Deterministic float generator. Seeded per call so independent tests
// see the same input sequence regardless of order.
std::vector<float> generateFloats(std::size_t n,
                                  std::uint32_t seed,
                                  float lo = -2.0F,
                                  float hi = 2.0F) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> v(n);
    for (auto& x : v) {
        x = dist(rng);
    }
    return v;
}

// Compare a GPU result to a CPU reference element-wise. Reports the
// first / max offender.
void expectArrayNearImpl(const char* file, int line,
                         const char* label,
                         const float* gpu,
                         const float* cpu,
                         std::size_t  n,
                         float        absTol) {
    float maxDiff = 0.0F;
    std::size_t  maxIdx = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const float d = std::fabs(gpu[i] - cpu[i]);
        if (d > maxDiff) {
            maxDiff = d;
            maxIdx  = i;
        }
    }
    if (!(maxDiff <= absTol)) {
        std::ostringstream os;
        os << file << ":" << line << " EXPECT_ARRAY_NEAR(" << label
           << ", n=" << n << ", tol=" << absTol
           << ") failed: maxDiff=" << maxDiff << " at i=" << maxIdx
           << " gpu=" << gpu[maxIdx] << " cpu=" << cpu[maxIdx];
        throw std::runtime_error(os.str());
    }
}

#define EXPECT_ARRAY_NEAR(label, gpu, cpu, n, tol) \
    expectArrayNearImpl(__FILE__, __LINE__, label, (gpu), (cpu), (n), (tol))

} // namespace

// =======================================================================
// RMSNorm — y = x * w / sqrt(mean(x^2) + eps)
// =======================================================================

TEST(rmsnorm_basic) {
    constexpr std::size_t M = 1;
    constexpr std::size_t K = 256;
    constexpr float eps     = 1e-6F;

    const auto x = generateFloats(M * K, 0xA1);
    const auto w = generateFloats(K, 0xA2);

    UsmBuf bufX(M * K * sizeof(float));
    UsmBuf bufW(K * sizeof(float));
    UsmBuf bufY(M * K * sizeof(float));
    std::memcpy(bufX.raw(), x.data(), x.size() * sizeof(float));
    std::memcpy(bufW.raw(), w.data(), w.size() * sizeof(float));

    fx().ops.rmsNormAsync(bufX.as<float>(), M, K,
                          bufW.as<float>(), eps, bufY.as<float>());
    fx().queue.flush();

    std::vector<float> cpu(M * K);
    mimirmind::compute::rmsNorm(x.data(), M, K, w.data(), eps, cpu.data());

    EXPECT_ARRAY_NEAR("rmsnorm_basic", bufY.as<float>(), cpu.data(),
                      M * K, 1e-5F);
}

TEST(rmsnorm_multiRow) {
    constexpr std::size_t M = 4;
    constexpr std::size_t K = 2816;     // Gemma 4 d_model
    constexpr float eps     = 1e-6F;

    const auto x = generateFloats(M * K, 0xB1);
    const auto w = generateFloats(K,     0xB2);

    UsmBuf bufX(M * K * sizeof(float));
    UsmBuf bufW(K * sizeof(float));
    UsmBuf bufY(M * K * sizeof(float));
    std::memcpy(bufX.raw(), x.data(), x.size() * sizeof(float));
    std::memcpy(bufW.raw(), w.data(), w.size() * sizeof(float));

    fx().ops.rmsNormAsync(bufX.as<float>(), M, K,
                          bufW.as<float>(), eps, bufY.as<float>());
    fx().queue.flush();

    std::vector<float> cpu(M * K);
    mimirmind::compute::rmsNorm(x.data(), M, K, w.data(), eps, cpu.data());

    EXPECT_ARRAY_NEAR("rmsnorm_multiRow", bufY.as<float>(), cpu.data(),
                      M * K, 5e-5F);
}

// (1+w)·norm Gemma variant: rewrite weight as (1+w), call standard rmsNorm
// reference, then compare. The GPU kernel does the addition internally.
TEST(rmsnorm_gemma_oneplus) {
    constexpr std::size_t M = 2;
    constexpr std::size_t K = 1024;
    constexpr float eps     = 1e-6F;

    const auto x = generateFloats(M * K, 0xC1);
    const auto w = generateFloats(K,     0xC2);

    UsmBuf bufX(M * K * sizeof(float));
    UsmBuf bufW(K * sizeof(float));
    UsmBuf bufY(M * K * sizeof(float));
    std::memcpy(bufX.raw(), x.data(), x.size() * sizeof(float));
    std::memcpy(bufW.raw(), w.data(), w.size() * sizeof(float));

    fx().ops.rmsNormGemmaAsync(bufX.as<float>(), M, K,
                               bufW.as<float>(), eps, bufY.as<float>());
    fx().queue.flush();

    std::vector<float> wPlusOne(K);
    for (std::size_t k = 0; k < K; ++k) wPlusOne[k] = 1.0F + w[k];
    std::vector<float> cpu(M * K);
    mimirmind::compute::rmsNorm(x.data(), M, K, wPlusOne.data(),
                                eps, cpu.data());

    EXPECT_ARRAY_NEAR("rmsnorm_gemma_oneplus", bufY.as<float>(), cpu.data(),
                      M * K, 5e-5F);
}

// Bare RMSNorm without learned weight: implicit weight = 1 vector.
TEST(rmsnorm_no_weight) {
    constexpr std::size_t M = 2;
    constexpr std::size_t K = 512;
    constexpr float eps     = 1e-6F;

    const auto x = generateFloats(M * K, 0xD1);

    UsmBuf bufX(M * K * sizeof(float));
    UsmBuf bufY(M * K * sizeof(float));
    std::memcpy(bufX.raw(), x.data(), x.size() * sizeof(float));

    fx().ops.rmsNormNoWeightAsync(bufX.as<float>(), M, K, eps,
                                  bufY.as<float>());
    fx().queue.flush();

    std::vector<float> ones(K, 1.0F);
    std::vector<float> cpu(M * K);
    mimirmind::compute::rmsNorm(x.data(), M, K, ones.data(), eps,
                                cpu.data());

    EXPECT_ARRAY_NEAR("rmsnorm_no_weight", bufY.as<float>(), cpu.data(),
                      M * K, 1e-5F);
}

// =======================================================================
// Element-wise: add_bias, add_residual, mul_scalar
// =======================================================================

TEST(commandQueue_unorderedScopeBarrierPublishesWrites) {
    // Two independent silu_mul launches inside an UnorderedScope, then
    // an addResidual that READS one of the scope's outputs. If the
    // pop's barrier didn't land, the addResidual could see stale data
    // and the output would mismatch the CPU reference.
    constexpr std::size_t N = 4096;

    auto a = generateFloats(N, 0xA1);   // gate input for scope op 1
    auto b = generateFloats(N, 0xA2);   // up input for scope op 1
    auto c = generateFloats(N, 0xA3);   // gate input for scope op 2
    auto d = generateFloats(N, 0xA4);   // up input for scope op 2
    auto e = generateFloats(N, 0xA5);   // residual target (read+written)

    UsmBuf bufA(N * sizeof(float));
    UsmBuf bufB(N * sizeof(float));
    UsmBuf bufC(N * sizeof(float));
    UsmBuf bufD(N * sizeof(float));
    UsmBuf bufE(N * sizeof(float));
    std::memcpy(bufA.raw(), a.data(), N * sizeof(float));
    std::memcpy(bufB.raw(), b.data(), N * sizeof(float));
    std::memcpy(bufC.raw(), c.data(), N * sizeof(float));
    std::memcpy(bufD.raw(), d.data(), N * sizeof(float));
    std::memcpy(bufE.raw(), e.data(), N * sizeof(float));

    EXPECT_EQ(fx().queue.unorderedDepth(), std::uint32_t{0});
    {
        mimirmind::runtime::UnorderedScope u{fx().queue};
        EXPECT_EQ(fx().queue.unorderedDepth(), std::uint32_t{1});
        fx().ops.siluMulAsync(bufA.as<float>(), bufB.as<float>(), N);
        fx().ops.siluMulAsync(bufC.as<float>(), bufD.as<float>(), N);
    }
    EXPECT_EQ(fx().queue.unorderedDepth(), std::uint32_t{0});

    // addResidual reads bufC which the second silu_mul wrote. With the
    // scope-pop barrier, bufC's writes are published before this read.
    fx().ops.addResidualAsync(bufE.as<float>(), bufC.as<float>(), N);
    fx().queue.flush();

    std::vector<float> aRef(N), cRef(N), eRef(N);
    for (std::size_t i = 0; i < N; ++i) {
        const float g1 = a[i];
        aRef[i] = (g1 / (1.0F + std::exp(-g1))) * b[i];
        const float g2 = c[i];
        cRef[i] = (g2 / (1.0F + std::exp(-g2))) * d[i];
        eRef[i] = e[i] + cRef[i];
    }

    EXPECT_ARRAY_NEAR("scope_siluA", bufA.as<float>(), aRef.data(),
                      N, 1e-5F);
    EXPECT_ARRAY_NEAR("scope_siluC", bufC.as<float>(), cRef.data(),
                      N, 1e-5F);
    EXPECT_ARRAY_NEAR("scope_residual_reads_scope_output",
                      bufE.as<float>(), eRef.data(), N, 1e-5F);
}

TEST(scaled_add_residual_basic) {
    // Fused dst[i] += scale * src[i]. Replaces a mulScalar(src,scale) +
    // addResidual(dst,src) pair in the Gemma 4 MoE per-expert loop.
    constexpr std::size_t N    = 4096;
    constexpr float       SCAL = 0.375F;

    auto dst = generateFloats(N, 0xB1);
    auto src = generateFloats(N, 0xB2);

    UsmBuf bufDst(N * sizeof(float));
    UsmBuf bufSrc(N * sizeof(float));
    std::memcpy(bufDst.raw(), dst.data(), N * sizeof(float));
    std::memcpy(bufSrc.raw(), src.data(), N * sizeof(float));

    fx().ops.scaledAddResidualAsync(bufDst.as<float>(), bufSrc.as<float>(),
                                    SCAL, N);
    fx().queue.flush();

    std::vector<float> cpu(N);
    for (std::size_t i = 0; i < N; ++i) {
        cpu[i] = dst[i] + SCAL * src[i];
    }

    EXPECT_ARRAY_NEAR("scaled_add_residual", bufDst.as<float>(), cpu.data(),
                      N, 1e-5F);
}

TEST(scaled_add_residual_zeroScale) {
    // scale=0 should leave dst unchanged. Edge case worth pinning.
    constexpr std::size_t N = 1024;
    auto dst = generateFloats(N, 0xB3);
    auto src = generateFloats(N, 0xB4);

    UsmBuf bufDst(N * sizeof(float));
    UsmBuf bufSrc(N * sizeof(float));
    std::memcpy(bufDst.raw(), dst.data(), N * sizeof(float));
    std::memcpy(bufSrc.raw(), src.data(), N * sizeof(float));

    fx().ops.scaledAddResidualAsync(bufDst.as<float>(), bufSrc.as<float>(),
                                    0.0F, N);
    fx().queue.flush();

    EXPECT_ARRAY_NEAR("scaled_add_residual_zeroScale",
                      bufDst.as<float>(), dst.data(), N, 0.0F);
}

TEST(add_residual_basic) {
    constexpr std::size_t N = 8192;

    auto y = generateFloats(N, 0x11);
    auto x = generateFloats(N, 0x12);

    UsmBuf bufY(N * sizeof(float));
    UsmBuf bufX(N * sizeof(float));
    std::memcpy(bufY.raw(), y.data(), y.size() * sizeof(float));
    std::memcpy(bufX.raw(), x.data(), x.size() * sizeof(float));

    fx().ops.addResidualAsync(bufY.as<float>(), bufX.as<float>(), N);
    fx().queue.flush();

    std::vector<float> cpu(N);
    for (std::size_t i = 0; i < N; ++i) cpu[i] = y[i] + x[i];

    EXPECT_ARRAY_NEAR("add_residual", bufY.as<float>(), cpu.data(),
                      N, 0.0F);
}

TEST(add_bias_basic) {
    constexpr std::size_t M = 4;
    constexpr std::size_t K = 512;

    auto y    = generateFloats(M * K, 0x21);
    auto bias = generateFloats(K,     0x22);

    UsmBuf bufY(M * K * sizeof(float));
    UsmBuf bufB(K * sizeof(float));
    std::memcpy(bufY.raw(), y.data(),    y.size() * sizeof(float));
    std::memcpy(bufB.raw(), bias.data(), bias.size() * sizeof(float));

    fx().ops.addBiasAsync(bufY.as<float>(), M, K, bufB.as<float>());
    fx().queue.flush();

    std::vector<float> cpu(M * K);
    for (std::size_t m = 0; m < M; ++m) {
        for (std::size_t k = 0; k < K; ++k) {
            cpu[m * K + k] = y[m * K + k] + bias[k];
        }
    }

    EXPECT_ARRAY_NEAR("add_bias", bufY.as<float>(), cpu.data(),
                      M * K, 0.0F);
}

TEST(mul_scalar_basic) {
    constexpr std::size_t N = 4096;
    constexpr float       s = 0.0823F;   // gemma4 layer_output_scale-ish

    auto y = generateFloats(N, 0x31);

    UsmBuf bufY(N * sizeof(float));
    std::memcpy(bufY.raw(), y.data(), y.size() * sizeof(float));

    fx().ops.mulScalarAsync(bufY.as<float>(), s, N);
    fx().queue.flush();

    std::vector<float> cpu(N);
    for (std::size_t i = 0; i < N; ++i) cpu[i] = y[i] * s;

    EXPECT_ARRAY_NEAR("mul_scalar", bufY.as<float>(), cpu.data(),
                      N, 0.0F);
}

// =======================================================================
// Activations: silu_mul + gelu_mul
// =======================================================================

TEST(silu_mul_basic) {
    constexpr std::size_t N = 4096;

    auto gate = generateFloats(N, 0x41);
    auto up   = generateFloats(N, 0x42);

    UsmBuf bufG(N * sizeof(float));
    UsmBuf bufU(N * sizeof(float));
    std::memcpy(bufG.raw(), gate.data(), gate.size() * sizeof(float));
    std::memcpy(bufU.raw(), up.data(),   up.size() * sizeof(float));

    fx().ops.siluMulAsync(bufG.as<float>(), bufU.as<float>(), N);
    fx().queue.flush();

    // CPU reference: silu(gate) then mul by up, in place on gate.
    std::vector<float> cpu = gate;
    mimirmind::compute::siluInPlace(cpu.data(), N);
    mimirmind::compute::mulInPlace(cpu.data(), up.data(), N);

    EXPECT_ARRAY_NEAR("silu_mul", bufG.as<float>(), cpu.data(),
                      N, 1e-5F);
}

// GELU-tanh approximation per llama.cpp's LLM_FFN_GELU:
//   y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
static float geluTanh(float x) {
    constexpr float kSqrt2OverPi = 0.7978845608028654F;
    constexpr float kCoeff       = 0.044715F;
    const float x3 = x * x * x;
    return 0.5F * x * (1.0F + std::tanh(kSqrt2OverPi * (x + kCoeff * x3)));
}

TEST(gelu_mul_basic) {
    constexpr std::size_t N = 4096;

    auto gate = generateFloats(N, 0x51);
    auto up   = generateFloats(N, 0x52);

    UsmBuf bufG(N * sizeof(float));
    UsmBuf bufU(N * sizeof(float));
    std::memcpy(bufG.raw(), gate.data(), gate.size() * sizeof(float));
    std::memcpy(bufU.raw(), up.data(),   up.size() * sizeof(float));

    fx().ops.geluMulAsync(bufG.as<float>(), bufU.as<float>(), N);
    fx().queue.flush();

    std::vector<float> cpu(N);
    for (std::size_t i = 0; i < N; ++i) cpu[i] = geluTanh(gate[i]) * up[i];

    EXPECT_ARRAY_NEAR("gelu_mul", bufG.as<float>(), cpu.data(),
                      N, 1e-5F);
}

// =======================================================================
// RoPE (in-place rotary positional embedding)
// =======================================================================

TEST(rope_inplace_basic) {
    constexpr std::size_t seqLen   = 4;
    constexpr std::size_t numHeads = 2;
    constexpr std::size_t headDim  = 64;
    constexpr std::size_t startPos = 0;
    constexpr float       base     = 10000.0F;

    const std::size_t n = seqLen * numHeads * headDim;
    auto x = generateFloats(n, 0x61);

    UsmBuf bufX(n * sizeof(float));
    std::memcpy(bufX.raw(), x.data(), n * sizeof(float));

    fx().ops.ropeInPlaceAsync(bufX.as<float>(), seqLen, numHeads, headDim,
                              startPos, base);
    fx().queue.flush();

    std::vector<float> cpu = x;
    mimirmind::compute::applyRopeInPlace(cpu.data(), seqLen, numHeads,
                                         headDim, startPos, base);

    EXPECT_ARRAY_NEAR("rope_inplace", bufX.as<float>(), cpu.data(),
                      n, 1e-5F);
}

TEST(rope_inplace_decodeOffset) {
    // startPos > 0 mimics decode (single-token, cache-length offset).
    constexpr std::size_t seqLen   = 1;
    constexpr std::size_t numHeads = 4;
    constexpr std::size_t headDim  = 128;
    constexpr std::size_t startPos = 100;
    constexpr float       base     = 1e6F;   // gemma4 full-attn rope base

    const std::size_t n = seqLen * numHeads * headDim;
    auto x = generateFloats(n, 0x62);

    UsmBuf bufX(n * sizeof(float));
    std::memcpy(bufX.raw(), x.data(), n * sizeof(float));

    fx().ops.ropeInPlaceAsync(bufX.as<float>(), seqLen, numHeads, headDim,
                              startPos, base);
    fx().queue.flush();

    std::vector<float> cpu = x;
    mimirmind::compute::applyRopeInPlace(cpu.data(), seqLen, numHeads,
                                         headDim, startPos, base);

    EXPECT_ARRAY_NEAR("rope_decode_offset", bufX.as<float>(), cpu.data(),
                      n, 1e-4F);
}

// =======================================================================
// Multi-head causal attention (GPU vs CPU)
// =======================================================================

namespace {

// Run GPU attention into `outGpu`, CPU attention into `outCpu`, then
// compare. q, k, v come from generateFloats. All buffers are USM so the
// kernel can read them directly.
void runAttentionParity(const char* label,
                        std::size_t T_q,
                        std::size_t T_k,
                        std::size_t nHeads,
                        std::size_t nKvHeads,
                        std::size_t headDim,
                        std::size_t positionOffset,
                        float       scale,
                        std::uint32_t seed,
                        float       tol)
{
    const std::size_t qN  = T_q * nHeads   * headDim;
    const std::size_t kvN = T_k * nKvHeads * headDim;
    const std::size_t oN  = T_q * nHeads   * headDim;

    auto qHost = generateFloats(qN,  seed + 0);
    auto kHost = generateFloats(kvN, seed + 1);
    auto vHost = generateFloats(kvN, seed + 2);

    UsmBuf qBuf(qN  * sizeof(float));
    UsmBuf kBuf(kvN * sizeof(float));
    UsmBuf vBuf(kvN * sizeof(float));
    UsmBuf oBuf(oN  * sizeof(float));
    std::memcpy(qBuf.raw(), qHost.data(), qN  * sizeof(float));
    std::memcpy(kBuf.raw(), kHost.data(), kvN * sizeof(float));
    std::memcpy(vBuf.raw(), vHost.data(), kvN * sizeof(float));
    std::memset(oBuf.raw(), 0,            oN  * sizeof(float));

    fx().ops.attentionAsync(qBuf.as<float>(), kBuf.as<float>(),
                            vBuf.as<float>(),
                            T_q, T_k, nHeads, nKvHeads, headDim,
                            positionOffset, scale,
                            oBuf.as<float>());
    fx().queue.flush();

    // CPU reference. compute::multiHeadAttention has the 1/sqrt(headDim)
    // baked in; the GPU kernel takes `scale` as a parameter. Pre-scale Q
    // by (scale * sqrt(headDim)) before calling the CPU reference so the
    // net Q·K scaling matches what the GPU did.
    const float cpuExpectedScale = 1.0F / std::sqrt(
        static_cast<float>(headDim));
    const float qPreScale        = scale / cpuExpectedScale;
    std::vector<float> qScaled = qHost;
    for (auto& x : qScaled) x *= qPreScale;

    std::vector<float> scratch(T_k);
    std::vector<float> outCpu(oN, 0.0F);
    mimirmind::compute::multiHeadAttention(
        qScaled.data(), kHost.data(), vHost.data(),
        T_q, T_k, nHeads, nKvHeads, headDim, positionOffset,
        scratch.data(), outCpu.data());

    EXPECT_ARRAY_NEAR(label, oBuf.as<float>(), outCpu.data(), oN, tol);
}

} // namespace

TEST(attention_prefill_mha) {
    // Plain multi-head (no GQA): nHeads == nKvHeads. Covers the prefill
    // path with positionOffset=0 and causal mask sweeping kMax = pq+1.
    runAttentionParity("attn_prefill_mha",
                       /*T_q=*/8, /*T_k=*/8,
                       /*nHeads=*/4, /*nKvHeads=*/4, /*headDim=*/32,
                       /*positionOffset=*/0,
                       /*scale=*/1.0F / std::sqrt(32.0F),
                       /*seed=*/0x71, /*tol=*/5e-5F);
}

TEST(attention_decode_mha) {
    // Single new query attending to a 7-position prefix already in KV
    // cache. T_q=1, positionOffset = cache_length, kMax = T_k.
    runAttentionParity("attn_decode_mha",
                       /*T_q=*/1, /*T_k=*/8,
                       /*nHeads=*/4, /*nKvHeads=*/4, /*headDim=*/32,
                       /*positionOffset=*/7,
                       /*scale=*/1.0F / std::sqrt(32.0F),
                       /*seed=*/0x72, /*tol=*/5e-5F);
}

TEST(attention_prefill_gqa) {
    // GQA: 4 query heads share 2 KV heads (2:1). Exercises the
    // hkv = (hq * nKvHeads) / nHeads index rewrite.
    runAttentionParity("attn_prefill_gqa",
                       /*T_q=*/4, /*T_k=*/4,
                       /*nHeads=*/4, /*nKvHeads=*/2, /*headDim=*/64,
                       /*positionOffset=*/0,
                       /*scale=*/1.0F / std::sqrt(64.0F),
                       /*seed=*/0x73, /*tol=*/5e-5F);
}

TEST(attention_decode_gqa_long_context) {
    // Decode with a longer cache (256 entries) and Gemma-like geometry
    // (nHeads=8, nKvHeads=2). Verifies the score row + softmax behaves
    // for non-trivial T_k.
    runAttentionParity("attn_decode_gqa_long",
                       /*T_q=*/1, /*T_k=*/256,
                       /*nHeads=*/8, /*nKvHeads=*/2, /*headDim=*/64,
                       /*positionOffset=*/255,
                       /*scale=*/1.0F / std::sqrt(64.0F),
                       /*seed=*/0x74, /*tol=*/1e-4F);
}

TEST(attention_decode_gemma_scale_one) {
    // Gemma 4 mode: backend passes scale = 1.0 (the model's
    // f_attention_scale). Same numerical behaviour as if Q had been
    // pre-scaled by sqrt(headDim) before a 1/sqrt(headDim) attention.
    runAttentionParity("attn_decode_gemma_scale1",
                       /*T_q=*/1, /*T_k=*/16,
                       /*nHeads=*/4, /*nKvHeads=*/2, /*headDim=*/64,
                       /*positionOffset=*/15,
                       /*scale=*/1.0F,
                       /*seed=*/0x75, /*tol=*/1e-4F);
}

// -- M5f.3.2 Flash-only stress cases ------------------------------------
//
// Decode (T_q == 1) routes through the K-tiled FlashAttention path. These
// cases push past the single-tile boundary (K_TILE_SIZE = 256) so the
// online-softmax merge across multiple tiles actually runs and gets
// compared to the CPU reference.

TEST(attention_decode_flash_twoTiles) {
    // T_k = 512 → 2 tiles. Position right at the end so kMax spans both.
    runAttentionParity("attn_flash_2tiles",
                       /*T_q=*/1, /*T_k=*/512,
                       /*nHeads=*/8, /*nKvHeads=*/2, /*headDim=*/64,
                       /*positionOffset=*/511,
                       /*scale=*/1.0F / std::sqrt(64.0F),
                       /*seed=*/0x81, /*tol=*/2e-4F);
}

TEST(attention_decode_flash_tileBoundary) {
    // positionOffset = 255 → kMax = 256, exactly fills tile 0; tile 1
    // is entirely past kMax and must emit a neutral partial that the
    // merge correctly ignores.
    runAttentionParity("attn_flash_boundary",
                       /*T_q=*/1, /*T_k=*/512,
                       /*nHeads=*/4, /*nKvHeads=*/2, /*headDim=*/64,
                       /*positionOffset=*/255,
                       /*scale=*/1.0F / std::sqrt(64.0F),
                       /*seed=*/0x82, /*tol=*/1e-4F);
}

TEST(attention_decode_flash_eightTiles) {
    // T_k = 2048 → 8 tiles. Larger headDim (128) covers the Qwen-like
    // shape under flash.
    runAttentionParity("attn_flash_8tiles",
                       /*T_q=*/1, /*T_k=*/2048,
                       /*nHeads=*/16, /*nKvHeads=*/4, /*headDim=*/128,
                       /*positionOffset=*/2047,
                       /*scale=*/1.0F / std::sqrt(128.0F),
                       /*seed=*/0x83, /*tol=*/5e-4F);
}

TEST(attention_decode_flash_maxKTiles) {
    // T_k = 16384 → 64 tiles, the compile-time MAX_K_TILES bound after
    // M9.8a. Exercises a Gemma-4-style 8q->2kv GQA group with
    // headDim=128. The previous max at T_k=8192/32-tiles moved down to
    // "half the max" — still covered indirectly by the decoder path in
    // production but no longer the sole boundary case.
    runAttentionParity("attn_flash_maxTiles",
                       /*T_q=*/1, /*T_k=*/16384,
                       /*nHeads=*/8, /*nKvHeads=*/2, /*headDim=*/128,
                       /*positionOffset=*/16383,
                       /*scale=*/1.0F / std::sqrt(128.0F),
                       /*seed=*/0x84, /*tol=*/2e-3F);
}

TEST(attention_decode_flash_gemma4FullAttn) {
    // Real-world geometry from Gemma 4 26B full-attention layers:
    // nHeads=16, nKvHeads=8 (GQA 2:1), headDim=512. Caught a missing
    // 512 in kFlashMaxHeadDim on first deploy 2026-06-30 — never let
    // it regress again.
    runAttentionParity("attn_flash_gemma4_full",
                       /*T_q=*/1, /*T_k=*/1024,
                       /*nHeads=*/16, /*nKvHeads=*/8, /*headDim=*/512,
                       /*positionOffset=*/1023,
                       /*scale=*/1.0F,
                       /*seed=*/0x86, /*tol=*/1e-3F);
}

TEST(attention_decode_flash_partialKTileTail) {
    // kMax = 1300 → tile 0..4 fully populated, tile 5 holds 1300-1280
    // = 20 keys, tiles 6+ are neutral. Mixed populated + partial +
    // empty exercises every branch of the partial kernel.
    runAttentionParity("attn_flash_partialTail",
                       /*T_q=*/1, /*T_k=*/2048,
                       /*nHeads=*/4, /*nKvHeads=*/2, /*headDim=*/64,
                       /*positionOffset=*/1299,
                       /*scale=*/1.0F / std::sqrt(64.0F),
                       /*seed=*/0x85, /*tol=*/5e-4F);
}

// =======================================================================
// Q4_K matmul (GPU vs CPU)
// =======================================================================

// Constructs a single Q4_K row of K elements: zero ql/qh, but set scales[0]
// = 1, d=1.0, dmin=0, qs[0..31]=0x0A (low nibble 10). Used as one
// weight row.  Verifies the GPU matvec against compute::matmul which
// internally dequants the same bytes and does a double-acc dot product.
TEST(matmul_q4k_singleRow) {
    constexpr std::size_t K = 256;          // one super-block
    constexpr std::size_t N = 1;

    // Build a single Q4_K row (144 bytes).
    std::array<std::uint8_t, 144> wRow{};
    constexpr std::uint16_t kHalfOne = 0x3C00U;
    std::memcpy(wRow.data(),     &kHalfOne, sizeof(std::uint16_t));   // d=1
    const std::uint16_t zero = 0;
    std::memcpy(wRow.data() + 2, &zero,     sizeof(std::uint16_t));   // dmin=0
    wRow[4] = 0x01U;                                                   // scales[0]
    for (std::size_t i = 16; i < 16 + 32; ++i) wRow[i] = 0x0AU;        // qs low nibble = 10

    const auto x = generateFloats(K, 0x71);

    UsmBuf bufW(wRow.size());
    UsmBuf bufX(K * sizeof(float));
    UsmBuf bufY(N * sizeof(float));
    UsmBuf bufScratch(K * sizeof(float));    // unused on GPU path
    std::memcpy(bufW.raw(), wRow.data(), wRow.size());
    std::memcpy(bufX.raw(), x.data(),    x.size() * sizeof(float));

    fx().gmm.matmul(mimirmind::model::GgmlType::Q4_K,
                    bufW.raw(), N, K,
                    bufX.as<float>(), 1,
                    bufY.as<float>(),
                    bufScratch.as<float>());

    // CPU reference: same input bytes, compute::matmul double-acc.
    std::vector<float> cpuW(K);
    std::vector<float> cpuY(N);
    std::vector<float> scratchCpu(K);
    mimirmind::compute::matmul(mimirmind::model::GgmlType::Q4_K,
                               wRow.data(), N, K,
                               x.data(), 1,
                               cpuY.data(),
                               scratchCpu.data());

    EXPECT_ARRAY_NEAR("matmul_q4k_singleRow", bufY.as<float>(), cpuY.data(),
                      N, 1e-3F);
}

// =======================================================================
// Q8_0 matmul (GPU vs CPU) — new in M8.G
// =======================================================================

// Q8_0 row: 32 elements per block, 34 bytes. d=1.0, qs alternating
// +5/-5 → dequant values alternate +5/-5. With X = all 1.0 → Y = 0.
// Sums every term but they cancel, exercising precision near zero.
TEST(matmul_q8_0_singleRow_alternating) {
    constexpr std::size_t K = 32;
    constexpr std::size_t N = 1;

    constexpr std::uint16_t kHalfOne = 0x3C00U;
    std::array<std::uint8_t, 34> wRow{};
    std::memcpy(wRow.data(), &kHalfOne, sizeof(std::uint16_t));
    for (std::size_t i = 0; i < 32; ++i) {
        wRow[2 + i] = (i % 2 == 0) ? static_cast<std::uint8_t>(5)
                                   : static_cast<std::uint8_t>(static_cast<std::int8_t>(-5));
    }

    const std::vector<float> x(K, 1.0F);

    UsmBuf bufW(wRow.size());
    UsmBuf bufX(K * sizeof(float));
    UsmBuf bufY(N * sizeof(float));
    UsmBuf bufScratch(K * sizeof(float));
    std::memcpy(bufW.raw(), wRow.data(), wRow.size());
    std::memcpy(bufX.raw(), x.data(),    x.size() * sizeof(float));

    fx().gmm.matmul(mimirmind::model::GgmlType::Q8_0,
                    bufW.raw(), N, K,
                    bufX.as<float>(), 1,
                    bufY.as<float>(),
                    bufScratch.as<float>());

    EXPECT_NEAR(bufY.as<float>()[0], 0.0F, 1e-5F);
}

// Larger matmul: K=2816 (Gemma 4 d_model), N=64. Verify GPU output
// matches CPU compute::matmul element-wise within tolerance.
TEST(matmul_q8_0_64rows_K2816) {
    constexpr std::size_t K = 2816;
    constexpr std::size_t N = 64;
    constexpr std::size_t blocksPerRow = K / 32;
    constexpr std::size_t bytesPerRow  = blocksPerRow * 34;

    constexpr std::uint16_t kHalfHalf = 0x3800U;  // 0.5

    std::vector<std::uint8_t> wAll(N * bytesPerRow, 0);
    for (std::size_t n = 0; n < N; ++n) {
        std::uint8_t* row = wAll.data() + n * bytesPerRow;
        for (std::size_t b = 0; b < blocksPerRow; ++b) {
            std::uint8_t* block = row + b * 34;
            std::memcpy(block, &kHalfHalf, sizeof(std::uint16_t));
            for (std::size_t l = 0; l < 32; ++l) {
                block[2 + l] = static_cast<std::uint8_t>(
                    static_cast<std::int8_t>((n + b + l) % 19 - 9));
            }
        }
    }

    const auto x = generateFloats(K, 0x90);

    UsmBuf bufW(wAll.size());
    UsmBuf bufX(K * sizeof(float));
    UsmBuf bufY(N * sizeof(float));
    UsmBuf bufScratch(K * sizeof(float));
    std::memcpy(bufW.raw(), wAll.data(), wAll.size());
    std::memcpy(bufX.raw(), x.data(),    x.size() * sizeof(float));

    fx().gmm.matmul(mimirmind::model::GgmlType::Q8_0,
                    bufW.raw(), N, K,
                    bufX.as<float>(), 1,
                    bufY.as<float>(),
                    bufScratch.as<float>());

    std::vector<float> cpuY(N);
    std::vector<float> scratchCpu(K);
    mimirmind::compute::matmul(mimirmind::model::GgmlType::Q8_0,
                               wAll.data(), N, K,
                               x.data(), 1,
                               cpuY.data(),
                               scratchCpu.data());

    EXPECT_ARRAY_NEAR("matmul_q8_0_64rows", bufY.as<float>(), cpuY.data(),
                      N, 5e-3F);
}

// =======================================================================
// Q6_K matmul (GPU vs CPU) — exercises the Kahan-accumulated kernel
// =======================================================================

TEST(matmul_q6k_singleRow) {
    constexpr std::size_t K = 256;          // one super-block
    constexpr std::size_t N = 1;

    // Build a single Q6_K row (210 bytes) with all scales = 1, d = 1,
    // ql all 0xAA, qh all 0x55 — produces a mix of quants per position.
    std::array<std::uint8_t, 210> wRow{};
    for (std::size_t i = 0; i < 128; ++i) wRow[i] = 0xAAU;       // ql
    for (std::size_t i = 128; i < 192; ++i) wRow[i] = 0x55U;     // qh
    for (std::size_t i = 192; i < 208; ++i) wRow[i] = 1;         // sc = 1
    constexpr std::uint16_t kHalfOne = 0x3C00U;
    std::memcpy(wRow.data() + 208, &kHalfOne, sizeof(std::uint16_t));

    const auto x = generateFloats(K, 0x81);

    UsmBuf bufW(wRow.size());
    UsmBuf bufX(K * sizeof(float));
    UsmBuf bufY(N * sizeof(float));
    UsmBuf bufScratch(K * sizeof(float));
    std::memcpy(bufW.raw(), wRow.data(), wRow.size());
    std::memcpy(bufX.raw(), x.data(),    x.size() * sizeof(float));

    fx().gmm.matmul(mimirmind::model::GgmlType::Q6_K,
                    bufW.raw(), N, K,
                    bufX.as<float>(), 1,
                    bufY.as<float>(),
                    bufScratch.as<float>());

    std::vector<float> cpuY(N);
    std::vector<float> scratchCpu(K);
    mimirmind::compute::matmul(mimirmind::model::GgmlType::Q6_K,
                               wRow.data(), N, K,
                               x.data(), 1,
                               cpuY.data(),
                               scratchCpu.data());

    EXPECT_ARRAY_NEAR("matmul_q6k_singleRow", bufY.as<float>(), cpuY.data(),
                      N, 1e-3F);
}

// Larger matmul stresses subgroup reduce + multiple workgroups.
// K = 2816 (Gemma 4 d_model), N = 64 output rows. Each row is the SAME
// Q6_K bytes so we get a well-defined N-way replicated output to compare.
TEST(matmul_q6k_64rows) {
    constexpr std::size_t K = 2816;
    constexpr std::size_t N = 64;
    constexpr std::size_t superBlocksPerRow = K / 256;
    constexpr std::size_t bytesPerRow       = superBlocksPerRow * 210;

    // Build one canonical row and replicate.
    std::vector<std::uint8_t> oneRow(bytesPerRow, 0);
    for (std::size_t sb = 0; sb < superBlocksPerRow; ++sb) {
        auto* block = oneRow.data() + sb * 210;
        for (std::size_t i = 0; i < 128; ++i) block[i] = static_cast<std::uint8_t>(sb * 13 + i);
        for (std::size_t i = 128; i < 192; ++i) block[i] = static_cast<std::uint8_t>(sb * 7 + i);
        for (std::size_t i = 192; i < 208; ++i) {
            block[i] = static_cast<std::uint8_t>((sb + i) % 8);
        }
        const std::uint16_t halfOne = 0x3C00U;
        std::memcpy(block + 208, &halfOne, sizeof(halfOne));
    }
    std::vector<std::uint8_t> wAll(N * bytesPerRow);
    for (std::size_t n = 0; n < N; ++n) {
        std::memcpy(wAll.data() + n * bytesPerRow, oneRow.data(), bytesPerRow);
    }

    const auto x = generateFloats(K, 0x82);

    UsmBuf bufW(wAll.size());
    UsmBuf bufX(K * sizeof(float));
    UsmBuf bufY(N * sizeof(float));
    UsmBuf bufScratch(K * sizeof(float));
    std::memcpy(bufW.raw(), wAll.data(), wAll.size());
    std::memcpy(bufX.raw(), x.data(),    x.size() * sizeof(float));

    fx().gmm.matmul(mimirmind::model::GgmlType::Q6_K,
                    bufW.raw(), N, K,
                    bufX.as<float>(), 1,
                    bufY.as<float>(),
                    bufScratch.as<float>());

    std::vector<float> cpuY(N);
    std::vector<float> scratchCpu(K);
    mimirmind::compute::matmul(mimirmind::model::GgmlType::Q6_K,
                               wAll.data(), N, K,
                               x.data(), 1,
                               cpuY.data(),
                               scratchCpu.data());

    EXPECT_ARRAY_NEAR("matmul_q6k_64rows", bufY.as<float>(), cpuY.data(),
                      N, 1e-2F);

    // All rows should be identical (replicated weights, same x).
    const float* gpu = bufY.as<float>();
    for (std::size_t n = 1; n < N; ++n) {
        if (std::fabs(gpu[n] - gpu[0]) > 1e-3F) {
            std::ostringstream os;
            os << "matmul_q6k_64rows: row " << n << " (" << gpu[n]
               << ") != row 0 (" << gpu[0] << ")";
            throw std::runtime_error(os.str());
        }
    }
}

int main() {
    return mm::test::run();
}