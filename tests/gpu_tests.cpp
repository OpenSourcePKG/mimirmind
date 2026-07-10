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
#include "compute/quant/Q8_0.hpp"
#include "model/GgufTypes.hpp"
#include "runtime/CommandQueue.hpp"
#include "runtime/Config.hpp"
#include "runtime/L0Context.hpp"
#include "runtime/UsmAllocator.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
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
    mimirmind::compute::GpuMatmul    gmm{ctx, ops, usm, queue};

    GpuFixture() {
        // Force the GEMM dispatch decision so the M>1 parity tests
        // actually exercise the GEMM kernels regardless of what the
        // dev-box iGPU would autotune-pick. M=1 regression tests still
        // hit the matvec path because dispatch fast-paths M==1 to vec.
        mimirmind::runtime::FeatureSettings features{};
        features.gemm = mimirmind::runtime::TriState::Force;
        gmm.autotune(usm, /*hiddenDim=*/2816, features);
    }
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
// x_quant_i8 — per-row symmetric int8 quantisation (M8.H.0 input path
// for the DP4A Q8_0 matmul).
// =======================================================================

namespace {

// CPU reference for a single row: scale = max(|x|) / 127, quantise
// with round-to-nearest ties-away-from-zero, clamp to [-127, 127].
// Matches kernels/x_quant_i8.cl.
void cpuXQuantI8Row(const float* x, std::int8_t* q, float& scale,
                    std::size_t K) {
    float amax = 0.0F;
    for (std::size_t k = 0; k < K; ++k) {
        amax = std::max(amax, std::fabs(x[k]));
    }
    scale = amax / 127.0F;
    const float invS = (amax > 0.0F) ? (127.0F / amax) : 0.0F;
    for (std::size_t k = 0; k < K; ++k) {
        float v = std::round(x[k] * invS);
        v = std::max(-127.0F, std::min(127.0F, v));
        q[k] = static_cast<std::int8_t>(static_cast<int>(v));
    }
}

} // namespace

TEST(x_quant_i8_basic) {
    // Single row at Gemma 4 26B-A4B hidden dim.
    constexpr std::size_t M = 1;
    constexpr std::size_t K = 2816;

    const auto x = generateFloats(M * K, 0xB1);

    UsmBuf bufX(M * K * sizeof(float));
    UsmBuf bufQ(M * K * sizeof(std::int8_t));
    UsmBuf bufS(M     * sizeof(float));
    std::memcpy(bufX.raw(), x.data(), x.size() * sizeof(float));

    fx().ops.xQuantI8Async(bufX.as<float>(), bufQ.as<std::int8_t>(),
                           bufS.as<float>(), M, K);
    fx().queue.flush();

    std::vector<std::int8_t> cpuQ(M * K);
    float                    cpuScale = 0.0F;
    cpuXQuantI8Row(x.data(), cpuQ.data(), cpuScale, K);

    // Scale is a single reduction — must match to the last ULP.
    const float gotScale = bufS.as<float>()[0];
    if (!(std::fabs(gotScale - cpuScale) <= 1e-6F * cpuScale)) {
        std::ostringstream os;
        os << "x_quant_i8_basic: scale mismatch — gpu=" << gotScale
           << " cpu=" << cpuScale;
        throw std::runtime_error(os.str());
    }
    // Quants match exactly (deterministic round + clamp on both sides).
    const std::int8_t* gotQ = bufQ.as<std::int8_t>();
    for (std::size_t k = 0; k < K; ++k) {
        if (gotQ[k] != cpuQ[k]) {
            std::ostringstream os;
            os << "x_quant_i8_basic: q mismatch at k=" << k
               << " gpu=" << static_cast<int>(gotQ[k])
               << " cpu=" << static_cast<int>(cpuQ[k])
               << " x=" << x[k] << " scale=" << cpuScale;
            throw std::runtime_error(os.str());
        }
    }

    // Dequant round-trip within 0.5 * scale.
    const float tol = 0.5F * cpuScale + 1e-6F;
    for (std::size_t k = 0; k < K; ++k) {
        const float deq = static_cast<float>(gotQ[k]) * gotScale;
        const float d   = std::fabs(deq - x[k]);
        if (!(d <= tol)) {
            std::ostringstream os;
            os << "x_quant_i8_basic: dequant error at k=" << k
               << " x=" << x[k] << " deq=" << deq
               << " diff=" << d << " tol=" << tol;
            throw std::runtime_error(os.str());
        }
    }
}

TEST(x_quant_i8_multiRow) {
    // Multiple rows with deliberately different dynamic ranges — each
    // row must get its own independent scale.
    constexpr std::size_t M = 4;
    constexpr std::size_t K = 1024;

    auto x = generateFloats(M * K, 0xB2);
    // Rescale each row so scales are visibly different.
    const float rowMul[M] = {0.1F, 1.0F, 5.0F, 20.0F};
    for (std::size_t m = 0; m < M; ++m) {
        for (std::size_t k = 0; k < K; ++k) {
            x[m * K + k] *= rowMul[m];
        }
    }

    UsmBuf bufX(M * K * sizeof(float));
    UsmBuf bufQ(M * K * sizeof(std::int8_t));
    UsmBuf bufS(M     * sizeof(float));
    std::memcpy(bufX.raw(), x.data(), x.size() * sizeof(float));

    fx().ops.xQuantI8Async(bufX.as<float>(), bufQ.as<std::int8_t>(),
                           bufS.as<float>(), M, K);
    fx().queue.flush();

    const std::int8_t* gotQ = bufQ.as<std::int8_t>();
    const float*       gotS = bufS.as<float>();
    for (std::size_t m = 0; m < M; ++m) {
        std::vector<std::int8_t> cpuQ(K);
        float                    cpuScale = 0.0F;
        cpuXQuantI8Row(x.data() + m * K, cpuQ.data(), cpuScale, K);

        if (!(std::fabs(gotS[m] - cpuScale) <= 1e-6F * cpuScale)) {
            std::ostringstream os;
            os << "x_quant_i8_multiRow: scale[" << m
               << "] gpu=" << gotS[m] << " cpu=" << cpuScale;
            throw std::runtime_error(os.str());
        }
        for (std::size_t k = 0; k < K; ++k) {
            if (gotQ[m * K + k] != cpuQ[k]) {
                std::ostringstream os;
                os << "x_quant_i8_multiRow: q mismatch row=" << m
                   << " k=" << k
                   << " gpu=" << static_cast<int>(gotQ[m * K + k])
                   << " cpu=" << static_cast<int>(cpuQ[k]);
                throw std::runtime_error(os.str());
            }
        }
    }
}

TEST(x_quant_i8_zeroRow) {
    // All-zero row must produce scale=0 and all-zero quants — the DP4A
    // matmul relies on this to produce an exactly-zero output row
    // rather than NaN from 0/0 = 0/0.
    constexpr std::size_t M = 1;
    constexpr std::size_t K = 128;

    std::vector<float> x(M * K, 0.0F);

    UsmBuf bufX(M * K * sizeof(float));
    UsmBuf bufQ(M * K * sizeof(std::int8_t));
    UsmBuf bufS(M     * sizeof(float));
    std::memcpy(bufX.raw(), x.data(), x.size() * sizeof(float));

    fx().ops.xQuantI8Async(bufX.as<float>(), bufQ.as<std::int8_t>(),
                           bufS.as<float>(), M, K);
    fx().queue.flush();

    if (bufS.as<float>()[0] != 0.0F) {
        std::ostringstream os;
        os << "x_quant_i8_zeroRow: expected scale=0, got="
           << bufS.as<float>()[0];
        throw std::runtime_error(os.str());
    }
    const std::int8_t* gotQ = bufQ.as<std::int8_t>();
    for (std::size_t k = 0; k < K; ++k) {
        if (gotQ[k] != 0) {
            std::ostringstream os;
            os << "x_quant_i8_zeroRow: q[" << k << "]="
               << static_cast<int>(gotQ[k]) << " (expected 0)";
            throw std::runtime_error(os.str());
        }
    }
}

// =======================================================================
// M10.2.0 Commit 3 — kv_quant_commit_q8_0 (fp32 K/V workspace → Q8_0 slot)
// =======================================================================
//
// Byte-exact parity against compute::quant::Q8_0::quantizeRow. The GPU
// kernel writes the same 34-byte block layout (fp16 scale + 32 int8
// quants) as the CPU reference, so `std::memcmp` gates the match.
// Zero-row inputs must survive as (scale=0, all quants=0) so the
// dequant path in the attention kernel round-trips to zero rather
// than reading a NaN.

TEST(kv_quant_commit_q8_0_basic) {
    // T=4 tokens × kvDim=128 elements (4 Q8_0 blocks per row).
    constexpr std::size_t T           = 4;
    constexpr std::size_t kvDim       = 128;
    constexpr std::size_t nBlocksRow  = kvDim / 32;
    constexpr std::size_t rowBytes    = nBlocksRow * 34;
    constexpr std::size_t writeOffset = 0;

    // Rows 0-2 get scaled random data with visibly different ranges;
    // row 3 is all-zero so the (scale=0, quants=0) branch is exercised.
    auto x = generateFloats(T * kvDim, 0xC1);
    const float rowMul[T] = {0.1F, 1.0F, 5.0F, 0.0F};
    for (std::size_t t = 0; t < T; ++t) {
        for (std::size_t k = 0; k < kvDim; ++k) {
            x[t * kvDim + k] *= rowMul[t];
        }
    }

    UsmBuf bufX (T * kvDim   * sizeof(float));
    UsmBuf bufKv(T * rowBytes);
    std::memcpy(bufX.raw(), x.data(), x.size() * sizeof(float));
    std::memset(bufKv.raw(), 0xAA, T * rowBytes);  // sentinel

    fx().ops.kvQuantCommitQ8Async(bufX.as<float>(), bufKv.raw(),
                                  T, kvDim, writeOffset);
    fx().queue.flush();

    // CPU reference — Q8_0::quantizeRow packs each kvDim-long row.
    std::vector<std::uint8_t> cpu(T * rowBytes, 0);
    for (std::size_t t = 0; t < T; ++t) {
        mimirmind::compute::quant::Q8_0::quantizeRow(
            x.data() + t * kvDim, kvDim, cpu.data() + t * rowBytes);
    }

    if (std::memcmp(bufKv.raw(), cpu.data(), T * rowBytes) != 0) {
        // Locate the first mismatching byte for the error message.
        const auto* gpu = static_cast<const std::uint8_t*>(bufKv.raw());
        std::size_t i = 0;
        while (i < T * rowBytes && gpu[i] == cpu[i]) ++i;
        const std::size_t t   = i / rowBytes;
        const std::size_t off = i - t * rowBytes;
        std::ostringstream os;
        os << "kv_quant_commit_q8_0_basic: first mismatch at t=" << t
           << " byte=" << off
           << " gpu=" << static_cast<int>(gpu[i])
           << " cpu=" << static_cast<int>(cpu[i]);
        throw std::runtime_error(os.str());
    }
}

TEST(kv_quant_commit_q8_0_writeOffset) {
    // Same shape but with writeOffset=5 — the kernel must skip the
    // first 5 rows and land the T new rows at slot [5, 5+T).
    constexpr std::size_t T           = 2;
    constexpr std::size_t kvDim       = 64;    // 2 blocks per row
    constexpr std::size_t nBlocksRow  = kvDim / 32;
    constexpr std::size_t rowBytes    = nBlocksRow * 34;
    constexpr std::size_t writeOffset = 5;
    constexpr std::size_t maxRows     = writeOffset + T;

    auto x = generateFloats(T * kvDim, 0xC2);

    UsmBuf bufX (T * kvDim * sizeof(float));
    UsmBuf bufKv(maxRows * rowBytes);
    std::memcpy(bufX.raw(), x.data(), x.size() * sizeof(float));
    std::memset(bufKv.raw(), 0x77, maxRows * rowBytes);  // sentinel

    fx().ops.kvQuantCommitQ8Async(bufX.as<float>(), bufKv.raw(),
                                  T, kvDim, writeOffset);
    fx().queue.flush();

    // Rows [0, writeOffset) must be untouched (sentinel still 0x77).
    const auto* gpu = static_cast<const std::uint8_t*>(bufKv.raw());
    for (std::size_t b = 0; b < writeOffset * rowBytes; ++b) {
        if (gpu[b] != 0x77) {
            std::ostringstream os;
            os << "kv_quant_commit_q8_0_writeOffset: pre-slot byte "
               << b << " overwritten — got=" << static_cast<int>(gpu[b])
               << " expected sentinel 0x77";
            throw std::runtime_error(os.str());
        }
    }

    // Rows [writeOffset, writeOffset+T) must match Q8_0::quantizeRow.
    std::vector<std::uint8_t> cpu(T * rowBytes, 0);
    for (std::size_t t = 0; t < T; ++t) {
        mimirmind::compute::quant::Q8_0::quantizeRow(
            x.data() + t * kvDim, kvDim, cpu.data() + t * rowBytes);
    }
    if (std::memcmp(gpu + writeOffset * rowBytes,
                    cpu.data(), T * rowBytes) != 0) {
        throw std::runtime_error(
            "kv_quant_commit_q8_0_writeOffset: written slot does not "
            "match Q8_0::quantizeRow output");
    }
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
                        float       tol,
                        std::size_t slidingWindow = 0)
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
                            oBuf.as<float>(),
                            slidingWindow);
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
        scratch.data(), outCpu.data(),
        slidingWindow);

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

// -- M5i.J Flash-Prefill stress cases -----------------------------------
//
// Prefill (T_q > 1) routes through the single-workgroup streaming
// FlashAttention kernel (kernels/attention_prefill_flash.cl). These
// cases exercise the K-tile online-softmax loop across all its interesting
// boundaries: single-tile (T_q ≤ K_TILE), multi-tile, unaligned tail, the
// GQA index rewrite at scale, and the Gemma 4 headDim=512 upper bound.
// Tolerance follows the decode-flash convention (2e-3 for many-tile runs)
// because online-softmax accumulates rescale-related roundoff — see the
// M5i.J ticket's numerics note.

TEST(attention_prefill_flash_singleTile) {
    // T_q = T_k = 64 → all within tile 0 (K_TILE = 128). No cross-tile
    // rescale — flash reduces exactly to the plain kernel's arithmetic.
    // Tight tolerance verifies the intra-tile path.
    runAttentionParity("attn_prefill_flash_singleTile",
                       /*T_q=*/64, /*T_k=*/64,
                       /*nHeads=*/8, /*nKvHeads=*/2, /*headDim=*/64,
                       /*positionOffset=*/0,
                       /*scale=*/1.0F / std::sqrt(64.0F),
                       /*seed=*/0x91, /*tol=*/2e-4F);
}

TEST(attention_prefill_flash_tileBoundary) {
    // Longest query row (pq = T_q-1 = 127) has kMax = 128 = exactly one
    // K_TILE. Every earlier row also fits in tile 0. First row past the
    // boundary appears at T_q=129 (below).
    runAttentionParity("attn_prefill_flash_boundary",
                       /*T_q=*/128, /*T_k=*/128,
                       /*nHeads=*/4, /*nKvHeads=*/2, /*headDim=*/64,
                       /*positionOffset=*/0,
                       /*scale=*/1.0F / std::sqrt(64.0F),
                       /*seed=*/0x92, /*tol=*/5e-4F);
}

TEST(attention_prefill_flash_twoTiles) {
    // T_q = T_k = 256 → tail rows sweep 2 K-tiles. This is the first
    // test where the online-softmax cross-tile merge is actually
    // exercised. Loosen tolerance vs. single-tile.
    runAttentionParity("attn_prefill_flash_twoTiles",
                       /*T_q=*/256, /*T_k=*/256,
                       /*nHeads=*/4, /*nKvHeads=*/2, /*headDim=*/64,
                       /*positionOffset=*/0,
                       /*scale=*/1.0F / std::sqrt(64.0F),
                       /*seed=*/0x93, /*tol=*/1e-3F);
}

TEST(attention_prefill_flash_fourTiles) {
    // T_q = T_k = 512 → 4 K-tiles. GQA 8:2 like Gemma 4 SWA layers.
    runAttentionParity("attn_prefill_flash_fourTiles",
                       /*T_q=*/512, /*T_k=*/512,
                       /*nHeads=*/8, /*nKvHeads=*/2, /*headDim=*/128,
                       /*positionOffset=*/0,
                       /*scale=*/1.0F / std::sqrt(128.0F),
                       /*seed=*/0x94, /*tol=*/2e-3F);
}

TEST(attention_prefill_flash_ragTypical) {
    // T_q = T_k = 900 → 8 K-tiles (128 * 8 = 1024 > 900). Matches the
    // RAG-typical baseline from f865d23 (2026-07-03) — 903-token prompt,
    // 397 ms/prefill-token. Real-world shape that M5i.J is here to make
    // fast.
    runAttentionParity("attn_prefill_flash_ragTypical",
                       /*T_q=*/900, /*T_k=*/900,
                       /*nHeads=*/8, /*nKvHeads=*/2, /*headDim=*/128,
                       /*positionOffset=*/0,
                       /*scale=*/1.0F / std::sqrt(128.0F),
                       /*seed=*/0x95, /*tol=*/2e-3F);
}

TEST(attention_prefill_flash_headdim512) {
    // Gemma 4 26B full-attention layer geometry (5 of 30 blocks):
    // nHeads=16, nKvHeads=8, headDim=512. Kept T_q small to bound the
    // CPU-reference cost (O(T_q * T_k * nHeads * headDim)) — the point
    // is to verify oRun[512] SLM usage and the MAX_HEADDIM upper bound.
    runAttentionParity("attn_prefill_flash_hd512",
                       /*T_q=*/64, /*T_k=*/64,
                       /*nHeads=*/16, /*nKvHeads=*/8, /*headDim=*/512,
                       /*positionOffset=*/0,
                       /*scale=*/1.0F,
                       /*seed=*/0x96, /*tol=*/1e-3F);
}

TEST(attention_prefill_flash_positionOffset) {
    // Non-zero positionOffset: prefill continuation with 32 prior
    // cached tokens. Query pq attends to keys [0, 32 + pq + 1). This
    // matches the multi-turn prefix-reuse path where the KV cache is
    // pre-filled from an earlier turn's tokens.
    runAttentionParity("attn_prefill_flash_posOffset",
                       /*T_q=*/128, /*T_k=*/160,
                       /*nHeads=*/8, /*nKvHeads=*/2, /*headDim=*/64,
                       /*positionOffset=*/32,
                       /*scale=*/1.0F / std::sqrt(64.0F),
                       /*seed=*/0x97, /*tol=*/1e-3F);
}

// -- M5i.J.1a Sliding-Window Attention (SWA) parity ---------------------
//
// SWA layers clamp each query's K-range to the last `slidingWindow`
// causal keys. Regression coverage against the CPU oracle for both the
// prefill-flash and decode-flash paths — the root cause of M5i.J.1 was
// that these kernels ignored the window and integrated over the full
// causal prefix, breaking Gemma 4 output past ~sliding_window tokens.

TEST(attention_prefill_flash_swa_windowExceedsCtx) {
    // sw >= kMax on every query row: SWA degenerates to pure causal.
    // Bit-parity guard against introducing a slow path for window=0
    // callers when the window is set but happens to not clamp anything.
    runAttentionParity("attn_prefill_flash_swa_noOp",
                       /*T_q=*/64, /*T_k=*/64,
                       /*nHeads=*/8, /*nKvHeads=*/2, /*headDim=*/64,
                       /*positionOffset=*/0,
                       /*scale=*/1.0F / std::sqrt(64.0F),
                       /*seed=*/0xA0, /*tol=*/2e-4F,
                       /*slidingWindow=*/128);
}

TEST(attention_prefill_flash_swa_intraTile) {
    // Window fits inside a single K-tile (sw=32, K_TILE=128). Exercises
    // the mid-tile kMin clamp with tileLen < K_TILE for the first live
    // tile of every query row.
    runAttentionParity("attn_prefill_flash_swa_intraTile",
                       /*T_q=*/64, /*T_k=*/64,
                       /*nHeads=*/4, /*nKvHeads=*/2, /*headDim=*/64,
                       /*positionOffset=*/0,
                       /*scale=*/1.0F / std::sqrt(64.0F),
                       /*seed=*/0xA1, /*tol=*/5e-4F,
                       /*slidingWindow=*/32);
}

TEST(attention_prefill_flash_swa_midTile) {
    // sw=200 with T_q=T_k=300 → for the last query row kMin=100, which
    // lands mid-tile-0 (K_TILE=128). ktStart skips no tiles but the
    // low-clamp inside tile 0 must trim scores[0..99] out of the softmax.
    runAttentionParity("attn_prefill_flash_swa_midTile",
                       /*T_q=*/300, /*T_k=*/300,
                       /*nHeads=*/4, /*nKvHeads=*/2, /*headDim=*/64,
                       /*positionOffset=*/0,
                       /*scale=*/1.0F / std::sqrt(64.0F),
                       /*seed=*/0xA2, /*tol=*/1e-3F,
                       /*slidingWindow=*/200);
}

TEST(attention_prefill_flash_swa_gemma4_e4b) {
    // Real-world Gemma 4 E4B SWA-layer geometry: sliding_window=512,
    // GQA head_count=8 / head_count_kv=2, head_dim=256. T_q=T_k=1024
    // makes ktStart advance past tile 0 for late query rows, so both
    // the tile-skip and mid-tile clamp branches fire on the same run.
    runAttentionParity("attn_prefill_flash_swa_e4b",
                       /*T_q=*/1024, /*T_k=*/1024,
                       /*nHeads=*/8, /*nKvHeads=*/2, /*headDim=*/256,
                       /*positionOffset=*/0,
                       /*scale=*/1.0F / std::sqrt(256.0F),
                       /*seed=*/0xA3, /*tol=*/3e-3F,
                       /*slidingWindow=*/512);
}

TEST(attention_decode_flash_swa) {
    // Decode-Flash SWA: T_q=1, positionOffset=1499, T_k=1500, sw=512.
    // kMin=988 lands in tile 15 (K_TILE=64) → tiles 0..14 must emit
    // neutral partials, tile 15 clamps its low boundary at kMin.
    runAttentionParity("attn_decode_flash_swa",
                       /*T_q=*/1, /*T_k=*/1500,
                       /*nHeads=*/8, /*nKvHeads=*/2, /*headDim=*/128,
                       /*positionOffset=*/1499,
                       /*scale=*/1.0F / std::sqrt(128.0F),
                       /*seed=*/0xA4, /*tol=*/1e-3F,
                       /*slidingWindow=*/512);
}

TEST(attention_decode_flash_swa_windowExceedsCtx) {
    // sw >= kMax: decode-flash with a wide window degenerates to pure
    // causal. Neutral-partial branch stays untouched — every tile
    // contributes to the merge.
    runAttentionParity("attn_decode_flash_swa_noOp",
                       /*T_q=*/1, /*T_k=*/256,
                       /*nHeads=*/8, /*nKvHeads=*/2, /*headDim=*/64,
                       /*positionOffset=*/255,
                       /*scale=*/1.0F / std::sqrt(64.0F),
                       /*seed=*/0xA5, /*tol=*/2e-4F,
                       /*slidingWindow=*/1024);
}

// =======================================================================
// M10.2.-1 FP16-KV attention parity
// =======================================================================
//
// Verify that the fp16-KV attention kernels (attention_fp16,
// attention_prefill_flash_fp16, attention_flash_partial_fp16) match the
// fp32-KV baseline on the *same* K/V source data. Baseline: run the f32
// kernel with fp32 K/V. Comparison: quantise K/V to fp16 on the host
// and run the fp16 kernel against the fp16 storage. Numerical delta is
// bounded by fp16 quantisation noise accumulated through the softmax +
// weighted V-sum. The design ticket (M10.2.-1) sets the gate at 2e-3.

namespace {

// IEEE-754 binary32 → binary16 (round-toward-zero on the mantissa,
// same as Gemma4E4BBackend::floatToHalf). vload_half on the GPU
// decodes the exact bit pattern this writes, so what the test
// quantises on the host is what the kernel loads.
inline std::uint16_t f32ToF16Bits(float f) noexcept {
    std::uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    const std::uint32_t sign = (bits >> 16) & 0x8000U;
    std::int32_t        exp  = static_cast<std::int32_t>((bits >> 23) & 0xFFU);
    const std::uint32_t mant = bits & 0x7FFFFFU;

    if (exp == 0xFF) {
        return static_cast<std::uint16_t>(
            sign | 0x7C00U | (mant != 0 ? 0x0200U : 0U));
    }
    exp -= 127 - 15;
    if (exp <= 0) {
        if (exp < -10) return static_cast<std::uint16_t>(sign);
        const std::uint32_t m = (mant | 0x800000U) >> (14 - exp);
        return static_cast<std::uint16_t>(sign | m);
    }
    if (exp >= 0x1F) {
        return static_cast<std::uint16_t>(sign | 0x7C00U);
    }
    return static_cast<std::uint16_t>(sign
        | (static_cast<std::uint32_t>(exp) << 10)
        | (mant >> 13));
}

std::vector<std::uint16_t> encodeHalf(const std::vector<float>& src) {
    std::vector<std::uint16_t> out(src.size());
    for (std::size_t i = 0; i < src.size(); ++i) {
        out[i] = f32ToF16Bits(src[i]);
    }
    return out;
}

// Baseline vs fp16-KV, same K/V source data. Both dispatches use the
// same physical K/V numbers (the fp16 side round-trips them through
// half-precision), so the delta is purely the accumulated fp16 rounding
// error inside the softmax + weighted V-sum.
void runAttentionFp16Parity(const char*  label,
                            std::size_t  T_q,
                            std::size_t  T_k,
                            std::size_t  nHeads,
                            std::size_t  nKvHeads,
                            std::size_t  headDim,
                            std::size_t  positionOffset,
                            float        scale,
                            std::uint32_t seed,
                            float        tol,
                            std::size_t  slidingWindow = 0)
{
    using mimirmind::runtime::KvDtype;

    const std::size_t qN  = T_q * nHeads   * headDim;
    const std::size_t kvN = T_k * nKvHeads * headDim;
    const std::size_t oN  = T_q * nHeads   * headDim;

    auto qHost = generateFloats(qN,  seed + 0);
    auto kHost = generateFloats(kvN, seed + 1);
    auto vHost = generateFloats(kvN, seed + 2);

    // -- Baseline: f32 kernel on the raw fp32 K/V. -----------------------
    UsmBuf qBufF32(qN  * sizeof(float));
    UsmBuf kBufF32(kvN * sizeof(float));
    UsmBuf vBufF32(kvN * sizeof(float));
    UsmBuf oBufF32(oN  * sizeof(float));
    std::memcpy(qBufF32.raw(), qHost.data(), qN  * sizeof(float));
    std::memcpy(kBufF32.raw(), kHost.data(), kvN * sizeof(float));
    std::memcpy(vBufF32.raw(), vHost.data(), kvN * sizeof(float));
    std::memset(oBufF32.raw(), 0,            oN  * sizeof(float));

    fx().ops.attentionAsync(qBufF32.as<float>(),
                            kBufF32.raw(), vBufF32.raw(),
                            T_q, T_k, nHeads, nKvHeads, headDim,
                            positionOffset, scale,
                            oBufF32.as<float>(),
                            slidingWindow, KvDtype::F32);
    fx().queue.flush();

    // -- Comparison: fp16 kernel on host-quantised K/V. ------------------
    const auto kHalf = encodeHalf(kHost);
    const auto vHalf = encodeHalf(vHost);

    UsmBuf qBufF16(qN  * sizeof(float));
    UsmBuf kBufF16(kvN * sizeof(std::uint16_t));
    UsmBuf vBufF16(kvN * sizeof(std::uint16_t));
    UsmBuf oBufF16(oN  * sizeof(float));
    std::memcpy(qBufF16.raw(), qHost.data(), qN  * sizeof(float));
    std::memcpy(kBufF16.raw(), kHalf.data(), kvN * sizeof(std::uint16_t));
    std::memcpy(vBufF16.raw(), vHalf.data(), kvN * sizeof(std::uint16_t));
    std::memset(oBufF16.raw(), 0,            oN  * sizeof(float));

    fx().ops.attentionAsync(qBufF16.as<float>(),
                            kBufF16.raw(), vBufF16.raw(),
                            T_q, T_k, nHeads, nKvHeads, headDim,
                            positionOffset, scale,
                            oBufF16.as<float>(),
                            slidingWindow, KvDtype::FP16);
    fx().queue.flush();

    EXPECT_ARRAY_NEAR(label, oBufF16.as<float>(), oBufF32.as<float>(), oN, tol);
}

} // namespace

TEST(attention_decode_flash_fp16_parity) {
    // T_q=1 → attention_flash_partial_fp16. Single-tile decode.
    runAttentionFp16Parity("attn_flash_partial_fp16",
                           /*T_q=*/1, /*T_k=*/64,
                           /*nHeads=*/4, /*nKvHeads=*/2, /*headDim=*/64,
                           /*positionOffset=*/63,
                           /*scale=*/1.0F / std::sqrt(64.0F),
                           /*seed=*/0xF1, /*tol=*/2e-3F);
}

TEST(attention_decode_flash_fp16_parity_multiTile) {
    // T_k=2048 → 32 fp16-decode tiles, GQA 4:1, headDim=128. Exercises
    // the cross-tile merge path under fp16 loads.
    runAttentionFp16Parity("attn_flash_partial_fp16_multiTile",
                           /*T_q=*/1, /*T_k=*/2048,
                           /*nHeads=*/16, /*nKvHeads=*/4, /*headDim=*/128,
                           /*positionOffset=*/2047,
                           /*scale=*/1.0F / std::sqrt(128.0F),
                           /*seed=*/0xF2, /*tol=*/2e-3F);
}

TEST(attention_prefill_flash_fp16_parity_singleTile) {
    // T_q=T_k=64 → single K-tile (K_TILE=128). No online-softmax
    // rescale. Tightest fp16 tolerance since only per-tile softmax
    // touches the rounded values.
    runAttentionFp16Parity("attn_prefill_flash_fp16_singleTile",
                           /*T_q=*/64, /*T_k=*/64,
                           /*nHeads=*/8, /*nKvHeads=*/2, /*headDim=*/64,
                           /*positionOffset=*/0,
                           /*scale=*/1.0F / std::sqrt(64.0F),
                           /*seed=*/0xF3, /*tol=*/2e-3F);
}

TEST(attention_prefill_flash_fp16_parity_multiTile) {
    // T_q=T_k=512, GQA 4:1, headDim=128 — matches Gemma-4 SWA layer
    // dims. Multi-tile online-softmax merge under fp16-KV.
    runAttentionFp16Parity("attn_prefill_flash_fp16_multiTile",
                           /*T_q=*/512, /*T_k=*/512,
                           /*nHeads=*/8, /*nKvHeads=*/2, /*headDim=*/128,
                           /*positionOffset=*/0,
                           /*scale=*/1.0F / std::sqrt(128.0F),
                           /*seed=*/0xF4, /*tol=*/2e-3F);
}

TEST(attention_prefill_flash_fp16_parity_ragTypical) {
    // 900-token prefill under fp16 — the RAG-typical M5i.J shape, most
    // common real-world case for chat prompts. Loosest tolerance
    // because 8 K-tiles compound the rescale error.
    runAttentionFp16Parity("attn_prefill_flash_fp16_ragTypical",
                           /*T_q=*/900, /*T_k=*/900,
                           /*nHeads=*/8, /*nKvHeads=*/2, /*headDim=*/128,
                           /*positionOffset=*/0,
                           /*scale=*/1.0F / std::sqrt(128.0F),
                           /*seed=*/0xF5, /*tol=*/2e-3F);
}

TEST(attention_prefill_flash_fp16_parity_swa) {
    // SWA under fp16-KV: sw=128, T_q=T_k=512. Verifies the fp16 kernel
    // honours the sliding-window low clamp exactly the same as the
    // f32 variant.
    runAttentionFp16Parity("attn_prefill_flash_fp16_swa",
                           /*T_q=*/512, /*T_k=*/512,
                           /*nHeads=*/8, /*nKvHeads=*/2, /*headDim=*/128,
                           /*positionOffset=*/0,
                           /*scale=*/1.0F / std::sqrt(128.0F),
                           /*seed=*/0xF6, /*tol=*/2e-3F,
                           /*slidingWindow=*/128);
}

// =======================================================================
// M10.2.0 Commit 6 — attention_q8_0 / attention_prefill_flash_q8_0 /
//                    attention_flash_partial_q8_0 parity
// =======================================================================
//
// Verify that the Q8_0-KV attention kernels match the fp32-KV baseline
// on the *same* K/V source data. Baseline: fp32 K/V through the f32
// kernel. Comparison: encode K/V to Q8_0 (compute::quant::Q8_0::quantizeRow
// — the exact byte layout the write kernel targets, so the read path
// sees the same bytes a real deploy would) and run the Q8_0 read
// variant. Numerical delta is bounded by per-block absmax + int8 quant
// noise accumulated through the softmax + weighted V-sum. The design
// ticket (M10.2.0) sets the gate at 5e-3 — 2.5× looser than fp16's
// 2e-3 to absorb K-channel-outlier per-block magnitude bleed.

namespace {

std::vector<std::uint8_t> encodeQ8_0(const std::vector<float>& src,
                                     std::size_t T_k,
                                     std::size_t kvDim) {
    constexpr std::size_t kBlockElems = 32;
    constexpr std::size_t kBlockBytes = 34;
    const std::size_t nBlocksPerRow = kvDim / kBlockElems;
    const std::size_t rowBytes      = nBlocksPerRow * kBlockBytes;
    std::vector<std::uint8_t> out(T_k * rowBytes, 0);
    for (std::size_t t = 0; t < T_k; ++t) {
        mimirmind::compute::quant::Q8_0::quantizeRow(
            src.data() + t * kvDim, kvDim, out.data() + t * rowBytes);
    }
    return out;
}

// Baseline vs Q8_0-KV, same K/V source data. Both dispatches use the
// same physical K/V numbers (the Q8_0 side round-trips them through
// per-32-block absmax + int8), so the delta is purely the accumulated
// Q8_0 quantisation error inside the softmax + weighted V-sum.
void runAttentionQ8Parity(const char*  label,
                          std::size_t  T_q,
                          std::size_t  T_k,
                          std::size_t  nHeads,
                          std::size_t  nKvHeads,
                          std::size_t  headDim,
                          std::size_t  positionOffset,
                          float        scale,
                          std::uint32_t seed,
                          float        tol,
                          std::size_t  slidingWindow = 0)
{
    using mimirmind::runtime::KvDtype;

    const std::size_t kvDim = nKvHeads * headDim;
    const std::size_t qN    = T_q * nHeads * headDim;
    const std::size_t kvN   = T_k * kvDim;
    const std::size_t oN    = T_q * nHeads * headDim;

    auto qHost = generateFloats(qN,  seed + 0);
    auto kHost = generateFloats(kvN, seed + 1);
    auto vHost = generateFloats(kvN, seed + 2);

    // -- Baseline: f32 kernel on the raw fp32 K/V. -----------------------
    UsmBuf qBufF32(qN  * sizeof(float));
    UsmBuf kBufF32(kvN * sizeof(float));
    UsmBuf vBufF32(kvN * sizeof(float));
    UsmBuf oBufF32(oN  * sizeof(float));
    std::memcpy(qBufF32.raw(), qHost.data(), qN  * sizeof(float));
    std::memcpy(kBufF32.raw(), kHost.data(), kvN * sizeof(float));
    std::memcpy(vBufF32.raw(), vHost.data(), kvN * sizeof(float));
    std::memset(oBufF32.raw(), 0,            oN  * sizeof(float));

    fx().ops.attentionAsync(qBufF32.as<float>(),
                            kBufF32.raw(), vBufF32.raw(),
                            T_q, T_k, nHeads, nKvHeads, headDim,
                            positionOffset, scale,
                            oBufF32.as<float>(),
                            slidingWindow, KvDtype::F32);
    fx().queue.flush();

    // -- Comparison: Q8_0 kernel on host-quantised K/V. ------------------
    const auto kQ8 = encodeQ8_0(kHost, T_k, kvDim);
    const auto vQ8 = encodeQ8_0(vHost, T_k, kvDim);

    UsmBuf qBufQ8(qN * sizeof(float));
    UsmBuf kBufQ8(kQ8.size());
    UsmBuf vBufQ8(vQ8.size());
    UsmBuf oBufQ8(oN * sizeof(float));
    std::memcpy(qBufQ8.raw(), qHost.data(), qN * sizeof(float));
    std::memcpy(kBufQ8.raw(), kQ8.data(),   kQ8.size());
    std::memcpy(vBufQ8.raw(), vQ8.data(),   vQ8.size());
    std::memset(oBufQ8.raw(), 0,            oN * sizeof(float));

    fx().ops.attentionAsync(qBufQ8.as<float>(),
                            kBufQ8.raw(), vBufQ8.raw(),
                            T_q, T_k, nHeads, nKvHeads, headDim,
                            positionOffset, scale,
                            oBufQ8.as<float>(),
                            slidingWindow, KvDtype::Q8_0);
    fx().queue.flush();

    EXPECT_ARRAY_NEAR(label, oBufQ8.as<float>(), oBufF32.as<float>(), oN, tol);
}

} // namespace

TEST(attention_decode_flash_q8_0_parity) {
    // T_q=1 → attention_flash_partial_q8_0. Single-tile decode, kvDim
    // = 2*64 = 128 = 4 Q8_0 blocks per row. Tightest Q8_0 tolerance —
    // only per-tile softmax + V-weighted sum touches the dequant.
    runAttentionQ8Parity("attn_flash_partial_q8_0",
                         /*T_q=*/1, /*T_k=*/64,
                         /*nHeads=*/4, /*nKvHeads=*/2, /*headDim=*/64,
                         /*positionOffset=*/63,
                         /*scale=*/1.0F / std::sqrt(64.0F),
                         /*seed=*/0xA1, /*tol=*/5e-3F);
}

TEST(attention_decode_flash_q8_0_parity_multiTile) {
    // T_k=2048 → 32 decode tiles, GQA 4:1, headDim=128 (= 4 blocks per
    // head). Exercises the cross-tile online-softmax merge path under
    // Q8_0 loads.
    runAttentionQ8Parity("attn_flash_partial_q8_0_multiTile",
                         /*T_q=*/1, /*T_k=*/2048,
                         /*nHeads=*/16, /*nKvHeads=*/4, /*headDim=*/128,
                         /*positionOffset=*/2047,
                         /*scale=*/1.0F / std::sqrt(128.0F),
                         /*seed=*/0xA2, /*tol=*/5e-3F);
}

TEST(attention_prefill_flash_q8_0_parity_singleTile) {
    // T_q=T_k=64 → single K-tile (K_TILE=128). No online-softmax
    // rescale across tiles. Tightest Q8_0 tolerance since only per-tile
    // softmax touches the dequantised values.
    runAttentionQ8Parity("attn_prefill_flash_q8_0_singleTile",
                         /*T_q=*/64, /*T_k=*/64,
                         /*nHeads=*/8, /*nKvHeads=*/2, /*headDim=*/64,
                         /*positionOffset=*/0,
                         /*scale=*/1.0F / std::sqrt(64.0F),
                         /*seed=*/0xA3, /*tol=*/5e-3F);
}

TEST(attention_prefill_flash_q8_0_parity_multiTile) {
    // T_q=T_k=512, GQA 4:1, headDim=128 — matches Gemma 4 SWA layer
    // dims. Multi-tile online-softmax merge under Q8_0-KV.
    runAttentionQ8Parity("attn_prefill_flash_q8_0_multiTile",
                         /*T_q=*/512, /*T_k=*/512,
                         /*nHeads=*/8, /*nKvHeads=*/2, /*headDim=*/128,
                         /*positionOffset=*/0,
                         /*scale=*/1.0F / std::sqrt(128.0F),
                         /*seed=*/0xA4, /*tol=*/5e-3F);
}

TEST(attention_prefill_flash_q8_0_parity_ragTypical) {
    // 900-token prefill under Q8_0 — the RAG-typical M5i.J shape, most
    // common real-world case for chat prompts. 8 K-tiles compound the
    // per-tile softmax rescale error, but the block-32 absmax noise is
    // the dominant term.
    runAttentionQ8Parity("attn_prefill_flash_q8_0_ragTypical",
                         /*T_q=*/900, /*T_k=*/900,
                         /*nHeads=*/8, /*nKvHeads=*/2, /*headDim=*/128,
                         /*positionOffset=*/0,
                         /*scale=*/1.0F / std::sqrt(128.0F),
                         /*seed=*/0xA5, /*tol=*/5e-3F);
}

TEST(attention_prefill_flash_q8_0_parity_swa) {
    // SWA under Q8_0-KV: sw=128, T_q=T_k=512. Verifies the Q8_0 kernel
    // honours the sliding-window low clamp exactly the same as the
    // f32 variant, and that the block-32 index arithmetic doesn't
    // break at the clamp boundary.
    runAttentionQ8Parity("attn_prefill_flash_q8_0_swa",
                         /*T_q=*/512, /*T_k=*/512,
                         /*nHeads=*/8, /*nKvHeads=*/2, /*headDim=*/128,
                         /*positionOffset=*/0,
                         /*scale=*/1.0F / std::sqrt(128.0F),
                         /*seed=*/0xA6, /*tol=*/5e-3F,
                         /*slidingWindow=*/128);
}

// =======================================================================
// R1 — attention_prefill_flash_q8_0_gqa parity (packed head variant)
// =======================================================================
//
// The head-packed kernel processes ALL Q-heads of a KV-group inside one
// workgroup so K/V dequant loads amortise across nHeads/nKvHeads. Same
// math as the plain per-Q-head kernel, different launch geometry and
// SLM layout — parity gates against the F32 baseline catch any
// regression in the packed dispatch path.
//
// runAttentionQ8Parity dispatches through attentionAsync, which picks
// the GQA kernel automatically for (Q8_0, nQPerKv > 1). Tests below
// exercise the auto-selected packed kernel; the final "plain_forced"
// case flips features.flashPrefillGqa off at runtime so the plain
// Q8_0 kernel keeps parity coverage even now that GQA-shaped tests
// route around it by default.

namespace {

// RAII: flip the GQA rollback flag for the duration of a test case,
// restore in the destructor. Keeps the fixture-wide GpuOps state
// consistent across tests regardless of failure paths.
class ScopedPrefillFlashGqaDisabled {
public:
    explicit ScopedPrefillFlashGqaDisabled(bool disabled)
        : _prev{!fx().ops.prefillFlashGqaEnabled()} {
        fx().ops.setPrefillFlashGqaDisabledForTest(disabled);
    }
    ~ScopedPrefillFlashGqaDisabled() {
        fx().ops.setPrefillFlashGqaDisabledForTest(_prev);
    }
    ScopedPrefillFlashGqaDisabled(const ScopedPrefillFlashGqaDisabled&) = delete;
    ScopedPrefillFlashGqaDisabled& operator=(const ScopedPrefillFlashGqaDisabled&) = delete;
private:
    bool _prev;
};

} // namespace

TEST(attention_prefill_flash_q8_0_gqa_singleTile) {
    // T_q=T_k=64 (single K-tile at K_TILE=128), GQA 4:1 (nHeads=8,
    // nKvHeads=2, nQPerKv=4). Tightest tolerance — only per-tile
    // softmax touches the dequantised values, and the packed inner
    // loop reuses the same K/V bytes across the 4 Q-heads per group.
    runAttentionQ8Parity("attn_prefill_flash_q8_0_gqa_singleTile",
                         /*T_q=*/64, /*T_k=*/64,
                         /*nHeads=*/8, /*nKvHeads=*/2, /*headDim=*/64,
                         /*positionOffset=*/0,
                         /*scale=*/1.0F / std::sqrt(64.0F),
                         /*seed=*/0xB1, /*tol=*/5e-3F);
}

TEST(attention_prefill_flash_q8_0_gqa_multiTile) {
    // T_q=T_k=512, GQA 4:1, headDim=128. Multi-tile online-softmax
    // merge under the packed kernel — verifies the per-Q-head
    // (m_reg, l_reg, oRun) state stays isolated across the K-tile
    // rescale steps.
    runAttentionQ8Parity("attn_prefill_flash_q8_0_gqa_multiTile",
                         /*T_q=*/512, /*T_k=*/512,
                         /*nHeads=*/8, /*nKvHeads=*/2, /*headDim=*/128,
                         /*positionOffset=*/0,
                         /*scale=*/1.0F / std::sqrt(128.0F),
                         /*seed=*/0xB2, /*tol=*/5e-3F);
}

TEST(attention_prefill_flash_q8_0_gqa_ratio2_gemma4) {
    // GQA 2:1 — Gemma 4 26B-A4B primary target (16 Q-heads /
    // 8 KV-heads = 2 per group). This is the production shape the
    // packed kernel was built for; nQPerKv=2 means the register-
    // array + SLM cost is exactly 2 slots, no dead per-Q-head
    // iterations.
    runAttentionQ8Parity("attn_prefill_flash_q8_0_gqa_ratio2_gemma4",
                         /*T_q=*/256, /*T_k=*/256,
                         /*nHeads=*/16, /*nKvHeads=*/8, /*headDim=*/128,
                         /*positionOffset=*/0,
                         /*scale=*/1.0F / std::sqrt(128.0F),
                         /*seed=*/0xB3, /*tol=*/5e-3F);
}

TEST(attention_prefill_flash_q8_0_gqa_ratio8_qwen) {
    // GQA 8:1 — Qwen-family shape at the upper bound of
    // ATTN_FLASH_PREFILL_N_Q_PER_KV_MAX. Guards the register-array
    // sizing against off-by-one when every slot is live.
    runAttentionQ8Parity("attn_prefill_flash_q8_0_gqa_ratio8_qwen",
                         /*T_q=*/256, /*T_k=*/256,
                         /*nHeads=*/8, /*nKvHeads=*/1, /*headDim=*/128,
                         /*positionOffset=*/0,
                         /*scale=*/1.0F / std::sqrt(128.0F),
                         /*seed=*/0xB4, /*tol=*/5e-3F);
}

TEST(attention_prefill_flash_q8_0_gqa_ragTypical) {
    // 900-token prefill under the packed kernel with Gemma-4-shaped
    // GQA (16 Q / 8 KV). Real-world Pegenaut-RAG shape; ~8 K-tile
    // iterations compound the online-softmax rescale error but the
    // block-32 absmax noise still dominates.
    runAttentionQ8Parity("attn_prefill_flash_q8_0_gqa_ragTypical",
                         /*T_q=*/900, /*T_k=*/900,
                         /*nHeads=*/16, /*nKvHeads=*/8, /*headDim=*/128,
                         /*positionOffset=*/0,
                         /*scale=*/1.0F / std::sqrt(128.0F),
                         /*seed=*/0xB5, /*tol=*/5e-3F);
}

TEST(attention_prefill_flash_q8_0_gqa_swa) {
    // SWA + GQA — Gemma-4 SWA layer shape (sliding window 128 tokens,
    // GQA 4:1). Verifies the per-Q-head accumulators clamp to the
    // sliding-window low bound identically for every Q-head in the
    // group, not just the leading one.
    runAttentionQ8Parity("attn_prefill_flash_q8_0_gqa_swa",
                         /*T_q=*/512, /*T_k=*/512,
                         /*nHeads=*/8, /*nKvHeads=*/2, /*headDim=*/128,
                         /*positionOffset=*/0,
                         /*scale=*/1.0F / std::sqrt(128.0F),
                         /*seed=*/0xB6, /*tol=*/5e-3F,
                         /*slidingWindow=*/128);
}

TEST(attention_prefill_flash_q8_0_gqa_positionOffset) {
    // Multi-turn prefix reuse: positionOffset=64 (prior turn's KV
    // stays in cache), then a T_q=32 continuation over T_k=96.
    // Verifies curLenPtr[0] is honoured for the absPos + kMax bounds
    // inside the packed kernel exactly as in the plain kernel.
    runAttentionQ8Parity("attn_prefill_flash_q8_0_gqa_positionOffset",
                         /*T_q=*/32, /*T_k=*/96,
                         /*nHeads=*/8, /*nKvHeads=*/2, /*headDim=*/128,
                         /*positionOffset=*/64,
                         /*scale=*/1.0F / std::sqrt(128.0F),
                         /*seed=*/0xB7, /*tol=*/5e-3F);
}

TEST(attention_prefill_flash_q8_0_plain_path_forced) {
    // Regression guard for the plain per-Q-head kernel now that the
    // dispatcher routes GQA shapes to the packed variant by default.
    // Force features.flashPrefillGqa=false for this case so the plain
    // Q8_0 kernel keeps a parity gate against the F32 baseline even
    // when nQPerKv > 1.
    ScopedPrefillFlashGqaDisabled disable{true};
    runAttentionQ8Parity("attn_prefill_flash_q8_0_plain_forced_gqa4",
                         /*T_q=*/256, /*T_k=*/256,
                         /*nHeads=*/8, /*nKvHeads=*/2, /*headDim=*/128,
                         /*positionOffset=*/0,
                         /*scale=*/1.0F / std::sqrt(128.0F),
                         /*seed=*/0xB8, /*tol=*/5e-3F);
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

// =======================================================================
// Q6_K GEMM (batched M>1) — parity vs CPU reference across M values that
// cover the M_TILE=8 tail-guard: aligned (8, 16), tail (4, 15, 17), and
// the M=1 baseline (which dispatches through the matvec kernel and thus
// re-tests the vec regression path via GpuMatmul::matmul).
// =======================================================================
namespace {

// Build W of shape [N, K] as Q6_K bytes. Every row gets a distinct byte
// pattern so N-way replication cannot hide a mis-indexed dequant.
std::vector<std::uint8_t> buildQ6kWeights(std::size_t N,
                                          std::size_t K,
                                          std::uint32_t seed) {
    const std::size_t superBlocksPerRow = K / 256;
    const std::size_t bytesPerRow       = superBlocksPerRow * 210;

    std::vector<std::uint8_t> w(N * bytesPerRow, 0);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> distByte(0, 255);
    std::uniform_int_distribution<int> distScale(-8, 8);

    for (std::size_t n = 0; n < N; ++n) {
        for (std::size_t sb = 0; sb < superBlocksPerRow; ++sb) {
            auto* block = w.data() + n * bytesPerRow + sb * 210;
            for (std::size_t i = 0; i < 128; ++i) {
                block[i] = static_cast<std::uint8_t>(distByte(rng));
            }
            for (std::size_t i = 128; i < 192; ++i) {
                block[i] = static_cast<std::uint8_t>(distByte(rng));
            }
            for (std::size_t i = 192; i < 208; ++i) {
                const int s = distScale(rng);
                block[i]    = static_cast<std::uint8_t>(static_cast<std::int8_t>(s));
            }
            // fp16 super-block scale — a small positive value keeps the
            // dot products in a comfortable float32 range.
            constexpr std::uint16_t kHalfSmall = 0x2E00U;   // ≈ 0.09
            std::memcpy(block + 208, &kHalfSmall, sizeof(std::uint16_t));
        }
    }
    return w;
}

// Run both CPU and GPU matmul for Q6_K with the given (N, K, M) and
// compare. Rejects a tolerance mismatch via EXPECT_ARRAY_NEAR.
void runQ6kMatmulParity(const char*   label,
                        std::size_t   N,
                        std::size_t   K,
                        std::size_t   M,
                        std::uint32_t seed,
                        float         tol) {
    const auto w = buildQ6kWeights(N, K, seed);
    const auto x = generateFloats(M * K, seed ^ 0x9E3779B9U);

    UsmBuf bufW(w.size());
    UsmBuf bufX(M * K * sizeof(float));
    UsmBuf bufY(M * N * sizeof(float));
    UsmBuf bufScratch(K * sizeof(float));

    std::memcpy(bufW.raw(), w.data(), w.size());
    std::memcpy(bufX.raw(), x.data(), x.size() * sizeof(float));

    fx().gmm.matmul(mimirmind::model::GgmlType::Q6_K,
                    bufW.raw(), N, K,
                    bufX.as<float>(), M,
                    bufY.as<float>(),
                    bufScratch.as<float>());

    std::vector<float> cpuY(M * N);
    std::vector<float> scratchCpu(K);
    mimirmind::compute::matmul(mimirmind::model::GgmlType::Q6_K,
                               w.data(), N, K,
                               x.data(), M,
                               cpuY.data(),
                               scratchCpu.data());

    EXPECT_ARRAY_NEAR(label, bufY.as<float>(), cpuY.data(), M * N, tol);
}

} // namespace

TEST(matmul_q6k_gemm_M1_regression) {
    // M==1 must still dispatch through the matvec kernel and give the
    // same result as before this change.
    runQ6kMatmulParity("matmul_q6k_gemm_M1", /*N=*/32, /*K=*/1024,
                       /*M=*/1, /*seed=*/0xC001U, /*tol=*/1e-2F);
}

TEST(matmul_q6k_gemm_M4_tail) {
    // M=4 < M_TILE=8: single M-workgroup with 4 tail slots.
    runQ6kMatmulParity("matmul_q6k_gemm_M4", /*N=*/32, /*K=*/1024,
                       /*M=*/4, /*seed=*/0xC002U, /*tol=*/1e-2F);
}

TEST(matmul_q6k_gemm_M8_aligned) {
    // Exactly one M-workgroup fully populated.
    runQ6kMatmulParity("matmul_q6k_gemm_M8", /*N=*/48, /*K=*/1024,
                       /*M=*/8, /*seed=*/0xC003U, /*tol=*/1e-2F);
}

TEST(matmul_q6k_gemm_M15_tail_almost_full) {
    // Two M-workgroups: first full (8), second nearly-full (7 tail).
    runQ6kMatmulParity("matmul_q6k_gemm_M15", /*N=*/32, /*K=*/1024,
                       /*M=*/15, /*seed=*/0xC004U, /*tol=*/1e-2F);
}

TEST(matmul_q6k_gemm_M16_aligned) {
    // Two M-workgroups, both fully populated.
    runQ6kMatmulParity("matmul_q6k_gemm_M16", /*N=*/40, /*K=*/1024,
                       /*M=*/16, /*seed=*/0xC005U, /*tol=*/1e-2F);
}

TEST(matmul_q6k_gemm_M17_tail_after_full) {
    // Three M-workgroups: two full + 1-row tail.
    runQ6kMatmulParity("matmul_q6k_gemm_M17", /*N=*/24, /*K=*/1024,
                       /*M=*/17, /*seed=*/0xC006U, /*tol=*/1e-2F);
}

TEST(matmul_q6k_gemm_realistic_prefill) {
    // Shape closer to a real Gemma-4-Q6_K attention projection at prefill:
    // N=d_model≈2560, K=d_model≈2816, M=16 tokens. Bigger K means the
    // matvec-vs-gemm precision drift shows up if we accidentally broke
    // Kahan compensation.
    runQ6kMatmulParity("matmul_q6k_gemm_prefill", /*N=*/2560, /*K=*/2816,
                       /*M=*/16, /*seed=*/0xC007U, /*tol=*/2e-2F);
}

// =======================================================================
// Q4_K GEMM (batched M>1) — same tail-coverage as Q6_K.
// =======================================================================
namespace {

std::vector<std::uint8_t> buildQ4kWeights(std::size_t N,
                                          std::size_t K,
                                          std::uint32_t seed) {
    const std::size_t superBlocksPerRow = K / 256;
    const std::size_t bytesPerRow       = superBlocksPerRow * 144;

    std::vector<std::uint8_t> w(N * bytesPerRow, 0);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> distByte(0, 255);

    // Small positive fp16 scales (~0.09) keep dot products in a
    // comfortable float32 range.
    constexpr std::uint16_t kHalfSmall = 0x2E00U;
    constexpr std::uint16_t kHalfTiny  = 0x2400U;   // ~0.02, dmin

    for (std::size_t n = 0; n < N; ++n) {
        for (std::size_t sb = 0; sb < superBlocksPerRow; ++sb) {
            auto* block = w.data() + n * bytesPerRow + sb * 144;

            std::memcpy(block,     &kHalfSmall, sizeof(std::uint16_t));  // d
            std::memcpy(block + 2, &kHalfTiny,  sizeof(std::uint16_t));  // dmin

            for (std::size_t i = 4; i < 16; ++i) {
                block[i] = static_cast<std::uint8_t>(distByte(rng));
            }
            for (std::size_t i = 16; i < 144; ++i) {
                block[i] = static_cast<std::uint8_t>(distByte(rng));
            }
        }
    }
    return w;
}

void runQ4kMatmulParity(const char*   label,
                        std::size_t   N,
                        std::size_t   K,
                        std::size_t   M,
                        std::uint32_t seed,
                        float         tol) {
    const auto w = buildQ4kWeights(N, K, seed);
    const auto x = generateFloats(M * K, seed ^ 0x9E3779B9U);

    UsmBuf bufW(w.size());
    UsmBuf bufX(M * K * sizeof(float));
    UsmBuf bufY(M * N * sizeof(float));
    UsmBuf bufScratch(K * sizeof(float));

    std::memcpy(bufW.raw(), w.data(), w.size());
    std::memcpy(bufX.raw(), x.data(), x.size() * sizeof(float));

    fx().gmm.matmul(mimirmind::model::GgmlType::Q4_K,
                    bufW.raw(), N, K,
                    bufX.as<float>(), M,
                    bufY.as<float>(),
                    bufScratch.as<float>());

    std::vector<float> cpuY(M * N);
    std::vector<float> scratchCpu(K);
    mimirmind::compute::matmul(mimirmind::model::GgmlType::Q4_K,
                               w.data(), N, K,
                               x.data(), M,
                               cpuY.data(),
                               scratchCpu.data());

    EXPECT_ARRAY_NEAR(label, bufY.as<float>(), cpuY.data(), M * N, tol);
}

} // namespace

TEST(matmul_q4k_gemm_M1_regression) {
    runQ4kMatmulParity("matmul_q4k_gemm_M1", /*N=*/32, /*K=*/1024,
                       /*M=*/1, /*seed=*/0xD001U, /*tol=*/1e-2F);
}

TEST(matmul_q4k_gemm_M8_aligned) {
    runQ4kMatmulParity("matmul_q4k_gemm_M8", /*N=*/48, /*K=*/1024,
                       /*M=*/8, /*seed=*/0xD002U, /*tol=*/1e-2F);
}

TEST(matmul_q4k_gemm_M15_tail) {
    runQ4kMatmulParity("matmul_q4k_gemm_M15", /*N=*/32, /*K=*/1024,
                       /*M=*/15, /*seed=*/0xD003U, /*tol=*/1e-2F);
}

TEST(matmul_q4k_gemm_M17_tail_after_full) {
    runQ4kMatmulParity("matmul_q4k_gemm_M17", /*N=*/24, /*K=*/1024,
                       /*M=*/17, /*seed=*/0xD004U, /*tol=*/1e-2F);
}

TEST(matmul_q4k_gemm_realistic_prefill) {
    // Qwen-style Q4_K attention proj at prefill: K=3584, M=16.
    runQ4kMatmulParity("matmul_q4k_gemm_prefill", /*N=*/2048, /*K=*/3584,
                       /*M=*/16, /*seed=*/0xD005U, /*tol=*/2e-2F);
}

// =======================================================================
// Q8_0 GEMM (batched M>1) — Q8_0 has 32-element blocks and the simplest
// dequant, so this variant benefits most from M-fold amortisation.
// =======================================================================
namespace {

std::vector<std::uint8_t> buildQ8_0Weights(std::size_t N,
                                           std::size_t K,
                                           std::uint32_t seed) {
    const std::size_t blocksPerRow = K / 32;
    const std::size_t bytesPerRow  = blocksPerRow * 34;

    std::vector<std::uint8_t> w(N * bytesPerRow, 0);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> distQ(-63, 63);

    // Positive fp16 scale ~0.02 so |value| stays small.
    constexpr std::uint16_t kHalfSmall = 0x2400U;

    for (std::size_t n = 0; n < N; ++n) {
        for (std::size_t b = 0; b < blocksPerRow; ++b) {
            auto* block = w.data() + n * bytesPerRow + b * 34;
            std::memcpy(block, &kHalfSmall, sizeof(std::uint16_t));
            for (std::size_t l = 0; l < 32; ++l) {
                block[2 + l] = static_cast<std::uint8_t>(
                    static_cast<std::int8_t>(distQ(rng)));
            }
        }
    }
    return w;
}

void runQ8_0MatmulParity(const char*   label,
                         std::size_t   N,
                         std::size_t   K,
                         std::size_t   M,
                         std::uint32_t seed,
                         float         tol) {
    const auto w = buildQ8_0Weights(N, K, seed);
    const auto x = generateFloats(M * K, seed ^ 0x9E3779B9U);

    UsmBuf bufW(w.size());
    UsmBuf bufX(M * K * sizeof(float));
    UsmBuf bufY(M * N * sizeof(float));
    UsmBuf bufScratch(K * sizeof(float));

    std::memcpy(bufW.raw(), w.data(), w.size());
    std::memcpy(bufX.raw(), x.data(), x.size() * sizeof(float));

    fx().gmm.matmul(mimirmind::model::GgmlType::Q8_0,
                    bufW.raw(), N, K,
                    bufX.as<float>(), M,
                    bufY.as<float>(),
                    bufScratch.as<float>());

    std::vector<float> cpuY(M * N);
    std::vector<float> scratchCpu(K);
    mimirmind::compute::matmul(mimirmind::model::GgmlType::Q8_0,
                               w.data(), N, K,
                               x.data(), M,
                               cpuY.data(),
                               scratchCpu.data());

    EXPECT_ARRAY_NEAR(label, bufY.as<float>(), cpuY.data(), M * N, tol);
}

} // namespace

TEST(matmul_q8_0_gemm_M1_regression) {
    runQ8_0MatmulParity("matmul_q8_0_gemm_M1", /*N=*/32, /*K=*/1024,
                        /*M=*/1, /*seed=*/0xE001U, /*tol=*/1e-2F);
}

TEST(matmul_q8_0_gemm_M8_aligned) {
    runQ8_0MatmulParity("matmul_q8_0_gemm_M8", /*N=*/48, /*K=*/1024,
                        /*M=*/8, /*seed=*/0xE002U, /*tol=*/1e-2F);
}

TEST(matmul_q8_0_gemm_M15_tail) {
    runQ8_0MatmulParity("matmul_q8_0_gemm_M15", /*N=*/32, /*K=*/1024,
                        /*M=*/15, /*seed=*/0xE003U, /*tol=*/1e-2F);
}

TEST(matmul_q8_0_gemm_M17_tail_after_full) {
    runQ8_0MatmulParity("matmul_q8_0_gemm_M17", /*N=*/24, /*K=*/1024,
                        /*M=*/17, /*seed=*/0xE004U, /*tol=*/1e-2F);
}

TEST(matmul_q8_0_gemm_realistic_prefill) {
    // Gemma-4 Q8_0 attention proj shape: N≈2560, K=2816, M=16.
    runQ8_0MatmulParity("matmul_q8_0_gemm_prefill", /*N=*/2560, /*K=*/2816,
                        /*M=*/16, /*seed=*/0xE005U, /*tol=*/2e-2F);
}

// =======================================================================
// M8.J — multi-M GEMM at Autotune-Buckets M=64 and M=256. The autotune
// parity gate at startup only checks the largest bucket; these tests
// belt-and-braces the intermediate + large paths against the CPU
// reference so a GEMM regression at any bucket lights up before deploy.
// =======================================================================

TEST(matmul_q8_0_gemm_M64_bucket) {
    runQ8_0MatmulParity("matmul_q8_0_gemm_M64", /*N=*/2560, /*K=*/2816,
                        /*M=*/64, /*seed=*/0xE064U, /*tol=*/2e-2F);
}

TEST(matmul_q8_0_gemm_M256_bucket) {
    runQ8_0MatmulParity("matmul_q8_0_gemm_M256", /*N=*/2560, /*K=*/2816,
                        /*M=*/256, /*seed=*/0xE256U, /*tol=*/2e-2F);
}

TEST(matmul_q6k_gemm_M64_bucket) {
    runQ6kMatmulParity("matmul_q6k_gemm_M64", /*N=*/2560, /*K=*/2816,
                       /*M=*/64, /*seed=*/0xC064U, /*tol=*/2e-2F);
}

TEST(matmul_q6k_gemm_M256_bucket) {
    runQ6kMatmulParity("matmul_q6k_gemm_M256", /*N=*/2560, /*K=*/2816,
                       /*M=*/256, /*seed=*/0xC256U, /*tol=*/2e-2F);
}

TEST(matmul_q4k_gemm_M64_bucket) {
    runQ4kMatmulParity("matmul_q4k_gemm_M64", /*N=*/2048, /*K=*/3584,
                       /*M=*/64, /*seed=*/0xD064U, /*tol=*/2e-2F);
}

TEST(matmul_q4k_gemm_M256_bucket) {
    runQ4kMatmulParity("matmul_q4k_gemm_M256", /*N=*/2048, /*K=*/3584,
                       /*M=*/256, /*seed=*/0xD256U, /*tol=*/2e-2F);
}

// =======================================================================
// M8.H.1 — matmul_q8_0_vec_dp4a: DP4A matvec with int8-quantised
// activation. Compared against the plain float-X Q8_0 matmul on the
// same weights; tolerance is driven by the int8 activation quant
// error (≈ K * amax * mean|w_dequant| / 254 in the worst case).
// =======================================================================
namespace {

void runQ8_0Dp4aParity(const char*   label,
                       std::size_t   N,
                       std::size_t   K,
                       std::size_t   M,
                       std::uint32_t seed) {
    if (!fx().gmm.dp4aAvailable()) {
        std::printf("[SKIP] %s (DP4A extension unavailable on this iGPU)\n",
                    label);
        return;
    }

    const auto w = buildQ8_0Weights(N, K, seed);
    const auto x = generateFloats(M * K, seed ^ 0x9E3779B9U);

    UsmBuf bufW (w.size());
    UsmBuf bufX (M * K * sizeof(float));
    UsmBuf bufXq(M * K * sizeof(std::int8_t));
    UsmBuf bufXs(M     * sizeof(float));
    UsmBuf bufYref (M * N * sizeof(float));
    UsmBuf bufYdp4a(M * N * sizeof(float));
    UsmBuf bufScratch(K * sizeof(float));

    std::memcpy(bufW.raw(), w.data(), w.size());
    std::memcpy(bufX.raw(), x.data(), x.size() * sizeof(float));

    // Reference: plain float-X matmul_q8_0 (whichever variant the
    // dispatcher picks — matvec-loop for M=1, GEMM otherwise). This is
    // what DP4A is replacing.
    fx().gmm.matmul(mimirmind::model::GgmlType::Q8_0,
                    bufW.raw(), N, K,
                    bufX.as<float>(), M,
                    bufYref.as<float>(),
                    bufScratch.as<float>());

    // Quantise X → Xq + Xscale on the GPU, then dispatch DP4A matvec.
    fx().ops.xQuantI8Async(bufX.as<float>(), bufXq.as<std::int8_t>(),
                           bufXs.as<float>(), M, K);
    fx().gmm.matmulDp4aAsync(mimirmind::model::GgmlType::Q8_0,
                             bufXq.as<std::int8_t>(),
                             bufXs.as<float>(),
                             bufW.raw(), N, K, M,
                             bufYdp4a.as<float>());
    fx().gmm.sync();

    // Tolerance model. The DP4A output differs from the float-X output
    // because each activation is int8-quantised (max per-element error
    // amax/254). Accumulated over K weights the worst-case bound is
    // K × amax/254 × max|w_deq|; the expected magnitude is closer to
    // √K × amax/254 × rms|w_deq|. We use a mixed absolute-plus-relative
    // bound: 5 % of max|ref| plus a small floor to catch outputs that
    // round to near-zero.
    float maxRefAbs = 0.0F;
    for (std::size_t i = 0; i < M * N; ++i) {
        maxRefAbs = std::max(maxRefAbs, std::fabs(bufYref.as<float>()[i]));
    }
    const float tol = std::max(0.05F * maxRefAbs, 1e-2F);
    EXPECT_ARRAY_NEAR(label, bufYdp4a.as<float>(), bufYref.as<float>(),
                      M * N, tol);
}

} // namespace

TEST(matmul_q8_0_dp4a_M1_basic) {
    // Single-row (decode hot path). N tail check: N=40 → 5 workgroups
    // of 8 outputs each.
    runQ8_0Dp4aParity("matmul_q8_0_dp4a_M1", /*N=*/40, /*K=*/1024,
                      /*M=*/1, /*seed=*/0xF101U);
}

TEST(matmul_q8_0_dp4a_M1_gemma4_shape) {
    // Realistic Gemma-4-Q8_0 attention proj shape at decode.
    runQ8_0Dp4aParity("matmul_q8_0_dp4a_gemma4", /*N=*/2560, /*K=*/2816,
                      /*M=*/1, /*seed=*/0xF102U);
}

TEST(matmul_q8_0_dp4a_M4_perRowScale) {
    // Multiple rows — each row has its own scale, so the row loop in
    // matmulDp4aAsync must advance Xscale by 1 per iteration.
    runQ8_0Dp4aParity("matmul_q8_0_dp4a_M4", /*N=*/32, /*K=*/1024,
                      /*M=*/4, /*seed=*/0xF103U);
}

TEST(matmul_q8_0_dp4a_N_tail) {
    // N not a multiple of the outputs-per-group (8) — last workgroup
    // has idle output slots.
    runQ8_0Dp4aParity("matmul_q8_0_dp4a_Ntail", /*N=*/33, /*K=*/1024,
                      /*M=*/1, /*seed=*/0xF104U);
}

// =======================================================================
// qkv_split — verify the scatter kernel routes the fused matmul output
// into Q / K / V sub-buffers correctly, including the hasV=false path.
// =======================================================================
namespace {

void runQkvSplitParity(const char* label,
                       std::size_t M,
                       std::size_t Nq,
                       std::size_t Nkv,
                       bool        hasV) {
    const std::size_t Nfused = Nq + Nkv * (hasV ? 2 : 1);
    const auto fused = generateFloats(M * Nfused, /*seed=*/0xF001U);

    UsmBuf bufFused(M * Nfused * sizeof(float));
    UsmBuf bufQ    (M * Nq     * sizeof(float));
    UsmBuf bufK    (M * Nkv    * sizeof(float));
    UsmBuf bufV    (M * Nkv    * sizeof(float));   // unused for !hasV

    std::memcpy(bufFused.raw(), fused.data(),
                fused.size() * sizeof(float));

    // Poison the destinations so we notice under-fills.
    std::vector<float> poison(M * Nkv, -12345.0F);
    std::memcpy(bufQ.raw(), poison.data(), M * Nq  * sizeof(float));
    std::memcpy(bufK.raw(), poison.data(), M * Nkv * sizeof(float));
    if (hasV) {
        std::memcpy(bufV.raw(), poison.data(), M * Nkv * sizeof(float));
    }

    fx().ops.qkvSplitAsync(bufFused.as<float>(),
                           bufQ.as<float>(),
                           bufK.as<float>(),
                           bufV.as<float>(),
                           M, Nq, Nkv, hasV);
    fx().queue.flush();

    // CPU reference: iterate the fused layout and check each split slot.
    std::vector<float> cpuQ(M * Nq);
    std::vector<float> cpuK(M * Nkv);
    std::vector<float> cpuV(M * Nkv, 0.0F);
    for (std::size_t m = 0; m < M; ++m) {
        for (std::size_t i = 0; i < Nq; ++i) {
            cpuQ[m * Nq + i] = fused[m * Nfused + i];
        }
        for (std::size_t j = 0; j < Nkv; ++j) {
            cpuK[m * Nkv + j] = fused[m * Nfused + Nq + j];
        }
        if (hasV) {
            for (std::size_t j = 0; j < Nkv; ++j) {
                cpuV[m * Nkv + j] =
                    fused[m * Nfused + Nq + Nkv + j];
            }
        }
    }

    EXPECT_ARRAY_NEAR((std::string{label} + "/Q").c_str(),
                      bufQ.as<float>(), cpuQ.data(), M * Nq,  0.0F);
    EXPECT_ARRAY_NEAR((std::string{label} + "/K").c_str(),
                      bufK.as<float>(), cpuK.data(), M * Nkv, 0.0F);
    if (hasV) {
        EXPECT_ARRAY_NEAR((std::string{label} + "/V").c_str(),
                          bufV.as<float>(), cpuV.data(), M * Nkv, 0.0F);
    }
}

} // namespace

TEST(qkv_split_full_qkv) {
    // Standard case: Q + K + V all present.
    runQkvSplitParity("qkv_split_full", /*M=*/8, /*Nq=*/256,
                      /*Nkv=*/64, /*hasV=*/true);
}

TEST(qkv_split_alt_attention_qk_only) {
    // Gemma 4 alt-attention layers: V derived downstream, kernel only
    // scatters Q + K. hasV=false must never touch the V slot.
    runQkvSplitParity("qkv_split_qk", /*M=*/8, /*Nq=*/256,
                      /*Nkv=*/64, /*hasV=*/false);
}

TEST(qkv_split_gqa_ratio) {
    // GQA-style: Nq is a multiple of Nkv (heads / kv_heads = 8/2).
    runQkvSplitParity("qkv_split_gqa", /*M=*/4, /*Nq=*/2048,
                      /*Nkv=*/512, /*hasV=*/true);
}

TEST(qkv_split_M1_decode) {
    // Decode path passes M=1. Trivial but must not misalign.
    runQkvSplitParity("qkv_split_M1", /*M=*/1, /*Nq=*/512,
                      /*Nkv=*/128, /*hasV=*/true);
}

// =======================================================================
// Perf micro-benchmarks — GEMM (batched) vs matvec-loop (per-token).
//
// Test-only opt-in via MIMIRMIND_GPU_BENCH=1 (test env only, not in the mimirmind config schema). Off by default so the regular test
// suite stays fast and deterministic.
//
// !!! HARDWARE WARNING !!!
// The dev workstation's iGPU (verify via `gpu_tests | grep 'target
// device'`) may be a much older µarch than the production target
// (L0_TARGET_HOST, Xe-LPG on Meteor Lake, Gen13). Numbers from a
// Gen9.5 HD Graphics 630 do NOT predict Xe-LPG behaviour — different
// register file, different SLM budget, different SIMD width, different
// compiler code-gen. Only use bench numbers from L0_TARGET_HOST when
// making kernel-design decisions.
//
// Bench flow per (type, N, K, M):
//   1. Warmup: 3 iterations of GEMM to JIT-compile + warm thermals.
//   2. Baseline: N_iter × (M × matmulAsync(M_arg=1) + 1 sync) —
//      forces matvec path with a single flush per iteration.
//   3. Batched:  N_iter × matmul(M_arg=M) — uses GEMM path.
//   4. Report median-of-N_iter for each, plus speedup ratio.
// =======================================================================
namespace {

bool benchEnabled() {
    const char* v = std::getenv("MIMIRMIND_GPU_BENCH");
    return v != nullptr && v[0] != '\0' && v[0] != '0';
}

// Median of a small vector of doubles. Destructive (sorts in place) —
// caller passes by value.
double median(std::vector<double> xs) {
    if (xs.empty()) return 0.0;
    std::sort(xs.begin(), xs.end());
    const std::size_t mid = xs.size() / 2;
    return (xs.size() % 2 == 1)
        ? xs[mid]
        : 0.5 * (xs[mid - 1] + xs[mid]);
}

struct BenchResult {
    double matvecMedianMs;
    double gemmMedianMs;
};

// Times both paths for a matmul at the given shape and returns the
// medians. `W` must already be populated on the USM buffer as GgmlType-
// encoded blocks; X/Y/scratch are (re-)used by every iteration.
//
// Fair-baseline note: the matvec-loop path calls matmulAsync M times
// but syncs only once per iteration (matching how the pre-GEMM
// InferenceEngine actually issued matmuls — one command-list flush at
// block end, not per-token).
BenchResult benchMatmulShape(mimirmind::model::GgmlType type,
                             const void* W,
                             std::size_t N,
                             std::size_t K,
                             std::size_t M,
                             const float* X,
                             float*       Y,
                             float*       scratch,
                             int          nIter = 8,
                             int          nWarmup = 3) {
    using clk = std::chrono::steady_clock;
    auto& gmm = fx().gmm;

    // Warmup — dispatch the GEMM path a few times so the SPV is JIT'd
    // and the iGPU clock has ramped up.
    for (int i = 0; i < nWarmup; ++i) {
        gmm.matmul(type, W, N, K, X, M, Y, scratch);
    }

    // Baseline: M × matmulAsync(M_arg=1) + 1 sync. Each single-row
    // call dispatches through the matvec kernel per our M==1 fast-path.
    // Only one flush at the end so we measure kernel throughput, not
    // per-token command-list overhead.
    std::vector<double> matvecMs;
    matvecMs.reserve(static_cast<std::size_t>(nIter));
    for (int it = 0; it < nIter; ++it) {
        const auto t0 = clk::now();
        for (std::size_t m = 0; m < M; ++m) {
            gmm.matmulAsync(type, W, N, K, X + m * K, /*M=*/1,
                            Y + m * N, scratch);
        }
        gmm.sync();
        const auto t1 = clk::now();
        const std::chrono::duration<double, std::milli> dt = t1 - t0;
        matvecMs.push_back(dt.count());
    }

    // Batched: 1 × matmul(M_arg=M). Uses the GEMM kernel. Same one
    // flush at the end via matmul()'s internal sync.
    std::vector<double> gemmMs;
    gemmMs.reserve(static_cast<std::size_t>(nIter));
    for (int it = 0; it < nIter; ++it) {
        const auto t0 = clk::now();
        gmm.matmul(type, W, N, K, X, M, Y, scratch);
        const auto t1 = clk::now();
        const std::chrono::duration<double, std::milli> dt = t1 - t0;
        gemmMs.push_back(dt.count());
    }

    return {median(std::move(matvecMs)), median(std::move(gemmMs))};
}

// Print a table row: "M=  16  matvec: 170.4 ms  gemm:  15.3 ms  speedup: 11.14×"
void printBenchRow(std::size_t M, BenchResult r) {
    const double speedup =
        (r.gemmMedianMs > 0.0) ? (r.matvecMedianMs / r.gemmMedianMs) : 0.0;
    std::printf("  M=%4zu  matvec: %8.2f ms   gemm: %8.2f ms   speedup: %5.2fx\n",
                M, r.matvecMedianMs, r.gemmMedianMs, speedup);
}

} // namespace

TEST(bench_matmul_q6k_prefill_shape) {
    if (!benchEnabled()) return;

    // Gemma-4 Q6_K attention-proj shape.
    constexpr std::size_t N = 2560;
    constexpr std::size_t K = 2816;

    const auto W = buildQ6kWeights(N, K, /*seed=*/0xB001U);
    const auto X = generateFloats(64 * K, /*seed=*/0xB101U);

    UsmBuf bufW(W.size());
    UsmBuf bufX(64 * K * sizeof(float));
    UsmBuf bufY(64 * N * sizeof(float));
    UsmBuf bufScratch(K * sizeof(float));
    std::memcpy(bufW.raw(), W.data(), W.size());
    std::memcpy(bufX.raw(), X.data(), X.size() * sizeof(float));

    std::printf("\n[bench] matmul_q6k  N=%zu  K=%zu  (Xe-LPG iGPU)\n", N, K);
    for (const std::size_t M : {std::size_t{4}, std::size_t{8},
                                std::size_t{16}, std::size_t{64}}) {
        const auto r = benchMatmulShape(mimirmind::model::GgmlType::Q6_K,
                                        bufW.raw(), N, K, M,
                                        bufX.as<float>(),
                                        bufY.as<float>(),
                                        bufScratch.as<float>());
        printBenchRow(M, r);
    }
}

TEST(bench_matmul_q4k_prefill_shape) {
    if (!benchEnabled()) return;

    // Qwen-style Q4_K attention proj shape.
    constexpr std::size_t N = 2048;
    constexpr std::size_t K = 3584;

    const auto W = buildQ4kWeights(N, K, /*seed=*/0xB002U);
    const auto X = generateFloats(64 * K, /*seed=*/0xB102U);

    UsmBuf bufW(W.size());
    UsmBuf bufX(64 * K * sizeof(float));
    UsmBuf bufY(64 * N * sizeof(float));
    UsmBuf bufScratch(K * sizeof(float));
    std::memcpy(bufW.raw(), W.data(), W.size());
    std::memcpy(bufX.raw(), X.data(), X.size() * sizeof(float));

    std::printf("\n[bench] matmul_q4k  N=%zu  K=%zu  (Xe-LPG iGPU)\n", N, K);
    for (const std::size_t M : {std::size_t{4}, std::size_t{8},
                                std::size_t{16}, std::size_t{64}}) {
        const auto r = benchMatmulShape(mimirmind::model::GgmlType::Q4_K,
                                        bufW.raw(), N, K, M,
                                        bufX.as<float>(),
                                        bufY.as<float>(),
                                        bufScratch.as<float>());
        printBenchRow(M, r);
    }
}

TEST(bench_matmul_q8_0_prefill_shape) {
    if (!benchEnabled()) return;

    // Gemma-4 Q8_0 attention proj shape.
    constexpr std::size_t N = 2560;
    constexpr std::size_t K = 2816;

    const auto W = buildQ8_0Weights(N, K, /*seed=*/0xB003U);
    const auto X = generateFloats(64 * K, /*seed=*/0xB103U);

    UsmBuf bufW(W.size());
    UsmBuf bufX(64 * K * sizeof(float));
    UsmBuf bufY(64 * N * sizeof(float));
    UsmBuf bufScratch(K * sizeof(float));
    std::memcpy(bufW.raw(), W.data(), W.size());
    std::memcpy(bufX.raw(), X.data(), X.size() * sizeof(float));

    std::printf("\n[bench] matmul_q8_0  N=%zu  K=%zu  (Xe-LPG iGPU)\n", N, K);
    for (const std::size_t M : {std::size_t{4}, std::size_t{8},
                                std::size_t{16}, std::size_t{64}}) {
        const auto r = benchMatmulShape(mimirmind::model::GgmlType::Q8_0,
                                        bufW.raw(), N, K, M,
                                        bufX.as<float>(),
                                        bufY.as<float>(),
                                        bufScratch.as<float>());
        printBenchRow(M, r);
    }
}

int main() {
    return mm::test::run();
}