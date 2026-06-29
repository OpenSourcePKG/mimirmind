// Pure-CPU unit tests for the arch-related code that lives outside the
// GPU pipeline:
//   - `isSupportedArchitecture()` — pure name predicate exposed by
//     ArchBackend for early-fail diagnostics.
//   - `compute::softmaxRows`     — numerically stable row-wise softmax,
//     causal-mask aware. Used by attention.
//   - `compute::multiHeadAttention` — CPU GQA-aware causal MHA. Pure
//     pointer math; testable end-to-end with hand-crafted Q/K/V.
//
// `runBlock` of the concrete backends requires real USM + kernel modules
// → not in scope here; that's an integration test.

#include "TestFramework.hpp"

#include "compute/Attention.hpp"
#include "compute/Softmax.hpp"
#include "runtime/arch/ArchBackend.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <string_view>
#include <vector>

// =======================================================================
// isSupportedArchitecture
// =======================================================================

TEST(arch_supportedNames) {
    using mimirmind::runtime::arch::isSupportedArchitecture;
    EXPECT_TRUE(isSupportedArchitecture("qwen2"));
    EXPECT_TRUE(isSupportedArchitecture("gemma4"));
}

TEST(arch_unsupportedNames) {
    using mimirmind::runtime::arch::isSupportedArchitecture;
    EXPECT_TRUE(!isSupportedArchitecture(""));
    EXPECT_TRUE(!isSupportedArchitecture("qwen3"));
    EXPECT_TRUE(!isSupportedArchitecture("gemma3"));
    EXPECT_TRUE(!isSupportedArchitecture("llama"));
    EXPECT_TRUE(!isSupportedArchitecture("Qwen2"));   // case-sensitive
    EXPECT_TRUE(!isSupportedArchitecture("Gemma4"));
}

// =======================================================================
// compute::softmaxRows
// =======================================================================

TEST(softmax_singleRow_uniform) {
    // All equal inputs → uniform distribution.
    std::array<float, 4> data{1.0F, 1.0F, 1.0F, 1.0F};
    mimirmind::compute::softmaxRows(data.data(), 1, 4, nullptr);

    for (float v : data) {
        EXPECT_NEAR(v, 0.25F, 1e-6F);
    }
}

TEST(softmax_singleRow_oneDominant) {
    // One huge value dominates → probability mass ~1 on that index.
    std::array<float, 3> data{0.0F, 100.0F, 0.0F};
    mimirmind::compute::softmaxRows(data.data(), 1, 3, nullptr);

    EXPECT_NEAR(data[0], 0.0F, 1e-6F);
    EXPECT_NEAR(data[1], 1.0F, 1e-6F);
    EXPECT_NEAR(data[2], 0.0F, 1e-6F);
}

TEST(softmax_numericallyStable_largeValues) {
    // Without max-subtract this would overflow exp().
    std::array<float, 3> data{1000.0F, 1000.0F, 1000.0F};
    mimirmind::compute::softmaxRows(data.data(), 1, 3, nullptr);

    for (float v : data) {
        EXPECT_NEAR(v, 1.0F / 3.0F, 1e-6F);
    }
}

TEST(softmax_multipleRows) {
    // Two independent rows, normalised separately.
    std::array<float, 6> data{
        1.0F, 1.0F, 1.0F,    // row 0 → all 1/3
        0.0F, 100.0F, 0.0F,  // row 1 → near [0, 1, 0]
    };
    mimirmind::compute::softmaxRows(data.data(), 2, 3, nullptr);

    EXPECT_NEAR(data[0], 1.0F / 3.0F, 1e-6F);
    EXPECT_NEAR(data[1], 1.0F / 3.0F, 1e-6F);
    EXPECT_NEAR(data[2], 1.0F / 3.0F, 1e-6F);
    EXPECT_NEAR(data[3], 0.0F,        1e-6F);
    EXPECT_NEAR(data[4], 1.0F,        1e-6F);
    EXPECT_NEAR(data[5], 0.0F,        1e-6F);
}

TEST(softmax_causalMask_firstRowOnlyAttendsFirst) {
    // Two rows of size 4. Row 0 attends to position 0 only; row 1 to
    // positions 0..1; the rest must be zero-ed.
    std::array<float, 8> data{
        5.0F, 5.0F, 5.0F, 5.0F,
        2.0F, 4.0F, 9.0F, 9.0F,
    };
    const std::array<std::size_t, 2> live{1, 2};
    mimirmind::compute::softmaxRows(data.data(), 2, 4, live.data());

    // Row 0: live=1 → [1, 0, 0, 0]
    EXPECT_NEAR(data[0], 1.0F, 1e-6F);
    EXPECT_NEAR(data[1], 0.0F, 1e-6F);
    EXPECT_NEAR(data[2], 0.0F, 1e-6F);
    EXPECT_NEAR(data[3], 0.0F, 1e-6F);

    // Row 1: live=2 → softmax of [2, 4], rest zero
    const float maxv = 4.0F;
    const float e0   = std::exp(2.0F - maxv);
    const float e1   = std::exp(4.0F - maxv);
    const float sum  = e0 + e1;
    EXPECT_NEAR(data[4], e0 / sum, 1e-6F);
    EXPECT_NEAR(data[5], e1 / sum, 1e-6F);
    EXPECT_NEAR(data[6], 0.0F,     1e-6F);
    EXPECT_NEAR(data[7], 0.0F,     1e-6F);
}

TEST(softmax_zeroLiveRowAllZeros) {
    std::array<float, 3> data{1.0F, 2.0F, 3.0F};
    const std::array<std::size_t, 1> live{0};
    mimirmind::compute::softmaxRows(data.data(), 1, 3, live.data());

    EXPECT_NEAR(data[0], 0.0F, 0.0F);
    EXPECT_NEAR(data[1], 0.0F, 0.0F);
    EXPECT_NEAR(data[2], 0.0F, 0.0F);
}

// =======================================================================
// compute::multiHeadAttention
// =======================================================================

// Smallest possible case: 1 token, 1 head, head_dim=2. Q attends to
// only itself (T_k = 1) so softmax([0.707]) = [1.0] and out = V[0].
TEST(mha_singleToken_singleHead) {
    constexpr std::size_t T = 1;
    constexpr std::size_t nHeads = 1;
    constexpr std::size_t nKvHeads = 1;
    constexpr std::size_t headDim = 2;

    const std::array<float, T * nHeads * headDim>   q{1.0F, 0.0F};
    const std::array<float, T * nKvHeads * headDim> k{1.0F, 0.0F};
    const std::array<float, T * nKvHeads * headDim> v{42.0F, 99.0F};
    std::array<float, T> scratch{};
    std::array<float, T * nHeads * headDim> out{};

    mimirmind::compute::multiHeadAttention(
        q.data(), k.data(), v.data(),
        T, T, nHeads, nKvHeads, headDim,
        /*positionOffset*/ 0,
        scratch.data(), out.data());

    // softmax([dot(q,k)/sqrt(2)]) = [1.0], out = V[0] = [42, 99]
    EXPECT_NEAR(out[0], 42.0F, 1e-5F);
    EXPECT_NEAR(out[1], 99.0F, 1e-5F);
}

// T=2, causal mask: row 0 sees only K[0], row 1 sees K[0] and K[1].
// Q[0] and Q[1] are orthogonal one-hot vectors so the scores reduce to
// dot products of 0 or 1.
TEST(mha_twoTokens_causalMask) {
    constexpr std::size_t T = 2;
    constexpr std::size_t nHeads = 1;
    constexpr std::size_t nKvHeads = 1;
    constexpr std::size_t headDim = 2;

    const std::array<float, T * nHeads * headDim>   q{
        1.0F, 0.0F,
        0.0F, 1.0F,
    };
    const std::array<float, T * nKvHeads * headDim> k{
        1.0F, 0.0F,   // K[0] = Q[0]
        0.0F, 1.0F,   // K[1] = Q[1]
    };
    const std::array<float, T * nKvHeads * headDim> v{
        10.0F, 100.0F,
        20.0F, 200.0F,
    };
    std::array<float, T> scratch{};
    std::array<float, T * nHeads * headDim> out{};

    mimirmind::compute::multiHeadAttention(
        q.data(), k.data(), v.data(),
        T, T, nHeads, nKvHeads, headDim,
        /*positionOffset*/ 0,
        scratch.data(), out.data());

    // Token 0: kMax=1, softmax over 1 entry = [1.0], out = V[0]
    EXPECT_NEAR(out[0], 10.0F,  1e-5F);
    EXPECT_NEAR(out[1], 100.0F, 1e-5F);

    // Token 1: kMax=2, scores = [Q1.K0, Q1.K1] / sqrt(2) = [0, 1/sqrt(2)]
    const float invSqrt = 1.0F / std::sqrt(2.0F);
    const float score0  = 0.0F;
    const float score1  = invSqrt;
    const float maxs    = score1;
    const float e0      = std::exp(score0 - maxs);
    const float e1      = std::exp(score1 - maxs);
    const float sum     = e0 + e1;
    const float w0      = e0 / sum;
    const float w1      = e1 / sum;
    EXPECT_NEAR(out[2], w0 * 10.0F + w1 * 20.0F,  1e-5F);
    EXPECT_NEAR(out[3], w0 * 100.0F + w1 * 200.0F, 1e-5F);
}

// GQA: nHeads=2, nKvHeads=1 → both query heads read from KV head 0.
// Q heads point at orthogonal K coords; both should produce the same V[0]
// blend at T=1 (only 1 KV position to attend to).
TEST(mha_groupedQuery_T1) {
    constexpr std::size_t T = 1;
    constexpr std::size_t nHeads = 2;
    constexpr std::size_t nKvHeads = 1;
    constexpr std::size_t headDim = 2;

    // Q: [head0=(1,0), head1=(0,1)]
    const std::array<float, T * nHeads * headDim>   q{
        1.0F, 0.0F,
        0.0F, 1.0F,
    };
    // K, V are single-head (nKvHeads=1)
    const std::array<float, T * nKvHeads * headDim> k{1.0F, 1.0F};
    const std::array<float, T * nKvHeads * headDim> v{42.0F, 99.0F};
    std::array<float, T> scratch{};
    std::array<float, T * nHeads * headDim> out{};

    mimirmind::compute::multiHeadAttention(
        q.data(), k.data(), v.data(),
        T, T, nHeads, nKvHeads, headDim,
        /*positionOffset*/ 0,
        scratch.data(), out.data());

    // Both heads attend only to KV pos 0 (T_k=1 → softmax=[1.0]).
    // out[head_q] = V[0] regardless of which Q head, because the only
    // weight is 1.0.
    EXPECT_NEAR(out[0], 42.0F, 1e-5F);   // head 0
    EXPECT_NEAR(out[1], 99.0F, 1e-5F);
    EXPECT_NEAR(out[2], 42.0F, 1e-5F);   // head 1 (same V via GQA)
    EXPECT_NEAR(out[3], 99.0F, 1e-5F);
}

// Decode-mode call shape: T_q=1, T_k=cache_length+1, positionOffset=cache_length.
// The new query at position 2 should attend to K[0..2].
TEST(mha_decodeMode) {
    constexpr std::size_t T_q = 1;
    constexpr std::size_t T_k = 3;
    constexpr std::size_t nHeads = 1;
    constexpr std::size_t nKvHeads = 1;
    constexpr std::size_t headDim = 1;

    const std::array<float, T_q * nHeads * headDim> q{1.0F};
    const std::array<float, T_k * nKvHeads * headDim> k{1.0F, 1.0F, 1.0F};
    const std::array<float, T_k * nKvHeads * headDim> v{10.0F, 20.0F, 30.0F};
    std::array<float, T_k> scratch{};
    std::array<float, T_q * nHeads * headDim> out{};

    mimirmind::compute::multiHeadAttention(
        q.data(), k.data(), v.data(),
        T_q, T_k, nHeads, nKvHeads, headDim,
        /*positionOffset*/ 2,
        scratch.data(), out.data());

    // All three K rows are 1.0 and Q=1.0, scores = [1, 1, 1] / sqrt(1) = 1,1,1
    // softmax → uniform 1/3, out = (10+20+30)/3 = 20.0
    EXPECT_NEAR(out[0], 20.0F, 1e-5F);
}

int main() {
    return mm::test::run();
}