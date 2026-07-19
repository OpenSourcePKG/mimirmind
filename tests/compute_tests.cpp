// Pure-CPU unit tests for the compute layer:
//   - compute::Sampler          — greedy + temperature/top-k/top-p sampling
//   - compute::matmul           — scalar CPU matmul with double accumulator
//   - compute::embeddingLookup  — quant-type-aware token row lookup
//   - runtime::NGramDrafter     — Prompt-Lookup Decoding tail-match core
//
// No Level Zero, no SPV kernels, no model file required.

#include "TestFramework.hpp"

#include "compute/Dequant.hpp"
#include "compute/Embedding.hpp"
#include "compute/Matmul.hpp"
#include "compute/Sampling.hpp"
#include "compute/quant/Q3K.hpp"
#include "compute/quant/Q6K.hpp"
#include "compute/quant/Q8_0.hpp"
#include "core/gguf/GgufTypes.hpp"
#include "runtime/spec/NGramDrafter.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <set>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

// fp16 bit patterns used in hand-crafted Q-type blocks.
constexpr std::uint16_t kHalfOne  = 0x3C00U;   // 1.0
constexpr std::uint16_t kHalfHalf = 0x3800U;   // 0.5
constexpr std::uint16_t kHalfZero = 0x0000U;

void writeHalfBits(std::uint8_t* dst, std::uint16_t bits) {
    std::memcpy(dst, &bits, sizeof(bits));
}

} // namespace

// =======================================================================
// compute::Sampler
// =======================================================================

TEST(sampler_greedy_alwaysArgmax) {
    // temperature defaults to 0 → greedy / argmax / no RNG.
    mimirmind::compute::Sampler s;
    const std::array<float, 5> logits{1.0F, 5.0F, 3.0F, -2.0F, 4.0F};
    const mimirmind::compute::SamplingParams params{};

    const auto id = s.sample(logits, params);
    EXPECT_EQ(id, 1);   // index of 5.0
}

TEST(sampler_greedy_topK1_forcesArgmax) {
    // topK == 1 short-circuits to argmax regardless of temperature.
    mimirmind::compute::Sampler s(42);
    const std::array<float, 4> logits{2.0F, 8.0F, 0.0F, 7.0F};
    mimirmind::compute::SamplingParams params{};
    params.temperature = 1.0F;
    params.topK        = 1;

    const auto id = s.sample(logits, params);
    EXPECT_EQ(id, 1);
}

TEST(sampler_seedDeterminism_sameSeedSameDraw) {
    // Two samplers seeded identically must produce the same sequence
    // of draws on the same logits.
    const std::array<float, 6> logits{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    mimirmind::compute::SamplingParams params{};
    params.temperature = 0.8F;
    params.seed        = 12345;

    mimirmind::compute::Sampler s1(params.seed);
    mimirmind::compute::Sampler s2(params.seed);

    for (int i = 0; i < 20; ++i) {
        const auto a = s1.sample(logits, params);
        const auto b = s2.sample(logits, params);
        EXPECT_EQ(a, b);
    }
}

TEST(sampler_topK_drawsOnlyFromTopK) {
    // Set up logits so the top-3 (indices 0, 4, 2) clearly dominate
    // and others are practically masked out.
    const std::array<float, 8> logits{
        10.0F, -50.0F, 9.0F, -50.0F,
         9.5F, -50.0F, -50.0F, -50.0F,
    };
    mimirmind::compute::SamplingParams params{};
    params.temperature = 1.0F;
    params.topK        = 3;
    params.seed        = 7;

    mimirmind::compute::Sampler s(params.seed);
    const std::set<std::int32_t> allowed{0, 2, 4};

    for (int i = 0; i < 200; ++i) {
        const auto id = s.sample(logits, params);
        EXPECT_TRUE(allowed.contains(id));
    }
}

TEST(sampler_topK_zero_doesNotFilter) {
    // topK=0 means disabled — full vocab participates. Verify by drawing
    // from a uniform distribution and observing values across the range.
    const std::array<float, 5> logits{0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    mimirmind::compute::SamplingParams params{};
    params.temperature = 1.0F;
    params.topK        = 0;
    params.seed        = 3;

    mimirmind::compute::Sampler s(params.seed);
    std::set<std::int32_t> seen;
    for (int i = 0; i < 200; ++i) {
        seen.insert(s.sample(logits, params));
    }
    // With uniform logits and 200 draws over 5 classes, all 5 should appear.
    EXPECT_EQ(seen.size(), 5U);
}

TEST(sampler_topP_drawsOnlyFromHighProbSet) {
    // Logits where indices 0 and 1 together carry ~99% of the mass at
    // T=1. With topP=0.9, the nucleus is {0, 1}; everything else must
    // never be drawn.
    const std::array<float, 5> logits{10.0F, 9.7F, -10.0F, -10.0F, -10.0F};
    mimirmind::compute::SamplingParams params{};
    params.temperature = 1.0F;
    params.topP        = 0.9F;
    params.seed        = 99;

    mimirmind::compute::Sampler s(params.seed);
    const std::set<std::int32_t> allowed{0, 1};

    for (int i = 0; i < 200; ++i) {
        const auto id = s.sample(logits, params);
        EXPECT_TRUE(allowed.contains(id));
    }
}

TEST(sampler_topP_one_doesNotFilter) {
    // topP=1.0 means disabled.
    const std::array<float, 5> logits{0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    mimirmind::compute::SamplingParams params{};
    params.temperature = 1.0F;
    params.topP        = 1.0F;
    params.seed        = 4;

    mimirmind::compute::Sampler s(params.seed);
    std::set<std::int32_t> seen;
    for (int i = 0; i < 200; ++i) {
        seen.insert(s.sample(logits, params));
    }
    EXPECT_EQ(seen.size(), 5U);
}

TEST(sampler_temperature_veryLow_approachesArgmax) {
    // At a tiny temperature the distribution becomes near-degenerate at
    // the argmax. Statistical: over many draws the argmax should win
    // virtually all of them.
    const std::array<float, 4> logits{1.0F, 1.0F, 5.0F, 1.0F};
    mimirmind::compute::SamplingParams params{};
    params.temperature = 0.01F;     // very sharp
    params.seed        = 17;

    mimirmind::compute::Sampler s(params.seed);
    int correct = 0;
    constexpr int kN = 200;
    for (int i = 0; i < kN; ++i) {
        if (s.sample(logits, params) == 2) ++correct;
    }
    // Allow some slop in case of numerical drift; correct should be >= 195.
    EXPECT_TRUE(correct >= 195);
}

TEST(sampler_emptyLogits_throws) {
    mimirmind::compute::Sampler s;
    const std::array<float, 0> logits{};
    const mimirmind::compute::SamplingParams params{};
    try {
        s.sample(std::span<const float>{logits.data(), logits.size()},
                 params);
        EXPECT_TRUE(false && "expected throw");
    } catch (const std::runtime_error&) {
        // expected
    }
}

// -----------------------------------------------------------------------
// M7f — repetition / frequency / presence penalties
// -----------------------------------------------------------------------

TEST(sampler_penalty_frequencyDivertsGreedyChoice) {
    // Two logits, index 0 slightly better than index 1. Greedy picks 0
    // when no history exists.
    const std::array<float, 2> logits{5.0F, 4.5F};
    mimirmind::compute::Sampler s;
    mimirmind::compute::SamplingParams p{};
    p.temperature       = 0.0F;   // greedy
    p.frequencyPenalty  = 2.0F;   // strong subtractive penalty
    p.penaltyWindow     = 64U;

    // Empty history — no penalty, greedy picks 0.
    EXPECT_EQ(s.sample(logits, p), 0);

    // History has index 0 appearing once. -2.0 penalty subtracts from
    // its logit → 5.0 - 2.0 = 3.0, less than 4.5 → argmax flips to 1.
    const std::array<std::int32_t, 1> hist1{0};
    EXPECT_EQ(s.sample(logits, std::span<const std::int32_t>{hist1}, p), 1);
}

TEST(sampler_penalty_repetitionDivertsGreedyChoice) {
    // Same setup as above but using multiplicative repetition_penalty.
    const std::array<float, 2> logits{5.0F, 3.0F};
    mimirmind::compute::Sampler s;
    mimirmind::compute::SamplingParams p{};
    p.temperature       = 0.0F;
    p.repetitionPenalty = 2.0F;   // divides positive logits by 2
    p.penaltyWindow     = 64U;

    // Empty history — greedy picks 0 (5.0 > 3.0).
    EXPECT_EQ(s.sample(logits, p), 0);

    // History has index 0 → 5.0 / 2.0 = 2.5 < 3.0, argmax flips to 1.
    const std::array<std::int32_t, 1> hist{0};
    EXPECT_EQ(s.sample(logits, std::span<const std::int32_t>{hist}, p), 1);
}

TEST(sampler_penalty_presenceIsBinary) {
    // Presence penalty triggers on any count > 0, not count.
    const std::array<float, 2> logits{5.0F, 4.5F};
    mimirmind::compute::Sampler s;
    mimirmind::compute::SamplingParams p{};
    p.temperature      = 0.0F;
    p.presencePenalty  = 1.0F;
    p.penaltyWindow    = 64U;

    // History has index 0 once. 5.0 - 1.0 = 4.0 < 4.5 → flip.
    const std::array<std::int32_t, 1> hist1{0};
    EXPECT_EQ(s.sample(logits, std::span<const std::int32_t>{hist1}, p), 1);

    // History has index 0 five times. Presence penalty applied once (binary),
    // not five times. Result unchanged from the single-hit case.
    const std::array<std::int32_t, 5> hist5{0, 0, 0, 0, 0};
    EXPECT_EQ(s.sample(logits, std::span<const std::int32_t>{hist5}, p), 1);
}

TEST(sampler_penalty_windowRestrictsHistory) {
    // penaltyWindow=2 → only the last two tokens are considered.
    const std::array<float, 2> logits{5.0F, 4.5F};
    mimirmind::compute::Sampler s;
    mimirmind::compute::SamplingParams p{};
    p.temperature       = 0.0F;
    p.frequencyPenalty  = 2.0F;
    p.penaltyWindow     = 2U;

    // History has index 0 way back, then only index 1 recently. Window
    // only sees {1, 1} → count(0)==0, no penalty on 0, greedy picks 0.
    const std::array<std::int32_t, 4> hist{0, 0, 1, 1};
    EXPECT_EQ(s.sample(logits, std::span<const std::int32_t>{hist}, p), 0);
}

TEST(sampler_penalty_zeroWindowDisables) {
    // penaltyWindow=0 means "no history considered" regardless of penalty
    // strength. Verifies the operator-side kill switch.
    const std::array<float, 2> logits{5.0F, 4.5F};
    mimirmind::compute::Sampler s;
    mimirmind::compute::SamplingParams p{};
    p.temperature       = 0.0F;
    p.frequencyPenalty  = 5.0F;     // would flip decision if applied
    p.penaltyWindow     = 0U;

    const std::array<std::int32_t, 3> hist{0, 0, 0};
    EXPECT_EQ(s.sample(logits, std::span<const std::int32_t>{hist}, p), 0);
}

TEST(sampler_penalty_neutralParamsAreNoOp) {
    // Default penalty values are neutral (rep=1.0, freq=0, pres=0).
    // History with a repeat token shouldn't change greedy choice.
    const std::array<float, 2> logits{5.0F, 4.5F};
    mimirmind::compute::Sampler s;
    mimirmind::compute::SamplingParams p{};
    p.temperature   = 0.0F;
    p.penaltyWindow = 64U;
    // repetitionPenalty=1.0, frequencyPenalty=0, presencePenalty=0 by default

    const std::array<std::int32_t, 3> hist{0, 0, 0};
    EXPECT_EQ(s.sample(logits, std::span<const std::int32_t>{hist}, p), 0);
}

// -----------------------------------------------------------------------
// Gemma-4 final-logit softcap (M-Softcap)
// -----------------------------------------------------------------------

TEST(sampler_softcap_greedyIsArgmaxInvariant) {
    // Softcap is monotonic (cap * tanh(x/cap)), so greedy argmax must
    // not shift regardless of whether the cap is applied. Verifies the
    // core correctness invariant that lets non-penalty greedy stay
    // bit-identical for token identity (values differ, choice does not).
    const std::array<float, 4> logits{45.0F, 50.0F, 12.0F, -30.0F};
    mimirmind::compute::Sampler s;
    mimirmind::compute::SamplingParams p{};
    p.temperature = 0.0F;

    // No softcap.
    EXPECT_EQ(s.sample(logits, p), 1);

    // Softcap 30 — same argmax.
    p.finalLogitSoftcap = 30.0F;
    EXPECT_EQ(s.sample(logits, p), 1);
}

TEST(sampler_softcap_zeroDisables) {
    // finalLogitSoftcap == 0 must fall through to the pre-fix behaviour:
    // no scratch buffer, no tanh calls, no allocation on the fast path.
    // Sampling result identical to a run without the field.
    const std::array<float, 3> logits{50.0F, 40.0F, 30.0F};
    mimirmind::compute::Sampler s;
    mimirmind::compute::SamplingParams p{};
    p.temperature       = 0.0F;
    p.finalLogitSoftcap = 0.0F;

    EXPECT_EQ(s.sample(logits, p), 0);
}

TEST(sampler_softcap_precedesFreqPenalty) {
    // Order-dependence: penalty AFTER softcap can produce a different
    // argmax than penalty on raw logits. This test constructs the flip.
    //
    //   raw     = {50, 45}
    //   raw + penalty(count={5, 1}, freq=1)      = {45,  44}  → argmax 0
    //   softcap(30) + penalty(count={5, 1}, freq=1):
    //     softcap gives ~{28.94, 27.15}
    //     minus {5, 1}                            ≈ {23.94, 26.15} → argmax 1
    //
    // The 1 answer is what target parity requires — llama.cpp applies the
    // cap in the compute graph, penalties in the sampler, in that order.
    const std::array<float, 2> logits{50.0F, 45.0F};
    mimirmind::compute::Sampler s;
    mimirmind::compute::SamplingParams p{};
    p.temperature       = 0.0F;
    p.frequencyPenalty  = 1.0F;
    p.penaltyWindow     = 64U;

    // History: index 0 appears 5×, index 1 appears once.
    const std::array<std::int32_t, 6> hist{0, 0, 0, 0, 0, 1};

    // Without softcap → penalty on raw → argmax stays 0.
    EXPECT_EQ(s.sample(logits, std::span<const std::int32_t>{hist}, p), 0);

    // With softcap 30 → softcap-then-penalty flips to 1. This is the
    // regression gate for the ordering bug that motivated the fix.
    p.finalLogitSoftcap = 30.0F;
    EXPECT_EQ(s.sample(logits, std::span<const std::int32_t>{hist}, p), 1);
}

TEST(sampler_softcap_softensSamplingDistribution) {
    // Softcap is argmax-invariant so the top-1 rank is preserved, but
    // it *does* narrow the logit gap. This test picks a regime where
    // that narrowing shows up in draw counts: logits {60, 40} with
    // temperature 5.
    //
    // No softcap:  scaled logits {12, 8} → softmax(0) ≈ 0.982.
    // Softcap 30:  softcap gives {29.19, 26.03} (tanh saturates),
    //              scaled {5.84, 5.21} → softmax(0) ≈ 0.653.
    //
    // With 500 draws at seed 42 we expect ~9 hits of token 1 without
    // softcap and ~170 with. The gap is enormous; even a loose bound
    // catches a broken softcap wiring.
    const std::array<float, 2> logits{60.0F, 40.0F};
    mimirmind::compute::SamplingParams p{};
    p.temperature = 5.0F;
    p.topK        = 2U;
    p.seed        = 42ULL;

    auto countTokenOne = [&](float cap) {
        p.finalLogitSoftcap = cap;
        mimirmind::compute::Sampler s(p.seed);
        int n = 0;
        for (int i = 0; i < 500; ++i) {
            if (s.sample(logits, p) == 1) {
                ++n;
            }
        }
        return n;
    };

    const int noCap    = countTokenOne(0.0F);
    const int withCap  = countTokenOne(30.0F);

    // Sanity floor / ceiling on both. Loose bounds to survive minor
    // RNG-state churn across libstdc++ versions.
    EXPECT_TRUE(noCap   < 40);          // ~9 expected
    EXPECT_TRUE(withCap > 80);          // ~170 expected

    // The signal we actually care about: softcap must widen the tail.
    // At minimum, softcap must produce strictly more token-1 draws than
    // the no-cap baseline. If this fails, softcap is either not applied
    // or applied after softmax (which would be a wiring bug).
    EXPECT_TRUE(withCap > noCap);
}

// =======================================================================
// compute::matmul (CPU, double accumulator)
// =======================================================================

// Matmul layout (per Matmul.hpp):
//   Y [M, N] = X [M, K] · W^T,  W is [N, K] in GGUF layout (rows of K
//   elements per output channel).
//
// F32 weights: blockSize=1, typeSize=4. So W is just float[N * K] with
// the n-th row at offset n*K.
TEST(matmul_f32_simpleDotProducts) {
    constexpr std::size_t M = 1;
    constexpr std::size_t N = 2;
    constexpr std::size_t K = 3;

    // W[0] = [1, 2, 3], W[1] = [4, 5, 6]
    // X[0] = [1, 1, 1]  → Y[0] = [1+2+3, 4+5+6] = [6, 15]
    const std::array<float, N * K> W{1.0F, 2.0F, 3.0F,
                                      4.0F, 5.0F, 6.0F};
    const std::array<float, M * K> X{1.0F, 1.0F, 1.0F};
    std::array<float, M * N>       Y{};
    std::array<float, K>           scratch{};

    mimirmind::compute::matmul(mimirmind::core::gguf::GgmlType::F32,
                               W.data(), N, K,
                               X.data(), M, Y.data(),
                               scratch.data());

    EXPECT_NEAR(Y[0],  6.0F, 0.0F);
    EXPECT_NEAR(Y[1], 15.0F, 0.0F);
}

TEST(matmul_f32_multipleRows) {
    constexpr std::size_t M = 2;
    constexpr std::size_t N = 2;
    constexpr std::size_t K = 4;

    // W[0] = [1, 0, 1, 0]  W[1] = [0, 1, 0, 1]
    const std::array<float, N * K> W{1.0F, 0.0F, 1.0F, 0.0F,
                                      0.0F, 1.0F, 0.0F, 1.0F};
    // X[0] = [10, 20, 30, 40] → Y[0,0]=10+30=40, Y[0,1]=20+40=60
    // X[1] = [1, 2, 3, 4]     → Y[1,0]=1+3=4,    Y[1,1]=2+4=6
    const std::array<float, M * K> X{10.0F, 20.0F, 30.0F, 40.0F,
                                       1.0F,  2.0F,  3.0F,  4.0F};
    std::array<float, M * N> Y{};
    std::array<float, K>     scratch{};

    mimirmind::compute::matmul(mimirmind::core::gguf::GgmlType::F32,
                               W.data(), N, K,
                               X.data(), M, Y.data(),
                               scratch.data());

    EXPECT_NEAR(Y[0], 40.0F, 0.0F);
    EXPECT_NEAR(Y[1], 60.0F, 0.0F);
    EXPECT_NEAR(Y[2],  4.0F, 0.0F);
    EXPECT_NEAR(Y[3],  6.0F, 0.0F);
}

// Hand-crafted Q8_0 row: d=0.5, qs[0..31] = 2 → dequanted W = [1.0, 1.0, ...].
// With X = [1.0, ..., 1.0] (32 elems), Y = sum = 32.0.
TEST(matmul_q8_0_singleBlock) {
    constexpr std::size_t M = 1;
    constexpr std::size_t N = 1;
    constexpr std::size_t K = 32;

    std::array<std::uint8_t, 34> W{};
    writeHalfBits(W.data(), kHalfHalf);              // d = 0.5
    for (std::size_t i = 2; i < 34; ++i) W[i] = 2;   // qs[i] = 2

    const std::array<float, K>    X{1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F,
                                    1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F,
                                    1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F,
                                    1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F};
    std::array<float, M * N>      Y{};
    std::array<float, K>          scratch{};

    mimirmind::compute::matmul(mimirmind::core::gguf::GgmlType::Q8_0,
                               W.data(), N, K,
                               X.data(), M, Y.data(),
                               scratch.data());

    // dequant value = 0.5 * 2 = 1.0  for each element, sum of K=32  → 32.
    EXPECT_NEAR(Y[0], 32.0F, 0.0F);
}

// Q6_K row with all-zeros bit pattern + d=1, sc[0..15]=1: every quant
// dequants to -32. With X = all ones over K=256 → Y = -32 * 256 = -8192.
TEST(matmul_q6k_singleSuperBlock) {
    constexpr std::size_t M = 1;
    constexpr std::size_t N = 1;
    constexpr std::size_t K = 256;

    std::array<std::uint8_t, 210> W{};
    for (std::size_t i = 192; i < 208; ++i) W[i] = 1;   // sc[i-192] = 1
    writeHalfBits(W.data() + 208, kHalfOne);             // d = 1

    std::vector<float> X(K, 1.0F);
    std::array<float, M * N> Y{};
    std::vector<float>       scratch(K);

    mimirmind::compute::matmul(mimirmind::core::gguf::GgmlType::Q6_K,
                               W.data(), N, K,
                               X.data(), M, Y.data(),
                               scratch.data());

    EXPECT_NEAR(Y[0], -32.0F * 256.0F, 0.0F);
}

// Q3_K super-block layout: 32 B hmask + 64 B qs + 12 B packed scales +
// 2 B fp16 d = 110 B / 256 elements.
//
// Signed 3-bit quant: qv = low_2 - (high_1 ? 0 : 4) ∈ [-4, 3].
// Value: d * (scale[sub_block] - 32) * qv.
//
// Scales packing trick: 16 x 6-bit values packed into 12 bytes via
// kmask1=0x03030303, kmask2=0x0f0f0f0f. To get all 16 scales = 33:
//   - low 4 bits: nibble = 1 → packed[0..7] = 0x11 (both nibbles 1)
//   - top 2 bits: value = 2 → tmp bytes = 0b10101010 = 0xAA
// So packed[0..11] = {0x11 * 8, 0xAA * 4}. (scale − 32) = 1 then, so
// the multiplier is just d * qv and we can read out the raw 3-bit
// numeric range cleanly.
TEST(dequant_q3k_singleSuperBlock_minValue) {
    // hmask=0, qs=0 → low_2 = 0, high_1 = 0 → qv = 0 - 4 = -4.
    // scales = 33 (so scale-32 = 1), d = 1 → output = -4 everywhere.
    std::array<std::uint8_t, 110> W{};
    // hmask (bytes 0..31)  = 0
    // qs    (bytes 32..95) = 0
    for (std::size_t i = 96; i < 104; ++i) W[i] = 0x11U;   // scales packed[0..7]
    for (std::size_t i = 104; i < 108; ++i) W[i] = 0xAAU;  // scales packed[8..11]
    writeHalfBits(W.data() + 108, kHalfOne);               // d = 1

    std::array<float, 256> out{};
    mimirmind::compute::quant::Q3K::instance().dequantToF32(
        W.data(), out.size(), out.data());

    for (std::size_t i = 0; i < out.size(); ++i) {
        EXPECT_NEAR(out[i], -4.0F, 0.0F);
    }
}

TEST(dequant_q3k_singleSuperBlock_maxValue) {
    // hmask=0xFF, qs=0xFF → low_2 = 3, high_1 = 1 → qv = 3 - 0 = 3.
    // scales = 33, d = 1 → output = 3 everywhere.
    std::array<std::uint8_t, 110> W{};
    for (std::size_t i = 0;  i < 32;  ++i) W[i] = 0xFFU;   // hmask
    for (std::size_t i = 32; i < 96;  ++i) W[i] = 0xFFU;   // qs
    for (std::size_t i = 96; i < 104; ++i) W[i] = 0x11U;   // scales packed[0..7]
    for (std::size_t i = 104; i < 108; ++i) W[i] = 0xAAU;  // scales packed[8..11]
    writeHalfBits(W.data() + 108, kHalfOne);               // d = 1

    std::array<float, 256> out{};
    mimirmind::compute::quant::Q3K::instance().dequantToF32(
        W.data(), out.size(), out.data());

    for (std::size_t i = 0; i < out.size(); ++i) {
        EXPECT_NEAR(out[i], 3.0F, 0.0F);
    }
}

// End-to-end via compute::matmul dispatch: verify the Q3_K path is
// reachable through the registry, not just via the direct
// Q3K::instance().dequantToF32 call.
TEST(matmul_q3k_singleSuperBlock) {
    constexpr std::size_t M = 1;
    constexpr std::size_t N = 1;
    constexpr std::size_t K = 256;

    // Same block as dequant_q3k_singleSuperBlock_minValue: dequants
    // to -4 for every element. Y = sum_k (X[k] * (-4)) with X = 1
    // → Y = -4 * 256 = -1024.
    std::array<std::uint8_t, 110> W{};
    for (std::size_t i = 96;  i < 104; ++i) W[i] = 0x11U;
    for (std::size_t i = 104; i < 108; ++i) W[i] = 0xAAU;
    writeHalfBits(W.data() + 108, kHalfOne);

    std::vector<float> X(K, 1.0F);
    std::array<float, M * N> Y{};
    std::vector<float>       scratch(K);

    mimirmind::compute::matmul(mimirmind::core::gguf::GgmlType::Q3_K,
                               W.data(), N, K,
                               X.data(), M, Y.data(),
                               scratch.data());

    EXPECT_NEAR(Y[0], -4.0F * 256.0F, 0.0F);
}

TEST(matmul_zeroN_noop) {
    constexpr std::size_t K = 4;
    const std::array<float, K>  X{1.0F, 2.0F, 3.0F, 4.0F};
    std::array<float, 1>        Y{42.0F};   // sentinel
    std::array<float, K>        scratch{};

    // N=0 should just return without touching Y.
    mimirmind::compute::matmul(mimirmind::core::gguf::GgmlType::F32,
                               X.data(), 0, K,
                               X.data(), 1, Y.data(),
                               scratch.data());
    EXPECT_NEAR(Y[0], 42.0F, 0.0F);
}

// =======================================================================
// compute::embeddingLookup
// =======================================================================

// F32 embedding table layout: [vocab, d_model] row-major. Row T at
// element offset T * d_model. Verify a known row is copied verbatim.
TEST(embedding_f32_singleToken) {
    constexpr std::size_t d_model    = 4;
    constexpr std::size_t vocab_size = 3;
    const std::array<float, vocab_size * d_model> table{
        0.10F, 0.20F, 0.30F, 0.40F,    // row 0
        1.10F, 1.20F, 1.30F, 1.40F,    // row 1
        2.10F, 2.20F, 2.30F, 2.40F,    // row 2
    };
    const std::array<std::int32_t, 1> ids{2};
    std::array<float, d_model>        dst{};

    mimirmind::compute::embeddingLookup(
        mimirmind::core::gguf::GgmlType::F32,
        table.data(), d_model, vocab_size,
        std::span<const std::int32_t>{ids},
        dst.data());

    EXPECT_NEAR(dst[0], 2.10F, 0.0F);
    EXPECT_NEAR(dst[1], 2.20F, 0.0F);
    EXPECT_NEAR(dst[2], 2.30F, 0.0F);
    EXPECT_NEAR(dst[3], 2.40F, 0.0F);
}

TEST(embedding_f32_multipleTokens) {
    constexpr std::size_t d_model    = 2;
    constexpr std::size_t vocab_size = 4;
    const std::array<float, vocab_size * d_model> table{
         0.0F,  1.0F,    // row 0
        10.0F, 11.0F,    // row 1
        20.0F, 21.0F,    // row 2
        30.0F, 31.0F,    // row 3
    };
    // Pick rows 3, 1, 0 — verify each row lands at the right slot.
    const std::array<std::int32_t, 3> ids{3, 1, 0};
    std::array<float, 3 * d_model>    dst{};

    mimirmind::compute::embeddingLookup(
        mimirmind::core::gguf::GgmlType::F32,
        table.data(), d_model, vocab_size,
        std::span<const std::int32_t>{ids},
        dst.data());

    EXPECT_NEAR(dst[0], 30.0F, 0.0F);
    EXPECT_NEAR(dst[1], 31.0F, 0.0F);
    EXPECT_NEAR(dst[2], 10.0F, 0.0F);
    EXPECT_NEAR(dst[3], 11.0F, 0.0F);
    EXPECT_NEAR(dst[4],  0.0F, 0.0F);
    EXPECT_NEAR(dst[5],  1.0F, 0.0F);
}

TEST(embedding_outOfRange_zeroFilled) {
    constexpr std::size_t d_model    = 3;
    constexpr std::size_t vocab_size = 2;
    const std::array<float, vocab_size * d_model> table{
        1.0F, 2.0F, 3.0F,
        4.0F, 5.0F, 6.0F,
    };
    // -1 (negative) and 99 (>= vocab_size) both must zero-fill, not throw.
    const std::array<std::int32_t, 3> ids{-1, 0, 99};
    std::array<float, 3 * d_model>    dst{};

    mimirmind::compute::embeddingLookup(
        mimirmind::core::gguf::GgmlType::F32,
        table.data(), d_model, vocab_size,
        std::span<const std::int32_t>{ids},
        dst.data());

    // ids[0] = -1 → row 0..2 all zeros
    EXPECT_NEAR(dst[0], 0.0F, 0.0F);
    EXPECT_NEAR(dst[1], 0.0F, 0.0F);
    EXPECT_NEAR(dst[2], 0.0F, 0.0F);
    // ids[1] = 0 → row [1, 2, 3]
    EXPECT_NEAR(dst[3], 1.0F, 0.0F);
    EXPECT_NEAR(dst[4], 2.0F, 0.0F);
    EXPECT_NEAR(dst[5], 3.0F, 0.0F);
    // ids[2] = 99 → row 6..8 all zeros
    EXPECT_NEAR(dst[6], 0.0F, 0.0F);
    EXPECT_NEAR(dst[7], 0.0F, 0.0F);
    EXPECT_NEAR(dst[8], 0.0F, 0.0F);
}

// Q6_K embedding table: each token row is one super-block (256 elements,
// 210 bytes). Build a 2-token vocab where token 0 dequants to all -32
// and token 1 dequants to all +31. Verify the lookup returns those.
TEST(embedding_q6k_dequants) {
    constexpr std::size_t d_model    = 256;
    constexpr std::size_t vocab_size = 2;

    std::array<std::uint8_t, vocab_size * 210> table{};
    // Token 0: all ql=0, qh=0, sc=1, d=1 → every output = -32
    {
        auto* row = table.data() + 0 * 210;
        for (std::size_t i = 192; i < 208; ++i) row[i] = 1;
        writeHalfBits(row + 208, kHalfOne);
    }
    // Token 1: all ql=0xFF, qh=0xFF, sc=1, d=1 → every output = +31
    {
        auto* row = table.data() + 1 * 210;
        for (std::size_t i = 0;   i < 128; ++i) row[i] = 0xFFU;
        for (std::size_t i = 128; i < 192; ++i) row[i] = 0xFFU;
        for (std::size_t i = 192; i < 208; ++i) row[i] = 1;
        writeHalfBits(row + 208, kHalfOne);
    }

    const std::array<std::int32_t, 2> ids{0, 1};
    std::array<float, 2 * d_model>    dst{};

    mimirmind::compute::embeddingLookup(
        mimirmind::core::gguf::GgmlType::Q6_K,
        table.data(), d_model, vocab_size,
        std::span<const std::int32_t>{ids},
        dst.data());

    for (std::size_t i = 0;       i < d_model;     ++i) {
        EXPECT_NEAR(dst[i], -32.0F, 0.0F);
    }
    for (std::size_t i = d_model; i < 2 * d_model; ++i) {
        EXPECT_NEAR(dst[i],  31.0F, 0.0F);
    }
}

TEST(embedding_zeroDModelThrows) {
    const std::array<float, 1>        table{0.0F};
    const std::array<std::int32_t, 1> ids{0};
    std::array<float, 1>              dst{};
    try {
        mimirmind::compute::embeddingLookup(
            mimirmind::core::gguf::GgmlType::F32,
            table.data(), 0, 1,
            std::span<const std::int32_t>{ids},
            dst.data());
        EXPECT_TRUE(false && "expected throw");
    } catch (const std::runtime_error&) {
        // expected
    }
}

// =======================================================================
// runtime::NGramDrafter — pure-function `lookup` covers PLD algorithm
// =======================================================================

namespace {

// The tiny test framework's EXPECT_EQ streams the compared values into
// an error message and has no operator<< for std::vector. This helper
// does an element-wise compare and hits EXPECT_EQ per element so the
// stock int streaming path handles the failure output.
void expectTokensEq(const std::vector<std::int32_t>& got,
                    const std::vector<std::int32_t>& expected) {
    EXPECT_EQ(got.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(got[i], expected[i]);
    }
}

}

TEST(ngram_lookup_emptyContext_returnsEmpty) {
    const std::vector<std::int32_t> ctx;
    const auto out = mimirmind::runtime::NGramDrafter::lookup(
        ctx, /*N=*/4, /*maxK=*/3, /*minK=*/2);
    EXPECT_TRUE(out.empty());
}

TEST(ngram_lookup_shorterThanMinKPlusOne_returnsEmpty) {
    // Need at least minK + 1 tokens (minK for needle + 1 for match).
    const std::vector<std::int32_t> ctx{7, 7};
    const auto out = mimirmind::runtime::NGramDrafter::lookup(
        ctx, /*N=*/4, /*maxK=*/3, /*minK=*/2);
    EXPECT_TRUE(out.empty());
}

TEST(ngram_lookup_noMatch_returnsEmpty) {
    // Tail [A, B] appears nowhere earlier in [X, Y, Z, A, B].
    const std::vector<std::int32_t> ctx{10, 20, 30, 1, 2};
    const auto out = mimirmind::runtime::NGramDrafter::lookup(
        ctx, /*N=*/3, /*maxK=*/2, /*minK=*/2);
    EXPECT_TRUE(out.empty());
}

TEST(ngram_lookup_exactMatch_returnsFollowUp) {
    // Tail [A, B, C] matches at position 0 in [A, B, C, X, Y, A, B, C]
    // → follow-up is [X, Y].
    const std::vector<std::int32_t> ctx{1, 2, 3, 42, 99, 1, 2, 3};
    const auto out = mimirmind::runtime::NGramDrafter::lookup(
        ctx, /*N=*/2, /*maxK=*/3, /*minK=*/2);
    const std::vector<std::int32_t> expected{42, 99};
    expectTokensEq(out, expected);
}

TEST(ngram_lookup_prefersLongerK) {
    // Tail [B, C]. At k=3 (needle [A, B, C]) match at pos 0 → follow-up
    // [X]. At k=2 (needle [B, C]) match at pos 1 → follow-up [X]. maxK
    // sweep starts at k=3, so the k=3 result wins even though both
    // exist. Here follow-ups happen to be identical, but the assertion
    // is that the sweep is longest-k-first.
    const std::vector<std::int32_t> ctx{1, 2, 3, 42, 99, 2, 3};
    const auto out = mimirmind::runtime::NGramDrafter::lookup(
        ctx, /*N=*/1, /*maxK=*/3, /*minK=*/2);
    const std::vector<std::int32_t> expected{42};
    expectTokensEq(out, expected);
}

TEST(ngram_lookup_fallsBackToSmallerK) {
    // Tail [B, C]. At k=3 (needle [A, B, C]) no match. At k=2 (needle
    // [B, C]) match at pos 2 → follow-up [99].
    const std::vector<std::int32_t> ctx{7, 8, 2, 3, 99, 5, 2, 3};
    const auto out = mimirmind::runtime::NGramDrafter::lookup(
        ctx, /*N=*/1, /*maxK=*/3, /*minK=*/2);
    const std::vector<std::int32_t> expected{99};
    expectTokensEq(out, expected);
}

TEST(ngram_lookup_prefersMostRecentMatch) {
    // Tail [1, 2] matches at positions 0 and 3. Backward scan picks the
    // more recent (pos 3), giving follow-up [77].
    const std::vector<std::int32_t> ctx{1, 2, 55, 1, 2, 77, 1, 2};
    const auto out = mimirmind::runtime::NGramDrafter::lookup(
        ctx, /*N=*/1, /*maxK=*/2, /*minK=*/2);
    const std::vector<std::int32_t> expected{77};
    expectTokensEq(out, expected);
}

TEST(ngram_lookup_truncatesFollowUpToN) {
    // Match has 3 follow-up tokens but caller asks for N=2 → truncate.
    const std::vector<std::int32_t> ctx{1, 2, 10, 20, 30, 99, 1, 2};
    const auto out = mimirmind::runtime::NGramDrafter::lookup(
        ctx, /*N=*/2, /*maxK=*/2, /*minK=*/2);
    const std::vector<std::int32_t> expected{10, 20};
    expectTokensEq(out, expected);
}

TEST(ngram_lookup_truncatesFollowUpToNeedleOverlap) {
    // Match at position 4 has follow-ups that would overlap the tail
    // needle at position 6. Only the non-overlapping tokens are
    // returned — here that's zero, so this match is skipped and the
    // earlier match at position 0 wins with follow-up [55, 66].
    const std::vector<std::int32_t> ctx{7, 8, 55, 66, 7, 8, 7, 8};
    const auto out = mimirmind::runtime::NGramDrafter::lookup(
        ctx, /*N=*/2, /*maxK=*/2, /*minK=*/2);
    const std::vector<std::int32_t> expected{55, 66};
    expectTokensEq(out, expected);
}

TEST(ngram_lookup_zeroN_returnsEmpty) {
    // Even with a perfect match, N=0 means no tokens requested.
    const std::vector<std::int32_t> ctx{1, 2, 3, 42, 1, 2, 3};
    const auto out = mimirmind::runtime::NGramDrafter::lookup(
        ctx, /*N=*/0, /*maxK=*/3, /*minK=*/2);
    EXPECT_TRUE(out.empty());
}

TEST(ngram_drafter_proposeWrapsLookup) {
    // Smoke-test the full Drafter interface: config, propose(), that
    // probs are filled to 1.0F per token.
    mimirmind::runtime::NGramDrafter drafter{{/*maxK=*/3, /*minK=*/2}};
    const std::vector<std::int32_t> ctx{1, 2, 3, 42, 99, 1, 2, 3};
    const mimirmind::compute::SamplingParams sampling{};
    const auto batch = drafter.propose(
        std::span<const std::int32_t>{ctx}, /*N=*/2, sampling);
    const std::vector<std::int32_t> expectedTokens{42, 99};
    expectTokensEq(batch.tokens, expectedTokens);
    EXPECT_EQ(batch.probs.size(), std::size_t{2});
    EXPECT_NEAR(batch.probs[0], 1.0F, 1e-6F);
    EXPECT_NEAR(batch.probs[1], 1.0F, 1e-6F);
    EXPECT_TRUE(!batch.hitStop);
}

TEST(ngram_drafter_ctorRejectsMinKZero) {
    try {
        mimirmind::runtime::NGramDrafter drafter{{/*maxK=*/3, /*minK=*/0}};
        EXPECT_TRUE(false && "expected throw for minK=0");
    } catch (const std::invalid_argument&) {
        // expected
    }
}

TEST(ngram_drafter_ctorRejectsMaxKBelowMinK) {
    try {
        mimirmind::runtime::NGramDrafter drafter{{/*maxK=*/1, /*minK=*/3}};
        EXPECT_TRUE(false && "expected throw for maxK<minK");
    } catch (const std::invalid_argument&) {
        // expected
    }
}

int main() {
    return mm::test::run();
}