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
#include "compute/MoeRouting.hpp"
#include "compute/Softmax.hpp"
#include "model/ResponseCleaner.hpp"
#include "runtime/arch/ArchBackend.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
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

// =======================================================================
// compute::moeTopKRoute (Gemma 4 path B routing)
// =======================================================================

TEST(moe_singleToken_clearWinners) {
    // 4 experts, K=2. logits = [10, 1, 8, 3]
    //   max = 10
    //   exp shift: [1, e^-9, e^-2, e^-7] ≈ [1, 1.23e-4, 0.1353, 9.12e-4]
    //   sum  = 1.13642...
    //   probs ≈ [0.8799, 1.08e-4, 0.1191, 8.02e-4]
    //   top-2 by descending = expert 0 then 2
    //   kept = 0.9990, weights = [0.8799/0.9990, 0.1191/0.9990] ≈ [0.8807, 0.1193]
    const std::array<float, 4> logits{10.0F, 1.0F, 8.0F, 3.0F};
    std::array<std::int32_t, 2> idx{};
    std::array<float, 2>        w{};

    mimirmind::compute::moeTopKRoute(logits.data(), 1, 4, 2,
                                     idx.data(), w.data());

    EXPECT_EQ(idx[0], 0);
    EXPECT_EQ(idx[1], 2);

    // Hand-compute expected renormalised weights:
    const float p0 = 1.0F;                     // exp(10-10)
    const float p2 = std::exp(8.0F - 10.0F);   // exp(-2)
    const float wsum = p0 + p2;
    EXPECT_NEAR(w[0], p0 / wsum, 1e-5F);
    EXPECT_NEAR(w[1], p2 / wsum, 1e-5F);
    EXPECT_NEAR(w[0] + w[1], 1.0F, 1e-5F);
}

TEST(moe_singleToken_uniformLogits_K1) {
    // All equal logits → softmax uniform → top-1 picks SOMEONE with weight 1.0.
    // Tie-breaking is implementation-defined; we just check validity.
    const std::array<float, 4> logits{2.5F, 2.5F, 2.5F, 2.5F};
    std::array<std::int32_t, 1> idx{};
    std::array<float, 1>        w{};

    mimirmind::compute::moeTopKRoute(logits.data(), 1, 4, 1,
                                     idx.data(), w.data());

    EXPECT_TRUE(idx[0] >= 0 && idx[0] < 4);
    EXPECT_NEAR(w[0], 1.0F, 1e-5F);
}

TEST(moe_singleToken_uniformLogits_topKEqualsN) {
    // K = nExperts, uniform logits → every expert selected with weight 1/N.
    // After partial_sort all indices are present; collect them as a set.
    const std::array<float, 4> logits{0.0F, 0.0F, 0.0F, 0.0F};
    std::array<std::int32_t, 4> idx{};
    std::array<float, 4>        w{};

    mimirmind::compute::moeTopKRoute(logits.data(), 1, 4, 4,
                                     idx.data(), w.data());

    std::set<std::int32_t> seen(idx.begin(), idx.end());
    EXPECT_EQ(seen.size(), 4U);   // all distinct
    EXPECT_TRUE(seen.contains(0));
    EXPECT_TRUE(seen.contains(1));
    EXPECT_TRUE(seen.contains(2));
    EXPECT_TRUE(seen.contains(3));

    float sumW = 0.0F;
    for (float wi : w) {
        EXPECT_NEAR(wi, 0.25F, 1e-5F);
        sumW += wi;
    }
    EXPECT_NEAR(sumW, 1.0F, 1e-5F);
}

TEST(moe_weightsSumToOne_renormalised) {
    // Even when topK < nExperts, the K chosen weights MUST sum to ~1
    // because we renormalise. This guards against drift in the divisor.
    const std::array<float, 6> logits{5.0F, 1.0F, 3.0F, 7.0F, 2.0F, 4.0F};
    std::array<std::int32_t, 3> idx{};
    std::array<float, 3>        w{};

    mimirmind::compute::moeTopKRoute(logits.data(), 1, 6, 3,
                                     idx.data(), w.data());

    const float sumW = w[0] + w[1] + w[2];
    EXPECT_NEAR(sumW, 1.0F, 1e-5F);

    // Top-3 should be experts 3, 0, 5 in descending order of logit.
    EXPECT_EQ(idx[0], 3);
    EXPECT_EQ(idx[1], 0);
    EXPECT_EQ(idx[2], 5);
}

TEST(moe_numericallyStable_largeLogits) {
    // Without max-subtract, exp(1000) would overflow.
    const std::array<float, 3> logits{1000.0F, 1003.0F, 999.0F};
    std::array<std::int32_t, 2> idx{};
    std::array<float, 2>        w{};

    mimirmind::compute::moeTopKRoute(logits.data(), 1, 3, 2,
                                     idx.data(), w.data());

    EXPECT_EQ(idx[0], 1);   // 1003 is largest
    EXPECT_EQ(idx[1], 0);   // 1000 is second
    EXPECT_NEAR(w[0] + w[1], 1.0F, 1e-5F);
    EXPECT_TRUE(std::isfinite(w[0]));
    EXPECT_TRUE(std::isfinite(w[1]));
}

TEST(moe_multipleTokens_independent) {
    // T=2 with different rows; verify each row is routed independently.
    // Row 0: [9, 0, 5]  → top-2 = (0, 2)
    // Row 1: [1, 4, 2]  → top-2 = (1, 2)
    const std::array<float, 6> logits{
        9.0F, 0.0F, 5.0F,
        1.0F, 4.0F, 2.0F,
    };
    std::array<std::int32_t, 4> idx{};
    std::array<float, 4>        w{};

    mimirmind::compute::moeTopKRoute(logits.data(), 2, 3, 2,
                                     idx.data(), w.data());

    EXPECT_EQ(idx[0], 0);
    EXPECT_EQ(idx[1], 2);
    EXPECT_NEAR(w[0] + w[1], 1.0F, 1e-5F);

    EXPECT_EQ(idx[2], 1);
    EXPECT_EQ(idx[3], 2);
    EXPECT_NEAR(w[2] + w[3], 1.0F, 1e-5F);
}

TEST(moe_descendingOrder) {
    // Routing returns indices in descending probability order.
    // logits = [3, 8, 1, 5, 9] → top-3 = (4, 1, 3)
    const std::array<float, 5> logits{3.0F, 8.0F, 1.0F, 5.0F, 9.0F};
    std::array<std::int32_t, 3> idx{};
    std::array<float, 3>        w{};

    mimirmind::compute::moeTopKRoute(logits.data(), 1, 5, 3,
                                     idx.data(), w.data());

    EXPECT_EQ(idx[0], 4);
    EXPECT_EQ(idx[1], 1);
    EXPECT_EQ(idx[2], 3);
    EXPECT_TRUE(w[0] >= w[1]);
    EXPECT_TRUE(w[1] >= w[2]);
}

TEST(moe_topKZeroThrows) {
    const std::array<float, 3>  logits{1.0F, 2.0F, 3.0F};
    std::array<std::int32_t, 1> idx{};
    std::array<float, 1>        w{};
    try {
        mimirmind::compute::moeTopKRoute(logits.data(), 1, 3, 0,
                                         idx.data(), w.data());
        EXPECT_TRUE(false && "expected throw");
    } catch (const std::runtime_error&) {
        // expected
    }
}

TEST(moe_topKExceedsNExpertsThrows) {
    const std::array<float, 3>  logits{1.0F, 2.0F, 3.0F};
    std::array<std::int32_t, 4> idx{};
    std::array<float, 4>        w{};
    try {
        mimirmind::compute::moeTopKRoute(logits.data(), 1, 3, 4,
                                         idx.data(), w.data());
        EXPECT_TRUE(false && "expected throw");
    } catch (const std::runtime_error&) {
        // expected
    }
}

// =======================================================================
// model::ResponseCleaner — Gemma 4 channel-wrapper strip at token level.
//
// The cleaner is the SSE-streaming counterpart to ChatTemplate::cleanResponse.
// It consumes (id, text) pairs and decides which to surface to the client.
//
// Token IDs in these tests are arbitrary fixtures — the cleaner only cares
// about equality against the constructor-supplied channel-start / channel-end
// ids. We use 1000 and 1001 to keep them distinct from any "real text" id.
// =======================================================================

namespace {
constexpr std::int32_t kFakeChannelStart = 1000;
constexpr std::int32_t kFakeChannelEnd   = 1001;
constexpr std::int32_t kFakeTextId       = 42;  // any non-special token

bool feedAndCapture(mimirmind::model::ResponseCleaner& c,
                    std::int32_t                       id,
                    std::string                        text,
                    std::string&                       captured) {
    const bool emit = c.feed(id, text);
    if (emit) {
        captured.append(text);
    }
    return emit;
}
} // namespace

TEST(responseCleaner_qwen_passThrough) {
    using mimirmind::model::ChatTemplate;
    using mimirmind::model::ResponseCleaner;
    ResponseCleaner c{ChatTemplate::Style::QwenChatML, -1, -1};

    std::string out;
    EXPECT_TRUE(feedAndCapture(c, kFakeTextId, "Hello", out));
    EXPECT_TRUE(feedAndCapture(c, kFakeTextId, ", ",    out));
    EXPECT_TRUE(feedAndCapture(c, kFakeTextId, "world", out));
    EXPECT_EQ(out, std::string{"Hello, world"});
}

TEST(responseCleaner_qwen_dropsEmptyText) {
    // Even in pass-through mode, an empty token text should not be emitted —
    // matches the existing onToken behaviour of skipping empty decodes.
    using mimirmind::model::ChatTemplate;
    using mimirmind::model::ResponseCleaner;
    ResponseCleaner c{ChatTemplate::Style::QwenChatML, -1, -1};

    std::string empty;
    EXPECT_TRUE(!c.feed(kFakeTextId, empty));
}

TEST(responseCleaner_gemma3_passThrough) {
    // Gemma 3 has no thinking-channel wrapper, so the cleaner is a no-op
    // even though the IDs were supplied.
    using mimirmind::model::ChatTemplate;
    using mimirmind::model::ResponseCleaner;
    ResponseCleaner c{ChatTemplate::Style::Gemma3,
                      kFakeChannelStart, kFakeChannelEnd};

    std::string out;
    // Even if the model somehow emitted the channel-start id, Gemma 3 mode
    // surfaces it as visible text (the id is just text in this style).
    EXPECT_TRUE(feedAndCapture(c, kFakeChannelStart, "<|channel>", out));
    EXPECT_TRUE(feedAndCapture(c, kFakeTextId,       "hi",         out));
    EXPECT_EQ(out, std::string{"<|channel>hi"});
}

TEST(responseCleaner_gemma4_stripsChannelWrapper) {
    using mimirmind::model::ChatTemplate;
    using mimirmind::model::ResponseCleaner;
    ResponseCleaner c{ChatTemplate::Style::Gemma4,
                      kFakeChannelStart, kFakeChannelEnd};

    std::string out;

    // Replay the exact sequence seen on the wire from the live Gemma 4 26B
    // smoke test against REDACTED_HOST:
    //
    //   <|channel>  thought  \n  <channel|>  The  capital ...
    //
    // Expected: only "The capital..." reaches `out`.
    EXPECT_TRUE(!feedAndCapture(c, kFakeChannelStart, "<|channel>", out));
    EXPECT_TRUE(!feedAndCapture(c, kFakeTextId,       "thought",    out));
    EXPECT_TRUE(!feedAndCapture(c, kFakeTextId,       "\n",         out));
    EXPECT_TRUE(!feedAndCapture(c, kFakeChannelEnd,   "<channel|>", out));
    EXPECT_TRUE(feedAndCapture (c, kFakeTextId,       "The",        out));
    EXPECT_TRUE(feedAndCapture (c, kFakeTextId,       " capital",   out));
    EXPECT_EQ(out, std::string{"The capital"});
}

TEST(responseCleaner_gemma4_stripsTrailingWhitespaceAfterChannel) {
    // Even if the channel close is immediately followed by several
    // whitespace-only tokens, all of them get dropped until visible
    // content arrives.
    using mimirmind::model::ChatTemplate;
    using mimirmind::model::ResponseCleaner;
    ResponseCleaner c{ChatTemplate::Style::Gemma4,
                      kFakeChannelStart, kFakeChannelEnd};

    std::string out;
    EXPECT_TRUE(!feedAndCapture(c, kFakeChannelStart, "<|channel>", out));
    EXPECT_TRUE(!feedAndCapture(c, kFakeChannelEnd,   "<channel|>", out));
    EXPECT_TRUE(!feedAndCapture(c, kFakeTextId,       "\n",         out));
    EXPECT_TRUE(!feedAndCapture(c, kFakeTextId,       " ",          out));
    EXPECT_TRUE(feedAndCapture (c, kFakeTextId,       "Hi",         out));
    EXPECT_EQ(out, std::string{"Hi"});
}

TEST(responseCleaner_gemma4_partialLeadingWhitespaceInToken) {
    // If the first non-whitespace token starts with whitespace bytes
    // (e.g. " Paris"), only the leading whitespace is stripped — the
    // rest is emitted.
    using mimirmind::model::ChatTemplate;
    using mimirmind::model::ResponseCleaner;
    ResponseCleaner c{ChatTemplate::Style::Gemma4,
                      kFakeChannelStart, kFakeChannelEnd};

    std::string out;
    EXPECT_TRUE(!feedAndCapture(c, kFakeChannelStart, "<|channel>", out));
    EXPECT_TRUE(!feedAndCapture(c, kFakeChannelEnd,   "<channel|>", out));
    EXPECT_TRUE(feedAndCapture (c, kFakeTextId,       "  Paris",    out));
    EXPECT_TRUE(feedAndCapture (c, kFakeTextId,       ".",          out));
    EXPECT_EQ(out, std::string{"Paris."});
}

TEST(responseCleaner_gemma4_missingIdsDisablesStrip) {
    // If the tokenizer does not have the channel-start/end specials, the
    // cleaner cannot recognise them and falls back to pass-through. This
    // mirrors the safety guarantee in forStyle() when findToken returns -1.
    using mimirmind::model::ChatTemplate;
    using mimirmind::model::ResponseCleaner;
    ResponseCleaner c{ChatTemplate::Style::Gemma4, -1, -1};

    std::string out;
    EXPECT_TRUE(feedAndCapture(c, kFakeChannelStart, "<|channel>", out));
    EXPECT_TRUE(feedAndCapture(c, kFakeTextId,       "thought",    out));
    EXPECT_EQ(out, std::string{"<|channel>thought"});
}

TEST(responseCleaner_gemma4_noChannelInOutput) {
    // If the model never emits the wrapper at all, the cleaner is fully
    // transparent — first token reaches the client unchanged.
    using mimirmind::model::ChatTemplate;
    using mimirmind::model::ResponseCleaner;
    ResponseCleaner c{ChatTemplate::Style::Gemma4,
                      kFakeChannelStart, kFakeChannelEnd};

    std::string out;
    EXPECT_TRUE(feedAndCapture(c, kFakeTextId, "Direct", out));
    EXPECT_TRUE(feedAndCapture(c, kFakeTextId, " answer", out));
    EXPECT_EQ(out, std::string{"Direct answer"});
}

int main() {
    return mm::test::run();
}