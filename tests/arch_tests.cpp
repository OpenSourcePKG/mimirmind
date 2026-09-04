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
#include "model/ToolCallParser.hpp"
#include "model/ToolCallStreamDetector.hpp"
#include "runtime/thermal/GpuClockGovernor.hpp"
#include "runtime/Lcp.hpp"
#include "runtime/thermal/PowerMonitor.hpp"
#include "runtime/thermal/SystemMonitor.hpp"
#include "runtime/thermal/ThermalGuard.hpp"
#include "runtime/thermal/ThermalProfile.hpp"
#include "runtime/arch/ArchBackend.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sys/stat.h>
#include <thread>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <set>
#include <span>
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
    EXPECT_TRUE(isSupportedArchitecture("llama"));  // Llama-3 via Qwen2Backend
}

TEST(arch_unsupportedNames) {
    using mimirmind::runtime::arch::isSupportedArchitecture;
    EXPECT_TRUE(!isSupportedArchitecture(""));
    EXPECT_TRUE(!isSupportedArchitecture("qwen3"));
    EXPECT_TRUE(!isSupportedArchitecture("gemma3"));
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

TEST(responseCleaner_qwen_stripsThinkingBlock) {
    // Qwen3 thinking models (qwen35moe) pre-open <think> in the prompt, so the
    // response begins inside the block and closes it with the first </think>,
    // emitted as literal text. A non-negative channelStartId marks the model as
    // thinking-capable. The cleaner swallows content + </think> and surfaces
    // only the post-</think> answer, stripping the trailing newlines.
    using mimirmind::model::ChatTemplate;
    using mimirmind::model::ResponseCleaner;
    ResponseCleaner c{ChatTemplate::Style::QwenChatML, /*thinkId=*/1, -1};

    std::string out;
    EXPECT_TRUE(!feedAndCapture(c, kFakeTextId, "Let me reason", out));
    EXPECT_TRUE(!feedAndCapture(c, kFakeTextId, "</think>",      out));
    EXPECT_TRUE(!feedAndCapture(c, kFakeTextId, "\n\n",          out));
    EXPECT_TRUE(feedAndCapture(c, kFakeTextId,  "The answer",    out));
    EXPECT_EQ(out, std::string{"The answer"});
}

TEST(responseCleaner_qwen_thinkTagsSplitAcrossTokens) {
    // The </think> closer can arrive split across tokens ("</thin" + "k>").
    // The state machine must still recognise it.
    using mimirmind::model::ChatTemplate;
    using mimirmind::model::ResponseCleaner;
    ResponseCleaner c{ChatTemplate::Style::QwenChatML, /*thinkId=*/1, -1};

    std::string out;
    EXPECT_TRUE(!feedAndCapture(c, kFakeTextId, "reasoning</", out));
    EXPECT_TRUE(!feedAndCapture(c, kFakeTextId, "thin",        out));
    EXPECT_TRUE(feedAndCapture(c,  kFakeTextId, "k>Paris",     out));
    EXPECT_EQ(out, std::string{"Paris"});
}

TEST(responseCleaner_qwen_noThinkPassesThrough) {
    // A response that does NOT open with <think> (Qwen2/2.5 or thinking off)
    // must pass through verbatim, including leading whitespace.
    using mimirmind::model::ChatTemplate;
    using mimirmind::model::ResponseCleaner;
    ResponseCleaner c{ChatTemplate::Style::QwenChatML, -1, -1};

    std::string out;
    EXPECT_TRUE(feedAndCapture(c, kFakeTextId, "The capital ", out));
    EXPECT_TRUE(feedAndCapture(c, kFakeTextId, "is Paris.",    out));
    EXPECT_EQ(out, std::string{"The capital is Paris."});
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

// =======================================================================
// runtime::longestCommonPrefix — M9.1 prefix-cache helper
// =======================================================================

namespace {
std::size_t lcp(const std::vector<std::int32_t>& a,
                const std::vector<std::int32_t>& b) {
    return mimirmind::runtime::longestCommonPrefix(
        std::span<const std::int32_t>{a},
        std::span<const std::int32_t>{b});
}
} // namespace

TEST(lcp_bothEmpty) {
    EXPECT_EQ(lcp({}, {}), std::size_t{0});
}

TEST(lcp_oneEmpty) {
    EXPECT_EQ(lcp({1, 2, 3}, {}),     std::size_t{0});
    EXPECT_EQ(lcp({},        {1, 2}), std::size_t{0});
}

TEST(lcp_identical) {
    EXPECT_EQ(lcp({1, 2, 3, 4}, {1, 2, 3, 4}), std::size_t{4});
}

TEST(lcp_partialPrefix) {
    EXPECT_EQ(lcp({1, 2, 3, 4, 5}, {1, 2, 3, 9, 9}), std::size_t{3});
}

TEST(lcp_aIsPrefixOfB) {
    // Caps at the shorter side. Used when the new prompt is exactly the
    // previous prompt plus a new turn — LCP = old prompt length.
    EXPECT_EQ(lcp({1, 2, 3}, {1, 2, 3, 4, 5, 6}), std::size_t{3});
}

TEST(lcp_bIsPrefixOfA) {
    EXPECT_EQ(lcp({1, 2, 3, 4, 5}, {1, 2}), std::size_t{2});
}

TEST(lcp_noCommonPrefix) {
    EXPECT_EQ(lcp({7, 8, 9}, {1, 2, 3}), std::size_t{0});
}

TEST(lcp_singleElementMatch) {
    EXPECT_EQ(lcp({42}, {42}),    std::size_t{1});
    EXPECT_EQ(lcp({42}, {42, 7}), std::size_t{1});
}

TEST(lcp_negativeIdsCompareLikeOthers) {
    // Token ids are signed (-1 is a common "missing" sentinel). The
    // function must treat them like any other int32_t.
    EXPECT_EQ(lcp({-1, -2, -3}, {-1, -2, -3}), std::size_t{3});
    EXPECT_EQ(lcp({-1, -2, -3}, {-1, -2,  7}), std::size_t{2});
}

// =======================================================================
// runtime::ThermalProfile JSON loader
// =======================================================================

namespace {

/// Drop a JSON string into a unique tmp file. The file outlives the
/// caller's frame so the path can be passed to loadThermalProfile().
struct TempJsonFile {
    std::string path;
    explicit TempJsonFile(const char* content) {
        char templ[] = "/tmp/mimirmind-test-profile-XXXXXX";
        const int fd = ::mkstemp(templ);
        if (fd >= 0) {
            ::close(fd);
        }
        path = templ;
        std::ofstream{path} << content;
    }
    ~TempJsonFile() { std::remove(path.c_str()); }
    TempJsonFile(const TempJsonFile&) = delete;
    TempJsonFile& operator=(const TempJsonFile&) = delete;
};

} // namespace

TEST(thermalProfile_minimumValidIsName) {
    TempJsonFile f{R"({"name":"empty-but-valid"})"};
    const auto p = mimirmind::runtime::loadThermalProfile(f.path);
    EXPECT_EQ(p.name, std::string{"empty-but-valid"});
    EXPECT_TRUE(!p.hasPackageLimits());
}

TEST(thermalProfile_loadsPackageLimits) {
    TempJsonFile f{R"({
        "name": "nuc14",
        "description": "test",
        "package_temp_soft_c": 75,
        "package_temp_hard_c": 82,
        "package_throttle_max_ms": 250
    })"};
    const auto p = mimirmind::runtime::loadThermalProfile(f.path);
    EXPECT_EQ(p.name, std::string{"nuc14"});
    EXPECT_TRUE(p.hasPackageLimits());
    EXPECT_NEAR(*p.package_temp_soft_c, 75.0F, 0.001F);
    EXPECT_NEAR(*p.package_temp_hard_c, 82.0F, 0.001F);
    EXPECT_EQ(p.package_throttle_max_ms, 250);
}

TEST(thermalProfile_rejectsMissingName) {
    TempJsonFile f{R"({"package_temp_soft_c": 75, "package_temp_hard_c": 82})"};
    try {
        (void)mimirmind::runtime::loadThermalProfile(f.path);
        EXPECT_TRUE(false && "expected throw");
    } catch (const std::runtime_error&) {
        // expected
    }
}

TEST(thermalProfile_rejectsLonelySoftWithoutHard) {
    TempJsonFile f{R"({"name":"x", "package_temp_soft_c": 75})"};
    try {
        (void)mimirmind::runtime::loadThermalProfile(f.path);
        EXPECT_TRUE(false && "expected throw");
    } catch (const std::runtime_error&) {
        // expected
    }
}

TEST(thermalProfile_rejectsSoftAboveHard) {
    TempJsonFile f{R"({
        "name":"x",
        "package_temp_soft_c": 90,
        "package_temp_hard_c": 80
    })"};
    try {
        (void)mimirmind::runtime::loadThermalProfile(f.path);
        EXPECT_TRUE(false && "expected throw");
    } catch (const std::runtime_error&) {
        // expected
    }
}

TEST(thermalProfile_ignoresUnknownFields) {
    // Forward-compatibility: future fields the loader does not understand
    // should just be ignored rather than rejected. RAM is not a thermal
    // concern — if someone adds RAM fields here by mistake, the loader
    // should not blow up, it should leave the profile temperature-only.
    TempJsonFile f{R"({
        "name": "x",
        "package_temp_soft_c": 70,
        "package_temp_hard_c": 80,
        "ram_available_min_mib": 4096,
        "future_unknown_knob": 42
    })"};
    const auto p = mimirmind::runtime::loadThermalProfile(f.path);
    EXPECT_EQ(p.name, std::string{"x"});
    EXPECT_TRUE(p.hasPackageLimits());
}

// =======================================================================
// runtime::ThermalGuard — decision state machine.
//
// SystemMonitor reads /sys + /proc directly so we cannot mock it cleanly
// inside this binary. Instead we drive the guard through a SystemMonitor
// constructed with no required sensors (it will still read host sensors,
// but the test does not care what's there) — and instead test the math
// via small inline-friendly helpers.
//
// What we CAN test end-to-end is the JSON → ThermalGuard flow plus the
// "no monitoring configured returns Ok" path.
// =======================================================================

TEST(thermalGuard_emptyProfileIsAlwaysOk) {
    using namespace mimirmind::runtime;
    ThermalProfile p;
    p.name = "empty";
    SystemMonitor m{};
    ThermalGuard  g{p, m};

    const auto d = g.decide();
    EXPECT_TRUE(d.admit_new_request);
    EXPECT_EQ(d.pause.count(), std::int64_t{0});
    EXPECT_EQ(static_cast<int>(d.state),
              static_cast<int>(ThermalDecision::State::Ok));
}

TEST(thermalGuard_checkAdmissionDoesNotThrowOnEmpty) {
    using namespace mimirmind::runtime;
    ThermalProfile p;
    p.name = "empty";
    SystemMonitor m{};
    ThermalGuard  g{p, m};
    g.checkAdmission();   // must not throw
}

// =======================================================================
// runtime::SystemMonitor — verify it can find at least one of the two
// expected sensor sources on a Linux host. This is a smoke test; it
// passes on the build container which has the standard sysfs layout.
// =======================================================================

TEST(systemMonitor_findsPackageTempSource) {
    using namespace mimirmind::runtime;
    SystemMonitor m{};
    // Either path is acceptable — the constructor probes both.
    // On a container without coretemp + without x86_pkg_temp this would
    // be empty; in that environment the test reports "(none detected)"
    // which is also a valid outcome, so just exercise the API.
    const auto src = m.packageTempSource();
    EXPECT_TRUE(!src.empty());
}

TEST(systemMonitor_readReturnsRamFigures) {
    using namespace mimirmind::runtime;
    SystemMonitor m{};
    const auto r = m.read();
    // /proc/meminfo is virtually always available; we don't assert on
    // package temp because exotic kernels may lack the thermal zone.
    EXPECT_TRUE(r.ram_total_mib.has_value());
    EXPECT_TRUE(r.ram_available_mib.has_value());
    if (r.ram_total_mib && r.ram_available_mib) {
        EXPECT_TRUE(*r.ram_available_mib <= *r.ram_total_mib);
    }
}

// =======================================================================
// runtime::PowerMonitor — RAPL probing + wrap math
//
// The build container does not have RAPL exposed (Docker masks
// /sys/devices/virtual/powercap by default), so the smoke test just
// verifies the unavailable path. To test the happy path we point the
// monitor at a synthetic powercap tree we build under /tmp.
// =======================================================================

namespace {

struct FakeRapl {
    std::string root;

    FakeRapl() {
        char templ[] = "/tmp/mimirmind-rapl-XXXXXX";
        root = ::mkdtemp(templ);
    }
    ~FakeRapl() {
        // Recursive rmdir via /bin/rm is fine in test scope.
        const std::string cmd = "rm -rf " + root;
        const int rc = std::system(cmd.c_str());
        (void)rc;
    }
    FakeRapl(const FakeRapl&) = delete;
    FakeRapl& operator=(const FakeRapl&) = delete;

    void writeFile(const std::string& rel, const std::string& body) const {
        const std::string full = root + "/" + rel;
        std::ofstream{full} << body;
    }
    void addDomain(const std::string& dirname,
                   const std::string& name,
                   std::uint64_t      energyUj,
                   std::uint64_t      maxRangeUj) const {
        const std::string sub = root + "/" + dirname;
        ::mkdir(sub.c_str(), 0755);
        std::ofstream{sub + "/name"}                  << name;
        std::ofstream{sub + "/energy_uj"}             << energyUj;
        std::ofstream{sub + "/max_energy_range_uj"}   << maxRangeUj;
    }
    void setEnergy(const std::string& dirname, std::uint64_t energyUj) const {
        std::ofstream{root + "/" + dirname + "/energy_uj"} << energyUj;
    }
};

} // namespace

TEST(powerMonitor_unavailableOnEmptyTree) {
    FakeRapl r;
    mimirmind::runtime::PowerMonitor m{r.root};
    EXPECT_TRUE(!m.available());
    EXPECT_TRUE(!m.unavailableReason().empty());
}

TEST(powerMonitor_findsPackageAndPsys) {
    FakeRapl r;
    r.addDomain("intel-rapl:0",   "package-0", 1'000'000ULL, 65'536'000'000ULL);
    r.addDomain("intel-rapl:0:0", "core",        500'000ULL, 65'536'000'000ULL);
    r.addDomain("intel-rapl:1",   "psys",      2'000'000ULL, 65'536'000'000ULL);

    mimirmind::runtime::PowerMonitor m{r.root};
    EXPECT_TRUE(m.available());
    const auto names = m.domainNames();
    EXPECT_EQ(names.size(), std::size_t{3});
}

TEST(powerMonitor_dedupesByName) {
    // Reproduces the observed Meteor Lake layout: package-0 is exposed
    // both at the top level (intel-rapl:0) AND as a sub-domain of psys
    // (intel-rapl:1:0). Both read the same physical counter — keeping
    // only one is the right behavior.
    FakeRapl r;
    r.addDomain("intel-rapl:0",   "package-0", 1'000'000ULL, 65'536'000'000ULL);
    r.addDomain("intel-rapl:0:0", "core",        500'000ULL, 65'536'000'000ULL);
    r.addDomain("intel-rapl:0:1", "uncore",      100'000ULL, 65'536'000'000ULL);
    r.addDomain("intel-rapl:1",   "psys",      3'000'000ULL, 65'536'000'000ULL);
    r.addDomain("intel-rapl:1:0", "package-0", 1'000'000ULL, 65'536'000'000ULL);

    mimirmind::runtime::PowerMonitor m{r.root};
    EXPECT_TRUE(m.available());
    const auto names = m.domainNames();
    // 4 distinct names — package-0 only counted once.
    EXPECT_EQ(names.size(), std::size_t{4});
    int packageCount = 0;
    for (const auto& n : names) {
        if (n == "package-0") ++packageCount;
    }
    EXPECT_EQ(packageCount, 1);
}

TEST(powerMonitor_energyBetweenSimpleDelta) {
    FakeRapl r;
    r.addDomain("intel-rapl:0", "package-0", 0ULL, 65'536'000'000ULL);

    mimirmind::runtime::PowerMonitor m{r.root};
    EXPECT_TRUE(m.available());

    const auto s0 = m.snapshot();
    // Bump energy by 2 J = 2_000_000 µJ.
    r.setEnergy("intel-rapl:0", 2'000'000ULL);
    const auto s1 = m.snapshot();

    const auto joules = m.energyBetween(s0, s1);
    EXPECT_EQ(joules.size(), std::size_t{1});
    EXPECT_NEAR(static_cast<float>(joules[0]), 2.0F, 1e-6F);
}

TEST(powerMonitor_energyBetweenHandlesWrap) {
    FakeRapl r;
    r.addDomain("intel-rapl:0", "package-0",
                65'536'000'000ULL - 1'000'000ULL,   // start 1 J below the cap
                65'536'000'000ULL);

    mimirmind::runtime::PowerMonitor m{r.root};
    const auto s0 = m.snapshot();
    // Wrap: counter resets and climbs to 500_000 µJ past zero.
    r.setEnergy("intel-rapl:0", 500'000ULL);
    const auto s1 = m.snapshot();

    const auto joules = m.energyBetween(s0, s1);
    EXPECT_EQ(joules.size(), std::size_t{1});
    // Expected delta = 1 J before wrap + 0.5 J after = 1.5 J.
    EXPECT_NEAR(static_cast<float>(joules[0]), 1.5F, 1e-6F);
}

TEST(powerMonitor_averageWattsBetween) {
    FakeRapl r;
    r.addDomain("intel-rapl:0", "package-0", 0ULL, 65'536'000'000ULL);

    mimirmind::runtime::PowerMonitor m{r.root};
    const auto s0 = m.snapshot();
    // Wait a little to get a measurable Δt.
    std::this_thread::sleep_for(std::chrono::milliseconds{20});
    r.setEnergy("intel-rapl:0", 100'000ULL);   // 0.1 J consumed
    const auto s1 = m.snapshot();

    const auto watts = m.averageWattsBetween(s0, s1);
    EXPECT_EQ(watts.size(), std::size_t{1});
    // 0.1 J / >=0.02 s = at most 5 W; sanity-check shape rather than
    // value because steady_clock resolution varies.
    EXPECT_TRUE(watts[0] > 0.0 && watts[0] < 50.0);
}

// =======================================================================
// runtime::GpuClockGovernor — probe + P-controller against a synthetic
// /sys/class/drm tree.
// =======================================================================

namespace {

struct FakeDrm {
    std::string root;

    FakeDrm() {
        char templ[] = "/tmp/mimirmind-drm-XXXXXX";
        root = ::mkdtemp(templ);
    }
    ~FakeDrm() {
        const std::string cmd = "rm -rf " + root;
        const int rc = std::system(cmd.c_str());
        (void)rc;
    }
    FakeDrm(const FakeDrm&) = delete;
    FakeDrm& operator=(const FakeDrm&) = delete;

    void addCard(const std::string& cardName,
                 std::uint32_t      rp0,
                 std::uint32_t      rpn,
                 std::uint32_t      currentMax) const {
        const std::string dir = root + "/" + cardName;
        ::mkdir(dir.c_str(), 0755);
        std::ofstream{dir + "/gt_RP0_freq_mhz"} << rp0;
        std::ofstream{dir + "/gt_RPn_freq_mhz"} << rpn;
        std::ofstream{dir + "/gt_max_freq_mhz"} << currentMax;
    }

    std::uint32_t readMax(const std::string& cardName) const {
        std::ifstream in{root + "/" + cardName + "/gt_max_freq_mhz"};
        std::uint32_t v = 0;
        in >> v;
        return v;
    }
};

} // namespace

TEST(gpuGovernor_unavailableOnEmptyTree) {
    FakeDrm d;
    mimirmind::runtime::GpuClockGovernor g{d.root};
    EXPECT_TRUE(!g.available());
    EXPECT_TRUE(!g.unavailableReason().empty());
}

TEST(gpuGovernor_findsCard1NotCard0) {
    // Real-world layout from a NUC14 — iGPU is card1, card0 is missing
    // (or is a virtual DRM device without gt_*_freq_mhz).
    FakeDrm d;
    d.addCard("card1", 2350, 800, 2350);
    mimirmind::runtime::GpuClockGovernor g{d.root};
    EXPECT_TRUE(g.available());
    EXPECT_EQ(g.rp0Mhz(), std::uint32_t{2350});
    EXPECT_EQ(g.rpnMhz(), std::uint32_t{800});
    EXPECT_EQ(g.currentCapMhz(), std::uint32_t{2350});
}

TEST(gpuGovernor_setMaxClampsToBounds) {
    FakeDrm d;
    d.addCard("card1", 2350, 800, 2350);
    mimirmind::runtime::GpuClockGovernor g{d.root};

    // Below RPn → clamp up to RPn.
    EXPECT_EQ(g.setMaxFreqMhz(100), std::uint32_t{800});
    EXPECT_EQ(d.readMax("card1"), std::uint32_t{800});

    // Above RP0 → clamp down to RP0.
    EXPECT_EQ(g.setMaxFreqMhz(9999), std::uint32_t{2350});
    EXPECT_EQ(d.readMax("card1"), std::uint32_t{2350});

    // In-range → applies as-is.
    EXPECT_EQ(g.setMaxFreqMhz(1500), std::uint32_t{1500});
    EXPECT_EQ(d.readMax("card1"), std::uint32_t{1500});
}

TEST(gpuGovernor_adjustForTempLowersCapWhenHot) {
    FakeDrm d;
    d.addCard("card1", 2350, 800, 2000);
    mimirmind::runtime::GpuClockGovernor g{d.root};
    g.setTargetTempC(72.0F);

    // Temp 8 °C above target, kGainDown = 100 → delta = -800 MHz,
    // new cap = 1200. (M9.6.2 briefly relaxed this to 50 but got
    // rolled back after a 2026-07-01 thermal shutdown — the paranoid
    // gain is the one that survives sustained load on NUC14.)
    EXPECT_EQ(g.adjustForTemp(80.0F), std::uint32_t{1200});
}

TEST(gpuGovernor_adjustForTempRaisesCapWhenCool) {
    FakeDrm d;
    d.addCard("card1", 2350, 800, 1500);
    mimirmind::runtime::GpuClockGovernor g{d.root};
    g.setTargetTempC(72.0F);

    // Temp 5 °C below target. M9.6.6 replaced the fixed kGainUp=10 with a
    // headroom-scaled adaptive up-gain, so with full headroom the 5 °C
    // error yields +125 MHz → cap = 1625. Slow on the cool side is still
    // intentional (heatsink takes 30-60 s to shed load) but headroom-aware.
    EXPECT_EQ(g.adjustForTemp(67.0F), std::uint32_t{1625});
}

TEST(gpuGovernor_adjustForTempAsymmetricGain) {
    // Same |error| in either direction must still produce different
    // deltas — that is the asymmetry baked into the controller.
    FakeDrm d;
    d.addCard("card1", 2350, 800, 1500);
    mimirmind::runtime::GpuClockGovernor g{d.root};
    g.setTargetTempC(72.0F);

    // +4 °C, kGainDown = 100 → delta = -400, cap → 1100.
    EXPECT_EQ(g.adjustForTemp(76.0F), std::uint32_t{1100});

    // Re-arm cap at 1500 and check the cool direction.
    g.setMaxFreqMhz(1500);
    // -4 °C, M9.6.6 headroom-scaled adaptive up-gain → delta = +80, cap
    // → 1580. Hot-side (kGainDown=100) still dominates the cool side —
    // the paranoid asymmetry from M9.6 is preserved.
    EXPECT_EQ(g.adjustForTemp(68.0F), std::uint32_t{1580});
}

TEST(gpuGovernor_adjustForTempDeadbandHoldsCap) {
    FakeDrm d;
    d.addCard("card1", 2350, 800, 1500);
    mimirmind::runtime::GpuClockGovernor g{d.root};
    g.setTargetTempC(72.0F);

    // Within ±0.5 °C of target — no movement.
    EXPECT_EQ(g.adjustForTemp(71.6F), std::uint32_t{1500});
    EXPECT_EQ(g.adjustForTemp(72.0F), std::uint32_t{1500});
    EXPECT_EQ(g.adjustForTemp(72.4F), std::uint32_t{1500});
    EXPECT_EQ(d.readMax("card1"), std::uint32_t{1500});

    // Just outside the band — exact-rep float, +1.0 °C, kGainDown = 100
    // → delta = -100, cap → 1400. (Picking 72.6F here would be misleading:
    // 72.6F-72.0F is 0.5999998F, not 0.6F, so int truncation gives -59
    // not -60.)
    EXPECT_EQ(g.adjustForTemp(73.0F), std::uint32_t{1400});
}

TEST(gpuGovernor_adjustForTempClampsToRpnOnExtremeHot) {
    // Pathological case: chip is 20 °C above target. The aggressive
    // down-gain wants to subtract 2000 MHz, which would underflow
    // below RPn — must clamp to RPn, not wrap.
    FakeDrm d;
    d.addCard("card1", 2350, 800, 1500);
    mimirmind::runtime::GpuClockGovernor g{d.root};
    g.setTargetTempC(72.0F);

    EXPECT_EQ(g.adjustForTemp(92.0F), std::uint32_t{800});
}

TEST(gpuGovernor_destructorRestoresRP0) {
    FakeDrm d;
    d.addCard("card1", 2350, 800, 2350);
    {
        mimirmind::runtime::GpuClockGovernor g{d.root};
        g.setMaxFreqMhz(1200);
        EXPECT_EQ(d.readMax("card1"), std::uint32_t{1200});
    }
    // After dtor — should be back to RP0.
    EXPECT_EQ(d.readMax("card1"), std::uint32_t{2350});
}

TEST(gpuGovernor_pinRecordsIntentAndClampsMhz) {
    // M9.11.a — pin() must clamp like setMaxFreqMhz and expose the
    // intent + raw env for /system/info reporting.
    FakeDrm d;
    d.addCard("card1", 2350, 800, 2350);
    mimirmind::runtime::GpuClockGovernor g{d.root};

    EXPECT_TRUE(!g.pinned());

    // Numeric in-range.
    EXPECT_EQ(g.pin(1200, "numeric", "1200"), std::uint32_t{1200});
    EXPECT_TRUE(g.pinned());
    EXPECT_EQ(g.pinnedMhz(), std::uint32_t{1200});
    EXPECT_EQ(g.pinIntent(), "numeric");
    EXPECT_EQ(g.pinRawEnv(), "1200");
    EXPECT_EQ(d.readMax("card1"), std::uint32_t{1200});

    // Below RPn — clamped up.
    EXPECT_EQ(g.pin(100, "numeric", "100"), std::uint32_t{800});
    EXPECT_EQ(g.pinnedMhz(), std::uint32_t{800});
    EXPECT_EQ(d.readMax("card1"), std::uint32_t{800});

    // rpn intent.
    EXPECT_EQ(g.pin(g.rpnMhz(), "rpn", "rpn"), std::uint32_t{800});
    EXPECT_EQ(g.pinIntent(), "rpn");
    EXPECT_EQ(g.pinRawEnv(), "rpn");
}

TEST(thermalProfile_loadsGpuTargetTempC) {
    TempJsonFile f{R"({
        "name": "test",
        "package_temp_soft_c": 78,
        "package_temp_hard_c": 85,
        "gpu_target_temp_c": 72
    })"};
    const auto p = mimirmind::runtime::loadThermalProfile(f.path);
    EXPECT_TRUE(p.hasGpuClockTarget());
    EXPECT_NEAR(*p.gpu_target_temp_c, 72.0F, 0.001F);
}

TEST(thermalProfile_rejectsGpuTargetAboveSoft) {
    // Sanity: target above soft means the per-token pause would fire
    // before the governor — defeats the point.
    TempJsonFile f{R"({
        "name": "bad",
        "package_temp_soft_c": 75,
        "package_temp_hard_c": 82,
        "gpu_target_temp_c": 80
    })"};
    try {
        (void)mimirmind::runtime::loadThermalProfile(f.path);
        EXPECT_TRUE(false && "expected throw");
    } catch (const std::runtime_error&) {
        // expected
    }
}

// ---- ToolCallParser — Qwen3-Coder XML dialect (8.19.6) ---------------------
//
// Qwen3.5/3.6/3.8 chat templates carry tool calls as
//   <tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n
//   </function>\n</tool_call>
// with parameter values typed via the offered tool's JSON schema.

namespace {

std::vector<mimirmind::model::ToolSpec> weatherSpecs() {
    return {mimirmind::model::ToolSpec{
        "get_weather",
        R"({"type":"function","function":{"name":"get_weather",)"
        R"("description":"d","parameters":{"type":"object","properties":)"
        R"({"city":{"type":"string"},"days":{"type":"integer"},)"
        R"("detailed":{"type":"boolean"}},"required":["city"]}}})"}};
}

} // namespace

TEST(toolParser_qwenXml_basicTypedCall) {
    using mimirmind::model::ToolCallParser;
    const auto calls = ToolCallParser::parseQwenXml(
        "<tool_call>\n<function=get_weather>\n"
        "<parameter=city>\nBerlin\n</parameter>\n"
        "<parameter=days>\n3\n</parameter>\n"
        "<parameter=detailed>\ntrue\n</parameter>\n"
        "</function>\n</tool_call>",
        weatherSpecs());
    EXPECT_EQ(calls.size(), std::size_t{1});
    EXPECT_TRUE(calls[0].name == "get_weather");
    EXPECT_TRUE(calls[0].argumentsJson ==
                R"({"city":"Berlin","days":3,"detailed":true})");
}

TEST(toolParser_qwenXml_multilineStringStaysRaw) {
    using mimirmind::model::ToolCallParser;
    const auto calls = ToolCallParser::parseQwenXml(
        "prose before\n<tool_call>\n<function=get_weather>\n"
        "<parameter=city>\nline one\nline two\n</parameter>\n"
        "</function>\n</tool_call>",
        weatherSpecs());
    EXPECT_EQ(calls.size(), std::size_t{1});
    EXPECT_TRUE(calls[0].argumentsJson == R"({"city":"line one\nline two"})");
}

TEST(toolParser_qwenXml_numericStringNotCoerced) {
    // A "string"-typed parameter whose value looks numeric must STAY a string.
    using mimirmind::model::ToolCallParser;
    const auto calls = ToolCallParser::parseQwenXml(
        "<tool_call>\n<function=get_weather>\n"
        "<parameter=city>\n1234\n</parameter>\n</function>\n</tool_call>",
        weatherSpecs());
    EXPECT_EQ(calls.size(), std::size_t{1});
    EXPECT_TRUE(calls[0].argumentsJson == R"({"city":"1234"})");
}

TEST(toolParser_qwenXml_truncatedBlockStillParses) {
    // Output cut at max_tokens: no </parameter>/</function>/</tool_call>.
    using mimirmind::model::ToolCallParser;
    const auto calls = ToolCallParser::parseQwenXml(
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis",
        weatherSpecs());
    EXPECT_EQ(calls.size(), std::size_t{1});
    EXPECT_TRUE(calls[0].argumentsJson == R"({"city":"Paris"})");
}

TEST(toolParser_qwenXml_multipleCallsAndIds) {
    using mimirmind::model::ToolCallParser;
    const auto calls = ToolCallParser::parseQwenXml(
        "<tool_call>\n<function=get_weather>\n"
        "<parameter=city>\nRome\n</parameter>\n</function>\n</tool_call>\n"
        "<tool_call>\n<function=get_weather>\n"
        "<parameter=city>\nOslo\n</parameter>\n</function>\n</tool_call>",
        weatherSpecs());
    EXPECT_EQ(calls.size(), std::size_t{2});
    EXPECT_TRUE(calls[0].id == "call_0");
    EXPECT_TRUE(calls[1].id == "call_1");
    EXPECT_TRUE(calls[1].argumentsJson == R"({"city":"Oslo"})");
}

TEST(toolParser_qwenXml_hermesBodyYieldsNothing) {
    // A Hermes JSON body has no <function=…> header — the XML parser must
    // skip it (the handler then falls back to parseQwen).
    using mimirmind::model::ToolCallParser;
    const auto calls = ToolCallParser::parseQwenXml(
        "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": "
        "{\"city\": \"Bonn\"}}\n</tool_call>",
        weatherSpecs());
    EXPECT_EQ(calls.size(), std::size_t{0});
}

TEST(toolParser_qwenHermes_braceRepairRecoversTruncatedJson) {
    // Seen live: Qwen3.6 imitating Hermes dropped the final '}' before
    // </tool_call>. The brace-balance repair pass must recover the call.
    using mimirmind::model::ToolCallParser;
    const auto calls = ToolCallParser::parseQwen(
        "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": "
        "{\"city\": \"Kiel\"}\n</tool_call>");
    EXPECT_EQ(calls.size(), std::size_t{1});
    EXPECT_TRUE(calls[0].name == "get_weather");
    EXPECT_TRUE(calls[0].argumentsJson == R"({"city":"Kiel"})");
}

TEST(toolParser_qwenXmlBare_missingEnvelopeParses) {
    // Captured live (8.19.9): third call after two error tool_responses —
    // the <tool_call> envelope is dropped but a dangling </tool_call>
    // trails the payload.
    using mimirmind::model::ToolCallParser;
    const auto calls = ToolCallParser::parseQwenXmlBare(
        "<function=get_weather>\n"
        "<parameter=city>\nMoers\n</parameter>\n"
        "</function>\n</tool_call>",
        weatherSpecs());
    EXPECT_EQ(calls.size(), std::size_t{1});
    EXPECT_TRUE(calls[0].name == "get_weather");
    EXPECT_TRUE(calls[0].argumentsJson == R"({"city":"Moers"})");
}

TEST(toolParser_qwenXmlBare_unknownNameIgnored) {
    using mimirmind::model::ToolCallParser;
    const std::string text =
        "<function=rm_rf>\n<parameter=path>\n/\n</parameter>\n</function>";
    EXPECT_TRUE(!ToolCallParser::looksLikeBareQwenXmlCall(text, weatherSpecs()));
    EXPECT_EQ(ToolCallParser::parseQwenXmlBare(text, weatherSpecs()).size(),
              std::size_t{0});
}

TEST(toolParser_qwenXmlBare_fencedExampleIgnored) {
    // Prose documenting the call format inside a code fence is NOT a call.
    using mimirmind::model::ToolCallParser;
    const auto calls = ToolCallParser::parseQwenXmlBare(
        "Use this format:\n```\n<function=get_weather>\n"
        "<parameter=city>\nBerlin\n</parameter>\n</function>\n```\nGot it?",
        weatherSpecs());
    EXPECT_EQ(calls.size(), std::size_t{0});
}

TEST(toolDetector_bareBlockBufferedAndDanglingSwallowed) {
    using mimirmind::model::ChatTemplate;
    using mimirmind::model::ToolCallStreamDetector;
    ToolCallStreamDetector d{ChatTemplate::Style::QwenChatML, /*enabled=*/true};

    std::string t1 = "Let me fix that.\n\n";
    EXPECT_TRUE(!d.feed(t1));               // prose passes through (partly held)
    std::string t2 = "<function=get_weather>\n<parameter=city>\nMoers\n"
                     "</parameter>\n</function>";
    const bool completed = d.feed(t2);
    EXPECT_TRUE(completed);
    EXPECT_TRUE(d.completedBare());
    EXPECT_TRUE(d.completedBlock().rfind("<function=", 0) == 0);
    d.reset();

    // The dangling closer of the envelope the model never opened must be
    // swallowed, not streamed as content.
    std::string t3 = "\n</tool_call>";
    EXPECT_TRUE(!d.feed(t3));
    EXPECT_TRUE(t3.empty());
    EXPECT_TRUE(d.takePendingFlush().empty());
}

TEST(toolDetector_wrappedBlockUnchanged) {
    using mimirmind::model::ChatTemplate;
    using mimirmind::model::ToolCallStreamDetector;
    ToolCallStreamDetector d{ChatTemplate::Style::QwenChatML, /*enabled=*/true};
    std::string t = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": "
                    "{\"city\": \"Bonn\"}}\n</tool_call>";
    EXPECT_TRUE(d.feed(t));
    EXPECT_TRUE(!d.completedBare());
    EXPECT_TRUE(d.completedBlock().rfind("<tool_call>", 0) == 0);
}

TEST(toolDetector_fencedBareOpenerNotCaptured) {
    using mimirmind::model::ChatTemplate;
    using mimirmind::model::ToolCallStreamDetector;
    ToolCallStreamDetector d{ChatTemplate::Style::QwenChatML, /*enabled=*/true};
    std::string t = "Example:\n```\n<function=get_weather>\nx\n</function>\n"
                    "```\ndone and finished";
    EXPECT_TRUE(!d.feed(t));                // never enters buffering
    std::string tail = d.takePendingFlush();
    const std::string all = t + tail;
    EXPECT_TRUE(all.find("<function=get_weather>") != std::string::npos);
}

TEST(toolParser_markupSalad_detected) {
    using mimirmind::model::ToolCallParser;
    // Captured live (round 7 of the multi-step E2E).
    EXPECT_TRUE(ToolCallParser::looksLikeToolMarkupSalad(
        "<tooltool_0>\n\n</invoke>"));
    EXPECT_TRUE(ToolCallParser::looksLikeToolMarkupSalad(
        "<|/tool_call_start|>"));
    // Keyword-FREE tag salad is deliberately NOT matched (surfacing beats
    // force-re-decoding — a live 44-byte HTML answer was eaten by the old
    // short-and-tag-dominated rule).
    EXPECT_TRUE(!ToolCallParser::looksLikeToolMarkupSalad("<x_1></y_2>"));
    EXPECT_TRUE(!ToolCallParser::looksLikeToolMarkupSalad(
        "<div class=greeting><span>Hello</span></div>"));
}

TEST(toolParser_markupSalad_notFooledByProseOrHtml) {
    using mimirmind::model::ToolCallParser;
    EXPECT_TRUE(!ToolCallParser::looksLikeToolMarkupSalad(
        "The answer is 4. No tags here."));
    // A long legitimate HTML answer: tag-heavy but no tool keywords and
    // too long for the keyword-free rule.
    std::string html = "<!doctype html>\n";
    for (int i = 0; i < 30; ++i) {
        html += "<div class=\"row\"><span>item</span><p>text body here "
                "with some real words</p></div>\n";
    }
    EXPECT_TRUE(!ToolCallParser::looksLikeToolMarkupSalad(html));
}

TEST(toolParser_qwenHermes_intactJsonStillParses) {
    using mimirmind::model::ToolCallParser;
    const auto calls = ToolCallParser::parseQwen(
        "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": "
        "{\"city\": \"Kiel\"}}\n</tool_call>");
    EXPECT_EQ(calls.size(), std::size_t{1});
    EXPECT_TRUE(calls[0].argumentsJson == R"({"city":"Kiel"})");
}

int main() {
    return mm::test::run();
}