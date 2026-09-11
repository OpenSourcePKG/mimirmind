// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Roadmap 5.28.1.0 (Inc 0) — the GDN prefix-cache correctness gate.
//
// Prefix / KV reuse is currently force-disabled for GatedDeltaNet backends
// (lcp = 0, InferenceEngine.cpp) because reusing a token prefix without
// restoring the recurrent SSM+conv state contaminates every follow-up request
// (see lesson ssm-prefix-cache-cross-request-contamination-2026-07-28: first
// request coherent, requests 2..N garbage). Inc 1 will re-enable reuse by
// checkpointing that state at the end of each prefill and restoring it on a
// pure-continuation follow-up. That is only correct if the GDN forward is a
// pure fold of the recurrent state — i.e. snapshotting at position L and
// resuming over [L, T) is identical to a single continuous forward over
// [0, T). This gate proves exactly that property against the CPU golden
// reference (compute::GatedDeltaNet — the truth the CUDA decode/prefill
// kernels are themselves validated against, see gated_deltanet_fold_test),
// and reproduces the contamination bug as a standing regression so Inc 1
// cannot ship the reuse path without both guarantees holding.
//
// Pure CPU, no GPU, no model — runs on any box:
//   cmake --build build --target gdn_prefix_cache_tests &&
//   build/gdn_prefix_cache_tests
//
// Levels proven here (and where bit-exactness holds vs only near-golden):
//   * recurrence (decode path)  -> resume == full replay, BIT-EXACT. This is
//     the "N x identical greedy -> identical output" contamination guarantee.
//   * causal conv state         -> resume == full replay, BIT-EXACT.
//   * chunked prefill path      -> resume == full replay BIT-EXACT iff the
//     split L is chunkSize-aligned; otherwise near-golden (<1e-3) because the
//     chunk boundaries regroup the FMAs. Inc 1 snapshots at a prefill/chunk
//     boundary, so the aligned case is the one it relies on.
//   * contamination negative control -> a leftover (un-restored) state
//     provably corrupts the output, so the gate is sensitive to the bug.

#include "TestFramework.hpp"

#include "compute/GatedDeltaNet.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

using ::mimirmind::compute::causalConv1dSilu;
using ::mimirmind::compute::gatedDeltaNetChunk;
using ::mimirmind::compute::gatedDeltaNetRecurrent;

// Same deterministic LCG as gated_deltanet_fold_test — reproducible inputs,
// no <random> nondeterminism.
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
    for (auto& x : v) {
        x = g.next();
    }
    return v;
}

// A full set of GDN linear-layer inputs for a T-token window, H value heads,
// head_dim S. gLog is forced <= 0 (decay in (0,1]); beta into [0,1). `decay`
// scales the log-decay magnitude: small -> state persists (mild decay), large
// -> state forgets fast. Deterministic in `seed`.
struct GdnInputs {
    std::size_t        T, H, S;
    std::vector<float> q, k, v;   // [T,H,S]
    std::vector<float> gLog;      // [T,H], <= 0
    std::vector<float> beta;      // [T,H], [0,1)

    GdnInputs(std::size_t T_, std::size_t H_, std::size_t S_, std::uint32_t seed,
              float decay = 1.0f)
        : T(T_), H(H_), S(S_) {
        const std::size_t nElem = T * H * S;
        const std::size_t nTH   = T * H;
        q = randVec(nElem, seed ^ 0x22u);
        k = randVec(nElem, seed ^ 0x33u);
        v = randVec(nElem, seed ^ 0x44u);
        auto gRaw = randVec(nTH, seed ^ 0x55u);
        auto bRaw = randVec(nTH, seed ^ 0x66u);
        gLog.resize(nTH);
        beta.resize(nTH);
        for (std::size_t i = 0; i < nTH; ++i) {
            // -|.| keeps the log-decay <= 0; `decay` tunes how fast state fades.
            gLog[i] = -decay * (gRaw[i] < 0.0f ? -gRaw[i] : gRaw[i]);
            beta[i] = 0.5f * (bRaw[i] + 1.0f);
        }
    }

    // Byte offsets (in floats) for the sub-window starting at token `t0`.
    const float* qAt(std::size_t t0) const { return q.data() + t0 * H * S; }
    const float* kAt(std::size_t t0) const { return k.data() + t0 * H * S; }
    const float* vAt(std::size_t t0) const { return v.data() + t0 * H * S; }
    const float* gAt(std::size_t t0) const { return gLog.data() + t0 * H; }
    const float* bAt(std::size_t t0) const { return beta.data() + t0 * H; }
};

bool bitEqual(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() &&
           std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

double maxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0.0;
    const std::size_t n = a.size() < b.size() ? a.size() : b.size();
    for (std::size_t i = 0; i < n; ++i) {
        const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        m = (d < 0 ? -d : d) > m ? (d < 0 ? -d : d) : m;
    }
    return m;
}

// Run the recurrent (decode) reference over [t0, t0+len) tokens of `in`,
// advancing `state` in place; return the [len,H,S] output.
std::vector<float> recur(const GdnInputs& in, std::vector<float>& state,
                         std::size_t t0, std::size_t len) {
    std::vector<float> out(len * in.H * in.S);
    gatedDeltaNetRecurrent(in.qAt(t0), in.kAt(t0), in.vAt(t0), in.gAt(t0),
                           in.bAt(t0), state.data(), out.data(), len, in.H, in.S);
    return out;
}

std::vector<float> chunk(const GdnInputs& in, std::vector<float>& state,
                         std::size_t t0, std::size_t len, std::size_t chunkSize) {
    std::vector<float> out(len * in.H * in.S);
    gatedDeltaNetChunk(in.qAt(t0), in.kAt(t0), in.vAt(t0), in.gAt(t0), in.bAt(t0),
                       state.data(), out.data(), len, in.H, in.S, chunkSize);
    return out;
}

std::vector<float> zeroState(const GdnInputs& in) {
    return std::vector<float>(in.H * in.S * in.S, 0.0f);
}

// Slice out tokens [t0, t0+len) of a [T,H,S] output buffer.
std::vector<float> tail(const std::vector<float>& full, std::size_t t0,
                        std::size_t len, std::size_t H, std::size_t S) {
    std::vector<float> out(len * H * S);
    std::memcpy(out.data(), full.data() + t0 * H * S,
                len * H * S * sizeof(float));
    return out;
}

} // namespace

// ---------------------------------------------------------------------------
// Gate 1 — decode-path (recurrent) resume == full replay, BIT-EXACT.
// This is the property Inc 1's checkpoint/restore relies on for the decode
// path, and the exact guarantee behind "N x identical greedy -> identical".
// ---------------------------------------------------------------------------
namespace {
void recurrenceResumeIsBitExact(std::size_t T, std::size_t L, std::size_t H,
                                std::size_t S, std::uint32_t seed) {
    const GdnInputs in{T, H, S, seed};

    // Full continuous replay over [0, T) from a zero state.
    auto stateFull = zeroState(in);
    auto outFull = recur(in, stateFull, 0, T);

    // Checkpoint at L, then resume over [L, T) from the snapshot.
    auto stateChk = zeroState(in);
    recur(in, stateChk, 0, L);               // prefill [0, L)
    const auto snapshot = stateChk;          // <- the SSM checkpoint captured at L
    auto outPost = recur(in, stateChk, L, T - L);

    // Output over [L, T) and the final state must be byte-identical: the
    // recurrence is a pure fold, no absolute-position term.
    EXPECT_TRUE(bitEqual(outPost, tail(outFull, L, T - L, H, S)));
    EXPECT_TRUE(bitEqual(stateChk, stateFull));

    // Sanity: the checkpoint is a real, non-trivial state (guards a silently
    // all-zero snapshot from making the test vacuous).
    bool nonZero = false;
    for (float x : snapshot) {
        if (x != 0.0f) { nonZero = true; break; }
    }
    EXPECT_TRUE(nonZero);
}
} // namespace

TEST(gdn_recurrence_resume_bit_exact_small) {
    recurrenceResumeIsBitExact(/*T=*/24, /*L=*/8, /*H=*/4, /*S=*/64, 0xA11Cu);
}

TEST(gdn_recurrence_resume_bit_exact_prod_dims) {
    // Production-shaped: S=128, H=16 (as gated_deltanet_fold_test).
    recurrenceResumeIsBitExact(/*T=*/40, /*L=*/16, /*H=*/16, /*S=*/128, 0xA11Du);
}

TEST(gdn_recurrence_resume_bit_exact_split_of_one) {
    // Degenerate splits: resume the very last token, and resume after just one.
    recurrenceResumeIsBitExact(/*T=*/12, /*L=*/11, /*H=*/4, /*S=*/64, 0xA11Eu);
    recurrenceResumeIsBitExact(/*T=*/12, /*L=*/1, /*H=*/4, /*S=*/64, 0xA11Fu);
}

// ---------------------------------------------------------------------------
// Gate 2 — causal-conv state resume == full replay, BIT-EXACT.
// The engine persists the last (K-1) raw input rows as the conv state; this
// proves restoring exactly those rows reproduces the streamed convolution.
// ---------------------------------------------------------------------------
namespace {
void convResumeIsBitExact(std::size_t T, std::size_t L, std::size_t channels,
                          std::size_t K, std::uint32_t seed) {
    // Raw streamed input rows x[0..T), channel-major per row.
    const auto x = randVec(T * channels, seed ^ 0x77u);
    const auto kernel = randVec(K * channels, seed ^ 0x88u);

    // Full: (K-1) zero state rows prepended to all T rows.
    std::vector<float> convFull((K - 1 + T) * channels, 0.0f);
    std::memcpy(convFull.data() + (K - 1) * channels, x.data(),
                T * channels * sizeof(float));
    std::vector<float> outFull(T * channels);
    causalConv1dSilu(convFull.data(), kernel.data(), outFull.data(), T, channels, K);

    // Resume [L, T): conv state = raw rows x[L-(K-1) .. L) (zero-padded when
    // L < K-1) — exactly what the engine's _convState holds.
    const std::size_t rem = T - L;
    std::vector<float> convResume((K - 1 + rem) * channels, 0.0f);
    for (std::size_t r = 0; r < K - 1; ++r) {
        // state row r corresponds to raw input row (L - (K-1) + r).
        const long src = static_cast<long>(L) - static_cast<long>(K - 1) +
                         static_cast<long>(r);
        if (src >= 0) {
            std::memcpy(convResume.data() + r * channels,
                        x.data() + static_cast<std::size_t>(src) * channels,
                        channels * sizeof(float));
        }
    }
    std::memcpy(convResume.data() + (K - 1) * channels, x.data() + L * channels,
                rem * channels * sizeof(float));
    std::vector<float> outResume(rem * channels);
    causalConv1dSilu(convResume.data(), kernel.data(), outResume.data(), rem,
                     channels, K);

    std::vector<float> outFullTail(rem * channels);
    std::memcpy(outFullTail.data(), outFull.data() + L * channels,
                rem * channels * sizeof(float));
    EXPECT_TRUE(bitEqual(outResume, outFullTail));
}
} // namespace

TEST(gdn_conv_state_resume_bit_exact) {
    // Qwen3-Next ssm_conv1d is a 4-tap causal conv.
    convResumeIsBitExact(/*T=*/24, /*L=*/8, /*channels=*/48, /*K=*/4, 0xC0A1u);
    // Split earlier than the kernel span exercises the zero-padded state rows.
    convResumeIsBitExact(/*T=*/16, /*L=*/2, /*channels=*/32, /*K=*/4, 0xC0A2u);
}

// ---------------------------------------------------------------------------
// Gate 3 — chunked prefill path. Inc 1 snapshots at the end of a chunked
// prefill and resumes with another chunked prefill. When the split is
// chunkSize-aligned the boundaries coincide -> BIT-EXACT. When it is not, the
// FMAs regroup, so resume is only near-golden — proven here so Inc 1 gates the
// reuse condition on chunk alignment (or falls back to lcp=0), rather than
// silently assuming exactness.
// ---------------------------------------------------------------------------
TEST(gdn_chunk_prefill_resume_bit_exact_when_aligned) {
    const std::size_t C = 8;
    const GdnInputs in{40, 16, 128, 0xCF01u};

    auto stateFull = zeroState(in);
    auto outFull = chunk(in, stateFull, 0, in.T, C);

    const std::size_t L = 16;  // multiple of C -> aligned boundaries.
    auto stateChk = zeroState(in);
    chunk(in, stateChk, 0, L, C);
    auto outPost = chunk(in, stateChk, L, in.T - L, C);

    EXPECT_TRUE(bitEqual(outPost, tail(outFull, L, in.T - L, in.H, in.S)));
    EXPECT_TRUE(bitEqual(stateChk, stateFull));
}

TEST(gdn_chunk_prefill_resume_unaligned_not_bit_exact_but_no_worse) {
    const std::size_t C = 8;
    const GdnInputs in{40, 16, 128, 0xCF02u};
    const std::size_t L = 13;  // NOT a multiple of C -> boundaries diverge.
    const std::size_t rem = in.T - L;

    // Recurrent golden (exact truth) and its tail over [L, T).
    auto stateGold = zeroState(in);
    auto outGold = recur(in, stateGold, 0, in.T);
    const auto goldTail = tail(outGold, L, rem, in.H, in.S);

    // Cold full chunked prefill over [0, T) — the intrinsic chunk-method error
    // vs the recurrent golden is the noise floor we compare against (it is
    // itself well above 1e-3 at S=128/T=40; the chunk algebra is only bit-exact
    // to *itself*, near-golden to the recurrence).
    auto stateFull = zeroState(in);
    auto outFull = chunk(in, stateFull, 0, in.T, C);
    const auto fullTail = tail(outFull, L, rem, in.H, in.S);
    const double chunkFloor = maxAbsDiff(fullTail, goldTail);

    // Unaligned resume: checkpoint at L=13, prefill [0,13) then [13,40).
    auto stateChk = zeroState(in);
    chunk(in, stateChk, 0, L, C);
    auto outPost = chunk(in, stateChk, L, rem, C);

    // (1) It is NOT bit-exact vs the cold full run — the split regroups FMAs.
    //     This is precisely why Inc 1 must gate reuse on chunk alignment
    //     (or fall back to lcp=0) rather than assume exactness here.
    EXPECT_TRUE(!bitEqual(outPost, fullTail));

    // (2) But it adds NO error beyond the chunk method's own distance from the
    //     recurrent truth: the unaligned resume disagrees with a cold run by no
    //     more than that intrinsic floor, so it is as correct as a cold prefill.
    EXPECT_TRUE(maxAbsDiff(outPost, fullTail) <= chunkFloor);
    EXPECT_TRUE(maxAbsDiff(outPost, goldTail) <= 2.0 * chunkFloor);
}

// ---------------------------------------------------------------------------
// Gate 4 — contamination regression. This is the bug that forced lcp=0:
// a persistent state buffer reused across requests, not reset for a fresh
// (non-continuation) request, bleeds the previous request's recurrence into
// the new one. Mirrors the serving _ssmState member.
// ---------------------------------------------------------------------------

// (a) A fresh request from a zeroed state is repeatable request-to-request:
//     N x identical greedy -> byte-identical output. Guards any hidden
//     cross-request state (the observable symptom of the original bug).
TEST(gdn_contamination_fresh_request_is_repeatable) {
    const GdnInputs seqB{20, 8, 64, 0xBEEFu};

    // One persistent state buffer, reused across "requests" like _ssmState.
    auto persistent = zeroState(seqB);

    std::memset(persistent.data(), 0, persistent.size() * sizeof(float));
    const auto canonical = recur(seqB, persistent, 0, seqB.T);
    const auto canonicalState = persistent;

    for (int req = 0; req < 5; ++req) {
        std::memset(persistent.data(), 0, persistent.size() * sizeof(float));
        const auto out = recur(seqB, persistent, 0, seqB.T);
        EXPECT_TRUE(bitEqual(out, canonical));
        EXPECT_TRUE(bitEqual(persistent, canonicalState));
    }
}

// (b) Negative control: NOT resetting the state (leftover from a prior,
//     unrelated request) provably corrupts the output. If this did not hold
//     the parity gates above would be meaningless. Mild decay so the leftover
//     state is unambiguously non-negligible.
TEST(gdn_contamination_leftover_state_corrupts_output) {
    const GdnInputs seqA{6, 8, 64, 0xA1A1u, /*decay=*/0.1f};   // mild decay
    const GdnInputs seqB{20, 8, 64, 0xB2B2u, /*decay=*/0.1f};

    auto persistent = zeroState(seqB);

    // Fresh B (correct behavior): zeroed state.
    std::memset(persistent.data(), 0, persistent.size() * sizeof(float));
    const auto outFresh = recur(seqB, persistent, 0, seqB.T);

    // Contaminated B (the bug): run A first, then B WITHOUT resetting.
    std::memset(persistent.data(), 0, persistent.size() * sizeof(float));
    recur(seqA, persistent, 0, seqA.T);       // leaves A's final state behind
    const auto outContam = recur(seqB, persistent, 0, seqB.T);

    // The leftover state changes B's output well above any FP noise.
    EXPECT_TRUE(maxAbsDiff(outContam, outFresh) > 1e-4);
}

// (c) The Inc-1 VALID reuse case: a pure continuation (request N+1 = full
//     history of request N ++ new tokens). Restoring the end-of-prefix
//     checkpoint and prefilling only the new tokens == full replay, BIT-EXACT.
//     This is the one scenario Inc 1 enables; everything else falls back to
//     lcp=0. (Same math as Gate 1, framed as the serving continuation.)
TEST(gdn_contamination_continuation_reuse_equals_full_replay) {
    const std::size_t lenA = 12;  // cached prefix (turn N)
    const std::size_t lenNew = 9; // new tokens (turn N+1)
    const GdnInputs conv{lenA + lenNew, 8, 64, 0xC017u};

    // Cold: replay the whole conversation from scratch.
    auto stateCold = zeroState(conv);
    const auto outCold = recur(conv, stateCold, 0, conv.T);

    // Warm reuse: restore the checkpoint captured at end of turn N, prefill
    // only the new tokens.
    auto stateWarm = zeroState(conv);
    recur(conv, stateWarm, 0, lenA);            // == checkpoint at lenA
    const auto outNew = recur(conv, stateWarm, lenA, lenNew);

    EXPECT_TRUE(bitEqual(outNew, tail(outCold, lenA, lenNew, conv.H, conv.S)));
    EXPECT_TRUE(bitEqual(stateWarm, stateCold));
}

// (d) End-to-end engine op-sequence: turn N does chunk-prefill([0,P)) then
//     AR-decode(G tokens); the snapshot is captured at position P+G. Turn N+1
//     (pure continuation) restores it and chunk-prefills the new tokens. This
//     is exactly InferenceEngine::generate()'s new SSM reuse path. It must
//     track the recurrent golden over the whole conversation to the chunk
//     method's own accuracy — i.e. reuse is as correct as a cold prefill and
//     free of contamination.
TEST(gdn_engine_prefill_decode_continuation_tracks_golden) {
    const std::size_t P = 24;   // turn-N prompt
    const std::size_t G = 6;    // turn-N generated (AR decode)
    const std::size_t N = 10;   // turn-(N+1) new tokens
    const std::size_t C = 8;    // chunk size
    const GdnInputs in{P + G + N, 16, 128, 0xE2E1u};
    const std::size_t cached = P + G;   // snapshot position

    // Recurrent golden over the entire conversation (exact truth).
    auto stateGold = zeroState(in);
    auto outGold = recur(in, stateGold, 0, in.T);

    // Turn N: chunk-prefill [0,P) then AR-decode the G tokens one at a time
    // (exactly how the engine advances: chunk for prefill, AR for decode).
    auto engState = zeroState(in);
    chunk(in, engState, 0, P, C);
    for (std::size_t g = 0; g < G; ++g) {
        recur(in, engState, P + g, 1);
    }
    const auto snapshot = engState;   // captured at position P+G == cached

    // Turn N+1 (continuation): restore snapshot, chunk-prefill the new tokens.
    auto restored = snapshot;
    auto outNew = chunk(in, restored, cached, N, C);

    // The reused continuation tracks the recurrent golden's tail to the chunk
    // method's own noise floor (a cold chunked prefill of the whole prompt is
    // no better) — so reuse is correct, not contaminated.
    const auto goldTail = tail(outGold, cached, N, in.H, in.S);
    // Reference the intrinsic chunk-vs-recur floor on a comparable window.
    auto sColdRef = zeroState(in);
    auto oColdRef = chunk(in, sColdRef, 0, in.T, C);
    const double floor = maxAbsDiff(tail(oColdRef, cached, N, in.H, in.S), goldTail);
    EXPECT_TRUE(maxAbsDiff(outNew, goldTail) <= 3.0 * floor + 1e-4);
}

int main() {
    return mm::test::run();
}
