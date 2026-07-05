#include "runtime/SpeculativeDecoder.hpp"

#include "compute/SpeculativeSampler.hpp"
#include "runtime/Log.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace mimirmind::runtime {

namespace {

/// Env kill-switch. `MIMIRMIND_SPEC_DEC=off` (or 0/false) forces the
/// target-only path even when a draft engine was successfully loaded.
[[nodiscard]] bool killSwitchEngaged() noexcept {
    const char* env = std::getenv("MIMIRMIND_SPEC_DEC");
    if (env == nullptr || env[0] == '\0') {
        return false;
    }
    const std::string_view v{env};
    return v == "off" || v == "0" || v == "false" || v == "no";
}

/// Penalty rules mirrored from compute::Sampler::applyPenalties — inline
/// here so the accept-check argmax matches what target's own Sampler
/// would have produced without exposing the internal buffer or making
/// applyPenalties public. The math is simple enough that a second copy
/// is cheaper than a plumbing round trip.
///
/// Mutates `logits` in place. `recentTokens` is the full history; only
/// its tail of at most `params.penaltyWindow` entries counts.
void applyPenaltiesInPlace(std::vector<float>&           logits,
                           std::span<const std::int32_t> recentTokens,
                           const compute::SamplingParams& params) {
    if (params.penaltyWindow == 0) {
        return;
    }
    if (params.repetitionPenalty == 1.0F
        && params.frequencyPenalty == 0.0F
        && params.presencePenalty == 0.0F) {
        return;
    }
    if (recentTokens.empty()) {
        return;
    }

    const std::size_t n   = recentTokens.size();
    const std::size_t win = std::min<std::size_t>(params.penaltyWindow, n);
    const auto window     = recentTokens.subspan(n - win);

    const std::size_t V = logits.size();
    // Count per token id. Reused per position via a fresh vector — the
    // decode-per-round count is bounded by draftN+1 so the alloc churn
    // does not matter next to a full-model forward.
    std::vector<std::uint32_t> counts(V, 0U);
    for (const std::int32_t t : window) {
        if (t < 0) {
            continue;
        }
        const auto ut = static_cast<std::size_t>(t);
        if (ut < V) {
            counts[ut] += 1U;
        }
    }

    const float rep  = params.repetitionPenalty;
    const float freq = params.frequencyPenalty;
    const float pres = params.presencePenalty;

    for (std::size_t i = 0; i < V; ++i) {
        const std::uint32_t c = counts[i];
        if (c == 0U) {
            continue;
        }
        float l = logits[i];
        if (rep != 1.0F) {
            l = (l > 0.0F) ? (l / rep) : (l * rep);
        }
        if (freq != 0.0F) {
            l -= freq * static_cast<float>(c);
        }
        if (pres != 0.0F) {
            l -= pres;
        }
        logits[i] = l;
    }
}

[[nodiscard]] std::int32_t argmaxOf(const std::vector<float>& v) {
    if (v.empty()) {
        throw std::runtime_error("SpeculativeDecoder: empty logits row");
    }
    std::int32_t best  = 0;
    float        bestV = v[0];
    for (std::size_t i = 1; i < v.size(); ++i) {
        if (v[i] > bestV) {
            bestV = v[i];
            best  = static_cast<std::int32_t>(i);
        }
    }
    return best;
}

} // namespace

SpeculativeDecoder::SpeculativeDecoder(InferenceEngine& target,
                                       InferenceEngine& draft,
                                       Config           cfg)
    : _target{target}, _draft{draft}, _cfg{cfg} {
    if (_cfg.draftN == 0) {
        MM_LOG_INFO("spec-dec",
                    "SpeculativeDecoder: draftN=0, spec-dec loop disabled "
                    "(target-only pass-through)");
    } else {
        MM_LOG_INFO("spec-dec",
                    "SpeculativeDecoder ready — draftN={}, kill_switch={}",
                    _cfg.draftN, killSwitchEngaged() ? "on" : "off");
    }
}

bool SpeculativeDecoder::wouldEngage(const GenerateParams& params) const noexcept {
    if (_cfg.draftN == 0) {
        return false;
    }
    if (killSwitchEngaged()) {
        return false;
    }
    // Sampled path is bit-inequivalent to modified rejection sampling
    // without full p_draft distributions — that's M9.11.4b work. Fall
    // through to target-only for anything with temperature > 0 so the
    // output stays identical to the baseline.
    if (params.sampling.temperature > 0.0F) {
        return false;
    }
    return true;
}

std::vector<std::int32_t>
SpeculativeDecoder::generate(
    std::span<const std::int32_t>                    promptIds,
    const GenerateParams&                            params,
    const InferenceEngine::TokenCallback&            onToken,
    GenerateStats*                                   outStats,
    const InferenceEngine::PrefillCallback&          onPrefillDone,
    const InferenceEngine::PrefillProgressCallback&  onPrefillProgress) {
    if (!wouldEngage(params)) {
        // Target-only fall-through. Byte-identical to the pre-M9.11.4
        // path so anything sampled or with the kill-switch flipped is
        // guaranteed to keep the same output.
        return _target.generate(promptIds, params, onToken, outStats,
                                onPrefillDone, onPrefillProgress);
    }
    return runSpeculative(promptIds, params, onToken, outStats,
                          onPrefillDone, onPrefillProgress);
}

std::vector<std::int32_t>
SpeculativeDecoder::runSpeculative(
    std::span<const std::int32_t>                    promptIds,
    const GenerateParams&                            params,
    const InferenceEngine::TokenCallback&            onToken,
    GenerateStats*                                   outStats,
    const InferenceEngine::PrefillCallback&          onPrefillDone,
    const InferenceEngine::PrefillProgressCallback&  onPrefillProgress) {
    using clock = std::chrono::steady_clock;

    if (promptIds.empty()) {
        throw std::runtime_error(
            "SpeculativeDecoder::generate: empty prompt");
    }

    const std::size_t maxNew = params.maxNewTokens;
    if (maxNew == 0) {
        if (outStats != nullptr) {
            outStats->promptTokens = promptIds.size();
        }
        return {};
    }

    // Stop-token predicate matches the target engine's semantics (EOS +
    // caller-supplied stop ids). Both sides of a spec-dec round consult
    // it: an accepted draft token that is a stop stops the run
    // immediately, no bonus emit.
    const std::int32_t eosId = _target.tokenizer().eosId();
    auto isStop = [&](std::int32_t id) -> bool {
        if (id == eosId) {
            return true;
        }
        for (auto s : params.stopIds) {
            if (id == s) {
                return true;
            }
        }
        return false;
    };

    // --- Phase 1: target prefill + first sample (t_last) --------------
    //
    // This is a one-token generate() so prefill callbacks fire the same
    // way they would on the vanilla path — client streaming UX is
    // preserved. The first token is emitted below after the draft
    // catch-up prefill so callers see the same "role → first token"
    // ordering as target-only.
    GenerateParams firstParams  = params;
    firstParams.maxNewTokens    = 1;

    GenerateStats firstStats{};
    std::vector<std::int32_t> firstOut = _target.generate(
        promptIds, firstParams,
        /*onToken=*/{},
        &firstStats,
        onPrefillDone,
        onPrefillProgress);

    if (outStats != nullptr) {
        outStats->promptTokens = firstStats.promptTokens;
        outStats->cachedTokens = firstStats.cachedTokens;
        outStats->prefillMs    = firstStats.prefillMs;
    }

    if (firstOut.empty()) {
        // Prefill aborted (client cancelled during prefill_progress).
        // Match the vanilla generate() behaviour: return nothing, leave
        // decodeMs at 0, hitStop stays false.
        return firstOut;
    }

    std::vector<std::int32_t> generated;
    generated.reserve(maxNew);

    std::int32_t tLast = firstOut.front();
    generated.push_back(tLast);

    bool aborted = false;
    if (onToken && !onToken(tLast)) {
        aborted = true;
    }
    bool hitStop = isStop(tLast);
    if (aborted || hitStop) {
        if (outStats != nullptr) {
            outStats->generatedTokens = generated.size();
            outStats->hitStop         = hitStop;
        }
        return generated;
    }

    // --- Phase 2: catch draft's KV up to [prompt + t_last] ------------
    //
    // A one-token draft.generate primes its prefix cache so subsequent
    // rounds only prefill the diverging suffix (which, with LCP, is at
    // most `1 + (accepted_last_round)` tokens). The single-token result
    // itself is discarded — we only care that the KV is warm.
    std::vector<std::int32_t> draftPrimePrompt;
    draftPrimePrompt.reserve(promptIds.size() + 1);
    draftPrimePrompt.insert(draftPrimePrompt.end(),
                            promptIds.begin(), promptIds.end());
    draftPrimePrompt.push_back(tLast);
    {
        GenerateParams primeParams  = params;
        primeParams.maxNewTokens    = 1;
        (void)_draft.generate(
            std::span<const std::int32_t>{draftPrimePrompt},
            primeParams, /*onToken=*/{}, /*outStats=*/nullptr,
            /*onPrefillDone=*/{}, /*onPrefillProgress=*/{});
    }

    // --- Phase 3: draft + verify + accept/reject rounds ---------------
    //
    // `context` mirrors target's committed history plus the pending
    // `tLast` — same shape as the `recentTokens` argument the sampler
    // consumes when applying penalties on the vanilla path.
    std::vector<std::int32_t> context;
    context.reserve(promptIds.size() + maxNew);
    context.insert(context.end(), promptIds.begin(), promptIds.end());

    compute::SpeculativeSampler specSampler;
    std::size_t rounds     = 0;
    std::size_t totalDraft = 0;
    std::size_t totalAcc   = 0;

    const auto decT0 = clock::now();

    while (!aborted && generated.size() < maxNew && !hitStop) {
        // Round budget: total tokens still to emit before we hit
        // maxNew. A round emits at most `roundN + 1` (K accepted +
        // recovery). When budget == 1 we drop to `roundN = 0` so the
        // round runs only the target's bonus argmax — no draft call,
        // no verify-batch waste for a single-token tail.
        const std::size_t budget = maxNew - generated.size();
        if (budget == 0) {
            break;
        }
        const std::size_t roundN =
            std::min(_cfg.draftN, budget - 1);

        // 3a. Draft speculates from [context + tLast]. The draft engine's
        //     own prefix cache absorbs the LCP so only the divergent
        //     suffix re-prefills. Kept as a stable vector across the
        //     verify call below.
        std::vector<std::int32_t> draftPrompt = context;
        draftPrompt.push_back(tLast);

        compute::SpeculativeBatch batch;
        if (roundN > 0) {
            batch = specSampler.speculate(
                _draft,
                std::span<const std::int32_t>{draftPrompt},
                roundN,
                params.sampling);
        }
        // Draft may have hit its EOS/stop before N tokens. We still
        // verify the shorter batch — target might disagree with the
        // draft's stop decision and the verify path handles that
        // uniformly.
        const std::size_t M = batch.tokens.size();

        // 3b. Target verify. Input is [tLast, d_0..d_{M-1}] length M+1
        //     — we need one logits vector per position so that
        //     logits[i] predicts the next token after d_{i-1} (or after
        //     tLast for i=0). logits[M] is target's bonus prediction
        //     if every draft token is accepted.
        std::vector<std::int32_t> verifyInput;
        verifyInput.reserve(M + 1);
        verifyInput.push_back(tLast);
        for (auto id : batch.tokens) {
            verifyInput.push_back(id);
        }

        std::vector<std::vector<float>> logits =
            _target.forwardVerify(
                std::span<const std::int32_t>{verifyInput});

        // 3c. Accept loop. At position i we compare argmax(target_logits[i]
        //     with penalties applied over the growing history) against
        //     the draft's d_i. checkHistory tracks the sequence of tokens
        //     that would already be committed when target's sampler
        //     would have chosen the token at position i — starts at
        //     context+tLast (the state right before d_0).
        std::vector<std::int32_t> checkHistory = context;
        checkHistory.push_back(tLast);

        std::size_t accepted = 0;
        for (std::size_t i = 0; i < M; ++i) {
            applyPenaltiesInPlace(logits[i],
                                  std::span<const std::int32_t>{checkHistory},
                                  params.sampling);
            const std::int32_t targetArg = argmaxOf(logits[i]);
            if (targetArg != batch.tokens[i]) {
                break;
            }
            ++accepted;
            checkHistory.push_back(batch.tokens[i]);
        }

        // 3d. Recovery / bonus. Position K (0..M) is the target's own
        //     next-token pick given the accepted prefix. When accepted
        //     == M this is the "all-accepted bonus"; when accepted < M
        //     it is the target's replacement for the first rejected
        //     draft token. Both cases pull from logits[accepted].
        applyPenaltiesInPlace(logits[accepted],
                              std::span<const std::int32_t>{checkHistory},
                              params.sampling);
        const std::int32_t recoveryTok = argmaxOf(logits[accepted]);

        // 3e. Commit K+1 rows to target's KV: [tLast, d_0..d_{K-1}].
        //     recoveryTok is the new pending sample — mirrors t_last in
        //     the vanilla generate() invariant (last sample uncommitted).
        std::vector<std::int32_t> committedThisRound;
        committedThisRound.reserve(accepted + 1);
        committedThisRound.push_back(tLast);
        for (std::size_t i = 0; i < accepted; ++i) {
            committedThisRound.push_back(batch.tokens[i]);
        }
        _target.commitVerified(
            std::span<const std::int32_t>{committedThisRound});

        // 3f. Mirror commitVerified()'s KV growth into `context`: the
        //     round-start tLast is now committed, and so are the K
        //     accepted draft tokens. Order matches
        //     committedThisRound. Only the recovery token stays
        //     uncommitted (like vanilla generate's last sample).
        context.push_back(tLast);

        // 3g. Emit the accepted draft tokens then the recovery. Order
        //     matches the semantic sequence — every emitted id is a
        //     token that becomes part of the final response.
        for (std::size_t i = 0; i < accepted && !aborted && !hitStop; ++i) {
            const std::int32_t id = batch.tokens[i];
            generated.push_back(id);
            context.push_back(id);
            if (isStop(id)) {
                hitStop = true;
            }
            if (onToken && !onToken(id)) {
                aborted = true;
            }
        }
        if (aborted || hitStop) {
            // A stop mid-round means we skip the bonus emit — matches
            // vanilla generate()'s "break as soon as a stop token is
            // sampled" semantics. context already reflects the last
            // committed emit.
            //
            // Also update t_last so the outer post-loop logging path
            // reads a sane last-token value if anything else looks at
            // it (nothing currently does, but keep the invariant).
            tLast = generated.back();
            ++rounds;
            totalDraft += M;
            totalAcc   += accepted;
            break;
        }

        generated.push_back(recoveryTok);
        // context is the "already committed" list; recoveryTok is the
        // new pending sample and should NOT be appended here — the next
        // round appends it via committedThisRound[0] like this round
        // did with the previous t_last.
        tLast = recoveryTok;
        if (isStop(recoveryTok)) {
            hitStop = true;
        }
        if (onToken && !onToken(recoveryTok)) {
            aborted = true;
        }

        ++rounds;
        totalDraft += M;
        totalAcc   += accepted;
    }

    const auto decT1 = clock::now();
    const double decodeMs =
        std::chrono::duration<double, std::milli>(decT1 - decT0).count();

    if (outStats != nullptr) {
        outStats->generatedTokens = generated.size();
        outStats->decodeMs        = decodeMs;
        outStats->hitStop         = hitStop;
        outStats->specDecRounds   = rounds;
        outStats->specDecDrafted  = totalDraft;
        outStats->specDecAccepted = totalAcc;
    }

    return generated;
}

} // namespace mimirmind::runtime