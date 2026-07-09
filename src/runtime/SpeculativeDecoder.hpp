#pragma once

#include "runtime/InferenceEngine.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace mimirmind::runtime {

/**
 * M9.11.4 — speculative-decoding orchestrator.
 *
 * Wraps a `target` (big, slow, authoritative) and a `draft` (small, fast)
 * InferenceEngine pair. Presents the same public shape as
 * `InferenceEngine::generate()` so the ApiServer can route through it
 * with no other change: eligibility check inside decides whether to
 * engage the spec-dec loop or delegate straight to `target.generate()`.
 *
 * Scope of M9.11.4 — greedy only:
 *
 * - Speculative-decoding is engaged only when `params.sampling.temperature
 *   <= 0` AND the draft engine is non-null AND `speculative.enabled` is
 *   true in config.json. Any other case falls through to plain
 *   `target.generate(...)` bit-for-bit — the streaming/blocking response
 *   is identical to the target-only baseline.
 *
 * - Accept/reject is `argmax(target_logits[i]) == draft_token[i]`.
 *   Recovery on first mismatch is `argmax(target_logits[K])`. Correct
 *   modified rejection sampling for the sampled path lands in M9.11.4b.
 *
 * - Sampler penalties (repetition/frequency/presence) are applied to the
 *   target's verify logits before the argmax comparison so an accepted
 *   token matches what target's own sampler would have picked. Draft's
 *   own decode already applies its penalties via its Sampler; the two
 *   argmaxes may still disagree because the underlying logits differ.
 *
 * Not thread-safe. Callers must serialise like they do for the underlying
 * InferenceEngine.
 */
class SpeculativeDecoder {
public:
    struct Config {
        /// From `speculative.enabled` in config.json. When false the
        /// wrapper always falls through to plain target.generate().
        bool        enabled{true};
        /// Tokens the draft speculates per verify round. From
        /// `speculative.n`. Typical range 2..8; default 4 matches the
        /// roadmap prediction.
        std::size_t draftN{4};
    };

    /// `target` and `draft` are non-owning. `draft` must have passed the
    /// vocab-compat check in main.cpp already — this class does no
    /// re-verification.
    SpeculativeDecoder(InferenceEngine& target,
                       InferenceEngine& draft,
                       Config           cfg);

    SpeculativeDecoder(const SpeculativeDecoder&)            = delete;
    SpeculativeDecoder& operator=(const SpeculativeDecoder&) = delete;
    SpeculativeDecoder(SpeculativeDecoder&&)                 = delete;
    SpeculativeDecoder& operator=(SpeculativeDecoder&&)      = delete;

    /// Same signature as `InferenceEngine::generate()`. Returns generated
    /// ids without the prompt. `outStats` is populated with prefill/decode
    /// wall times AND (M9.11.4) with `specDecRounds`,
    /// `specDecDrafted`, `specDecAccepted` when the spec-dec path
    /// engaged; those stay 0 on the fall-through target-only path.
    std::vector<std::int32_t> generate(
        std::span<const std::int32_t>                    promptIds,
        const GenerateParams&                            params,
        const InferenceEngine::TokenCallback&            onToken            = {},
        GenerateStats*                                   outStats           = nullptr,
        const InferenceEngine::PrefillCallback&          onPrefillDone      = {},
        const InferenceEngine::PrefillProgressCallback&  onPrefillProgress  = {});

    /// True when the spec-dec loop would engage for these params. Cheap;
    /// ApiServer uses it purely for a log line.
    [[nodiscard]] bool wouldEngage(const GenerateParams& params) const noexcept;

    [[nodiscard]] const Config& config() const noexcept { return _cfg; }

private:
    /// Greedy spec-dec loop. Returns the final `generated` list and
    /// populates the spec-dec fields of `outStats` in place.
    std::vector<std::int32_t> runSpeculative(
        std::span<const std::int32_t>                    promptIds,
        const GenerateParams&                            params,
        const InferenceEngine::TokenCallback&            onToken,
        GenerateStats*                                   outStats,
        const InferenceEngine::PrefillCallback&          onPrefillDone,
        const InferenceEngine::PrefillProgressCallback&  onPrefillProgress);

    InferenceEngine& _target;
    InferenceEngine& _draft;
    Config           _cfg;
};

} // namespace mimirmind::runtime