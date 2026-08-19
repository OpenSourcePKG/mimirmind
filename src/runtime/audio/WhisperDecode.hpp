// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace mimirmind::runtime::audio {

/**
 * Whisper special-token ids. Defaults are the multilingual vocab (large/tiny,
 * non-.en). They are model-specific in general and should be resolved from the
 * tokenizer's added-tokens on the real path; parameterised here so the decode
 * logic is independent of any single checkpoint.
 */
struct WhisperSpecialTokens {
    std::int32_t sot{50258};            // <|startoftranscript|>
    std::int32_t eot{50257};            // <|endoftext|>
    std::int32_t transcribe{50359};     // <|transcribe|>
    std::int32_t translate{50358};      // <|translate|>
    std::int32_t noTimestamps{50363};   // <|notimestamps|>
    std::int32_t langEn{50259};         // <|en|> (first language token)
};

/// Options driving greedy transcription.
struct WhisperDecodeOptions {
    WhisperSpecialTokens special{};
    std::int32_t langToken{50259};      // language token to force (default en)
    bool         translate{false};      // task: transcribe (false) / translate
    bool         timestamps{false};     // emit timestamp tokens (unsupported yet)
    // Auto-detect the spoken language: instead of forcing `langToken`, let the
    // model predict the language token at decoder position 1 (argmax over the
    // language-token range), then force the task + no-timestamps tokens. This
    // is what OpenAI's Whisper does when no language is given — forcing the
    // wrong language (e.g. "en" on German audio) yields hallucinated output.
    bool         detectLanguage{false};
    std::size_t  maxNewTokens{224};     // hard cap (half of max_target_positions)
};

/**
 * The forced decoder prompt Whisper conditions on before free generation:
 *   [ <|startoftranscript|>, <|lang|>, <|transcribe|/|translate|>,
 *     (<|notimestamps|> unless timestamps) ]
 * Free decoding continues from here until <|endoftext|> or the length cap.
 * When `detectLanguage` is set the prompt is just [ <|startoftranscript|> ];
 * WhisperRunner then predicts the language and forces the task/no-timestamps
 * tokens at positions 2/3.
 */
[[nodiscard]] std::vector<std::int32_t>
whisperInitialPromptTokens(const WhisperDecodeOptions& opt);

/// Greedy argmax over one logits row [vocab]. Returns the token id.
[[nodiscard]] std::int32_t argmaxRow(std::span<const float> logits);

/// Argmax restricted to the half-open id range [lo, hi) — used for Whisper
/// language detection (mask everything but the language tokens). Returns `lo`
/// if the range is empty or out of bounds.
[[nodiscard]] std::int32_t
argmaxRange(std::span<const float> logits, std::int32_t lo, std::int32_t hi);

} // namespace mimirmind::runtime::audio
