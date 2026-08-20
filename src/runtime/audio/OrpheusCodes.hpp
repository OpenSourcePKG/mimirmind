// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mimirmind::runtime::audio {

/**
 * Orpheus TTS (Llama-3.2 backbone + SNAC 24 kHz codec) token conventions —
 * the pure, model-independent bridge between the acoustic LM's emitted token
 * ids and the SnacDecoder's three hierarchical code levels (roadmap 8.13.2).
 *
 * Everything here mirrors the reference canopyai/Orpheus-TTS exactly:
 *   prompt      = [start] + BPE("{voice}: {text}") + end[]
 *   token->code = tid - codeOffset - (index % 7) * codebookSize
 *   regroup     = per 7-token frame: coarse=[f0], mid=[f1,f4],
 *                 fine=[f2,f3,f5,f6]   (== SNAC vq_strides [4,2,1] layout)
 *
 * The default ids are for the 3B `canopylabs/orpheus-tts-0.1-finetune-prod`
 * model; codeOffset 128266 == the id of `<custom_token_10>` (the first audio
 * codebook token, the 10 below it being reserved specials).
 */
struct OrpheusTokens {
    std::int32_t              startToken   = 128259;
    std::array<std::int32_t,4> endTokens   = {128009, 128260, 128261, 128257};
    std::int32_t              audioEos     = 128258;   // stops generation
    std::int32_t              codeOffset   = 128266;   // <custom_token_10>
    std::int32_t              codebookSize = 4096;
};

/// The eight English Orpheus voices, most-realistic first.
[[nodiscard]] const std::vector<std::string>& orpheusVoices();

/// Convert one emitted LM token id to a SNAC codebook index, given its 0-based
/// position `index` in the audio-token stream. Returns the code in
/// [0, codebookSize) or -1 if `tid` is not a valid audio code at this slot
/// (e.g. an EOS/special/out-of-range token). Slot = index % 7.
[[nodiscard]] std::int32_t
orpheusTokenToCode(std::int32_t tid, std::size_t index,
                   const OrpheusTokens& t = {});

/// Regroup a flat per-frame code stream (as produced by orpheusTokenToCode over
/// the emitted tokens) into the three SNAC hierarchical levels, exactly as the
/// reference convert_to_audio does. `flat` length need not be a multiple of 7;
/// a trailing partial frame is dropped. Returns {coarse, mid, fine} with sizes
/// {N, 2N, 4N} for N complete frames — the shape SnacDecoder::decode expects.
[[nodiscard]] std::vector<std::vector<std::int32_t>>
regroupOrpheusFrames(const std::vector<std::int32_t>& flat);

} // namespace mimirmind::runtime::audio
