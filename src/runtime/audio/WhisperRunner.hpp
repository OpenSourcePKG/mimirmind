// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "compute/ComputeBuffer.hpp"
#include "runtime/audio/WhisperDecode.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mimirmind::compute {
class ComputeOps;
class ComputeMatmul;
} // namespace mimirmind::compute

namespace mimirmind::runtime::audio {

class WhisperModel;

/**
 * Greedy Whisper transcription over the abstract compute backend, mirroring
 * EncoderRunner's op discipline. Pipeline:
 *
 *   host log-mel  -> host conv stem  -> device encoder (pre-norm self-attn+FFN)
 *   -> device decoder greedy loop (masked self-attn + cross-attn + FFN)
 *   -> tied output projection -> argmax -> next token.
 *
 * The decoder recomputes the whole prefix each step (no KV cache) — correct and
 * simple for a first reference; a KV-cached decode is a later optimisation. All
 * heavy math (attention, GEMM, norm, GELU) reuses the existing ComputeOps /
 * ComputeMatmul primitives; only the conv stem runs on the host.
 */
class WhisperRunner {
public:
    WhisperRunner(const WhisperModel& model, compute::ComputeOps& ops,
                  compute::ComputeMatmul& matmul);

    /// Transcribe a host log-mel spectrogram [nMels x nFrames] (mel-major),
    /// returning the generated token ids (the forced prompt prefix through the
    /// token before <|endoftext|>; the prompt/special tokens are included).
    [[nodiscard]] std::vector<std::int32_t>
    transcribeGreedy(const float* mel, std::size_t nMels, std::size_t nFrames,
                     const WhisperDecodeOptions& opt);

private:
    /// Conv stem (host) + encoder (device). Returns the [nCtx x dModel] encoder
    /// states on device and sets nCtxOut.
    compute::ComputeBuffer runEncoder(const float* mel, std::size_t nMels,
                                      std::size_t nFrames, std::size_t& nCtxOut);

    const WhisperModel&     _m;
    compute::ComputeOps&    _ops;
    compute::ComputeMatmul& _mm;
};

} // namespace mimirmind::runtime::audio
