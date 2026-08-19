// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstddef>
#include <vector>

namespace mimirmind::runtime::audio {

/**
 * Result of the Whisper convolutional stem: the encoder input embeddings,
 * time-major [nCtx x dModel] row-major (row t = frame t's d_model vector),
 * ready to have the encoder positional table added and be fed to the encoder.
 */
struct ConvStemOutput {
    std::vector<float> data;   // nCtx * dModel, row-major (time-major)
    std::size_t nCtx{0};       // = (nFrames - 1) / 2 + 1  (conv2 stride 2)
    std::size_t dModel{0};
};

/**
 * Whisper convolutional stem (host reference), matching HF WhisperEncoder:
 *
 *   x = gelu(conv1(mel))     conv1: [dModel, nMels, 3], stride 1, pad 1
 *   x = gelu(conv2(x))       conv2: [dModel, dModel, 3], stride 2, pad 1
 *   x = x.transpose(time, channel)   -> [nCtx, dModel]
 *
 * GELU is the exact (erf) variant. The stem is tiny next to the encoder, so it
 * runs on the host; the transformer body runs on the device backend.
 *
 * Inputs:
 *   mel     : [nMels x nFrames] row-major (mel-major) — the log-mel from the
 *             front-end (compute::dsp::logMelSpectrogram).
 *   conv1W  : [dModel, nMels, 3]  row-major
 *   conv1B  : [dModel]
 *   conv2W  : [dModel, dModel, 3] row-major
 *   conv2B  : [dModel]
 *
 * All weight pointers must be host-readable F32.
 */
[[nodiscard]] ConvStemOutput whisperConvStem(const float* mel,
                                             std::size_t  nMels,
                                             std::size_t  nFrames,
                                             std::size_t  dModel,
                                             const float* conv1W,
                                             const float* conv1B,
                                             const float* conv2W,
                                             const float* conv2B);

} // namespace mimirmind::runtime::audio
