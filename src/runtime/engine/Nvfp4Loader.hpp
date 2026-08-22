// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <string_view>

namespace mimirmind::runtime {
class InferenceEngine;
}
namespace mimirmind::core::safetensors {
class SafetensorsModel;
}

namespace mimirmind::runtime::engine {

/**
 * The NVFP4 checkpoint load pipeline, extracted from InferenceEngine so the
 * engine's translation unit stays focused on generation. Runs on the engine
 * as a friend collaborator: reads config.json + tokenizer, uploads the
 * NVFP4/FP8 weights, dequantises them to BF16 on device, applies the
 * GatedDeltaNet value-head regroup, and re-quantises the MoE experts to
 * K-quants (and, gated off, the attention projections to Q8_0). It leaves the
 * engine with `_config`, `_tokenizer`, `_materializedBf16` and `_weights`
 * populated; the caller runs `finalizeLoad()` afterwards.
 *
 * CUDA-only — the body is compiled only under MIMIRMIND_HAVE_CUDA.
 */
class Nvfp4Loader {
public:
    /// Populate `engine` from the NVFP4 checkpoint at `checkpointDir`, taking
    /// the tokenizer from `tokenizerGguf`. Throws on a non-CUDA backend or a
    /// malformed checkpoint. Does NOT call finalizeLoad().
    ///
    /// `attachedSm` is the M-Munin.CUDA attach hook: when non-null, the
    /// safetensors shards are read from that already-open SafetensorsModel
    /// (reconstructed from shm memfd chunks) instead of from disk. The small
    /// text sidecars (config.json / hf_quant_config.json / tokenizer.json)
    /// are still read locally from `checkpointDir`. `attachedSm` must outlive
    /// the call.
    static void load(InferenceEngine&                     engine,
                     std::string_view                     checkpointDir,
                     std::string_view                     tokenizerGguf,
                     core::safetensors::SafetensorsModel* attachedSm = nullptr);
};

} // namespace mimirmind::runtime::engine
