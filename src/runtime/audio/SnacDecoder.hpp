// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace mimirmind::runtime::audio {

/**
 * Host-side decoder for the SNAC 24 kHz multi-scale neural audio codec
 * (hubertsiuzdak/snac_24khz) — the vocoder half of the Orpheus TTS pipeline
 * (roadmap 8.13.2). Turns the hierarchical audio codes the Llama-3.2 acoustic
 * model emits into a 24 kHz mono waveform. CPU-only and backend-neutral: the
 * decoder runs once per utterance and is tiny next to the LM, so it mirrors the
 * host-side WhisperConvStem rather than a GPU kernel.
 *
 * Architecture (exact, from the reference config + source):
 *   latent_dim 768, decoder_dim 1024, decoder_rates [8,8,4,2] (512x upsample),
 *   codebook_size 4096, codebook_dim 8, vq_strides [4,2,1] (3 RVQ levels),
 *   noise + depthwise.
 *   decode(codes) = quantizer.from_codes -> Decoder:
 *     from_codes: for each level i, embed codes[i] -> out_proj Conv1d(8->768,k1)
 *       -> repeat_interleave by stride[i] -> sum  =>  z_q [768, T]
 *     Decoder (depthwise): Conv1d(768->768,k7,groups768) -> Conv1d(768->1024,k1)
 *       -> 4x DecoderBlock -> Snake(64) -> Conv1d(64->1,k7) -> tanh
 *     DecoderBlock: Snake -> ConvTranspose1d(k=2s,stride=s,pad=ceil(s/2),
 *       output_padding=s%2) -> NoiseBlock -> 3x ResidualUnit(dil 1/3/9)
 *     ResidualUnit: Snake -> depthwise Conv1d(k7,dilated) -> Snake ->
 *       Conv1d(k1) -> + residual
 *
 * Weights are read from a converted safetensors file whose tensors are the
 * EFFECTIVE (weight-norm already resolved) conv weights/biases, Snake alphas,
 * and codebook embeddings, under the semantic names this class expects (see
 * scripts/convert-snac.py). The reference `pytorch_model.bin` uses weight-norm
 * parametrizations; the conversion materialises `.weight` so C++ never has to
 * reconstruct the norm.
 */
class SnacDecoder {
public:
    SnacDecoder() = default;

    /// Load the converted SNAC safetensors from `dir` (a directory holding
    /// `model.safetensors`, or a direct file path). Throws std::runtime_error
    /// naming any missing tensor. `noise` enables the (stochastic) NoiseBlock;
    /// default false so decode() is deterministic and parity-checkable.
    void load(std::string_view path, bool noise = false);

    [[nodiscard]] bool isLoaded() const noexcept { return _loaded; }
    [[nodiscard]] int  sampleRate() const noexcept { return 24000; }

    /// Decode the three hierarchical code levels into 24 kHz mono f32 PCM in
    /// [-1, 1]. `codes[0]` is the coarse level (stride 4, T/4 entries),
    /// `codes[1]` the mid (stride 2, T/2), `codes[2]` the fine (stride 1, T);
    /// each entry is a codebook index in [0, 4096). Output length is 512*T.
    /// Throws if the code-level lengths are inconsistent with vq_strides.
    [[nodiscard]] std::vector<float>
    decode(const std::vector<std::vector<std::int32_t>>& codes) const;

    // --- Test-only exposure of the numeric primitives (the error-prone core).
    // Signals are [channels, time] row-major (x[c*T + t]). These wrap the same
    // host implementations decode() uses. ---
    [[nodiscard]] static std::vector<float>
    conv1dTest(const std::vector<float>& x, std::size_t cin, std::size_t T,
               const std::vector<float>& w, const std::vector<float>& bias,
               std::size_t cout, std::size_t k, std::size_t stride,
               std::size_t pad, std::size_t dil, std::size_t groups,
               std::size_t& tout);
    [[nodiscard]] static std::vector<float>
    convTranspose1dTest(const std::vector<float>& x, std::size_t cin,
                        std::size_t T, const std::vector<float>& w,
                        const std::vector<float>& bias, std::size_t cout,
                        std::size_t k, std::size_t stride, std::size_t pad,
                        std::size_t outPad, std::size_t& tout);
    [[nodiscard]] static std::vector<float>
    snakeTest(std::vector<float> x, std::size_t c, std::size_t T,
              const std::vector<float>& alpha);

private:
    struct HostTensor {
        std::vector<float>       data;
        std::vector<std::size_t> shape;
    };

    [[nodiscard]] const HostTensor& t(const std::string& name) const;

    std::unordered_map<std::string, HostTensor> _w;
    bool _loaded{false};
    bool _noise{false};

    static constexpr int kStrides[3]   = {4, 2, 1};   // vq_strides
    static constexpr int kDecRates[4]  = {8, 8, 4, 2};
    static constexpr int kLatentDim    = 768;
    static constexpr int kCodebookDim  = 8;
};

} // namespace mimirmind::runtime::audio
