#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace mimirmind::model {

class GgufReader;

/**
 * Architecture-agnostic config for a decoder-only LLM stored in GGUF.
 * Works for any architecture that exposes the standard llama.cpp metadata
 * keys (gemma, gemma2, gemma3, gemma4, llama, mistral, qwen2, phi3, ...).
 * The `architecture` field disambiguates downstream code paths that care
 * about per-arch quirks (Gemma 3 sliding window, Llama RoPE scaling, ...).
 *
 * Required keys throw if missing; optional keys fall back to sensible
 * defaults and are logged at INFO so it's visible *which* defaults
 * kicked in.
 *
 * Reference: github.com/ggerganov/llama.cpp src/llama-model.cpp.
 */
struct LlmConfig {
    std::string architecture;

    // Required — throws if missing from metadata.
    std::uint32_t blockCount        {0};
    std::uint32_t contextLength     {0};
    std::uint32_t embeddingLength   {0};
    std::uint32_t feedForwardLength {0};
    std::uint32_t headCount         {0};
    std::uint32_t headCountKv       {0};

    // Optional — defaults applied + logged.
    std::uint32_t keyLength         {0};        // 0 = derive embedding/head_count
    std::uint32_t valueLength       {0};        // 0 = same as keyLength
    std::uint32_t keyLengthSwa      {0};        // SWA-layer key length (Gemma 4)
    std::uint32_t valueLengthSwa    {0};        // SWA-layer value length (Gemma 4)
    float         rmsNormEps        {1e-6F};
    float         ropeFreqBase      {10000.0F}; // global / full-attention layers
    float         ropeFreqBaseSwa   {10000.0F}; // sliding-window layers (Gemma 3/4)
    std::uint32_t slidingWindow     {0};        // 0 = disabled (Gemma 3+ uses 4096)

    // Gemma 3/4 sliding-window-attention pattern. Empty = all layers are
    // full attention. Otherwise size == blockCount and entry b is true if
    // block b is SWA, false if it's a global-attention block.
    std::vector<bool> slidingWindowPattern{};

    // Per-layer KV head count. Empty = uniform headCountKv across layers
    // (Qwen / Llama). On Gemma 4 it's typically e.g. 8 for SWA layers and
    // 2 for the (sparser) full-attention layers.
    std::vector<std::uint32_t> headCountKvPerLayer{};

    // MoE (Gemma 4). 0 = dense model (no expert routing).
    std::uint32_t expertCount       {0};        // total experts per block
    std::uint32_t expertUsedCount   {0};        // top-K experts activated per token

    // Gemma 4 "shared KV layers": last N layers reuse an earlier layer's
    // K/V cache instead of computing their own. GGUF key
    // `<arch>.attention.shared_kv_layers`. 0 = every layer computes its
    // own K/V (the historical default we've been assuming).
    // Populated for E4B (=18) and 26B-A4B (=some value); zero elsewhere.
    // Consumers should treat `blockCount - sharedKvLayers` as the
    // "n_layer_kv_from_start" threshold — layers >= this reuse an
    // earlier layer's K/V per the llama.cpp reuse callback
    // `n_kv_from_start - (is_swa(il) ? 2 : 1)`.
    std::uint32_t sharedKvLayers    {0};

    [[nodiscard]] std::uint32_t headDim() const noexcept {
        if (keyLength > 0) {
            return keyLength;
        }
        return headCount > 0 ? embeddingLength / headCount : 0;
    }

    /// Per-layer head_dim. For Gemma 4 returns `keyLengthSwa` on SWA
    /// layers and `keyLength` (full) on global-attention layers. For
    /// every other arch (uniform attention) it just returns `headDim()`.
    [[nodiscard]] std::uint32_t headDim(std::size_t blockIdx) const noexcept {
        if (blockIdx < slidingWindowPattern.size() && keyLengthSwa > 0) {
            return slidingWindowPattern[blockIdx] ? keyLengthSwa : keyLength;
        }
        return headDim();
    }

    /// Per-layer KV head count. Reads `headCountKvPerLayer[blockIdx]` if
    /// the model ships a per-layer array, else falls back to the global
    /// `headCountKv` value.
    [[nodiscard]] std::uint32_t headCountKvFor(std::size_t blockIdx) const noexcept {
        if (blockIdx < headCountKvPerLayer.size()) {
            return headCountKvPerLayer[blockIdx];
        }
        return headCountKv;
    }

    /// Populate from the reader's metadata. Throws on missing required
    /// keys; logs every value found at INFO.
    void parseFromGguf(const GgufReader& reader);
};

} // namespace mimirmind::model