#pragma once

#include <cstdint>
#include <string>

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
    float         rmsNormEps        {1e-6F};
    float         ropeFreqBase      {10000.0F};
    std::uint32_t slidingWindow     {0};        // 0 = disabled (Gemma 3+ uses 4096)

    [[nodiscard]] std::uint32_t headDim() const noexcept {
        if (keyLength > 0) {
            return keyLength;
        }
        return headCount > 0 ? embeddingLength / headCount : 0;
    }

    /// Populate from the reader's metadata. Throws on missing required
    /// keys; logs every value found at INFO.
    void parseFromGguf(const GgufReader& reader);
};

} // namespace mimirmind::model