#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace mimirmind::model {

class GgufReader;

/**
 * SentencePiece-style BPE tokenizer driven directly off GGUF metadata.
 * The vocab + per-token scores live in `tokenizer.ggml.tokens` and
 * `tokenizer.ggml.scores` — no external SentencePiece library involved.
 *
 * Encode uses the canonical SP algorithm: split into UTF-8 code points,
 * then repeatedly merge the adjacent pair whose combined token has the
 * highest score, with leftmost-wins tiebreak. Decode concatenates token
 * text and converts the SentencePiece space marker (U+2581) back to ASCII
 * space.
 *
 * Known limitations (will be revisited when parity testing kicks in):
 *   - Byte-fallback for OOV characters is recognised but not yet
 *     synthesised — unknown code points map to `unknownId` instead.
 *   - Decode does not yet reconstitute raw bytes from `<0xXX>` byte
 *     tokens — they are emitted as-is.
 *   - Only the "llama" (SPM) model type is exercised; "gpt2" (merges
 *     array) and "bert" (WordPiece) are loaded but not encoded.
 */
class Tokenizer {
public:
    Tokenizer() = default;

    /// Populate vocab and special-token IDs from GGUF metadata. Throws if
    /// `tokenizer.ggml.tokens` is missing.
    void loadFromGguf(const GgufReader& reader);

    [[nodiscard]] std::vector<std::int32_t> encode(std::string_view text,
                                                   bool addBos = true) const;
    [[nodiscard]] std::string                decode(std::span<const std::int32_t> ids,
                                                    bool skipSpecial = false) const;

    [[nodiscard]] std::string_view tokenText(std::int32_t id) const noexcept;

    [[nodiscard]] std::size_t      vocabSize() const noexcept { return _tokens.size(); }
    [[nodiscard]] std::string_view modelType() const noexcept { return _modelType; }

    [[nodiscard]] std::int32_t bosId()     const noexcept { return _bosId;     }
    [[nodiscard]] std::int32_t eosId()     const noexcept { return _eosId;     }
    [[nodiscard]] std::int32_t unknownId() const noexcept { return _unknownId; }
    [[nodiscard]] std::int32_t padId()     const noexcept { return _padId;     }

private:
    struct TokenInfo {
        std::string  text;
        float        score{0.0F};
        std::int32_t type{1};
    };

    std::vector<TokenInfo>                       _tokens{};
    std::unordered_map<std::string, std::int32_t> _byText{};
    std::string                                  _modelType{};

    std::int32_t _bosId    {-1};
    std::int32_t _eosId    {-1};
    std::int32_t _unknownId{-1};
    std::int32_t _padId    {-1};
};

} // namespace mimirmind::model