// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace mimirmind::runtime::encoder::laya {

/**
 * The Laya multilingual tokenizer — the Gemma/mmBERT metaspace BPE, parsed
 * straight from the checkpoint's HF `tokenizer/tokenizer.json` (no HuggingFace
 * tokenizers library). mimirmind's model::Tokenizer only does byte-level
 * (GPT-2) BPE from tokenizer.json, so this is the metaspace (SentencePiece-
 * style) variant Gemma uses.
 *
 * Pipeline (matches `add_special_tokens=False`, as build_head/build_sequence use):
 *   - normalize: replace ' ' with the metaspace char '▁' (U+2581);
 *   - metaspace pre-tokenize: ensure exactly one leading '▁', split into pieces
 *     at every '▁' (each piece begins with '▁');
 *   - per piece: rank-ordered BPE merge over UTF-8 characters; map each final
 *     symbol to its vocab id, or byte-fallback to `<0xXX>` tokens when a symbol
 *     is not in the vocab (byte_fallback=true). No ignore_merges, no affixes.
 *
 * Special-token splitting is NOT done (the question/option/state text has its
 * mask token replaced by a space before encoding), so encode() is pure BPE.
 */
class LayaTokenizer {
public:
    LayaTokenizer() = default;

    /// Load from `<modelDir>/tokenizer/tokenizer.json` (+ tokenizer_config.json
    /// for the cls/sep/mask/pad/unk token strings). Throws on missing/malformed.
    void load(std::string_view modelDir);

    /// Encode `text` to token ids (no special tokens added), metaspace BPE.
    [[nodiscard]] std::vector<std::int32_t> encode(std::string_view text) const;

    [[nodiscard]] std::int32_t clsId()  const noexcept { return _clsId; }
    [[nodiscard]] std::int32_t sepId()  const noexcept { return _sepId; }
    [[nodiscard]] std::int32_t maskId() const noexcept { return _maskId; }
    [[nodiscard]] std::int32_t padId()  const noexcept { return _padId; }
    [[nodiscard]] std::int32_t unkId()  const noexcept { return _unkId; }
    [[nodiscard]] std::size_t  vocabSize() const noexcept { return _vocab.size(); }

private:
    void bpeMerge(std::vector<std::string>& syms) const;

    std::unordered_map<std::string, std::int32_t> _vocab;     // token text -> id
    std::unordered_map<std::string, std::int32_t> _mergeRank; // "a b" -> rank
    std::int32_t _byteToken[256]{};                           // <0xXX> ids, -1 if absent

    std::int32_t _clsId{-1};
    std::int32_t _sepId{-1};
    std::int32_t _maskId{-1};
    std::int32_t _padId{-1};
    std::int32_t _unkId{-1};
};

} // namespace mimirmind::runtime::encoder::laya
