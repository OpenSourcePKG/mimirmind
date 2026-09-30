// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/gguf/GgufReader.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mimirmind::model {

using ::mimirmind::core::gguf::GgufReader;
using ::mimirmind::core::gguf::GgufArray;
using ::mimirmind::core::gguf::GgufValueType;
using ::mimirmind::core::gguf::MetadataValue;

/**
 * BPE tokenizer that dispatches on `tokenizer.ggml.model`:
 *
 *   - `llama` — SentencePiece-style: ▁-normalisation, per-token scores,
 *     best-score-pair merging with leftmost-on-tie. Used by Llama/Gemma.
 *   - `gpt2`  — Byte-level BPE: GPT-2 byte-to-unicode mapping, merges
 *     ranked by their order in `tokenizer.ggml.merges`, lowest-rank-pair
 *     wins. Used by Qwen2/3, GPT-NeoX, etc.
 *   - `bert`  — Loaded but not encodeable (WordPiece, no plan to support).
 *
 * Vocab + special tokens come straight from GGUF metadata — no external
 * SentencePiece or HuggingFace tokenizer library involved.
 *
 * Known limitations (revisit when llama-cli parity testing kicks in):
 *   - Llama: byte-fallback for OOV codepoints not synthesised — they
 *     map to `unknownId` instead.
 *   - GPT-2: no pre-tokenization regex split. The full input string is
 *     byte-encoded then BPE'd in one go. For most inputs this matches
 *     because merges built during training already respect the trainer's
 *     pre-split — but edge cases (long whitespace runs, mixed
 *     punctuation) may diverge from `llama-tokenize`.
 *   - Decode does not yet expand `<0xXX>` byte tokens for llama vocabs.
 */
class Tokenizer {
public:
    Tokenizer() = default;

    /// Populate vocab and special-token IDs from GGUF metadata. Throws if
    /// `tokenizer.ggml.tokens` is missing.
    void loadFromGguf(const GgufReader& reader);

    /// Populate vocab, merges and special-token IDs from a HuggingFace
    /// `tokenizer.json` (+ `tokenizer_config.json` / `generation_config.json`)
    /// found in `checkpointDir`. Byte-level BPE only (Qwen / GPT-2 family) —
    /// it fills exactly the same internal state as the gpt2 GGUF path, so
    /// encode/decode behave identically. Lets NVFP4 checkpoints (which ship
    /// no GGUF tokenizer) tokenise without a separate GGUF. Throws if the
    /// tokenizer is not byte-level BPE or the files are missing/malformed.
    void loadFromHfJson(const std::string& checkpointDir);

    [[nodiscard]] std::vector<std::int32_t> encode(std::string_view text,
                                                   bool addBos = true) const;

    /// Special-token-aware encode (8.24.3, for the Jinja chat-template path):
    /// the input carries special markup (`<|im_start|>` …) as literal text, so
    /// with `parseSpecial=true` the string is split on the known special tokens
    /// — each special becomes its single vocab id, each free-text run is BPE'd
    /// (addBos=false) — mirroring llama.cpp `parse_special=true`. With
    /// `parseSpecial=false` this is identical to `encode(text, addBos)`.
    /// NOTE: for SentencePiece vocabs the per-run space-prefix is a known
    /// parity nuance validated per-family in 8.24.5; the GPT-2 (Qwen) path is
    /// exact.
    [[nodiscard]] std::vector<std::int32_t> encode(std::string_view text,
                                                   bool addBos,
                                                   bool parseSpecial) const;

    /// Split `text` into (isSpecial, piece) segments by greedy longest-first
    /// match against `specialsLongestFirst` (must be pre-sorted longest-first).
    /// Pure + static so the scan is unit-testable without a loaded vocab.
    [[nodiscard]] static std::vector<std::pair<bool, std::string>> splitOnSpecials(
        std::string_view text, const std::vector<std::string>& specialsLongestFirst);

    /// The special/added tokens (bos/eos/pad/unk + GGUF control/user-defined +
    /// HF special added_tokens), sorted longest-first for the parse-special scan.
    [[nodiscard]] const std::vector<std::string>& specialTexts() const noexcept {
        return _specialTexts;
    }

    [[nodiscard]] std::string                decode(std::span<const std::int32_t> ids,
                                                    bool skipSpecial = false) const;

    [[nodiscard]] std::string_view tokenText(std::int32_t id) const noexcept;

    /// Look up a token id by its exact text. Returns -1 if no vocab entry
    /// matches. Useful for special tokens like `<|im_start|>` where the
    /// id is model-specific and we want to bypass BPE for them.
    [[nodiscard]] std::int32_t     findToken(std::string_view text) const noexcept;

    [[nodiscard]] std::size_t      vocabSize() const noexcept { return _tokens.size(); }
    [[nodiscard]] std::string_view modelType() const noexcept { return _modelType; }
    [[nodiscard]] std::size_t      mergesCount() const noexcept { return _mergesRank.size(); }

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

    [[nodiscard]] std::vector<std::int32_t> encodeSpm (std::string_view text, bool addBos) const;
    [[nodiscard]] std::vector<std::int32_t> encodeGpt2(std::string_view text, bool addBos) const;
    [[nodiscard]] std::string                decodeSpm (std::span<const std::int32_t> ids, bool skipSpecial) const;
    [[nodiscard]] std::string                decodeGpt2(std::span<const std::int32_t> ids, bool skipSpecial) const;

    /// Collect the special/added tokens into `_specialTexts` (longest-first).
    /// Called at the end of both load paths. Uses TokenInfo.type (GGUF sets
    /// 3=control/4=user-defined; loadFromHfJson marks special added_tokens
    /// type=3) plus the resolved bos/eos/pad/unk texts.
    void buildSpecialTexts();

    std::vector<TokenInfo>                        _tokens{};
    std::unordered_map<std::string, std::int32_t> _byText{};
    std::string                                   _modelType{};

    /// Special/added token texts, sorted longest-first for the parse-special
    /// greedy scan (8.24.3). Empty until buildSpecialTexts() runs.
    std::vector<std::string>                      _specialTexts{};

    // Populated only for the gpt2 model. Maps "first second" (the merges
    // file format with a literal space separator) to its rank (lower = preferred).
    std::unordered_map<std::string, std::int32_t> _mergesRank{};

    std::int32_t _bosId    {-1};
    std::int32_t _eosId    {-1};
    std::int32_t _unknownId{-1};
    std::int32_t _padId    {-1};

    /// SentencePiece's "add a ▁ to the very front of the input" behaviour.
    /// Defaults to true (legacy SPM). Gemma 4's GGUF sets it to false via
    /// `tokenizer.ggml.add_space_prefix` so "Hello, world!" tokenises to
    /// [Hello][,][▁world][!] instead of [▁Hello][,][▁world][!].
    bool         _addSpacePrefix{true};
};

} // namespace mimirmind::model