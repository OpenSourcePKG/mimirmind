// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace mimirmind::model {

/**
 * SentencePiece **Unigram** tokenizer for XLM-RoBERTa checkpoints — the
 * cross-encoder reranker path (bge-reranker-v2-m3 / EncoderRunner). Distinct
 * from `Tokenizer`, which only does llama-SPM (bigram best-merge) and gpt2-BPE
 * off GGUF metadata; XLM-R needs true Unigram Viterbi over the pieces in a
 * `sentencepiece.bpe.model` protobuf, plus the fairseq id remapping.
 *
 * Pipeline (validated bit-exact against the HF tokenizer on a real DE pair):
 *   normalize  : collapse whitespace, add dummy prefix, escape space -> U+2581
 *   segment    : Unigram Viterbi maximising the sum of piece log-scores
 *   remap      : HF_id = sp_id + 1 (fairseq offset); sp <unk> -> HF unk (3)
 *   frame pair : [<s>] q [</s>] [</s>] p [</s>]   (XLM-R sentence-pair format)
 *
 * fairseq special ids (fixed for all XLM-R): <s>=0, <pad>=1, </s>=2, <unk>=3.
 *
 * Limitation: the model's `nmt_nfkc` precompiled charsmap is approximated by
 * whitespace normalization only (no full NFKC). This is identity for normal
 * Latin/German/CJK text — matters only for compatibility/exotic codepoints;
 * revisit if a parity case needs it.
 */
class XlmRobertaTokenizer {
public:
    // fairseq special ids (HF XLMRobertaTokenizer convention).
    static constexpr std::int32_t kBos = 0;   // <s>
    static constexpr std::int32_t kPad = 1;   // <pad>
    static constexpr std::int32_t kEos = 2;   // </s>
    static constexpr std::int32_t kUnk = 3;   // <unk>

    XlmRobertaTokenizer() = default;

    /// Parse a `sentencepiece.bpe.model` (protobuf ModelProto) file. Throws
    /// std::runtime_error on an unreadable/malformed file.
    void load(std::string_view modelPath);

    /// Segment `text` into HF token ids (no special tokens added).
    [[nodiscard]] std::vector<std::int32_t> encode(std::string_view text) const;

    /// XLM-R sentence-pair framing for a cross-encoder:
    ///   [<s>] encode(query) [</s>][</s>] encode(passage) [</s>]
    [[nodiscard]] std::vector<std::int32_t>
    encodePair(std::string_view query, std::string_view passage) const;

    /// Single sequence framed as [<s>] encode(text) [</s>].
    [[nodiscard]] std::vector<std::int32_t>
    encodeSingle(std::string_view text) const;

    [[nodiscard]] std::size_t vocabSize() const noexcept { return _pieces.size(); }

private:
    struct Piece {
        std::string text;
        float       score{0.0F};
        std::int32_t type{1};
    };

    [[nodiscard]] std::string normalize(std::string_view text) const;
    [[nodiscard]] std::vector<std::int32_t> viterbi(const std::string& norm) const;

    std::vector<Piece>                            _pieces;
    std::unordered_map<std::string, std::int32_t> _pieceToId;
    std::int32_t _unkSpId{0};
    std::size_t  _maxPieceChars{1};
};

} // namespace mimirmind::model
