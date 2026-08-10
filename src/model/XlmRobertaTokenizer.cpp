// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "model/XlmRobertaTokenizer.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace mimirmind::model {

namespace {

constexpr std::string_view kMeta = "\xE2\x96\x81";   // U+2581 "LOWER ONE EIGHTH BLOCK" (SP space)

// ---- minimal protobuf wire reader (varint + length-delimited + fixed32) ----
struct PbCursor {
    const std::uint8_t* p;
    const std::uint8_t* end;
};

std::uint64_t readVarint(PbCursor& c) {
    std::uint64_t val = 0;
    int shift = 0;
    while (c.p < c.end) {
        const std::uint8_t b = *c.p++;
        val |= static_cast<std::uint64_t>(b & 0x7F) << shift;
        if ((b & 0x80) == 0) {
            return val;
        }
        shift += 7;
        if (shift > 63) {
            break;
        }
    }
    throw std::runtime_error("XlmRobertaTokenizer: truncated varint");
}

// Codepoint count of a UTF-8 string (number of non-continuation bytes).
std::size_t utf8Len(std::string_view s) {
    std::size_t n = 0;
    for (unsigned char b : s) {
        if ((b & 0xC0) != 0x80) {
            ++n;
        }
    }
    return n;
}

} // namespace

void XlmRobertaTokenizer::load(std::string_view modelPath) {
    std::ifstream in(std::string{modelPath}, std::ios::binary);
    if (!in) {
        throw std::runtime_error("XlmRobertaTokenizer: cannot open " +
                                 std::string{modelPath});
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    const std::string buf = ss.str();

    PbCursor top{reinterpret_cast<const std::uint8_t*>(buf.data()),
                 reinterpret_cast<const std::uint8_t*>(buf.data()) + buf.size()};

    // ModelProto: field 1 = repeated SentencePiece {1:piece,2:score,3:type}.
    while (top.p < top.end) {
        const std::uint64_t tag = readVarint(top);
        const std::uint32_t fn  = static_cast<std::uint32_t>(tag >> 3);
        const std::uint32_t wt  = static_cast<std::uint32_t>(tag & 7);

        if (wt == 2) {   // length-delimited
            const std::uint64_t len = readVarint(top);
            if (static_cast<std::uint64_t>(top.end - top.p) < len) {
                throw std::runtime_error("XlmRobertaTokenizer: truncated field");
            }
            const std::uint8_t* sub = top.p;
            top.p += len;

            if (fn == 1) {   // one SentencePiece
                PbCursor pc{sub, sub + len};
                Piece piece{};
                while (pc.p < pc.end) {
                    const std::uint64_t t2 = readVarint(pc);
                    const std::uint32_t f2 = static_cast<std::uint32_t>(t2 >> 3);
                    const std::uint32_t w2 = static_cast<std::uint32_t>(t2 & 7);
                    if (f2 == 1 && w2 == 2) {          // piece string
                        const std::uint64_t sl = readVarint(pc);
                        piece.text.assign(reinterpret_cast<const char*>(pc.p),
                                          static_cast<std::size_t>(sl));
                        pc.p += sl;
                    } else if (f2 == 2 && w2 == 5) {   // score (fixed32 float)
                        std::memcpy(&piece.score, pc.p, 4);
                        pc.p += 4;
                    } else if (f2 == 3 && w2 == 0) {   // type (varint)
                        piece.type = static_cast<std::int32_t>(readVarint(pc));
                    } else {
                        // skip unknown sub-field
                        if (w2 == 0) { (void)readVarint(pc); }
                        else if (w2 == 2) { const std::uint64_t l = readVarint(pc); pc.p += l; }
                        else if (w2 == 5) { pc.p += 4; }
                        else if (w2 == 1) { pc.p += 8; }
                        else { throw std::runtime_error("XlmRobertaTokenizer: bad wire"); }
                    }
                }
                _pieces.push_back(std::move(piece));
            }
            // other length-delimited fields (trainer/normalizer) are skipped
        } else if (wt == 0) {
            (void)readVarint(top);
        } else if (wt == 5) {
            top.p += 4;
        } else if (wt == 1) {
            top.p += 8;
        } else {
            throw std::runtime_error("XlmRobertaTokenizer: bad top-level wire");
        }
    }

    if (_pieces.empty()) {
        throw std::runtime_error("XlmRobertaTokenizer: no pieces parsed");
    }

    _pieceToId.reserve(_pieces.size() * 2);
    _maxPieceChars = 1;
    for (std::size_t i = 0; i < _pieces.size(); ++i) {
        const Piece& p = _pieces[i];
        _pieceToId.emplace(p.text, static_cast<std::int32_t>(i));
        if (p.type == 2) {   // SentencePiece UNKNOWN
            _unkSpId = static_cast<std::int32_t>(i);
        }
        const std::size_t clen = utf8Len(p.text);
        if (clen > _maxPieceChars) {
            _maxPieceChars = clen;
        }
    }
}

std::string XlmRobertaTokenizer::normalize(std::string_view text) const {
    // remove_extra_whitespaces: collapse ASCII whitespace runs, strip ends.
    std::string collapsed;
    collapsed.reserve(text.size());
    bool inWs = false;
    bool started = false;
    for (char ch : text) {
        const bool ws = ch == ' ' || ch == '\t' || ch == '\n' ||
                        ch == '\r' || ch == '\f' || ch == '\v';
        if (ws) {
            inWs = true;
            continue;
        }
        if (inWs && started) {
            collapsed.push_back(' ');
        }
        inWs = false;
        started = true;
        collapsed.push_back(ch);
    }

    // add_dummy_prefix + escape_whitespace: prepend U+2581, space -> U+2581.
    std::string out;
    out.reserve(collapsed.size() + kMeta.size() * 2);
    out.append(kMeta);
    for (char ch : collapsed) {
        if (ch == ' ') {
            out.append(kMeta);
        } else {
            out.push_back(ch);
        }
    }
    return out;
}

std::vector<std::int32_t>
XlmRobertaTokenizer::viterbi(const std::string& norm) const {
    // Codepoint start byte-offsets (starts[n] == norm.size()).
    std::vector<std::size_t> starts;
    starts.reserve(norm.size() + 1);
    for (std::size_t i = 0; i < norm.size(); ++i) {
        if ((static_cast<unsigned char>(norm[i]) & 0xC0) != 0x80) {
            starts.push_back(i);
        }
    }
    starts.push_back(norm.size());
    const std::size_t n = starts.size() - 1;

    constexpr float kNegInf = -std::numeric_limits<float>::infinity();
    constexpr float kUnkPenalty = -100.0F;
    std::vector<float>        best(n + 1, kNegInf);
    std::vector<std::size_t>  backJ(n + 1, 0);
    std::vector<std::int32_t> backId(n + 1, _unkSpId);
    best[0] = 0.0F;

    for (std::size_t i = 1; i <= n; ++i) {
        const std::size_t lo = (i > _maxPieceChars) ? (i - _maxPieceChars) : 0;
        for (std::size_t j = lo; j < i; ++j) {
            if (best[j] == kNegInf) {
                continue;
            }
            const std::string_view sub{norm.data() + starts[j],
                                       starts[i] - starts[j]};
            const auto it = _pieceToId.find(std::string{sub});
            if (it != _pieceToId.end()) {
                const float sc = best[j] + _pieces[static_cast<std::size_t>(
                                               it->second)].score;
                if (sc > best[i]) {
                    best[i]   = sc;
                    backJ[i]  = j;
                    backId[i] = it->second;
                }
            }
        }
        if (best[i] == kNegInf) {   // no piece ends here -> single-cp <unk>
            best[i]   = best[i - 1] + kUnkPenalty;
            backJ[i]  = i - 1;
            backId[i] = _unkSpId;
        }
    }

    std::vector<std::int32_t> ids;
    std::size_t i = n;
    while (i > 0) {
        ids.push_back(backId[i]);
        i = backJ[i];
    }
    std::reverse(ids.begin(), ids.end());
    return ids;
}

std::vector<std::int32_t>
XlmRobertaTokenizer::encode(std::string_view text) const {
    const std::vector<std::int32_t> sp = viterbi(normalize(text));
    std::vector<std::int32_t> out;
    out.reserve(sp.size());
    for (std::int32_t id : sp) {
        // fairseq: HF_id = sp_id + 1; sp <unk> -> HF unk (3).
        out.push_back(id == _unkSpId ? kUnk : id + 1);
    }
    return out;
}

std::vector<std::int32_t>
XlmRobertaTokenizer::encodePair(std::string_view query,
                                std::string_view passage) const {
    std::vector<std::int32_t> ids;
    ids.push_back(kBos);
    for (std::int32_t id : encode(query)) {
        ids.push_back(id);
    }
    ids.push_back(kEos);
    ids.push_back(kEos);
    for (std::int32_t id : encode(passage)) {
        ids.push_back(id);
    }
    ids.push_back(kEos);
    return ids;
}

std::vector<std::int32_t>
XlmRobertaTokenizer::encodeSingle(std::string_view text) const {
    std::vector<std::int32_t> ids;
    ids.push_back(kBos);
    for (std::int32_t id : encode(text)) {
        ids.push_back(id);
    }
    ids.push_back(kEos);
    return ids;
}

} // namespace mimirmind::model
