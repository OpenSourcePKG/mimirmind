// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/encoder/laya/LayaTokenizer.hpp"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace mimirmind::runtime::encoder::laya {

namespace {

constexpr const char* kMetaspace = "\xE2\x96\x81";   // U+2581 '▁'

std::string readFile(const std::filesystem::path& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) {
        throw std::runtime_error("LayaTokenizer: cannot read " + p.string());
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

bool startsWithMetaspace(std::string_view s) {
    return s.size() >= 3 && static_cast<unsigned char>(s[0]) == 0xE2 &&
           static_cast<unsigned char>(s[1]) == 0x96 &&
           static_cast<unsigned char>(s[2]) == 0x81;
}

// UTF-8 code-point byte length from a lead byte (1..4; 1 for a stray byte).
std::size_t utf8Len(unsigned char lead) {
    if (lead < 0x80) return 1;
    if ((lead & 0xE0) == 0xC0) return 2;
    if ((lead & 0xF0) == 0xE0) return 3;
    if ((lead & 0xF8) == 0xF0) return 4;
    return 1;
}

} // namespace

void LayaTokenizer::load(std::string_view modelDir) {
    const std::filesystem::path dir = std::filesystem::path(modelDir) / "tokenizer";
    const nlohmann::json j = nlohmann::json::parse(readFile(dir / "tokenizer.json"));

    const auto& model = j.at("model");
    _vocab.reserve(model.at("vocab").size() * 2);
    for (const auto& [tok, id] : model.at("vocab").items()) {
        _vocab.emplace(tok, id.get<std::int32_t>());
    }
    // merges: array of ["a","b"] pairs (older dumps store "a b" strings — handle both).
    const auto& merges = model.at("merges");
    _mergeRank.reserve(merges.size() * 2);
    std::int32_t rank = 0;
    for (const auto& m : merges) {
        std::string key;
        if (m.is_array() && m.size() == 2) {
            key = m[0].get<std::string>() + " " + m[1].get<std::string>();
        } else if (m.is_string()) {
            key = m.get<std::string>();   // already "a b"
        } else {
            continue;
        }
        _mergeRank.emplace(std::move(key), rank++);
    }

    // byte-fallback tokens <0x00>..<0xFF>.
    for (int b = 0; b < 256; ++b) {
        char buf[8];
        std::snprintf(buf, sizeof(buf), "<0x%02X>", b);
        const auto it = _vocab.find(buf);
        _byteToken[b] = (it != _vocab.end()) ? it->second : -1;
    }

    // Special-token ids from tokenizer_config.json (strings like "<bos>").
    auto idOf = [&](const std::string& tokenText) -> std::int32_t {
        const auto it = _vocab.find(tokenText);
        return (it != _vocab.end()) ? it->second : -1;
    };
    const nlohmann::json tc =
        nlohmann::json::parse(readFile(dir / "tokenizer_config.json"));
    auto special = [&](const char* key, const char* fallback) -> std::int32_t {
        if (tc.contains(key) && tc.at(key).is_string()) {
            return idOf(tc.at(key).get<std::string>());
        }
        return idOf(fallback);
    };
    _clsId  = special("cls_token", "<bos>");
    _sepId  = special("sep_token", "<eos>");
    _maskId = special("mask_token", "<mask>");
    _padId  = special("pad_token", "<pad>");
    _unkId  = special("unk_token", "<unk>");
}

void LayaTokenizer::bpeMerge(std::vector<std::string>& syms) const {
    while (syms.size() > 1) {
        std::int32_t bestRank = std::numeric_limits<std::int32_t>::max();
        std::size_t bestI = 0;
        bool found = false;
        for (std::size_t i = 0; i + 1 < syms.size(); ++i) {
            const auto it = _mergeRank.find(syms[i] + " " + syms[i + 1]);
            if (it != _mergeRank.end() && it->second < bestRank) {
                bestRank = it->second;
                bestI = i;
                found = true;
            }
        }
        if (!found) {
            break;
        }
        syms[bestI] += syms[bestI + 1];
        syms.erase(syms.begin() + static_cast<std::ptrdiff_t>(bestI + 1));
    }
}

std::vector<std::int32_t> LayaTokenizer::encode(std::string_view text) const {
    // 1) normalize: ' ' -> metaspace.
    std::string norm;
    norm.reserve(text.size() + 8);
    for (char c : text) {
        if (c == ' ') {
            norm += kMetaspace;
        } else {
            norm += c;
        }
    }
    // 2) metaspace prepend "always": exactly one leading metaspace.
    if (!startsWithMetaspace(norm)) {
        norm = std::string(kMetaspace) + norm;
    }

    std::vector<std::int32_t> out;
    // 3) split into pieces at each metaspace boundary; BPE each piece.
    std::size_t i = 0;
    const std::size_t n = norm.size();
    while (i < n) {
        // piece = metaspace + following chars up to the next metaspace.
        std::size_t end = i + 3;   // the leading metaspace is 3 bytes
        while (end < n && !startsWithMetaspace(std::string_view(norm).substr(end))) {
            end += utf8Len(static_cast<unsigned char>(norm[end]));
        }
        const std::string_view piece = std::string_view(norm).substr(i, end - i);

        // split the piece into UTF-8 characters.
        std::vector<std::string> syms;
        for (std::size_t p = 0; p < piece.size();) {
            const std::size_t len = utf8Len(static_cast<unsigned char>(piece[p]));
            syms.emplace_back(piece.substr(p, len));
            p += len;
        }
        bpeMerge(syms);

        for (const std::string& sym : syms) {
            const auto it = _vocab.find(sym);
            if (it != _vocab.end()) {
                out.push_back(it->second);
            } else {
                // byte-fallback: emit <0xXX> per UTF-8 byte.
                for (unsigned char b : sym) {
                    const std::int32_t bt = _byteToken[b];
                    out.push_back(bt >= 0 ? bt : _unkId);
                }
            }
        }
        i = end;
    }
    return out;
}

} // namespace mimirmind::runtime::encoder::laya
