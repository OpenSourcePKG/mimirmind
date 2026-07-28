// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "model/Tokenizer.hpp"

#include "core/gguf/GgufReader.hpp"
#include "core/log/Log.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <climits>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <queue>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

namespace mimirmind::model {

namespace {

// SentencePiece's "lower one-eighth block" — used as the visible
// representation of a space inside a token (▁).
constexpr std::string_view kSpaceMarker = "\xe2\x96\x81";

template <typename T>
std::optional<T> asNumeric(const MetadataValue& v) {
    return std::visit([](const auto& x) -> std::optional<T> {
        using U = std::decay_t<decltype(x)>;
        if constexpr (std::is_arithmetic_v<U>) {
            return static_cast<T>(x);
        } else {
            return std::nullopt;
        }
    }, v);
}

std::optional<std::string> asString(const MetadataValue& v) {
    if (std::holds_alternative<std::string>(v)) {
        return std::get<std::string>(v);
    }
    return std::nullopt;
}

const GgufArray* asArray(const MetadataValue* v) noexcept {
    if (v == nullptr) {
        return nullptr;
    }
    if (std::holds_alternative<GgufArray>(*v)) {
        return &std::get<GgufArray>(*v);
    }
    return nullptr;
}

template <typename T>
std::optional<T> readArrayElement(const GgufArray& arr, std::size_t i,
                                  GgufValueType expectedType) {
    if (arr.elementType != expectedType) {
        return std::nullopt;
    }
    const std::size_t off = i * sizeof(T);
    if (off + sizeof(T) > arr.raw.size()) {
        return std::nullopt;
    }
    T v;
    std::memcpy(&v, arr.raw.data() + off, sizeof(T));
    return v;
}

std::size_t utf8CodepointLen(unsigned char c) noexcept {
    if ((c & 0x80) == 0)    return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1; // malformed leading byte — treat as a single byte
}

std::string codepointToUtf8(int cp) {
    std::string out;
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    return out;
}

// GPT-2 byte-to-unicode mapping (Karpathy / OpenAI). Printable ASCII +
// Latin-1 byte values stay as themselves; control / unicode-conflicting
// bytes get remapped to U+0100..U+0142 so the entire 0..255 byte range
// has a visible unicode representation.
struct ByteUnicodeMap {
    std::array<std::string, 256>            byteToUtf8;
    std::unordered_map<std::string, std::uint8_t> utf8ToByte;
};

const ByteUnicodeMap& byteMap() {
    static const ByteUnicodeMap m = [] {
        std::vector<int> bs;
        std::vector<int> cs;
        for (int i = '!';  i <= '~';  ++i) { bs.push_back(i); cs.push_back(i); }
        for (int i = 0xa1; i <= 0xac; ++i) { bs.push_back(i); cs.push_back(i); }
        for (int i = 0xae; i <= 0xff; ++i) { bs.push_back(i); cs.push_back(i); }

        int n = 0;
        for (int b = 0; b < 256; ++b) {
            if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
                bs.push_back(b);
                cs.push_back(256 + n);
                ++n;
            }
        }

        ByteUnicodeMap out;
        out.utf8ToByte.reserve(256);
        for (std::size_t i = 0; i < bs.size(); ++i) {
            const std::string s = codepointToUtf8(cs[i]);
            out.byteToUtf8[static_cast<std::size_t>(bs[i])] = s;
            out.utf8ToByte.emplace(s, static_cast<std::uint8_t>(bs[i]));
        }
        return out;
    }();
    return m;
}

std::string normalizeForSpm(std::string_view text, bool addSpacePrefix) {
    // SentencePiece replaces internal ASCII spaces with ▁. Whether it also
    // *prepends* a ▁ to the very first character depends on the model:
    // Llama/Mistral/Gemma 3 default to yes, Gemma 4's GGUF sets it to no.
    std::string out;
    out.reserve(text.size() + kSpaceMarker.size());
    if (addSpacePrefix) {
        out.append(kSpaceMarker);
    }
    for (char c : text) {
        if (c == ' ') {
            out.append(kSpaceMarker);
        } else {
            out.push_back(c);
        }
    }
    return out;
}

} // namespace

// ---------------------------------------------------------------------------
// Loader
// ---------------------------------------------------------------------------

void Tokenizer::loadFromGguf(const GgufReader& reader) {
    if (const auto* v = reader.findMetadata("tokenizer.ggml.model")) {
        if (auto m = asString(*v)) {
            _modelType = std::move(*m);
        }
    }
    MM_LOG_INFO("tok", "tokenizer.ggml.model = '{}'", _modelType);

    // Decide dispatcher once. Anything not 'gpt2' goes through SPM —
    // 'llama', 'gemma', 'gemma4' all use SentencePiece with the ▁ marker
    // and produce identical encodings via our SPM path. Log unknown
    // models exactly once so the encode loop stays quiet.
    if (_modelType != "gpt2" &&
        _modelType != "llama" &&
        _modelType != "gemma" &&
        _modelType != "gemma4" &&
        !_modelType.empty()) {
        MM_LOG_INFO("tok",
                    "tokenizer model '{}' not in recognised list — using SPM dispatch",
                    _modelType);
    }

    const auto* tokensArr = asArray(reader.findMetadata("tokenizer.ggml.tokens"));
    if (tokensArr == nullptr || tokensArr->elementType != GgufValueType::String) {
        MM_LOG_ERROR("tok", "tokenizer.ggml.tokens missing or not a string array");
        throw std::runtime_error("Tokenizer: tokenizer.ggml.tokens missing");
    }

    const auto* scoresArr = asArray(reader.findMetadata("tokenizer.ggml.scores"));
    const auto* typesArr  = asArray(reader.findMetadata("tokenizer.ggml.token_type"));

    const std::size_t vocabN = tokensArr->strings.size();
    _tokens.reserve(vocabN);
    _byText.reserve(vocabN * 2);

    for (std::size_t i = 0; i < vocabN; ++i) {
        TokenInfo info;
        info.text  = tokensArr->strings[i];

        if (scoresArr != nullptr) {
            if (auto v = readArrayElement<float>(*scoresArr, i, GgufValueType::Float32)) {
                info.score = *v;
            }
        }
        if (typesArr != nullptr) {
            if (auto v = readArrayElement<std::int32_t>(*typesArr, i, GgufValueType::Int32)) {
                info.type = *v;
            }
        }

        // Last-write wins for duplicate strings (matches llama.cpp behaviour
        // for vocabs that intentionally repeat — rare but happens).
        _byText[info.text] = static_cast<std::int32_t>(i);
        _tokens.push_back(std::move(info));
    }

    auto readSpecialId = [&](const char* key, std::int32_t& dst) {
        if (const auto* v = reader.findMetadata(key)) {
            if (auto n = asNumeric<std::int32_t>(*v)) {
                dst = *n;
            }
        }
    };
    readSpecialId("tokenizer.ggml.bos_token_id",     _bosId);
    readSpecialId("tokenizer.ggml.eos_token_id",     _eosId);
    readSpecialId("tokenizer.ggml.unknown_token_id", _unknownId);
    readSpecialId("tokenizer.ggml.padding_token_id", _padId);

    // `tokenizer.ggml.add_space_prefix` — SentencePiece "prepend ▁ at the
    // start of the input" flag. Defaults to true (legacy Llama/Mistral
    // behaviour); Gemma 4 sets it to false in its converter
    // (conversion/gemma.py: `add_add_space_prefix(False)`).
    if (const auto* v = reader.findMetadata("tokenizer.ggml.add_space_prefix")) {
        if (std::holds_alternative<bool>(*v)) {
            _addSpacePrefix = std::get<bool>(*v);
        }
    }

    // For GPT-2 the merges array is what drives encoding — load it.
    if (const auto* mergesArr = asArray(reader.findMetadata("tokenizer.ggml.merges"))) {
        if (mergesArr->elementType == GgufValueType::String) {
            _mergesRank.reserve(mergesArr->strings.size());
            for (std::size_t i = 0; i < mergesArr->strings.size(); ++i) {
                _mergesRank.emplace(mergesArr->strings[i],
                                    static_cast<std::int32_t>(i));
            }
        }
    }

    MM_LOG_INFO("tok",
                "vocab loaded: {} tokens, {} merges, bos={} eos={} unk={} "
                "pad={} add_space_prefix={}",
                _tokens.size(), _mergesRank.size(),
                _bosId, _eosId, _unknownId, _padId, _addSpacePrefix);
}

void Tokenizer::loadFromHfJson(const std::string& checkpointDir) {
    namespace fs = std::filesystem;
    const fs::path dir{checkpointDir};

    auto readJson = [](const fs::path& p) -> nlohmann::json {
        std::ifstream f(p);
        if (!f) {
            throw std::runtime_error(
                "Tokenizer::loadFromHfJson: cannot open " + p.string());
        }
        nlohmann::json j;
        f >> j;
        return j;
    };

    const nlohmann::json tj = readJson(dir / "tokenizer.json");
    const auto modelIt = tj.find("model");
    if (modelIt == tj.end() || !modelIt->is_object()) {
        throw std::runtime_error(
            "Tokenizer::loadFromHfJson: tokenizer.json has no 'model' object");
    }
    const nlohmann::json& model = *modelIt;

    // Byte-level BPE only (Qwen / GPT-2 family) — the one encode path this
    // populates. SentencePiece / Unigram / WordPiece checkpoints must keep
    // using a GGUF tokenizer (models[].tokenizerGguf).
    const std::string bpeType = model.value("type", std::string{});
    if (bpeType != "BPE") {
        throw std::runtime_error(
            "Tokenizer::loadFromHfJson: only byte-level BPE tokenizer.json is "
            "supported (model.type='" + bpeType + "')");
    }
    _modelType = "gpt2";

    // ---- 1. vocab {token: id} (+ added_tokens) -> _tokens (indexed by id)
    //         and _byText. Same layout the gpt2 GGUF path builds.
    const auto vocabIt = model.find("vocab");
    if (vocabIt == model.end() || !vocabIt->is_object()) {
        throw std::runtime_error(
            "Tokenizer::loadFromHfJson: tokenizer.json model.vocab missing");
    }

    std::vector<std::pair<std::int32_t, std::string>> entries;
    entries.reserve(vocabIt->size() + 64);
    std::int32_t maxId = -1;
    for (auto it = vocabIt->begin(); it != vocabIt->end(); ++it) {
        const auto id = it.value().get<std::int32_t>();
        entries.emplace_back(id, it.key());
        maxId = std::max(maxId, id);
    }
    if (const auto addedIt = tj.find("added_tokens");
        addedIt != tj.end() && addedIt->is_array()) {
        for (const auto& a : *addedIt) {
            if (!a.contains("id") || !a.contains("content")) {
                continue;
            }
            const auto id = a["id"].get<std::int32_t>();
            entries.emplace_back(id, a["content"].get<std::string>());
            maxId = std::max(maxId, id);
        }
    }
    if (maxId < 0) {
        throw std::runtime_error(
            "Tokenizer::loadFromHfJson: empty vocab");
    }

    _tokens.assign(static_cast<std::size_t>(maxId) + 1, TokenInfo{});
    _byText.clear();
    _byText.reserve(entries.size() * 2);
    for (auto& [id, text] : entries) {
        _tokens[static_cast<std::size_t>(id)].text = text;  // score=0, type=1
        _byText[text] = id;  // last-write-wins, mirrors the GGUF path
    }

    // ---- 2. merges -> rank (index). tokenizer.json stores them either as
    //         space-joined strings ("Ġ Ġ", identical to the GGUF format) or
    //         as ["Ġ","Ġ"] pairs; accept both.
    _mergesRank.clear();
    if (const auto mergesIt = model.find("merges");
        mergesIt != model.end() && mergesIt->is_array()) {
        _mergesRank.reserve(mergesIt->size());
        std::int32_t rank = 0;
        for (const auto& m : *mergesIt) {
            std::string key;
            if (m.is_string()) {
                key = m.get<std::string>();
            } else if (m.is_array() && m.size() == 2) {
                key = m[0].get<std::string>() + ' ' + m[1].get<std::string>();
            } else {
                continue;
            }
            _mergesRank.emplace(std::move(key), rank++);
        }
    }

    // ---- 3. special-token ids. tokenizer_config.json is authoritative:
    //         Qwen sets bos_token=null (no BOS prepend) -> _bosId = -1, which
    //         the gpt2 encode path honours. generation_config.json is a
    //         numeric fallback for eos only (it may be a list -> take first).
    auto idOf = [&](const std::string& text) -> std::int32_t {
        const auto it = _byText.find(text);
        return it != _byText.end() ? it->second : -1;
    };
    _bosId = _eosId = _unknownId = _padId = -1;

    if (const fs::path tcPath = dir / "tokenizer_config.json"; fs::exists(tcPath)) {
        const nlohmann::json tc = readJson(tcPath);
        auto strTok = [&](const char* key) -> std::string {
            if (const auto it = tc.find(key); it != tc.end() && it->is_string()) {
                return it->get<std::string>();
            }
            return {};
        };
        if (const std::string s = strTok("bos_token"); !s.empty()) _bosId     = idOf(s);
        if (const std::string s = strTok("eos_token"); !s.empty()) _eosId     = idOf(s);
        if (const std::string s = strTok("pad_token"); !s.empty()) _padId     = idOf(s);
        if (const std::string s = strTok("unk_token"); !s.empty()) _unknownId = idOf(s);
    }

    if (const fs::path gcPath = dir / "generation_config.json";
        _eosId < 0 && fs::exists(gcPath)) {
        const nlohmann::json gc = readJson(gcPath);
        if (const auto it = gc.find("eos_token_id"); it != gc.end()) {
            if (it->is_number_integer()) {
                _eosId = it->get<std::int32_t>();
            } else if (it->is_array() && !it->empty()
                       && it->front().is_number_integer()) {
                _eosId = it->front().get<std::int32_t>();
            }
        }
    }

    _addSpacePrefix = false;  // gpt2 byte-level BPE never prepends a ▁.

    MM_LOG_INFO("tok",
                "HF tokenizer.json loaded: {} tokens, {} merges, bos={} eos={} "
                "unk={} pad={}",
                _tokens.size(), _mergesRank.size(),
                _bosId, _eosId, _unknownId, _padId);
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

std::vector<std::int32_t> Tokenizer::encode(std::string_view text, bool addBos) const {
    if (_tokens.empty()) {
        MM_LOG_WARN("tok", "encode called before loadFromGguf");
        return {};
    }
    if (_modelType == "gpt2") {
        return encodeGpt2(text, addBos);
    }
    return encodeSpm(text, addBos);
}

std::string Tokenizer::decode(std::span<const std::int32_t> ids, bool skipSpecial) const {
    if (_modelType == "gpt2") {
        return decodeGpt2(ids, skipSpecial);
    }
    return decodeSpm(ids, skipSpecial);
}

// ---------------------------------------------------------------------------
// SentencePiece path (llama / gemma / mistral)
// ---------------------------------------------------------------------------

std::vector<std::int32_t> Tokenizer::encodeSpm(std::string_view text, bool addBos) const {
    const std::string normalized = normalizeForSpm(text, _addSpacePrefix);

    // Initial segmentation: one segment per UTF-8 codepoint, looked up in
    // the vocab. Anything not in the vocab becomes _unknownId (proper
    // byte-fallback synthesis lands when we have a model to validate
    // against).
    struct Seg {
        std::string  text;
        std::int32_t id;
        bool         active;
        std::size_t  prev;
        std::size_t  next;
    };
    std::vector<Seg> segs;
    segs.reserve(normalized.size());

    {
        std::size_t i = 0;
        while (i < normalized.size()) {
            const std::size_t cpLen = utf8CodepointLen(
                static_cast<unsigned char>(normalized[i]));
            const std::size_t take = std::min(cpLen, normalized.size() - i);
            std::string cp = normalized.substr(i, take);
            std::int32_t id = _unknownId;
            if (auto it = _byText.find(cp); it != _byText.end()) {
                id = it->second;
            }
            const std::size_t idx = segs.size();
            segs.push_back({std::move(cp), id, true, idx - 1, idx + 1});
            i += take;
        }
    }

    if (segs.empty()) {
        std::vector<std::int32_t> out;
        if (addBos && _bosId >= 0) {
            out.push_back(_bosId);
        }
        return out;
    }
    segs.front().prev = SIZE_MAX;
    segs.back().next  = SIZE_MAX;

    // Best-pair-merge priority queue. We push (score, leftIdx, mergedId);
    // top is highest score, leftmost on tie.
    struct Candidate {
        float        score;
        std::size_t  left;
        std::int32_t id;
        bool operator<(const Candidate& o) const noexcept {
            if (score != o.score) {
                return score < o.score;
            }
            return left > o.left;   // smaller left wins on tie
        }
    };
    std::priority_queue<Candidate> pq;

    auto tryAddCandidate = [&](std::size_t leftIdx) {
        if (leftIdx == SIZE_MAX || leftIdx >= segs.size() || !segs[leftIdx].active) {
            return;
        }
        const std::size_t rightIdx = segs[leftIdx].next;
        if (rightIdx == SIZE_MAX || rightIdx >= segs.size() || !segs[rightIdx].active) {
            return;
        }
        std::string combined = segs[leftIdx].text + segs[rightIdx].text;
        auto it = _byText.find(combined);
        if (it == _byText.end()) {
            return;
        }
        const std::int32_t mergedId = it->second;
        pq.push({_tokens[static_cast<std::size_t>(mergedId)].score, leftIdx, mergedId});
    };

    for (std::size_t i = 0; i < segs.size(); ++i) {
        tryAddCandidate(i);
    }

    std::size_t merges = 0;
    while (!pq.empty()) {
        const Candidate c = pq.top();
        pq.pop();

        auto& left = segs[c.left];
        if (!left.active) {
            continue;
        }
        const std::size_t rIdx = left.next;
        if (rIdx == SIZE_MAX || rIdx >= segs.size() || !segs[rIdx].active) {
            continue;
        }
        auto& right = segs[rIdx];

        std::string combined = left.text + right.text;
        auto it = _byText.find(combined);
        if (it == _byText.end() || it->second != c.id) {
            continue;
        }

        left.text = std::move(combined);
        left.id   = c.id;
        left.next = right.next;
        if (right.next != SIZE_MAX && right.next < segs.size()) {
            segs[right.next].prev = c.left;
        }
        right.active = false;
        ++merges;

        tryAddCandidate(c.left);
        tryAddCandidate(left.prev);
    }

    std::vector<std::int32_t> out;
    out.reserve(segs.size() + 1);
    if (addBos && _bosId >= 0) {
        out.push_back(_bosId);
    }
    std::size_t cursor = 0;
    while (cursor < segs.size() && !segs[cursor].active) {
        ++cursor;
    }
    while (cursor != SIZE_MAX && cursor < segs.size()) {
        out.push_back(segs[cursor].id);
        cursor = segs[cursor].next;
    }

    MM_LOG_DEBUG("tok",
                 "encodeSpm('{}{}') -> {} tokens (after {} merges, addBos={})",
                 text.substr(0, std::min<std::size_t>(text.size(), 40)),
                 text.size() > 40 ? "..." : "",
                 out.size(), merges, addBos);

    return out;
}

std::string Tokenizer::decodeSpm(std::span<const std::int32_t> ids,
                                 bool skipSpecial) const {
    std::string raw;
    raw.reserve(ids.size() * 4);

    for (auto id : ids) {
        if (id < 0 || static_cast<std::size_t>(id) >= _tokens.size()) {
            continue;
        }
        if (skipSpecial && (id == _bosId || id == _eosId || id == _padId)) {
            continue;
        }
        raw.append(_tokens[static_cast<std::size_t>(id)].text);
    }

    std::string out;
    out.reserve(raw.size());
    for (std::size_t i = 0; i < raw.size();) {
        if (i + kSpaceMarker.size() <= raw.size() &&
            std::memcmp(raw.data() + i, kSpaceMarker.data(), kSpaceMarker.size()) == 0) {
            out.push_back(' ');
            i += kSpaceMarker.size();
        } else {
            out.push_back(raw[i]);
            ++i;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// GPT-2 byte-level BPE path (qwen2 / gpt-neox / ...)
// ---------------------------------------------------------------------------

std::vector<std::int32_t> Tokenizer::encodeGpt2(std::string_view text, bool addBos) const {
    if (_mergesRank.empty()) {
        MM_LOG_WARN("tok",
                    "gpt2 tokenizer has no merges loaded — falling back to "
                    "per-byte lookup");
    }

    const auto& bm = byteMap();

    // Byte-encode: each input byte becomes its UTF-8 representation under
    // the GPT-2 byte-to-unicode mapping. Each entry of `word` is a piece
    // (which may grow during merging).
    std::vector<std::string> word;
    word.reserve(text.size());
    for (char c : text) {
        word.push_back(bm.byteToUtf8[static_cast<unsigned char>(c)]);
    }

    // Greedy best-rank-pair BPE. Iteratively pick the adjacent pair with
    // the lowest rank in `_mergesRank` and merge it; stop when no
    // adjacent pair maps to a merge.
    while (word.size() > 1) {
        std::int32_t bestRank = INT32_MAX;
        std::size_t  bestIdx  = SIZE_MAX;
        for (std::size_t i = 0; i + 1 < word.size(); ++i) {
            std::string key;
            key.reserve(word[i].size() + 1 + word[i + 1].size());
            key.append(word[i]);
            key.push_back(' ');
            key.append(word[i + 1]);
            auto it = _mergesRank.find(key);
            if (it != _mergesRank.end() && it->second < bestRank) {
                bestRank = it->second;
                bestIdx  = i;
            }
        }
        if (bestIdx == SIZE_MAX) {
            break;
        }
        word[bestIdx] = word[bestIdx] + word[bestIdx + 1];
        word.erase(word.begin() + static_cast<std::ptrdiff_t>(bestIdx) + 1);
    }

    std::vector<std::int32_t> out;
    out.reserve(word.size() + 1);
    if (addBos && _bosId >= 0) {
        out.push_back(_bosId);
    }
    std::size_t oov = 0;
    for (const auto& w : word) {
        auto it = _byText.find(w);
        if (it != _byText.end()) {
            out.push_back(it->second);
        } else {
            out.push_back(_unknownId);
            ++oov;
        }
    }

    MM_LOG_DEBUG("tok",
                 "encodeGpt2('{}{}') -> {} tokens ({} oov, addBos={})",
                 text.substr(0, std::min<std::size_t>(text.size(), 40)),
                 text.size() > 40 ? "..." : "",
                 out.size(), oov, addBos);

    return out;
}

std::string Tokenizer::decodeGpt2(std::span<const std::int32_t> ids,
                                  bool skipSpecial) const {
    const auto& bm = byteMap();

    // Concatenate token text (in GPT-2 byte-encoded form), then walk
    // codepoint-by-codepoint and invert the byte mapping. Unknown
    // codepoints are passed through verbatim — that surfaces special
    // tokens like <|endoftext|> rather than silently dropping them.
    std::string raw;
    raw.reserve(ids.size() * 4);
    for (auto id : ids) {
        if (id < 0 || static_cast<std::size_t>(id) >= _tokens.size()) {
            continue;
        }
        if (skipSpecial && (id == _bosId || id == _eosId || id == _padId)) {
            continue;
        }
        raw.append(_tokens[static_cast<std::size_t>(id)].text);
    }

    std::string out;
    out.reserve(raw.size());
    std::size_t i = 0;
    while (i < raw.size()) {
        const std::size_t cpLen = utf8CodepointLen(
            static_cast<unsigned char>(raw[i]));
        const std::size_t take = std::min(cpLen, raw.size() - i);
        std::string cp = raw.substr(i, take);
        if (auto it = bm.utf8ToByte.find(cp); it != bm.utf8ToByte.end()) {
            out.push_back(static_cast<char>(it->second));
        } else {
            out.append(cp);
        }
        i += take;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Misc
// ---------------------------------------------------------------------------

std::string_view Tokenizer::tokenText(std::int32_t id) const noexcept {
    if (id < 0 || static_cast<std::size_t>(id) >= _tokens.size()) {
        return {};
    }
    return _tokens[static_cast<std::size_t>(id)].text;
}

std::int32_t Tokenizer::findToken(std::string_view text) const noexcept {
    const auto it = _byText.find(std::string{text});
    if (it == _byText.end()) {
        return -1;
    }
    return it->second;
}

} // namespace mimirmind::model