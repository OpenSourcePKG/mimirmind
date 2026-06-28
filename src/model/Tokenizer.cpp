#include "model/Tokenizer.hpp"

#include "model/GgufReader.hpp"
#include "runtime/Log.hpp"

#include <algorithm>
#include <cstring>
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

std::string normalizeForSpm(std::string_view text) {
    // SentencePiece (and llama.cpp's SPM tokenizer) prepends a single ▁
    // and replaces internal ASCII spaces with ▁.
    std::string out;
    out.reserve(text.size() + kSpaceMarker.size());
    out.append(kSpaceMarker);
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

void Tokenizer::loadFromGguf(const GgufReader& reader) {
    if (const auto* v = reader.findMetadata("tokenizer.ggml.model")) {
        if (auto m = asString(*v)) {
            _modelType = std::move(*m);
        }
    }
    MM_LOG_INFO("tok", "tokenizer.ggml.model = '{}'", _modelType);

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

    MM_LOG_INFO("tok",
                "vocab loaded: {} tokens, bos={} eos={} unk={} pad={}",
                _tokens.size(), _bosId, _eosId, _unknownId, _padId);
}

std::vector<std::int32_t> Tokenizer::encode(std::string_view text, bool addBos) const {
    if (_tokens.empty()) {
        MM_LOG_WARN("tok", "encode called before loadFromGguf");
        return {};
    }

    const std::string normalized = normalizeForSpm(text);

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

        // Verify the candidate still describes the current adjacency
        // (could be stale after intervening merges changed neighbours).
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
                 "encode('{}{}') -> {} tokens (after {} merges, addBos={})",
                 text.substr(0, std::min<std::size_t>(text.size(), 40)),
                 text.size() > 40 ? "..." : "",
                 out.size(), merges, addBos);

    return out;
}

std::string Tokenizer::decode(std::span<const std::int32_t> ids,
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

    // Replace SentencePiece space marker with ASCII space.
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

std::string_view Tokenizer::tokenText(std::int32_t id) const noexcept {
    if (id < 0 || static_cast<std::size_t>(id) >= _tokens.size()) {
        return {};
    }
    return _tokens[static_cast<std::size_t>(id)].text;
}

} // namespace mimirmind::model