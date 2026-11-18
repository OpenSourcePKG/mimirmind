// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "hugin/GenerationCheck.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>

namespace mimirmind::hugin {

using nlohmann::json;

namespace {

// Function words that are frequent in one language and (nearly) absent in the
// other. Words that are common in both ("in", "an", "was", "also", "am") are
// left out on purpose; a word with an umlaut or ß counts as German as well.
constexpr std::array<std::string_view, 52> kGermanMarkers{
    "der", "die", "das", "den", "dem", "des", "und", "ist", "sind", "wird", "werden", "wurde",
    "ein", "eine", "einer", "eines", "einem", "nicht", "auf", "mit", "von", "zum", "zur",
    "bei", "auch", "oder", "sich", "es", "ich", "wir", "sie", "kann", "hat", "gibt", "liegt",
    "im", "grad", "tage", "sprache", "hauptstadt", "beine", "siedet", "schrieb", "entspricht",
    "wasser", "geschrieben", "entdeckt", "metern", "insekten", "gesprochen", "jahr", "jahre"};

constexpr std::array<std::string_view, 50> kEnglishMarkers{
    "the", "is", "are", "and", "of", "it", "its", "this", "that", "with", "for", "which",
    "have", "has", "be", "can", "from", "they", "their", "these", "there", "by", "not",
    "spoken", "days", "capital", "legs", "largest", "tallest", "discovered", "wrote",
    "written", "boils", "water", "elevation", "located", "insects", "official", "language",
    "symbol", "freezing", "point", "year", "years", "people", "country", "called", "known",
    "about", "what"};

bool isWordByte(unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 0x80;   // UTF-8 letters
}

bool hasGermanLetter(std::string_view w) {
    // ä ö ü ß Ä Ö Ü in UTF-8: C3 A4 / B6 / BC / 9F / 84 / 96 / 9C.
    for (std::size_t i = 0; i + 1 < w.size(); ++i) {
        if (static_cast<unsigned char>(w[i]) == 0xC3) {
            switch (static_cast<unsigned char>(w[i + 1])) {
            case 0xA4: case 0xB6: case 0xBC: case 0x9F: case 0x84: case 0x96: case 0x9C:
                return true;
            default:
                break;
            }
        }
    }
    return false;
}

template <std::size_t N>
bool contains(const std::array<std::string_view, N>& set, std::string_view w) {
    return std::find(set.begin(), set.end(), w) != set.end();
}

std::string flattenNewlines(std::string_view text) {
    std::string s{text};
    std::replace(s.begin(), s.end(), '\n', ' ');
    std::replace(s.begin(), s.end(), '\r', ' ');
    return s;
}

} // namespace

GenerationCheck::GenerationCheck(const json& spec) {
    if (!spec.is_object()) {
        throw std::runtime_error("'generate' must be an object");
    }
    if (spec.contains("max_tokens")) {
        const auto n = spec["max_tokens"].get<std::int64_t>();
        if (n < 1 || n > static_cast<std::int64_t>(kMaxTokensCap)) {
            throw std::runtime_error("generate.max_tokens must be in [1, " +
                                     std::to_string(kMaxTokensCap) + "]");
        }
        _maxTokens = static_cast<std::size_t>(n);
    }
    if (spec.contains("lang")) {
        _lang = spec["lang"].get<std::string>();
        if (_lang != "de" && _lang != "en") {
            throw std::runtime_error("generate.lang must be \"de\" or \"en\"");
        }
    }
    if (spec.contains("max_other_markers")) {
        const auto n = spec["max_other_markers"].get<std::int64_t>();
        if (n < 0) {
            throw std::runtime_error("generate.max_other_markers must be >= 0");
        }
        _maxOtherMarkers = static_cast<std::size_t>(n);
    }
    _match   = patterns(spec, "match");
    _noMatch = patterns(spec, "no_match");
    if (_lang.empty() && _match.empty() && _noMatch.empty()) {
        throw std::runtime_error("generate item needs at least one of lang / match / no_match");
    }
}

std::vector<GenerationCheck::Pattern> GenerationCheck::patterns(const json& spec, const char* key) {
    std::vector<Pattern> out;
    if (!spec.contains(key)) {
        return out;
    }
    const json& v = spec[key];
    const std::vector<std::string> sources =
        v.is_string() ? std::vector<std::string>{v.get<std::string>()} : v.get<std::vector<std::string>>();
    for (const auto& s : sources) {
        try {
            out.push_back({s, std::regex(s, std::regex::ECMAScript | std::regex::icase)});
        } catch (const std::regex_error& ex) {
            throw std::runtime_error(std::string{"generate."} + key + ": bad regex '" + s + "': " + ex.what());
        }
    }
    return out;
}

GenerationCheck::Markers GenerationCheck::countMarkers(std::string_view text) {
    Markers m;
    std::size_t i = 0;
    while (i < text.size()) {
        if (!isWordByte(static_cast<unsigned char>(text[i]))) {
            ++i;
            continue;
        }
        std::size_t j = i;
        while (j < text.size() && isWordByte(static_cast<unsigned char>(text[j]))) {
            ++j;
        }
        std::string w{text.substr(i, j - i)};
        std::transform(w.begin(), w.end(), w.begin(), [](char c) {
            return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
        });
        if (hasGermanLetter(w) || contains(kGermanMarkers, w)) {
            ++m.de;
        } else if (contains(kEnglishMarkers, w)) {
            ++m.en;
        }
        i = j;
    }
    return m;
}

std::string GenerationCheck::detectLanguage(const Markers& m) {
    return m.de > m.en ? "de" : m.en > m.de ? "en" : "?";
}

json GenerationCheck::evaluate(std::string_view text) const {
    const Markers m    = countMarkers(text);
    const std::string got = detectLanguage(m);
    json checks = json::object();
    if (!_lang.empty()) {
        checks["lang"] = got == _lang;
        const std::size_t other = _lang == "de" ? m.en : m.de;
        checks["no_mixing"] = other <= _maxOtherMarkers;
    }
    const std::string flat = flattenNewlines(text);
    if (!_match.empty()) {
        checks["match"] = std::all_of(_match.begin(), _match.end(),
                                      [&](const Pattern& p) { return std::regex_search(flat, p.re); });
    }
    if (!_noMatch.empty()) {
        checks["no_match"] = std::none_of(_noMatch.begin(), _noMatch.end(),
                                          [&](const Pattern& p) { return std::regex_search(flat, p.re); });
    }
    bool all = true;
    for (const auto& [k, v] : checks.items()) {
        all = all && v.get<bool>();
    }
    return json{{"checks", checks}, {"correct", all}, {"lang", got},
                {"markers", {{"de", m.de}, {"en", m.en}}}};
}

} // namespace mimirmind::hugin
