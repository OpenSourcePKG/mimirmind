// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <nlohmann/json.hpp>

#include <cstddef>
#include <regex>
#include <string>
#include <string_view>
#include <vector>

namespace mimirmind::hugin {

/**
 * Checks on a freely generated answer, for goldset items of kind "generate"
 * (7.3.4). A multiple-choice item only scores fixed sentences; it cannot see
 * an edit that makes the model mix languages or get a fact wrong. These
 * checks look at what the model actually writes:
 *
 *   lang        answer language ("de" / "en") by function-word markers
 *   no_mixing   at most `max_other_markers` markers of the other language
 *               (only checked together with `lang`)
 *   match       every regex in `match` occurs (case-insensitive)
 *   no_match    no regex in `no_match` occurs
 *
 * Spec JSON: {"max_tokens": 64, "lang": "de", "max_other_markers": 1,
 *             "match": ["Ottawa"], "no_match": ["Toronto"]}.
 * Regexes are ECMAScript; newlines in the answer are matched as spaces.
 */
class GenerationCheck {
public:
    /// Parses and validates a spec (throws std::runtime_error with the
    /// offending field on a bad language, threshold or regex).
    explicit GenerationCheck(const nlohmann::json& spec);

    /// Greedy token budget of the item.
    [[nodiscard]] std::size_t maxTokens() const noexcept { return _maxTokens; }

    /// {"checks": {name: bool}, "correct": all checks pass, "lang": detected,
    ///  "markers": {"de": n, "en": n}}.
    [[nodiscard]] nlohmann::json evaluate(std::string_view text) const;

    /// Marker counts of one text: {de, en}.
    struct Markers {
        std::size_t de{0};
        std::size_t en{0};
    };
    [[nodiscard]] static Markers countMarkers(std::string_view text);

    /// "de", "en" or "?" (tie) from marker counts.
    [[nodiscard]] static std::string detectLanguage(const Markers& m);

    static constexpr std::size_t kDefaultMaxTokens = 64;
    static constexpr std::size_t kMaxTokensCap     = 512;

private:
    struct Pattern {
        std::string source;
        std::regex  re;
    };
    static std::vector<Pattern> patterns(const nlohmann::json& spec, const char* key);

    std::size_t          _maxTokens{kDefaultMaxTokens};
    std::string          _lang;              // empty = no language check
    std::size_t          _maxOtherMarkers{1};
    std::vector<Pattern> _match;
    std::vector<Pattern> _noMatch;
};

} // namespace mimirmind::hugin
