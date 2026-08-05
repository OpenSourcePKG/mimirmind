// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "model/ToolCallParser.hpp"

#include <nlohmann/json.hpp>

#include <cctype>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace mimirmind::model {

namespace {

constexpr std::string_view kOpen  = "<tool_call>";
constexpr std::string_view kClose = "</tool_call>";

// Gemma 4 markers (note the leading/trailing pipe — distinct from Qwen).
constexpr std::string_view kGOpen  = "<|tool_call>";
constexpr std::string_view kGClose = "<tool_call|>";
constexpr std::string_view kGQuote = "<|\"|>";

using json = nlohmann::json;

std::string trimws(std::string_view v) {
    const std::size_t a = v.find_first_not_of(" \t\n\r");
    if (a == std::string_view::npos) {
        return "";
    }
    const std::size_t b = v.find_last_not_of(" \t\n\r");
    return std::string{v.substr(a, b - a + 1)};
}

/// Recursive-descent parser for Gemma's tool-argument DSL over a string_view
/// with a moving cursor. Values: `<|"|>string<|"|>` (or a plain "string"),
/// `true`/`false`, numbers, `{k:v,...}` objects (bare keys), `[...]` arrays.
/// Throws std::runtime_error on a malformed value — parseGemma catches it.
struct GemmaValParser {
    std::string_view s;
    std::size_t      i = 0;

    void skipws() {
        while (i < s.size() &&
               (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) {
            ++i;
        }
    }
    [[nodiscard]] bool eof() const { return i >= s.size(); }
    [[nodiscard]] bool starts(std::string_view p) const {
        return s.substr(i, p.size()) == p;
    }

    json value() {
        skipws();
        if (starts(kGQuote))              return gstring();
        if (!eof() && s[i] == '"')        return dquote();
        if (!eof() && s[i] == '{')        return object();
        if (!eof() && s[i] == '[')        return array();
        if (starts("true"))  { i += 4;    return true; }
        if (starts("false")) { i += 5;    return false; }
        return scalar();
    }

    json gstring() {
        i += kGQuote.size();
        const std::size_t end = s.find(kGQuote, i);
        if (end == std::string_view::npos) {
            throw std::runtime_error("unterminated gemma string");
        }
        std::string out{s.substr(i, end - i)};
        i = end + kGQuote.size();
        return out;
    }

    json dquote() {
        ++i;  // opening "
        std::string out;
        while (i < s.size() && s[i] != '"') {
            if (s[i] == '\\' && i + 1 < s.size()) {
                out.push_back(s[i + 1]);
                i += 2;
            } else {
                out.push_back(s[i]);
                ++i;
            }
        }
        if (i >= s.size()) {
            throw std::runtime_error("unterminated string");
        }
        ++i;  // closing "
        return out;
    }

    json object() {
        ++i;  // {
        json obj = json::object();
        skipws();
        if (!eof() && s[i] == '}') { ++i; return obj; }
        while (true) {
            skipws();
            const std::size_t ks = i;
            while (i < s.size() && s[i] != ':' && s[i] != '}') { ++i; }
            const std::string key = trimws(s.substr(ks, i - ks));
            if (eof() || s[i] != ':') {
                throw std::runtime_error("expected ':' in object");
            }
            ++i;  // :
            obj[key] = value();
            skipws();
            if (!eof() && s[i] == ',') { ++i; continue; }
            if (!eof() && s[i] == '}') { ++i; break; }
            throw std::runtime_error("expected ',' or '}'");
        }
        return obj;
    }

    json array() {
        ++i;  // [
        json arr = json::array();
        skipws();
        if (!eof() && s[i] == ']') { ++i; return arr; }
        while (true) {
            arr.push_back(value());
            skipws();
            if (!eof() && s[i] == ',') { ++i; continue; }
            if (!eof() && s[i] == ']') { ++i; break; }
            throw std::runtime_error("expected ',' or ']'");
        }
        return arr;
    }

    json scalar() {
        const std::size_t vs = i;
        while (i < s.size() && s[i] != ',' && s[i] != '}' && s[i] != ']') {
            ++i;
        }
        const std::string tok = trimws(s.substr(vs, i - vs));
        try {
            if (tok.find_first_of(".eE") != std::string::npos) {
                return std::stod(tok);
            }
            return static_cast<std::int64_t>(std::stoll(tok));
        } catch (...) {
            return tok;  // bare token → treat as string
        }
    }
};

/// Normalise the `arguments` field to a compact JSON *string*: OpenAI carries
/// arguments stringified. Qwen usually emits an object; some models emit an
/// already-stringified value. Anything else falls back to "{}".
std::string argumentsToString(const json& args) {
    if (args.is_string()) {
        return args.get<std::string>();
    }
    if (args.is_object() || args.is_array()) {
        return args.dump();
    }
    return "{}";
}

} // namespace

bool ToolCallParser::looksLikeQwenToolCall(std::string_view text) noexcept {
    return text.find(kOpen) != std::string_view::npos;
}

std::vector<ToolCall> ToolCallParser::parseQwen(std::string_view text) {
    std::vector<ToolCall> calls;

    std::size_t pos = 0;
    while (true) {
        const std::size_t open = text.find(kOpen, pos);
        if (open == std::string_view::npos) {
            break;
        }
        const std::size_t bodyStart = open + kOpen.size();
        const std::size_t close     = text.find(kClose, bodyStart);
        // An unterminated final block (e.g. output truncated at max_tokens):
        // take the rest of the string so a complete JSON object still parses.
        const std::size_t bodyEnd =
            (close == std::string_view::npos) ? text.size() : close;

        const std::string_view body = text.substr(bodyStart, bodyEnd - bodyStart);

        // Parse the inner JSON; skip this block on any error rather than
        // aborting the whole response.
        json parsed = json::parse(body, /*cb=*/nullptr, /*allow_exceptions=*/false);
        if (!parsed.is_discarded() && parsed.is_object()) {
            const auto nameIt = parsed.find("name");
            if (nameIt != parsed.end() && nameIt->is_string()) {
                ToolCall call;
                call.id   = "call_" + std::to_string(calls.size());
                call.name = nameIt->get<std::string>();

                const auto argsIt = parsed.find("arguments");
                call.argumentsJson =
                    (argsIt != parsed.end()) ? argumentsToString(*argsIt) : "{}";

                calls.push_back(std::move(call));
            }
        }

        if (close == std::string_view::npos) {
            break;
        }
        pos = close + kClose.size();
    }

    return calls;
}

bool ToolCallParser::looksLikeGemmaToolCall(std::string_view text) noexcept {
    return text.find(kGOpen) != std::string_view::npos;
}

std::vector<ToolCall> ToolCallParser::parseGemma(std::string_view text) {
    std::vector<ToolCall> calls;

    std::size_t pos = 0;
    while (true) {
        const std::size_t open = text.find(kGOpen, pos);
        if (open == std::string_view::npos) {
            break;
        }
        const std::size_t bodyStart = open + kGOpen.size();
        const std::size_t close     = text.find(kGClose, bodyStart);
        const std::size_t bodyEnd =
            (close == std::string_view::npos) ? text.size() : close;
        const std::string_view body = text.substr(bodyStart, bodyEnd - bodyStart);

        // Expect `call:NAME{...}` (leading whitespace tolerated).
        std::size_t p = 0;
        while (p < body.size() &&
               std::isspace(static_cast<unsigned char>(body[p]))) {
            ++p;
        }
        constexpr std::string_view kCall = "call:";
        if (body.substr(p, kCall.size()) == kCall) {
            p += kCall.size();
            const std::size_t nameStart = p;
            while (p < body.size() && body[p] != '{') {
                ++p;
            }
            const std::string name =
                trimws(body.substr(nameStart, p - nameStart));
            if (!name.empty() && p < body.size() && body[p] == '{') {
                GemmaValParser vp{body, p};
                try {
                    const json args = vp.object();
                    ToolCall call;
                    call.id            = "call_" + std::to_string(calls.size());
                    call.name          = name;
                    call.argumentsJson = args.dump();
                    calls.push_back(std::move(call));
                } catch (...) {
                    // malformed args → skip this block
                }
            }
        }

        if (close == std::string_view::npos) {
            break;
        }
        pos = close + kGClose.size();
    }

    return calls;
}

bool ToolCallParser::looksLikeBareJsonToolCall(
        std::string_view text,
        const std::vector<std::string>& knownNames) noexcept {
    if (text.find("\"name\"") == std::string_view::npos) {
        return false;
    }
    for (const std::string& n : knownNames) {
        if (!n.empty() && text.find(n) != std::string_view::npos) {
            return true;
        }
    }
    return false;
}

std::vector<ToolCall> ToolCallParser::parseBareJson(
        std::string_view text, const std::vector<std::string>& knownNames) {
    std::vector<ToolCall> calls;

    const auto isKnown = [&knownNames](const std::string& name) {
        for (const std::string& n : knownNames) {
            if (n == name) return true;
        }
        return false;
    };

    // Walk every `{`, find its string-aware matching `}`, and try that span as
    // a tool-call object. Accept only objects whose `name` is an offered tool —
    // that guard is what keeps a genuine JSON answer from being parsed as a
    // call. Continue after each object so several concatenated calls all parse.
    std::size_t i = 0;
    while (i < text.size()) {
        const std::size_t open = text.find('{', i);
        if (open == std::string_view::npos) {
            break;
        }

        std::size_t depth = 0;
        bool        inStr = false;
        bool        esc   = false;
        std::size_t closeIdx = std::string_view::npos;
        for (std::size_t j = open; j < text.size(); ++j) {
            const char c = text[j];
            if (inStr) {
                if (esc)            esc = false;
                else if (c == '\\') esc = true;
                else if (c == '"')  inStr = false;
            } else if (c == '"') {
                inStr = true;
            } else if (c == '{') {
                ++depth;
            } else if (c == '}') {
                if (--depth == 0) { closeIdx = j; break; }
            }
        }
        if (closeIdx == std::string_view::npos) {
            break;   // unbalanced from here on — stop
        }

        const std::string_view obj = text.substr(open, closeIdx - open + 1);
        json parsed = json::parse(obj, /*cb=*/nullptr, /*allow_exceptions=*/false);
        if (!parsed.is_discarded() && parsed.is_object()) {
            const auto nameIt = parsed.find("name");
            if (nameIt != parsed.end() && nameIt->is_string() &&
                isKnown(nameIt->get<std::string>())) {
                ToolCall call;
                call.id   = "call_" + std::to_string(calls.size());
                call.name = nameIt->get<std::string>();
                const auto argsIt = parsed.find("arguments");
                call.argumentsJson =
                    (argsIt != parsed.end()) ? argumentsToString(*argsIt) : "{}";
                calls.push_back(std::move(call));
            }
        }

        i = closeIdx + 1;
    }

    return calls;
}

} // namespace mimirmind::model
