// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "model/ToolCallParser.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <utility>
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

/// Repair pass for a trivially-truncated Hermes JSON body (a model dropping
/// the final `}` before `</tool_call>` is a seen-in-the-wild Qwen3.6 failure):
/// count braces outside strings and append the missing closers. Returns the
/// discarded sentinel when the body is not salvageable this way.
json parseWithBraceRepair(std::string_view body) {
    json parsed = json::parse(body, /*cb=*/nullptr, /*allow_exceptions=*/false);
    if (!parsed.is_discarded()) {
        return parsed;
    }
    std::int32_t depth = 0;
    bool         inStr = false;
    bool         esc   = false;
    for (char c : body) {
        if (inStr) {
            if (esc)            esc = false;
            else if (c == '\\') esc = true;
            else if (c == '"')  inStr = false;
        } else if (c == '"') {
            inStr = true;
        } else if (c == '{') {
            ++depth;
        } else if (c == '}') {
            --depth;
        }
    }
    if (inStr || depth <= 0 || depth > 4) {
        return json(json::value_t::discarded);
    }
    std::string repaired = trimws(body);
    repaired.append(static_cast<std::size_t>(depth), '}');
    return json::parse(repaired, /*cb=*/nullptr, /*allow_exceptions=*/false);
}

/// Coerce one raw XML parameter value to the JSON type the tool's schema
/// declares for it (vLLM qwen3coder-parser behaviour). Unknown or "string"
/// type keeps the raw text; a failed conversion falls back to the raw text
/// rather than dropping the call.
json coerceXmlParamValue(const std::string& raw, const std::string& type) {
    if (type.empty() || type == "string") {
        return raw;
    }
    const std::string t = trimws(raw);
    try {
        if (type == "integer") {
            return static_cast<std::int64_t>(std::stoll(t));
        }
        if (type == "number") {
            return std::stod(t);
        }
    } catch (...) {
        return raw;
    }
    if (type == "boolean") {
        if (t == "true")  return true;
        if (t == "false") return false;
        return raw;
    }
    // object / array / null / anything else: accept valid JSON verbatim.
    json v = json::parse(t, /*cb=*/nullptr, /*allow_exceptions=*/false);
    return v.is_discarded() ? json(raw) : v;
}

/// Normalise the `arguments` field to a compact JSON *string*: OpenAI carries
/// arguments stringified. Qwen usually emits an object; some models emit an
/// already-stringified value. Anything else falls back to "{}".
std::string argumentsToString(const json& args) {
    if (args.is_string()) {
        return args.get<std::string>();
    }
    if (args.is_object() || args.is_array()) {
        // Model-derived bytes may hold an invalid UTF-8 sequence
        // (token-boundary truncation) — dump() must replace, not
        // throw (a throw here escaped the SSE provider, 8.19.10).
        return args.dump(-1, ' ', false,
                         json::error_handler_t::replace);
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

        // Parse the inner JSON (with a brace-balance repair retry for a
        // truncated closer); skip this block on any error rather than
        // aborting the whole response.
        json parsed = parseWithBraceRepair(body);
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

namespace {

constexpr std::string_view kFunc     = "<function=";
constexpr std::string_view kFuncEnd  = "</function>";
constexpr std::string_view kParam    = "<parameter=";
constexpr std::string_view kParamEnd = "</parameter>";

/// Per-tool parameter-type lookup from the offered schemas:
/// function.parameters.properties.<key>.type. Missing/unparseable schemas
/// simply leave every value a string.
std::string paramTypeFor(std::span<const ToolSpec> specs,
                         const std::string&        fn,
                         const std::string&        key) {
    for (const auto& spec : specs) {
        if (spec.name != fn) {
            continue;
        }
        const json tool = json::parse(spec.toolJson, /*cb=*/nullptr,
                                      /*allow_exceptions=*/false);
        if (tool.is_discarded()) {
            return {};
        }
        const json* props = nullptr;
        if (tool.contains("function") &&
            tool["function"].contains("parameters") &&
            tool["function"]["parameters"].contains("properties")) {
            props = &tool["function"]["parameters"]["properties"];
        }
        if (props != nullptr && props->contains(key) &&
            (*props)[key].contains("type") &&
            (*props)[key]["type"].is_string()) {
            return (*props)[key]["type"].get<std::string>();
        }
        return {};
    }
    return {};
}

/// The template writes values as `<parameter=k>\nVALUE\n</parameter>` —
/// strip exactly one framing newline from each side, preserving interior
/// (and any additional intentional) whitespace of a multi-line value.
std::string stripFrame(std::string_view v) {
    if (!v.empty() && v.front() == '\n') {
        v.remove_prefix(1);
    }
    if (!v.empty() && v.back() == '\n') {
        v.remove_suffix(1);
    }
    return std::string{v};
}

/// Parse one `<function=NAME>…</function>` span whose "<function=" starts at
/// `fn` within `scope`. On success fills `out` (id is left to the caller)
/// and sets `endPos` just past the consumed span (scope end when the closer
/// is missing — truncation tolerance). Returns false when no parseable
/// function name is present.
bool parseFunctionSpan(std::string_view          scope,
                       std::size_t               fn,
                       std::span<const ToolSpec> specs,
                       ToolCall&                 out,
                       std::size_t&              endPos) {
    const std::size_t nameStart = fn + kFunc.size();
    const std::size_t nameEnd   = scope.find('>', nameStart);
    const std::string name = (nameEnd != std::string_view::npos)
        ? trimws(scope.substr(nameStart, nameEnd - nameStart))
        : std::string{};
    if (name.empty()) {
        return false;
    }
    json args = json::object();
    std::size_t p = nameEnd + 1;
    const std::size_t funcEnd = scope.find(kFuncEnd, p);
    const std::size_t paramsEnd =
        (funcEnd != std::string_view::npos) ? funcEnd : scope.size();
    while (true) {
        const std::size_t ps = scope.find(kParam, p);
        if (ps == std::string_view::npos || ps >= paramsEnd) {
            break;
        }
        const std::size_t ks = ps + kParam.size();
        const std::size_t ke = scope.find('>', ks);
        if (ke == std::string_view::npos) {
            break;
        }
        const std::string key = trimws(scope.substr(ks, ke - ks));
        const std::size_t vs  = ke + 1;
        std::size_t ve = scope.find(kParamEnd, vs);
        // Missing closer (truncation): value runs to the next parameter
        // opener / function closer / end of scope.
        std::size_t next = ve;
        if (ve == std::string_view::npos || ve > paramsEnd) {
            const std::size_t vNext = scope.find(kParam, vs);
            ve   = std::min(paramsEnd,
                            (vNext == std::string_view::npos) ? paramsEnd
                                                              : vNext);
            next = ve;
        } else {
            next = ve + kParamEnd.size();
        }
        if (!key.empty()) {
            args[key] = coerceXmlParamValue(
                stripFrame(scope.substr(vs, ve - vs)),
                paramTypeFor(specs, name, key));
        }
        p = next;
    }
    out.name          = name;
    out.argumentsJson = args.dump(-1, ' ', false,
                                  json::error_handler_t::replace);
    endPos = (funcEnd != std::string_view::npos) ? funcEnd + kFuncEnd.size()
                                                 : scope.size();
    return true;
}

/// Byte ranges of ```fenced``` markdown code blocks — a bare <function=…>
/// inside one is documentation, not a call. Conservative: an unclosed final
/// fence extends to the end of the text.
std::vector<std::pair<std::size_t, std::size_t>>
fencedRanges(std::string_view text) {
    std::vector<std::pair<std::size_t, std::size_t>> ranges;
    std::size_t pos = 0;
    while (true) {
        const std::size_t open = text.find("```", pos);
        if (open == std::string_view::npos) {
            break;
        }
        const std::size_t close = text.find("```", open + 3);
        const std::size_t end   = (close == std::string_view::npos)
            ? text.size() : close + 3;
        ranges.emplace_back(open, end);
        if (close == std::string_view::npos) {
            break;
        }
        pos = end;
    }
    return ranges;
}

bool insideRange(
        const std::vector<std::pair<std::size_t, std::size_t>>& ranges,
        std::size_t pos) {
    for (const auto& [a, b] : ranges) {
        if (pos >= a && pos < b) {
            return true;
        }
    }
    return false;
}

} // namespace

std::vector<ToolCall> ToolCallParser::parseQwenXml(
        std::string_view text, std::span<const ToolSpec> specs) {
    std::vector<ToolCall> calls;

    std::size_t pos = 0;
    while (true) {
        const std::size_t open = text.find(kOpen, pos);
        if (open == std::string_view::npos) {
            break;
        }
        const std::size_t bodyStart = open + kOpen.size();
        const std::size_t close     = text.find(kClose, bodyStart);
        // Truncated final block (cut at max_tokens): parse what is there.
        const std::size_t bodyEnd =
            (close == std::string_view::npos) ? text.size() : close;
        const std::string_view body = text.substr(bodyStart, bodyEnd - bodyStart);

        // <function=NAME> header. A body without one (e.g. a Hermes JSON
        // block) is not this dialect — skip it.
        const std::size_t fn = body.find(kFunc);
        if (fn != std::string_view::npos) {
            ToolCall    call;
            std::size_t spanEnd = 0;
            if (parseFunctionSpan(body, fn, specs, call, spanEnd)) {
                call.id = "call_" + std::to_string(calls.size());
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

bool ToolCallParser::looksLikeBareQwenXmlCall(
        std::string_view text, std::span<const ToolSpec> specs) noexcept {
    std::size_t fn = text.find(kFunc);
    while (fn != std::string_view::npos) {
        const std::string_view after = text.substr(fn + kFunc.size());
        for (const auto& spec : specs) {
            if (!spec.name.empty() &&
                after.substr(0, spec.name.size()) == spec.name) {
                return true;
            }
        }
        fn = text.find(kFunc, fn + kFunc.size());
    }
    return false;
}

namespace {

/// Ordered parameter names of a tool (JSON schema `properties`, insertion
/// order), for mapping positional Python-call args.
std::vector<std::string> orderedParams(std::span<const ToolSpec> specs,
                                       const std::string& name) {
    for (const auto& s : specs) {
        if (s.name != name) { continue; }
        const json tool = json::parse(s.toolJson, nullptr, false);
        if (!tool.is_discarded() && tool.contains("function")
            && tool["function"].is_object()
            && tool["function"].contains("parameters")
            && tool["function"]["parameters"].is_object()
            && tool["function"]["parameters"].contains("properties")
            && tool["function"]["parameters"]["properties"].is_object()) {
            std::vector<std::string> keys;
            const json& props = tool["function"]["parameters"]["properties"];
            for (auto it = props.begin(); it != props.end(); ++it) {
                keys.push_back(it.key());
            }
            return keys;
        }
    }
    return {};
}

/// Split a call arg-list on top-level commas (respecting quotes + brackets).
std::vector<std::string> splitTopLevel(const std::string& s) {
    std::vector<std::string> out;
    std::size_t start = 0, depth = 0;
    bool inStr = false; char q = 0;
    for (std::size_t i = 0; i < s.size(); ++i) {
        const char c = s[i];
        if (inStr) {
            if (c == '\\') { ++i; continue; }
            if (c == q) { inStr = false; }
            continue;
        }
        if (c == '"' || c == '\'') { inStr = true; q = c; }
        else if (c == '(' || c == '[' || c == '{') { ++depth; }
        else if (c == ')' || c == ']' || c == '}') { if (depth) --depth; }
        else if (c == ',' && depth == 0) {
            out.push_back(s.substr(start, i - start));
            start = i + 1;
        }
    }
    if (start <= s.size()) { out.push_back(s.substr(start)); }
    return out;
}

/// Parse a Python/JSON literal value into JSON (best effort; unknown -> string).
json parsePyValue(const std::string& raw) {
    const std::string v = trimws(raw);
    if (v.size() >= 2 && (v.front() == '"' || v.front() == '\'')
        && v.back() == v.front()) {
        // String literal: unescape the common sequences.
        std::string out;
        for (std::size_t i = 1; i + 1 < v.size(); ++i) {
            if (v[i] == '\\' && i + 2 < v.size()) {
                const char n = v[i + 1];
                out += (n == 'n') ? '\n' : (n == 't') ? '\t'
                     : (n == 'r') ? '\r' : n;
                ++i;
            } else {
                out += v[i];
            }
        }
        return out;
    }
    if (v == "true" || v == "True")   { return true; }
    if (v == "false" || v == "False") { return false; }
    if (v == "None" || v == "null")   { return json(nullptr); }
    // JSON array/object (Python single-quote variants won't always parse — fall
    // back to the raw string on failure).
    if (!v.empty() && (v.front() == '[' || v.front() == '{')) {
        const json j = json::parse(v, nullptr, false);
        if (!j.is_discarded()) { return j; }
    }
    // number?
    try {
        std::size_t used = 0;
        if (v.find('.') != std::string::npos || v.find('e') != std::string::npos
            || v.find('E') != std::string::npos) {
            const double d = std::stod(v, &used);
            if (used == v.size()) { return d; }
        } else {
            const long long n = std::stoll(v, &used);
            if (used == v.size()) { return static_cast<std::int64_t>(n); }
        }
    } catch (...) {
    }
    return v;   // bare word / identifier / anything else -> string
}

} // namespace

std::vector<ToolCall> ToolCallParser::parseToolCodeCall(
        std::string_view text, std::span<const ToolSpec> specs) {
    // Coder-Next's native auto reply is a Gemma/Gemini `tool_code` fence with a
    // Python-style NAME(args) call, e.g.  ```tool_code\nrun_command("ls -la")\n```
    // (also ```python/```bash/```json, or unfenced). Match an OFFERED tool name
    // immediately followed by a balanced `(...)`, map positional args to the
    // schema's parameter order and `key=value` to named, and emit a call. The
    // offered-name gate makes a false positive (prose mentioning a tool) require
    // an actual call syntax `name(...)`.
    const std::string s{text};
    std::vector<ToolCall> calls;
    for (const auto& spec : specs) {
        const std::string& name = spec.name;
        if (name.empty()) { continue; }
        std::size_t p = 0;
        while ((p = s.find(name, p)) != std::string::npos) {
            const std::size_t after = p + name.size();
            // identifier boundary before the name
            if (p > 0 && (std::isalnum(static_cast<unsigned char>(s[p - 1]))
                          || s[p - 1] == '_' || s[p - 1] == '.')) {
                p = after; continue;
            }
            std::size_t q = after;
            while (q < s.size() && (s[q] == ' ' || s[q] == '\t')) { ++q; }
            if (q >= s.size() || s[q] != '(') { p = after; continue; }
            // balanced parens
            std::size_t depth = 0, i = q, end = std::string::npos;
            bool inStr = false; char qc = 0;
            for (; i < s.size(); ++i) {
                const char c = s[i];
                if (inStr) {
                    if (c == '\\') { ++i; continue; }
                    if (c == qc) { inStr = false; }
                    continue;
                }
                if (c == '"' || c == '\'') { inStr = true; qc = c; }
                else if (c == '(') { ++depth; }
                else if (c == ')') { if (--depth == 0) { end = i; break; } }
            }
            if (end == std::string::npos) { p = after; continue; }
            const std::string argstr = s.substr(q + 1, end - q - 1);
            json args = json::object();
            const std::vector<std::string> ordered = orderedParams(specs, name);
            std::size_t posIdx = 0;
            for (const std::string& part : splitTopLevel(argstr)) {
                const std::string a = trimws(part);
                if (a.empty()) { continue; }
                // kwarg detection: leading identifier then '=' (not '==').
                std::size_t eq = std::string::npos;
                {
                    bool istr = false; char c2 = 0;
                    for (std::size_t j = 0; j < a.size(); ++j) {
                        const char c = a[j];
                        if (istr) { if (c == '\\') { ++j; continue; }
                                    if (c == c2) istr = false; continue; }
                        if (c == '"' || c == '\'') { istr = true; c2 = c; }
                        else if (c == '=' && (j + 1 >= a.size() || a[j + 1] != '=')
                                 && (j == 0 || a[j - 1] != '!'
                                     && a[j - 1] != '<' && a[j - 1] != '>')) {
                            eq = j; break;
                        }
                        else if (c == '(' || c == '[' || c == '{') { break; }
                    }
                }
                std::string key, val;
                if (eq != std::string::npos) {
                    key = trimws(a.substr(0, eq));
                    val = a.substr(eq + 1);
                } else {
                    key = (posIdx < ordered.size())
                              ? ordered[posIdx]
                              : ("arg" + std::to_string(posIdx));
                    val = a;
                    ++posIdx;
                }
                if (!key.empty()) { args[key] = parsePyValue(val); }
            }
            ToolCall call;
            call.name = name;
            call.argumentsJson = args.dump(-1, ' ', false,
                                           json::error_handler_t::replace);
            call.id = "call_" + std::to_string(calls.size());
            calls.push_back(std::move(call));
            break;   // one call per offered tool
        }
        if (!calls.empty()) { break; }   // first matched call wins
    }
    return calls;
}

std::vector<ToolCall> ToolCallParser::parseQwenXmlNoisy(
        std::string_view text, std::span<const ToolSpec> specs) {
    // 1. Drop special-token literals the model spuriously emitted mid-call.
    std::string s{text};
    for (const char* sp : {"<|im_start|>", "<|im_end|>", "<|endoftext|>"}) {
        const std::string m{sp};
        std::size_t p = 0;
        while ((p = s.find(m, p)) != std::string::npos) {
            s.erase(p, m.size());
        }
    }
    // 2. Repair a `<` that a stripped special token had replaced, so the XML
    //    markers are well-formed again. Only touch an occurrence that is not
    //    already preceded by '<'.
    const auto repair = [&s](const std::string& marker) {
        std::size_t p = 0;
        while ((p = s.find(marker, p)) != std::string::npos) {
            if (p == 0 || s[p - 1] != '<') {
                s.insert(p, "<");
                p += 1 + marker.size();
            } else {
                p += marker.size();
            }
        }
    };
    repair("function=");
    repair("parameter=");
    // 3. Parse the repaired text, then keep only calls whose name is offered —
    //    the guard against turning arbitrary noise into a call.
    std::vector<ToolCall> calls = parseQwenXml(s, specs);
    std::vector<ToolCall> kept;
    for (auto& c : calls) {
        bool offered = false;
        for (const auto& spec : specs) {
            if (spec.name == c.name) { offered = true; break; }
        }
        if (offered) {
            kept.push_back(std::move(c));
        }
    }
    return kept;
}

std::vector<ToolCall> ToolCallParser::parseQwenXmlBare(
        std::string_view text, std::span<const ToolSpec> specs) {
    std::vector<ToolCall> calls;

    const auto isOffered = [&specs](const std::string& name) {
        for (const auto& spec : specs) {
            if (spec.name == name) {
                return true;
            }
        }
        return false;
    };

    const auto fences = fencedRanges(text);

    std::size_t pos = 0;
    while (true) {
        const std::size_t fn = text.find(kFunc, pos);
        if (fn == std::string_view::npos) {
            break;
        }
        if (insideRange(fences, fn)) {
            pos = fn + kFunc.size();
            continue;
        }
        ToolCall    call;
        std::size_t spanEnd = 0;
        if (parseFunctionSpan(text, fn, specs, call, spanEnd) &&
            isOffered(call.name)) {
            call.id = "call_" + std::to_string(calls.size());
            calls.push_back(std::move(call));
            pos = spanEnd;
        } else {
            pos = fn + kFunc.size();
        }
    }

    return calls;
}

bool ToolCallParser::looksLikeToolMarkupSalad(std::string_view text) noexcept {
    // Keyword-only on purpose: an earlier "short and tag-dominated" fallback
    // rule ate a legitimate 44-byte HTML answer live (`<div><span>…`). Every
    // observed salad carries a tool keyword inside a tag (`<tooltool_0>`,
    // `</invoke>`, `<|/tool_call_start|>`, `<function=…` variants); anything
    // keyword-free is safer surfaced as content than force-re-decoded.
    // "call" alone is deliberately NOT a keyword ("tool_call" is covered by
    // "tool") — onclick="callFoo()"-style attributes would false-positive.
    for (std::size_t i = 0; i < text.size(); ++i) {
        if (text[i] != '<') {
            continue;
        }
        // Tag-like span: `<` … `>` within 64 chars.
        const std::size_t close = text.find('>', i + 1);
        if (close == std::string_view::npos || close - i > 64) {
            continue;
        }
        std::string low;
        low.reserve(close - i);
        for (std::size_t j = i + 1; j < close; ++j) {
            low.push_back(static_cast<char>(
                std::tolower(static_cast<unsigned char>(text[j]))));
        }
        static constexpr std::string_view kWords[] = {
            "tool", "invoke", "function"};
        for (const auto w : kWords) {
            if (low.find(w) != std::string::npos) {
                return true;
            }
        }
    }
    return false;
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
                    call.argumentsJson = args.dump(
                        -1, ' ', false, json::error_handler_t::replace);
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
