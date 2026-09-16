// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "model/ToolCallConstraint.hpp"

#include "model/Tokenizer.hpp"
#include "core/log/Log.hpp"

#include <nlohmann/json.hpp>
#include <xgrammar/xgrammar.h>
#include <dlpack/dlpack.h>

#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace mimirmind::model {

using nlohmann::json;

namespace {

// Ordered schema-property names of a tool from its OpenAI tool JSON, each
// tagged with whether the schema lists it as required and carrying the full
// per-property JSON subschema — the subschema is compiled into a per-value
// grammar rule (like vLLM compiling the tool JSON Schema), so `type`, `enum`
// and `maxLength` all constrain the value "for free" from what the schema
// declares, instead of a hardcoded per-arch value rule.
struct SchemaParams {
    std::vector<std::string> keys;      // schema-property order
    std::vector<bool>        required;  // parallel to keys
    std::vector<json>        schema;    // parallel to keys; the property subschema
};

SchemaParams schemaParams(const std::string& toolJson) {
    SchemaParams sp;
    const json t = json::parse(toolJson, nullptr, false);
    if (t.is_discarded() || !t.contains("function")
        || !t["function"].is_object()
        || !t["function"].contains("parameters")
        || !t["function"]["parameters"].is_object()
        || !t["function"]["parameters"].contains("properties")
        || !t["function"]["parameters"]["properties"].is_object()) {
        return sp;
    }
    const json& params = t["function"]["parameters"];
    std::set<std::string> req;
    if (params.contains("required") && params["required"].is_array()) {
        for (const auto& r : params["required"]) {
            if (r.is_string()) { req.insert(r.get<std::string>()); }
        }
    }
    const json& props = params["properties"];
    for (auto it = props.begin(); it != props.end(); ++it) {
        sp.keys.push_back(it.key());
        sp.required.push_back(req.count(it.key()) > 0);
        sp.schema.push_back(it.value());
    }
    return sp;
}

// A name/key emitted verbatim into an EBNF double-quoted literal. Tool names and
// JSON keys are identifiers in practice; escape the two chars that would break a
// literal defensively.
std::string ebnfLiteralSafe(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; }
        o += c;
    }
    return o;
}

// Per-(tokenizer) xgrammar preprocessing: the TokenizerInfo + a GrammarCompiler
// bound to it. Building these scans the whole vocab, so cache and share.
struct TokContext {
    std::shared_ptr<xgrammar::TokenizerInfo>  info;
    std::shared_ptr<xgrammar::GrammarCompiler> compiler;
    int                                       vocab{0};
    // Compiled-grammar cache keyed by the EBNF string — CompileGrammar
    // preprocesses against the whole (~150k) vocab, so it must run once per
    // tool-set, not once per request.
    std::mutex                                 compileMtx;
    std::map<std::string,
             std::shared_ptr<xgrammar::CompiledGrammar>> compiledCache;
};

std::shared_ptr<TokContext> tokContext(const Tokenizer& tok) {
    static std::mutex mtx;
    static std::map<const Tokenizer*, std::shared_ptr<TokContext>> cache;
    std::lock_guard<std::mutex> lk{mtx};
    const auto it = cache.find(&tok);
    if (it != cache.end()) { return it->second; }

    auto ctx = std::make_shared<TokContext>();
    const int vocab = static_cast<int>(tok.vocabSize());
    std::vector<std::string> encoded(static_cast<std::size_t>(vocab));
    for (int i = 0; i < vocab; ++i) {
        const std::int32_t id = i;
        // Proper decoded text of each single token (byte-level -> utf8); with
        // VocabType::RAW xgrammar uses it verbatim, so the ASCII tool-call
        // markers/names/keys reconstruct exactly regardless of the tokenizer's
        // byte-level storage form.
        encoded[static_cast<std::size_t>(i)] =
            tok.decode(std::span<const std::int32_t>(&id, 1),
                       /*skipSpecial=*/false);
    }
    std::vector<std::int32_t> stops;
    if (tok.eosId() >= 0) { stops.push_back(tok.eosId()); }
    ctx->info = std::make_shared<xgrammar::TokenizerInfo>(
        encoded, xgrammar::VocabType::RAW, vocab,
        stops.empty() ? std::nullopt
                      : std::optional<std::vector<std::int32_t>>{stops});
    ctx->compiler = std::make_shared<xgrammar::GrammarCompiler>(*ctx->info);
    // Authoritative size for the bitmask: xgrammar may pad the vocab, and
    // FillNextTokenBitmask writes (GetVocabSize()+31)/32 int32 words — sizing
    // the buffer by the raw tokenizer vocab would let it write out of range.
    ctx->vocab = ctx->info->GetVocabSize();
    cache.emplace(&tok, ctx);
    return ctx;
}

} // namespace

struct ToolCallConstraint::Impl {
    std::shared_ptr<TokContext>                   tctx;
    std::shared_ptr<xgrammar::CompiledGrammar>    compiled;
    std::unique_ptr<xgrammar::GrammarMatcher>     matcher;   // non-null while in a call
    std::vector<std::int32_t>                     bitmask;   // (vocab+31)/32 ints
    int                                           vocab{0};
    bool                                          active{false};
};

ToolCallConstraint::ToolCallConstraint(std::span<const ToolSpec> tools,
                                       const Tokenizer& tok,
                                       bool assumeOpenerConsumed)
    : _impl(std::make_unique<Impl>()) {
    if (tools.empty()) { _impl.reset(); return; }
    _impl->tctx  = tokContext(tok);
    _impl->vocab = _impl->tctx->vocab;

    // EBNF: free prose that dispatches to a grammar-forced call body on the
    // `<tool_call>` trigger (xgrammar's TagDispatch = token-robust AC automaton,
    // so it fires whether the opener is one token or several). The call body has
    // one alternative per offered tool with that tool's keys, so NAME and KEY
    // are both forced; VALUE is any text (the parser reads it); prose outside a
    // call stays free (loop_after_dispatch). This is exactly how a constrained
    // decoder handles prose-XOR-call.
    // This checkpoint emits the tool call in the BARE `<function=NAME>…
    // </function>\n</tool_call>` form (no leading `<tool_call>`), so the trigger
    // is `<function=` and the dispatched body begins with the NAME. Each body
    // alternative is one offered tool with that tool's keys -> NAME and KEY are
    // both grammar-forced; VALUE is any text.
    std::string ebnf;
    // Value = a run of chars up to the closer's '<', that must LEAD with a
    // non-whitespace character. Crucially the parameter terminator starts with
    // '<' (`</parameter>`), NOT with '\n', so the value ends DETERMINISTICALLY
    // at the first '<' — an ambiguous "\n</parameter>" terminator lets the value
    // also eat the '\n', which explodes the Earley parser state (and hangs
    // CompileGrammar). Values that legitimately contain '<' just end the
    // parameter early, which the downstream parser tolerates.
    //
    // 5.31 — a plain `[^<]+` forbids only the ZERO-length value; it still ACCEPTS
    // a whitespace-only value (`"\n"`, `"\n\n"`, spaces), because [^<] matches
    // every char except '<' — including \t\r\n and space. On a minimal-context
    // forced tool call the 3B-active model's cheapest way to satisfy the mask is
    // exactly that whitespace filler (raw logits peak on the newline that closes
    // the empty parameter), so it shipped dead calls like `{"query":"\n"}`
    // ~40-60% of the time even through the grammar-masked salvage re-decode
    // (confirmed live via cuda-gdb, chat-trace 2bd53443). Requiring a leading
    // NON-whitespace char removes that escape hatch grammatically at any
    // temperature: the model must commit a real character, and once it does it
    // pulls content from context (e.g. the author name). Value = one leading
    // "content" char then any run of non-'<'.
    //
    // Expressing "non-whitespace" is constrained by xgrammar's char-class lexer:
    // it does NOT accept `\t`/`\n`/`\r` escapes (only `^ $ \ . * + ? ( ) [ ] { }
    // | / -`), rejects a literal newline in a class, and its PARSER rejects the
    // regex escapes `\s`/`\S`/`\d`/`\w` ("escape not supported yet in EBNF"). So
    // whitespace cannot be named at all. Instead the leading char is a POSITIVE
    // class of printable-ASCII content via range endpoints only: [!-;] = 0x21..
    // 0x3B and [=-~] = 0x3D..0x7E — every printable ASCII char EXCEPT space
    // (0x20), the C0 controls (incl. \t \n \r), and '<' (0x3C). The REST stays
    // `[^<]*` (any non-'<', incl. spaces/UTF-8), so only the FIRST char is
    // constrained. Limitation: a value whose very first char is non-ASCII (e.g.
    // "Über…") is disallowed — acceptable for web-search queries, which lead with
    // ASCII; the tail is unrestricted.
    ebnf += "value ::= [!-;=-~] [^<]*\n";
    // Shared base rules for the primitive JSON types — an integer must be digits
    // only, a number a plain decimal, a boolean true/false, so a forced tool call
    // can't degenerate e.g. `count` into a 150-digit string or a text ramble.
    ebnf += "value_int ::= \"-\"? [0-9]+\n";
    ebnf += "value_num ::= \"-\"? [0-9]+ (\".\" [0-9]+)?\n";
    ebnf += "value_bool ::= \"true\" | \"false\"\n";
    std::string bodyAlts;
    // Compile a property's JSON subschema into the grammar rule its value must
    // follow — the vLLM "feed the schema, the type comes for free" idea, adapted
    // to the RAW (unquoted) Qwen-XML parameter value (vLLM emits JSON, so it can
    // hand the whole schema to xgrammar's JSON compiler; our wire value carries
    // no JSON quotes, so we read the schema keywords ourselves and emit EBNF):
    //   enum            -> the exact literal set (a value that cannot ramble);
    //   integer/number/boolean -> the typed base rule;
    //   string+maxLength -> a bounded content run ([^<]{0,N-1});
    //   everything else -> permissive `value`.
    // (maxLength/pattern/numeric-range are NOT expressible via xgrammar's own JSON
    // Schema path — a documented gap — so even vLLM leans on this only when the
    // schema declares them; enum + type are the reliable wins.)
    // Emits a dedicated rule (named `id`) into `rules` when the schema needs one
    // and returns the rule name to reference; otherwise returns a shared rule.
    const auto valueRuleFromSchema =
        [](const json& s, const std::string& id, std::string& rules) -> std::string {
        if (!s.is_object()) { return "value"; }
        if (s.contains("enum") && s["enum"].is_array() && !s["enum"].empty()) {
            std::string alts;
            for (const auto& e : s["enum"]) {
                std::string lit;
                if (e.is_string())              { lit = e.get<std::string>(); }
                else if (e.is_number_integer()) { lit = std::to_string(e.get<std::int64_t>()); }
                else if (e.is_boolean())        { lit = e.get<bool>() ? "true" : "false"; }
                else { continue; }   // float/object/array enum member: no literal
                if (!alts.empty()) { alts += " | "; }
                alts += "\"" + ebnfLiteralSafe(lit) + "\"";
            }
            if (!alts.empty()) { rules += id + " ::= " + alts + "\n"; return id; }
        }
        const std::string type =
            (s.contains("type") && s["type"].is_string())
                ? s["type"].get<std::string>() : std::string{};
        if (type == "integer") { return "value_int"; }
        if (type == "number")  { return "value_num"; }
        if (type == "boolean") { return "value_bool"; }
        if (type == "string" && s.contains("maxLength")
            && s["maxLength"].is_number_integer()) {
            const std::int64_t n = s["maxLength"].get<std::int64_t>();
            if (n >= 1 && n <= 8192) {
                rules += id + " ::= [!-;=-~] [^<]{0," + std::to_string(n - 1) + "}\n";
                return id;
            }
        }
        return "value";   // string (unbounded) / array / object / unknown
    };
    for (std::size_t i = 0; i < tools.size(); ++i) {
        const std::string nm = ebnfLiteralSafe(tools[i].name);
        const std::string pr = "params_" + std::to_string(i);
        if (!bodyAlts.empty()) { bodyAlts += " | "; }
        bodyAlts += "\"" + nm + ">\\n\" " + pr
                  + " \"</function>\\n</tool_call>\"";

        const SchemaParams sp = schemaParams(tools[i].toolJson);
        // Compile each property's subschema into its value rule ONCE (a key can
        // be referenced by both the optional and the required block below).
        std::vector<std::string> keyRule(sp.keys.size());
        for (std::size_t j = 0; j < sp.keys.size(); ++j) {
            keyRule[j] = valueRuleFromSchema(
                sp.schema[j],
                "pv_" + std::to_string(i) + "_" + std::to_string(j), ebnf);
        }
        // A parameter block for key j, using its schema-compiled value rule.
        const auto keyBlock = [&](std::size_t j) {
            return "\"<parameter=" + ebnfLiteralSafe(sp.keys[j]) + ">\\n\" "
                 + keyRule[j] + " \"</parameter>\\n\"";
        };

        // opt_i = any run of the OPTIONAL keys (each omittable, any order).
        const std::string optRule = "opt_" + std::to_string(i);
        std::string optAlts;
        for (std::size_t j = 0; j < sp.keys.size(); ++j) {
            if (sp.required[j]) { continue; }
            if (!optAlts.empty()) { optAlts += " | "; }
            optAlts += keyBlock(j);
        }
        if (optAlts.empty()) {
            ebnf += optRule + " ::= \"\"\n";
        } else {
            ebnf += optRule + " ::= (" + optAlts + ")*\n";
        }

        // params_i forces EVERY required key (in schema order) to appear before
        // the arguments object can close, with optionals free to interleave:
        //   opt_i  req1  opt_i  req2  opt_i ...  reqN  opt_i
        // No required keys -> params_i is just opt_i (may be empty).
        std::string paramsBody = optRule;
        for (std::size_t j = 0; j < sp.keys.size(); ++j) {
            if (!sp.required[j]) { continue; }
            paramsBody += " " + keyBlock(j) + " " + optRule;
        }
        ebnf += pr + " ::= " + paramsBody + "\n";
    }
    ebnf += "call_body ::= " + bodyAlts + "\n";
    if (assumeOpenerConsumed) {
        // Forced-opener re-decode: `<tool_call>\n<function=` was prefilled into
        // the prompt (so it never reaches this matcher, which only sees GENERATED
        // tokens). Root at the body directly -> the NAME is forced from the very
        // first generated token, whatever dialect the free decode had drifted to.
        ebnf += "root ::= call_body\n";
    } else {
        // Auto mode: free prose until a call opens ITSELF with the `<function=`
        // trigger (token-robust AC automaton); prose outside a call stays free.
        ebnf += "root ::= TagDispatch((\"<function=\", call_body), "
                "loop_after_dispatch=true)\n";
    }

    try {
        auto& tc = *_impl->tctx;
        std::lock_guard<std::mutex> lk{tc.compileMtx};
        const auto cit = tc.compiledCache.find(ebnf);
        if (cit != tc.compiledCache.end()) {
            _impl->compiled = cit->second;   // reuse across requests
        } else {
            const xgrammar::Grammar g =
                xgrammar::Grammar::FromEBNF(ebnf, "root");
            _impl->compiled = std::make_shared<xgrammar::CompiledGrammar>(
                tc.compiler->CompileGrammar(g));
            tc.compiledCache.emplace(ebnf, _impl->compiled);
            MM_LOG_INFO("tgram",
                        "compiled tool-call grammar ({} tools, {} cached)",
                        tools.size(), tc.compiledCache.size());
        }
    } catch (const std::exception& e) {
        MM_LOG_WARN("tgram", "tool-call grammar compile failed: {}", e.what());
        _impl.reset();
        return;
    }
    _impl->bitmask.assign(static_cast<std::size_t>((_impl->vocab + 31) / 32), 0);
    // One matcher for the whole response: TagDispatch keeps prose free and only
    // constrains a call body once the trigger fires.
    _impl->matcher =
        std::make_unique<xgrammar::GrammarMatcher>(*_impl->compiled);
    _impl->active = true;
}

std::shared_ptr<ToolCallConstraint>
ToolCallConstraint::forResponseFormatJson(const Tokenizer& tok,
                                          std::string_view jsonSchema) {
    auto c      = std::shared_ptr<ToolCallConstraint>(new ToolCallConstraint());
    c->_impl    = std::make_unique<Impl>();
    auto& impl  = *c->_impl;
    impl.tctx   = tokContext(tok);
    impl.vocab  = impl.tctx->vocab;

    // Cache key: distinct from any tool EBNF (which starts with "value ::=").
    const std::string key = "\x01json\x01" + std::string{jsonSchema};
    try {
        auto& tc = *impl.tctx;
        std::lock_guard<std::mutex> lk{tc.compileMtx};
        const auto cit = tc.compiledCache.find(key);
        if (cit != tc.compiledCache.end()) {
            impl.compiled = cit->second;
        } else {
            // Rooted at token 0 (no TagDispatch): the WHOLE output must be a
            // single JSON value. Empty schema => any JSON (json_object);
            // otherwise the exact JSON-Schema (json_schema).
            const xgrammar::Grammar g =
                jsonSchema.empty()
                    ? xgrammar::Grammar::BuiltinJSONGrammar()
                    : xgrammar::Grammar::FromJSONSchema(std::string{jsonSchema});
            impl.compiled = std::make_shared<xgrammar::CompiledGrammar>(
                tc.compiler->CompileGrammar(g));
            tc.compiledCache.emplace(key, impl.compiled);
            MM_LOG_INFO("tgram", "compiled response_format JSON grammar ({})",
                        jsonSchema.empty() ? "any" : "schema");
        }
    } catch (const std::exception& e) {
        MM_LOG_WARN("tgram", "response_format JSON grammar compile failed: {}",
                    e.what());
        return nullptr;
    }
    impl.bitmask.assign(static_cast<std::size_t>((impl.vocab + 31) / 32), 0);
    impl.matcher =
        std::make_unique<xgrammar::GrammarMatcher>(*impl.compiled);
    impl.active = true;
    return c;
}

ToolCallConstraint::~ToolCallConstraint() = default;
ToolCallConstraint::ToolCallConstraint(ToolCallConstraint&&) noexcept = default;
ToolCallConstraint&
ToolCallConstraint::operator=(ToolCallConstraint&&) noexcept = default;

bool ToolCallConstraint::active() const noexcept {
    return _impl && _impl->active;
}

void ToolCallConstraint::reset() noexcept {
    if (!_impl || !_impl->matcher) { return; }
    try {
        _impl->matcher->Reset();
    } catch (...) {
        // Reset is noexcept-intended; swallow to keep the sampler alive.
    }
}

void ToolCallConstraint::advance(std::int32_t token) {
    if (!_impl || !_impl->active || !_impl->matcher) { return; }
    // The lm-head vocab can be padded past the tokenizer vocab the matcher was
    // built with; such padding ids never appear in a valid call (masked out
    // inside a call body) and are not in xgrammar's tables, so skip them rather
    // than index out of range. Feed every in-range token: TagDispatch keeps
    // prose free and only constrains inside a dispatched call body.
    if (token < 0 || token >= _impl->vocab) { return; }
    // Once the grammar has accepted a stop token the matcher is TERMINATED;
    // driving it further asserts inside xgrammar (grammar_matcher.cc:1648
    // "IsStopTokenAccepted"). After termination the tool call / JSON is complete
    // and the model is free — stop feeding the matcher. try/catch so a matcher
    // fault degrades to "unconstrained" instead of 500-ing the request.
    try {
        if (_impl->matcher->IsTerminated()) { return; }
        _impl->matcher->AcceptToken(token);
    } catch (...) {
        // keep the sampler alive; grammar faults must never crash the request
    }
}

void ToolCallConstraint::maskLogits(float* logits, std::size_t vocab) const {
    if (!_impl || !_impl->active || !_impl->matcher) { return; }
    // After the grammar accepts a stop token the matcher is TERMINATED;
    // FillNextTokenBitmask then asserts (grammar_matcher.cc:1648) and 500s the
    // request. This is the exact bug behind bad tool-mode answers: the tool-call
    // grammar stayed engaged into the free-text final answer and crashed at EOS.
    // Once terminated (call/JSON complete) the model is free — no mask. The whole
    // body is wrapped so ANY xgrammar fault degrades to "unconstrained", never 500.
    try {
        if (_impl->matcher->IsTerminated()) { return; }
    } catch (...) {
        return;
    }
    try {

    // ApplyTokenBitmaskInplaceCPU reads shape[0]/strides[0] as a batch row: it
    // dereferences strides[0] UNCONDITIONALLY (no compact/nullptr fast path), so
    // both tensors MUST carry an explicit strides array — a nullptr strides
    // (valid DLPack for a contiguous tensor) segfaults inside xgrammar.
    DLTensor bm{};
    std::int64_t bmShape[1] = {
        static_cast<std::int64_t>(_impl->bitmask.size())};
    std::int64_t bmStride[1] = {
        static_cast<std::int64_t>(_impl->bitmask.size())};
    bm.data        = _impl->bitmask.data();
    bm.device      = DLDevice{kDLCPU, 0};
    bm.ndim        = 1;
    bm.dtype       = DLDataType{kDLInt, 32, 1};
    bm.shape       = bmShape;
    bm.strides     = bmStride;
    bm.byte_offset = 0;

    if (!_impl->matcher->FillNextTokenBitmask(&bm)) {
        return;   // all-allowed this step
    }

    // The bitmask is sized to the tokenizer vocab; bound the logits view to the
    // smaller of (logits width, tokenizer vocab) so neither is indexed out of
    // range when the lm-head vocab is padded past the tokenizer vocab.
    const int n = std::min(static_cast<int>(vocab), _impl->vocab);
    DLTensor lt{};
    std::int64_t ltShape[1]  = {static_cast<std::int64_t>(n)};
    std::int64_t ltStride[1] = {static_cast<std::int64_t>(n)};
    lt.data        = logits;
    lt.device      = DLDevice{kDLCPU, 0};
    lt.ndim        = 1;
    lt.dtype       = DLDataType{kDLFloat, 32, 1};
    lt.shape       = ltShape;
    lt.strides     = ltStride;
    lt.byte_offset = 0;

    xgrammar::ApplyTokenBitmaskInplaceCPU(&lt, bm, n);
    } catch (...) {
        // grammar/xgrammar fault -> leave logits untouched (unconstrained) rather
        // than crash the request.
    }
}

} // namespace mimirmind::model
