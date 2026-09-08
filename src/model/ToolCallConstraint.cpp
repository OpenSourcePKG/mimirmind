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
// tagged with whether the schema lists it as required.
struct SchemaParams {
    std::vector<std::string> keys;      // schema-property order
    std::vector<bool>        required;  // parallel to keys
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
                                       const Tokenizer& tok)
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
    // Value = a NON-EMPTY run of chars up to the closer's '<'. Crucially the
    // parameter terminator starts with '<' (`</parameter>`), NOT with '\n', so
    // [^<]+ ends DETERMINISTICALLY at the first '<' — an ambiguous
    // "\n</parameter>" terminator lets the value also eat the '\n', which
    // explodes the Earley parser state (and hangs CompileGrammar). Values that
    // legitimately contain '<' just end the parameter early, which the
    // downstream parser tolerates. `+` (not `*`) forbids an empty value, so a
    // required key never collapses to `{"file_path":""}`.
    ebnf += "value ::= [^<]+\n";
    std::string bodyAlts;
    for (std::size_t i = 0; i < tools.size(); ++i) {
        const std::string nm = ebnfLiteralSafe(tools[i].name);
        const std::string pr = "params_" + std::to_string(i);
        if (!bodyAlts.empty()) { bodyAlts += " | "; }
        bodyAlts += "\"" + nm + ">\\n\" " + pr
                  + " \"</function>\\n</tool_call>\"";

        // A parameter block for key k.
        const auto keyBlock = [](const std::string& k) {
            return "\"<parameter=" + ebnfLiteralSafe(k)
                 + ">\\n\" value \"</parameter>\\n\"";
        };
        const SchemaParams sp = schemaParams(tools[i].toolJson);

        // opt_i = any run of the OPTIONAL keys (each omittable, any order).
        const std::string optRule = "opt_" + std::to_string(i);
        std::string optAlts;
        for (std::size_t j = 0; j < sp.keys.size(); ++j) {
            if (sp.required[j]) { continue; }
            if (!optAlts.empty()) { optAlts += " | "; }
            optAlts += keyBlock(sp.keys[j]);
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
            paramsBody += " " + keyBlock(sp.keys[j]) + " " + optRule;
        }
        ebnf += pr + " ::= " + paramsBody + "\n";
    }
    ebnf += "call_body ::= " + bodyAlts + "\n";
    ebnf += "root ::= TagDispatch((\"<function=\", call_body), "
            "loop_after_dispatch=true)\n";

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
    _impl->matcher->AcceptToken(token);
}

void ToolCallConstraint::maskLogits(float* logits, std::size_t vocab) const {
    if (!_impl || !_impl->active || !_impl->matcher) { return; }

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
}

} // namespace mimirmind::model
