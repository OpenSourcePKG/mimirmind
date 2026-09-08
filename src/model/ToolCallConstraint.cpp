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
#include <string>
#include <vector>

namespace mimirmind::model {

using nlohmann::json;

namespace {

// Ordered schema-property names of a tool from its OpenAI tool JSON.
std::vector<std::string> schemaKeys(const std::string& toolJson) {
    std::vector<std::string> keys;
    const json t = json::parse(toolJson, nullptr, false);
    if (t.is_discarded() || !t.contains("function")
        || !t["function"].is_object()
        || !t["function"].contains("parameters")
        || !t["function"]["parameters"].is_object()
        || !t["function"]["parameters"].contains("properties")
        || !t["function"]["parameters"]["properties"].is_object()) {
        return keys;
    }
    const json& props = t["function"]["parameters"]["properties"];
    for (auto it = props.begin(); it != props.end(); ++it) {
        keys.push_back(it.key());
    }
    return keys;
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
    std::int32_t                              triggerToken{-1};
    int                                       vocab{0};
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
    ctx->triggerToken = tok.findToken("<tool_call>");
    ctx->vocab = vocab;
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
    std::int32_t                                  trigger{-1};
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
    // Value = any run of chars up to the closer's '<'. Bounding the class (vs
    // full unicode) keeps the compiled grammar + per-step mask cheap; values
    // that legitimately contain '<' just end the parameter early, which the
    // downstream parser tolerates.
    ebnf += "value ::= [^<]*\n";
    std::string bodyAlts;
    for (std::size_t i = 0; i < tools.size(); ++i) {
        const std::string nm = ebnfLiteralSafe(tools[i].name);
        const std::string pr = "params_" + std::to_string(i);
        if (!bodyAlts.empty()) { bodyAlts += " | "; }
        bodyAlts += "\"" + nm + ">\\n\" " + pr
                  + " \"</function>\\n</tool_call>\"";
        const auto keys = schemaKeys(tools[i].toolJson);
        std::string paramAlts;
        for (const std::string& k : keys) {
            if (!paramAlts.empty()) { paramAlts += " | "; }
            paramAlts += "\"<parameter=" + ebnfLiteralSafe(k)
                       + ">\\n\" value \"\\n</parameter>\\n\"";
        }
        if (paramAlts.empty()) {
            ebnf += pr + " ::= \"\"\n";
        } else {
            ebnf += pr + " ::= (" + paramAlts + ")*\n";
        }
    }
    ebnf += "call_body ::= " + bodyAlts + "\n";
    ebnf += "root ::= TagDispatch((\"<function=\", call_body), "
            "loop_after_dispatch=true)\n";

    try {
        const xgrammar::Grammar g = xgrammar::Grammar::FromEBNF(ebnf, "root");
        _impl->compiled = std::make_shared<xgrammar::CompiledGrammar>(
            _impl->tctx->compiler->CompileGrammar(g));
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
    // Feed every token; TagDispatch keeps prose free and only constrains inside
    // a dispatched call body. A rejected token (should not happen when the mask
    // is applied) is ignored so the sampler never stalls.
    _impl->matcher->AcceptToken(token);
}

void ToolCallConstraint::maskLogits(float* logits, std::size_t vocab) const {
    if (!_impl || !_impl->active || !_impl->matcher) { return; }

    DLTensor bm{};
    std::int64_t bmShape[1] = {
        static_cast<std::int64_t>(_impl->bitmask.size())};
    bm.data        = _impl->bitmask.data();
    bm.device      = DLDevice{kDLCPU, 0};
    bm.ndim        = 1;
    bm.dtype       = DLDataType{kDLInt, 32, 1};
    bm.shape       = bmShape;
    bm.strides     = nullptr;
    bm.byte_offset = 0;

    if (!_impl->matcher->FillNextTokenBitmask(&bm)) {
        return;   // all-allowed this step
    }

    // The bitmask is sized to the tokenizer vocab; bound the logits view to the
    // smaller of (logits width, tokenizer vocab) so neither is indexed out of
    // range when the lm-head vocab is padded past the tokenizer vocab.
    const int n = std::min(static_cast<int>(vocab), _impl->vocab);
    DLTensor lt{};
    std::int64_t ltShape[1] = {static_cast<std::int64_t>(n)};
    lt.data        = logits;
    lt.device      = DLDevice{kDLCPU, 0};
    lt.ndim        = 1;
    lt.dtype       = DLDataType{kDLFloat, 32, 1};
    lt.shape       = ltShape;
    lt.strides     = nullptr;
    lt.byte_offset = 0;

    xgrammar::ApplyTokenBitmaskInplaceCPU(&lt, bm, n);
}

} // namespace mimirmind::model
