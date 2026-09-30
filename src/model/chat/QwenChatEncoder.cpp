// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "model/chat/QwenChatEncoder.hpp"

#include "model/Tokenizer.hpp"
#include "model/chat/ChatEncoderCommon.hpp"

#include <nlohmann/json.hpp>

#include <initializer_list>
#include <string>

namespace mimirmind::model::chat {

namespace {

// Qwen ChatML start-of-turn token (encoder-local; the end token kQwenImEnd is
// shared and lives in the header).
constexpr std::string_view kQwenImStart = "<|im_start|>";

// Mirrors the official Qwen2.5 default when the conversation has no
// explicit system message. Kept identical to llama.cpp / HF Jinja
// template so encoded bytes match.
constexpr std::string_view kQwenDefaultSystem =
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";

/// Render one tool call in the Qwen3-Coder XML shape the Qwen3.5/3.6/3.8
/// chat templates use. `argumentsJson` is the OpenAI stringified arguments
/// object; each key becomes a <parameter=…> block whose value is the raw
/// string for string arguments and compact JSON for everything else —
/// mirroring the template's `args_value|string if string else tojson`.
std::string renderQwenXmlCall(const ToolCall& call) {
    std::string out = "<tool_call>\n<function=" + call.name + ">\n";
    const auto args = nlohmann::json::parse(call.argumentsJson,
                                            /*cb=*/nullptr,
                                            /*allow_exceptions=*/false);
    if (!args.is_discarded() && args.is_object()) {
        for (const auto& [key, value] : args.items()) {
            out += "<parameter=" + key + ">\n";
            out += value.is_string() ? value.get<std::string>() : value.dump();
            out += "\n</parameter>\n";
        }
    }
    out += "</function>\n</tool_call>";
    return out;
}

// jinja `tojson`-style serialisation: compact but with ", " / ": " separators
// (nlohmann's dump() has no space option). Used for the <required>/<enum>/...
// extra-key values in the structured tool-def form so the rendered prompt
// tracks the model's chat_template output.
std::string jinjaJson(const nlohmann::json& j) {
    if (j.is_array()) {
        std::string s = "[";
        for (std::size_t i = 0; i < j.size(); ++i) {
            if (i) { s += ", "; }
            s += jinjaJson(j[i]);
        }
        return s + "]";
    }
    if (j.is_object()) {
        std::string s = "{";
        bool first = true;
        for (auto it = j.begin(); it != j.end(); ++it) {
            if (!first) { s += ", "; }
            first = false;
            s += nlohmann::json(it.key()).dump() + ": " + jinjaJson(it.value());
        }
        return s + "}";
    }
    return j.dump();   // scalar/string, properly escaped
}

// modeling_qwen3_next's render_extra_keys: any object key not in `handled`
// is spliced as `\n<key>value</key>` (raw for strings, jinjaJson otherwise).
std::string renderExtraKeys(const nlohmann::json& obj,
                            std::initializer_list<const char*> handled) {
    std::string s;
    if (!obj.is_object()) { return s; }
    for (auto it = obj.begin(); it != obj.end(); ++it) {
        bool isHandled = false;
        for (const char* h : handled) {
            if (it.key() == h) { isHandled = true; break; }
        }
        if (isHandled) { continue; }
        s += "\n<" + it.key() + ">";
        s += it.value().is_string() ? it.value().get<std::string>()
                                    : jinjaJson(it.value());
        s += "</" + it.key() + ">";
    }
    return s;
}

// Render one tool as the STRUCTURED-XML function definition Qwen3-Coder-Next's
// chat template expects (`<function><name>…<parameters><parameter>…<required>`),
// as opposed to qwen3.6's `tool | tojson`. Mirrors the model's Jinja exactly so
// auto-mode tool calling stays in-distribution.
std::string renderQwenXmlStructuredToolDef(const ToolSpec& t) {
    const nlohmann::json root =
        nlohmann::json::parse(t.toolJson, nullptr, /*allow_exceptions=*/false);
    const nlohmann::json fn =
        (!root.is_discarded() && root.contains("function")
         && root["function"].is_object())
            ? root["function"] : nlohmann::json::object();
    const std::string name = fn.value("name", t.name);
    auto trimmed = [](std::string v) {
        const auto b = v.find_first_not_of(" \t\r\n");
        const auto e = v.find_last_not_of(" \t\r\n");
        return b == std::string::npos ? std::string{} : v.substr(b, e - b + 1);
    };
    std::string s = "\n<function>\n<name>" + name + "</name>";
    if (fn.contains("description") && fn["description"].is_string()) {
        s += "\n<description>" + trimmed(fn["description"].get<std::string>())
             + "</description>";
    }
    s += "\n<parameters>";
    const nlohmann::json params =
        (fn.contains("parameters") && fn["parameters"].is_object())
            ? fn["parameters"] : nlohmann::json::object();
    if (params.contains("properties") && params["properties"].is_object()) {
        for (auto it = params["properties"].begin();
             it != params["properties"].end(); ++it) {
            const nlohmann::json& pf = it.value();
            s += "\n<parameter>\n<name>" + it.key() + "</name>";
            if (pf.contains("type") && pf["type"].is_string()) {
                s += "\n<type>" + pf["type"].get<std::string>() + "</type>";
            }
            if (pf.contains("description") && pf["description"].is_string()) {
                s += "\n<description>"
                     + trimmed(pf["description"].get<std::string>())
                     + "</description>";
            }
            s += renderExtraKeys(pf, {"name", "type", "description"});
            s += "\n</parameter>";
        }
    }
    // params-level extra keys (not type/properties) — emits <required>[…]</required>.
    s += renderExtraKeys(params, {"type", "properties"});
    s += "\n</parameters>";
    // function-level extra keys.
    s += renderExtraKeys(fn, {"type", "name", "description", "parameters"});
    s += "\n</function>";
    return s;
}

// Build the Qwen tools system block. Two trained-on dialects (8.19.6):
//   HermesJson (Qwen2.5/Qwen3, !xmlTools): a "# Tools" section listing the
//     tool JSON inside <tools></tools>, calls as <tool_call>{json}</tool_call>.
//   QwenXml (Qwen3.5/3.6/3.8 Coder format, xmlTools): the <tools> block with a
//     fixed call-format tail; `structuredXml` (coder-next) additionally renders
//     each tool as a <function> definition and appends a firm XML-call
//     reinforcement. All wording is copied byte-for-byte from the model's
//     chat_template.jinja — do NOT paraphrase, auto-mode calling is sensitive
//     to the exact prompt. Returns "" when no tools are offered so a plain chat
//     renders byte-identically to the pre-8.19.6 encoder.
std::string buildQwenToolsBlock(std::span<const ToolSpec> tools,
                                bool xmlTools, bool structuredXml) {
    // The fixed call-format tail, identical across both Qwen XML dialects.
    constexpr const char* kQwenXmlTail =
        "\n</tools>\n\nIf you choose to call a function ONLY reply in the "
        "following format with NO suffix:\n\n<tool_call>\n"
        "<function=example_function_name>\n"
        "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
        "<parameter=example_parameter_2>\nThis is the value for the second "
        "parameter\nthat can span\nmultiple lines\n</parameter>\n"
        "</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n"
        "- Function calls MUST follow the specified format: an inner "
        "<function=...></function> block must be nested within "
        "<tool_call></tool_call> XML tags\n"
        "- Required parameters MUST be specified\n"
        "- You may provide optional reasoning for your function call in "
        "natural language BEFORE the function call, but NOT after\n"
        "- If there is no function call available, answer the question "
        "like normal with your current knowledge and do not tell the user "
        "about function calls\n</IMPORTANT>";
    // Coder-Next has a strong Gemma/Gemini `tool_code` prior: given the
    // template's own (soft) format hint it still replies with a
    // ```tool_code / ```python / ```bash / ```json fence containing a
    // Python-style NAME(args) call instead of the <tool_call><function=…>
    // XML the parser needs. A firm directive flips it to the XML shape (the
    // model is proven capable — an explicit system prompt elicits it). Only
    // appended for the structured (coder-next) dialect; qwen3.6 is untouched.
    constexpr const char* kXmlCallReinforce =
        "\n\nCRITICAL OUTPUT FORMAT: When you decide to use a function, you "
        "MUST reply with ONLY the XML block below and nothing else:\n"
        "<tool_call>\n<function=THE_FUNCTION_NAME>\n<parameter=THE_ARG_NAME>\n"
        "the value\n</parameter>\n</function>\n</tool_call>\n"
        "Do NOT write the call as Python, bash, JSON, or a ```tool_code / "
        "```python / ```bash / ```json code fence. Do NOT write it as "
        "name(args). Use ONLY exactly one offered function name. If no "
        "function is needed, answer normally in prose.";

    std::string toolsBlock;
    if (tools.empty()) {
        return toolsBlock;
    }
    if (!xmlTools) {
        toolsBlock =
            "\n\n# Tools\n\nYou may call one or more functions to assist with "
            "the user query.\n\nYou are provided with function signatures "
            "within <tools></tools> XML tags:\n<tools>";
        for (const auto& t : tools) {
            toolsBlock += "\n";
            toolsBlock += t.toolJson;
        }
        toolsBlock +=
            "\n</tools>\n\nFor each function call, return a json object with "
            "function name and arguments within <tool_call></tool_call> XML "
            "tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": "
            "<args-json-object>}\n</tool_call>";
    } else {
        toolsBlock = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
        for (const auto& t : tools) {
            if (structuredXml) {
                toolsBlock += renderQwenXmlStructuredToolDef(t);
            } else {
                toolsBlock += "\n";
                toolsBlock += t.toolJson;
            }
        }
        toolsBlock += kQwenXmlTail;
        if (structuredXml) {
            toolsBlock += kXmlCallReinforce;
        }
    }
    return toolsBlock;
}

// Append the assistant generation prompt (and, for thinking models, the
// pre-opened / pre-closed <think> block) to `ids`.
void appendQwenGenerationPrompt(const Tokenizer&           tok,
                               std::int32_t                imStart,
                               std::optional<bool>         enableThinking,
                               std::optional<bool>         templateUsesThink,
                               std::vector<std::int32_t>&  ids) {
    ids.push_back(imStart);
    encodeText(tok, "assistant\n", ids);
    // Qwen3 "thinking" models (Qwen3 / Qwen3.5 / Qwen3.6 `qwen35moe`)
    // pre-open a <think> block in the default generation prompt — the
    // model is trained to continue *inside* it and close with </think>.
    // Their HF/GGUF chat_template appends '<think>\n' after
    // 'assistant\n' (default) or '<think>\n\n</think>\n\n' when thinking
    // is disabled. Qwen2 / Qwen2.5 have no <think> token and use a plain
    // ChatML generation prompt. Auto-detect by the presence of the
    // <think> special token so both families work without an arch
    // switch. Without this, a thinking model emits a spurious </think>
    // and stops immediately.
    // Inject the <think> block only for models whose template actually
    // uses it. templateUsesThink is authoritative when the checkpoint was
    // probed; nullopt falls back to the token-presence heuristic (GGUF /
    // unprobed models keep their prior behaviour). Coder-Next ships the
    // token but sets this false -> plain `assistant\n`, no OOD block.
    const std::int32_t think = tok.findToken("<think>");
    if (think >= 0 && templateUsesThink.value_or(true)) {
        ids.push_back(think);
        // Explicit enable_thinking (OpenAI chat_template_kwargs, like vLLM)
        // overrides: enable_thinking=true opts INTO reasoning; =false forces
        // the empty pre-closed block (direct answer, no reasoning).
        // Unset (nullopt) => DEFAULT OFF: an interactive chat gets a direct
        // answer, not a multi-thousand-token reasoning trace (which also
        // removes the thinking-loop surface). Reasoning is opt-in per request
        // via enable_thinking=true. Tool rounds were already pre-closed, so
        // this only changes the plain-chat/no-tools default. MUST stay in
        // sync with ChatCompletionHandler's streaming `thinkPreClosed`
        // (= !thinkOn), else the answer is mislabelled as reasoning_content.
        const bool thinkOn = enableThinking.value_or(false);
        if (thinkOn) {
            // pre-OPEN <think>: the model reasons, then closes
            // with </think>. This is the answer path; the reasoning is
            // surfaced as reasoning_content.
            encodeText(tok, "\n", ids);
        } else {
            // Tool round → pre-CLOSE an empty think block (Qwen3's
            // "thinking disabled" prompt shape: <think>\n\n</think>\n\n)
            // so the model emits the tool call directly instead of a
            // multi-thousand-token chain-of-thought. Tool selection does
            // not need deep reasoning, and those think blocks made agentic
            // RAG unusably slow (~4096 tokens / round). The final answer
            // (sent without tools) still reasons.
            encodeText(tok, "\n\n", ids);
            const std::int32_t thinkEnd = tok.findToken("</think>");
            if (thinkEnd >= 0) {
                ids.push_back(thinkEnd);
            } else {
                encodeText(tok, "</think>", ids);
            }
            encodeText(tok, "\n\n", ids);
        }
    }
}

} // namespace

std::vector<std::int32_t> QwenChatEncoder::encode(
        const Tokenizer&              tok,
        std::span<const ChatMessage>  messages,
        bool                          addGenerationPrompt,
        std::span<const ToolSpec>     tools,
        std::optional<bool>           enableThinking,
        ChatTemplate::ToolFormat      toolFormat,
        std::optional<bool>           templateUsesThink,
        std::optional<bool>           toolDefsStructuredXml) {
    const std::int32_t imStart = requireToken(tok, kQwenImStart);
    const std::int32_t imEnd   = requireToken(tok, kQwenImEnd);

    std::vector<std::int32_t> ids;
    ids.reserve(64);  // chat headers + a short message fit comfortably

    const bool hasExplicitSystem =
        !messages.empty() && messages.front().role == ChatRole::System;

    // Qwen2 / Qwen2.5 inject a default system turn ("You are Qwen, created
    // by Alibaba Cloud. …") when the caller provides none — their HF Jinja
    // template does exactly this. Qwen3 / Qwen3.5 / Qwen3.6 (the "thinking"
    // models, `qwen35moe`) do NOT: their template renders no system turn at
    // all unless one is supplied. Prepending the 2.5 default to a 3.x model
    // is an out-of-distribution context it was never trained on and yields
    // off-topic / early-EOS garbage. Discriminate on the <think> special
    // token, exactly as the generation-prompt branch below already does.
    // Qwen3-family detection (drives ONLY the default-system suppression:
    // Qwen2/2.5 inject a default system turn, Qwen3+ do not). Coder-Next is
    // Qwen3-family — it ships <think> in vocab — so the token-presence probe
    // is the right signal here and must stay independent of whether the
    // template actually USES think (that is templateUsesThink, below).
    const bool isThinkingFamily = tok.findToken("<think>") >= 0;

    // M-FunctionCalling: the tool-spec block a model only honours in
    // tool_choice:"auto" when it matches its own chat template verbatim.
    // Two trained-on dialects (8.19.6):
    //   HermesJson — Qwen2.5/Qwen3: block APPENDED to the system content,
    //     calls as <tool_call>{json}</tool_call>.
    //   QwenXml — Qwen3.5/3.6/3.8 (Qwen3-Coder format): block PREFIXES the
    //     system turn (system content follows after a blank line), calls in
    //     the <function=…>/<parameter=…> XML shape. Wording below is copied
    //     byte-for-byte from the model's chat_template.jinja — do not
    //     paraphrase it, auto-mode calling is sensitive to the exact prompt.
    // Gated on tools actually being offered so a plain chat renders
    // byte-identically to the pre-8.19.6 encoder regardless of format.
    const bool xmlTools =
        (toolFormat == ChatTemplate::ToolFormat::QwenXml) && !tools.empty();
    // Qwen3-Coder-Next renders tool DEFINITIONS as structured XML
    // (<function><name>…<parameters><parameter>…<required>) and opens the
    // system turn with a "You are Qwen…" preamble, whereas qwen3.6 uses
    // `tool | tojson` and no preamble. Both share the CALL format and the
    // "If you choose…"/<IMPORTANT> tail. Detected per-model from the checkpoint
    // template (see LlmConfig::toolDefsStructuredXml); nullopt = the qwen3.6
    // form, so unprobed/GGUF models are byte-identical to before.
    const bool structuredXml = xmlTools && toolDefsStructuredXml.value_or(false);
    // Qwen3-Coder-Next default system preamble when the caller sends none.
    constexpr const char* kQwenAgentPreamble =
        "You are Qwen, a helpful AI assistant that can interact with a "
        "computer to solve tasks.";

    const std::string toolsBlock =
        buildQwenToolsBlock(tools, xmlTools, structuredXml);

    auto emitTurn = [&](std::string_view role, std::string_view content) {
        ids.push_back(imStart);
        std::string head{role};
        head.push_back('\n');
        head.append(content);
        encodeText(tok, head, ids);
        ids.push_back(imEnd);
        encodeText(tok, "\n", ids);
    };

    // System turn. Without an explicit system: Qwen2.5 gets its default, a
    // thinking model gets none — but if tools are present they still need a
    // home, so emit a (possibly bare) system turn carrying the tools block.
    // QwenXml placement is inverted: the tools block OPENS the system turn
    // and an explicit system message follows after a blank line, exactly as
    // the Qwen3.5/3.6 chat template renders it.
    if (xmlTools && !toolsBlock.empty() && structuredXml) {
        // Qwen3-Coder-Next: system content (or the default agent preamble) comes
        // FIRST, then a blank line, then the tools block — the inverse of the
        // qwen3.6 ordering below.
        std::string sys =
            (hasExplicitSystem && !messages.front().content.empty())
                ? messages.front().content
                : std::string{kQwenAgentPreamble};
        sys += "\n\n";
        sys += toolsBlock;
        emitTurn("system", sys);
    } else if (xmlTools && !toolsBlock.empty()) {
        std::string sys = toolsBlock;
        if (hasExplicitSystem && !messages.front().content.empty()) {
            sys += "\n\n";
            sys += messages.front().content;
        }
        emitTurn("system", sys);
    } else if (!hasExplicitSystem) {
        std::string sys =
            isThinkingFamily ? std::string{} : std::string{kQwenDefaultSystem};
        sys += toolsBlock;
        if (!sys.empty()) {
            emitTurn("system", sys);
        }
    }

    // When the caller DID supply a system turn, the Hermes tools block is
    // appended to it on its first occurrence (QwenXml consumed it above).
    bool explicitSystemToolsPending =
        hasExplicitSystem && !toolsBlock.empty() && !xmlTools;

    // Qwen3.5/3.6 templates render an EMPTY pre-closed think block onto every
    // assistant turn AFTER the last real user query (their last_query_index
    // logic) — the tool-round assistant turn is in-distribution only with it.
    // messages.size() sentinel = "no user turn" → no prefix anywhere.
    std::size_t lastUserIdx = messages.size();
    if (xmlTools) {
        for (std::size_t i = messages.size(); i-- > 0;) {
            if (messages[i].role == ChatRole::User) {
                lastUserIdx = i;
                break;
            }
        }
    }

    for (std::size_t mi = 0; mi < messages.size(); ++mi) {
        const auto& m = messages[mi];
        if (xmlTools && !toolsBlock.empty() && mi == 0 &&
            m.role == ChatRole::System) {
            continue;  // folded into the tools system turn above
        }
        if (explicitSystemToolsPending && m.role == ChatRole::System) {
            emitTurn("system", m.content + toolsBlock);
            explicitSystemToolsPending = false;
            continue;
        }
        if (xmlTools && m.role == ChatRole::Assistant) {
            std::string body =
                (lastUserIdx < messages.size() && mi > lastUserIdx)
                    ? "<think>\n\n</think>\n\n" : "";
            body += m.content;
            for (const auto& call : m.toolCalls) {
                if (&call == &m.toolCalls.front()) {
                    if (!m.content.empty()) {
                        body += "\n\n";
                    }
                } else {
                    body += "\n";
                }
                body += renderQwenXmlCall(call);
            }
            emitTurn("assistant", body);
            continue;
        }
        if (xmlTools && m.role == ChatRole::Tool) {
            // Consecutive tool results merge into ONE user turn:
            // <|im_start|>user\n<tool_response>\nA\n</tool_response>\n
            // <tool_response>\nB\n</tool_response><|im_end|>.
            std::string body;
            std::size_t j = mi;
            for (; j < messages.size() && messages[j].role == ChatRole::Tool; ++j) {
                if (!body.empty()) {
                    body += "\n";
                }
                body += "<tool_response>\n" + messages[j].content +
                        "\n</tool_response>";
            }
            emitTurn("user", body);
            mi = j - 1;
            continue;
        }
        // Assistant turn that invoked tools: render each call as a
        // <tool_call>{...}</tool_call> block after any visible content.
        if (m.role == ChatRole::Assistant && !m.toolCalls.empty()) {
            std::string body = m.content;
            for (const auto& call : m.toolCalls) {
                if (!body.empty()) {
                    body += "\n";
                }
                body += "<tool_call>\n{\"name\": \"";
                body += call.name;
                body += "\", \"arguments\": ";
                body += call.argumentsJson;
                body += "}\n</tool_call>";
            }
            emitTurn("assistant", body);
            continue;
        }
        // Tool result: Qwen2.5 carries it in a user turn wrapped in
        // <tool_response></tool_response>.
        if (m.role == ChatRole::Tool) {
            emitTurn("user",
                     "<tool_response>\n" + m.content + "\n</tool_response>");
            continue;
        }
        emitTurn(chatRoleName(m.role), m.content);
    }

    if (addGenerationPrompt) {
        appendQwenGenerationPrompt(tok, imStart, enableThinking,
                                   templateUsesThink, ids);
    }

    return ids;
}

} // namespace mimirmind::model::chat
