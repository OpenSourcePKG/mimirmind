// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "model/ChatTemplate.hpp"   // ChatMessage, ChatRole
#include "model/ToolCall.hpp"       // ToolSpec

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace minja {
class chat_template;
}

namespace mimirmind::model {
class Tokenizer;
}

namespace mimirmind::model::chat {

/**
 * Data-driven chat-template renderer (roadmap 8.24.3): interprets the model's
 * own embedded Jinja `chat_template` at runtime via the vendored minja engine
 * (third_party/minja), exactly as vLLM `apply_chat_template` and llama.cpp
 * `--jinja` do — instead of hand-transcribing the prompt structure per family
 * in the ChatTemplate encoders. The model template becomes the single source of
 * truth, eliminating the hand-copy-vs-checkpoint drift bug class.
 *
 * This is the RENDER core: it binds the chat context (messages / tools /
 * add_generation_prompt / bos_token / eos_token / chat_template_kwargs) into the
 * Jinja context and returns the prompt STRING. Turning that string into token
 * IDs needs special-token-aware tokenisation (parse-special) — a separate step
 * (8.24.3b) — because the rendered prompt carries special markup (`<|im_start|>`
 * …) as literal text.
 *
 * pimpl: minja.hpp is large and pulls nlohmann/json, so it stays out of this
 * (widely-included) header. Backend-agnostic; lives under src/model/chat/.
 */
class JinjaChatTemplate {
public:
    /// Compile `chatTemplate` (the model's Jinja source) once. `bosToken` /
    /// `eosToken` are bound into the Jinja context (templates reference them);
    /// either may be empty when the model has none. Throws on a malformed
    /// template (minja parse error).
    JinjaChatTemplate(std::string chatTemplate,
                      std::string bosToken,
                      std::string eosToken);
    ~JinjaChatTemplate();
    JinjaChatTemplate(JinjaChatTemplate&&) noexcept;
    JinjaChatTemplate& operator=(JinjaChatTemplate&&) noexcept;
    JinjaChatTemplate(const JinjaChatTemplate&)            = delete;
    JinjaChatTemplate& operator=(const JinjaChatTemplate&) = delete;

    /// Render `messages` (+ optional `tools`) through the template. Mirrors the
    /// OpenAI/HF context: each message → {role, content[, tool_calls][,
    /// tool_call_id]}; each tool → its parsed `toolJson` object; the
    /// `add_generation_prompt` flag; and `enable_thinking` passed through as a
    /// chat_template_kwargs-style extra-context var (like vLLM) when set.
    /// Returns the rendered prompt string. Throws on a Jinja render error.
    [[nodiscard]] std::string render(
        std::span<const ChatMessage> messages,
        std::span<const ToolSpec>    tools,
        bool                         addGenerationPrompt,
        std::optional<bool>          enableThinking = std::nullopt) const;

    /// render() + special-token-aware tokenisation of the result — the rendered
    /// prompt carries special markup (`<|im_start|>` …) as literal text, so it
    /// is tokenised with parseSpecial=true (addBos=false; the template already
    /// emits the bos token where the model wants it). This is the prompt the
    /// decoder continues from.
    [[nodiscard]] std::vector<std::int32_t> encode(
        const Tokenizer&             tok,
        std::span<const ChatMessage> messages,
        std::span<const ToolSpec>    tools,
        bool                         addGenerationPrompt,
        std::optional<bool>          enableThinking = std::nullopt) const;

private:
    std::unique_ptr<minja::chat_template> _tmpl;
};

} // namespace mimirmind::model::chat
