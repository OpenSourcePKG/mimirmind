// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "model/ChatTemplate.hpp"             // Style, ToolFormat
#include "model/ResponseCleaner.hpp"
#include "model/ToolCall.hpp"                 // ToolSpec
#include "model/ToolCallStreamDetector.hpp"
#include "runtime/InferenceEngine.hpp"        // GenerateParams
#include "server/ChatRequestParser.hpp"       // ResponseFormat

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mimirmind::model {
class Tokenizer;
}

namespace mimirmind::server {

class IToolCallParser;   // 8.25.6 — resolved tool-call parser (registry)

/// Per-stream state for ChatCompletionHandler::handleStream — the whole
/// SSE-chat state machine in one shared object so the async content-provider
/// (which may run on a different pool thread and outlives the synchronous part
/// of handleStream) has a stable home. Extracted verbatim from the former
/// inline `StreamState` (roadmap 8.30.11.2); pure data + a small ctor that
/// seeds the token-level response cleaner.
struct ChatStreamState {
    std::vector<std::int32_t>     promptIds;
    std::vector<std::int32_t>     stopIds;
    runtime::GenerateParams       params;
    std::string                   respId;
    std::int64_t                  created{};
    std::string                   echoModel;
    // Owning tenant, snapshotted on the request thread: the chunked
    // content-provider that submits to the batcher may be invoked on a
    // different pool thread where the thread_local ScopedTenant no longer
    // holds this request's label.
    std::string                   tenantId;
    // Buffers a trailing incomplete UTF-8 codepoint between tokens
    // so SSE deltas always carry valid UTF-8.
    std::string                   utf8Pending;
    // Same UTF-8 hold buffer, for the reasoning_content stream.
    std::string                   reasoningPending;
    // Per-stream filter that swallows the Gemma 4
    // <|channel>thought<channel|> wrapper at the token level,
    // matching the behaviour ChatTemplate::cleanResponse applies in
    // the non-streaming path. No-op for other chat styles.
    model::ResponseCleaner        cleaner;
    // M-FunctionCalling: mirrors the blocking path's marker-dispatch, but
    // token-by-token. toolCallsEnabled gates it off entirely for requests
    // without tools (or tool_choice:"none") so plain-text streaming is
    // byte-for-byte unchanged. style picks which parser
    // (parseQwen/parseGemma) a completed block goes through.
    bool                          toolCallsEnabled{false};
    model::ChatTemplate::Style    style{model::ChatTemplate::Style::QwenChatML};
    // 8.19.6: which Qwen dialect a completed block is parsed as first
    // (Hermes JSON vs Qwen3-Coder XML), plus the offered tool schemas the
    // XML parser needs for parameter-type coercion.
    model::ChatTemplate::ToolFormat toolFormat{
        model::ChatTemplate::ToolFormat::HermesJson};
    std::vector<model::ToolSpec>  toolSpecs;
    // 8.25.6 — the server-resolved tool-call parser for this model (registry,
    // keyed by config.serve.json tool_call_parser + arch auto-detect). nullptr
    // for styles without a registered parser -> the stream block path keeps the
    // legacy ToolCallExtractor::extractBlock dispatch. Non-owning: the registry
    // owns the process-lifetime const instance.
    const IToolCallParser*        toolParser{nullptr};
    // 8.19.13.3: needed to build the primary-decode grammar constraint with
    // the right root ("required" prefills the opener → body-rooted).
    std::string                   toolChoice;
    // 8.19.13.4: response_format JSON enforcement on the streaming path.
    ResponseFormat                responseFormat{ResponseFormat::Text};
    std::string                   jsonSchema;
    // 8.19.14 part B: emit delta.logprobs chunks on the streaming path.
    bool                          logprobsWanted{false};
    model::ToolCallStreamDetector toolCallDetector;
    int                           nextToolCallIndex{0};
    bool                          anyToolCallEmitted{false};
    // 8.19.10 — held-content mode: when tools are offered and the FIRST
    // visible answer character is '<', content is held server-side until
    // end-of-turn so an unparseable tool-markup response can be salvaged
    // (or suppressed) instead of having leaked to the client chunk by
    // chunk. Plain prose (not starting with '<') streams as before.
    bool                          contentHoldActive{false};
    bool                          firstContentSeen{false};
    std::string                   heldContent;
    bool                          done{false};
    // OpenAI stream_options.include_usage: emit a terminal usage chunk
    // (empty choices + usage) just before [DONE].
    bool                          includeUsage{false};
    // M-Munin.3 (full): keeps a pool-mode target's worker-side slot
    // pinned for the WHOLE async stream, not just the synchronous part
    // of handleStream() that resolves it. Without this the slot could
    // be evicted (and targetEngine/targetMutex/targetSpec below would
    // dangle) as soon as the local `target` in handleStream() goes out
    // of scope, well before the chunked content-provider lambda — which
    // outlives that scope — actually runs. Empty/no-op on the eager
    // path, where engines are process-lifetime resident.
    std::shared_ptr<void>        pin{};

    ChatStreamState(model::ChatTemplate::Style chatStyle,
                    const model::Tokenizer&    tok,
                    bool                       preserveThinking,
                    bool                       thinkPreClosed)
        // A `<think>` token makes forStyle() start the cleaner INSIDE the
        // think block (thinking-on pre-opens <think>). But when thinking is
        // off — now the DEFAULT, or an explicit enable_thinking=false — the
        // chat template pre-CLOSES the block (`<think>\n\n</think>\n\n`) so
        // generation is answer content from the first token — start the
        // cleaner in content mode (channelStartId = -1), else the whole
        // answer is mislabelled as reasoning_content and delta.content
        // streams empty. thinkPreClosed MUST equal !thinkOn in ChatTemplate.
        //
        // thinkPreClosed is a QwenChatML-only concept (whether <think> was
        // pre-closed in the PROMPT) — it says nothing about Gemma4, which
        // emits its <|channel>thought\n<channel|> wrapper unconditionally,
        // thinking on or off. Applying the bypass to every style meant a
        // plain Gemma4 request (enable_thinking unset → thinkPreClosed=true
        // by construction below) never engaged forStyle()'s channel
        // detection at all, leaking the raw wrapper into delta.content.
        // Only preserveThinking (explicit debug passthrough) should bypass
        // Gemma4's cleaner; Qwen keeps the thinkPreClosed shortcut.
        : cleaner{(preserveThinking ||
                   (chatStyle == model::ChatTemplate::Style::QwenChatML && thinkPreClosed))
            ? model::ResponseCleaner{chatStyle, -1, -1}
            : model::ResponseCleaner::forStyle(chatStyle, tok)},
          style{chatStyle} {}
};

} // namespace mimirmind::server
