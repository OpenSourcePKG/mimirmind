// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "model/ChatTemplate.hpp"

#include <string>
#include <string_view>

namespace mimirmind::model {

/**
 * Streaming counterpart to ToolCallParser for the SSE chat-completion path.
 *
 * The blocking path decodes the whole generated span at once and hands it to
 * ToolCallParser::parseQwen/parseGemma in one shot. The SSE path emits text
 * token by token and must decide, as each token arrives, whether it is plain
 * answer content or part of a model's native tool-call marker span
 * (`<tool_call>...</tool_call>` for Qwen, `<|tool_call>...<tool_call|>` for
 * Gemma 4) — this class is that decision, mirroring ResponseCleaner's
 * per-token state-machine shape for the (unrelated) thinking/channel markup.
 *
 * Both families' markers are plain literal text once decoded (skipSpecial
 * only strips BOS/EOS/PAD, not other vocab entries), so detection is a
 * straightforward substring scan over an accumulating buffer — the same
 * technique ResponseCleaner::feedQwenThink uses for `<think>`/`</think>`.
 * Gemma 4's closing marker is additionally a configured stop id (see
 * ChatTemplate::toolCallStopIds) that halts decoding before its text ever
 * reaches feed() — closeOnStop() covers that path.
 *
 * Construct once per chat turn — state is reset by construction.
 */
class ToolCallStreamDetector {
public:
    /// Build a detector for `style`. `enabled` gates the whole thing (pass
    /// false when the request carries no tools, or tool_choice:"none" — feed()
    /// then never buffers, matching plain-text streaming exactly as before
    /// this class existed). `forcedOpener` is the tool_choice:"required" text
    /// that was prefilled onto the PROMPT (empty for "auto"/"none", and always
    /// empty for Gemma 4 — see ChatTemplate::toolCallOpenerText): since that
    /// text lives in the prompt it never reaches feed(), so when non-empty the
    /// detector starts already inside a call instead of waiting for a fresh
    /// opener marker that will never come.
    ToolCallStreamDetector(ChatTemplate::Style style, bool enabled,
                            std::string forcedOpener = {});

    /// Cheap inactive default — assign a real instance over this once the
    /// per-request style/gate is known.
    ToolCallStreamDetector() noexcept = default;

    /// Feed one token's already-cleaned visible answer text (call only when
    /// the caller's own content filtering — e.g. ResponseCleaner — says this
    /// token IS answer content, not reasoning/channel markup). May clear or
    /// shrink `text` in place: content before an opener, or when disabled, is
    /// left untouched; a marker-delimited span is removed from `text`
    /// entirely and held internally instead.
    ///
    /// Returns true exactly when this call completed a full tool-call block —
    /// completedBlock() then holds the opener..closer span, ready for
    /// ToolCallParser::parseQwen/parseGemma. The caller must call reset()
    /// afterwards before feeding further tokens (there may be more plain text
    /// or another call later in the same turn).
    [[nodiscard]] bool feed(std::string& text);

    /// Gemma 4's closing marker is a configured stop id — the engine's
    /// isStop() check intercepts it before its text ever reaches feed().
    /// Call this from that branch instead, only when buffering() is true, to
    /// finalize the block. Also the right call for a plain EOS/stop hit
    /// mid-buffer (e.g. Qwen truncated at max_tokens without a closer) —
    /// finalizes with whatever was buffered so far, matching
    /// ToolCallParser::parseQwen's tolerance for an unterminated final block.
    void closeOnStop() noexcept;

    /// True while inside a marker-delimited span that hasn't closed yet.
    [[nodiscard]] bool buffering() const noexcept { return _state == State::Buffering; }

    /// Valid once feed() returned true or closeOnStop() was called while
    /// buffering() — the buffered opener..closer span.
    [[nodiscard]] const std::string& completedBlock() const noexcept { return _buffer; }

    /// Reset to pass-through after handling a completed block, so later
    /// tokens in the same turn are neither buffered nor treated as a new
    /// forced-opener seed.
    void reset() noexcept;

    /// Drain text that was held back purely as a defensive tail against a
    /// marker split across tokens (see feed()) and turned out never to be
    /// part of an opener — call once at end-of-stream so it is not silently
    /// dropped. Empty in the common no-tool-call-in-flight case.
    [[nodiscard]] std::string takePendingFlush();

private:
    enum class State { PassThrough, Buffering, Completed };

    std::string_view _open;
    std::string_view _close;
    bool             _active{false};
    State            _state{State::PassThrough};
    // Held-back tail while scanning for an opener (PassThrough) — may end in
    // a partial opener prefix that the next token's text completes.
    std::string      _pending;
    // Accumulated opener..closer span once buffering has started.
    std::string      _buffer;
    // Text found after the closer within the same feed() call (closers are
    // effectively always their own token in practice, so this is a rare
    // defensive case) — folded into `_pending` by reset() so the next feed()
    // (or takePendingFlush() at end-of-stream) surfaces it as normal content.
    std::string      _trailingAfterClose;
};

} // namespace mimirmind::model
