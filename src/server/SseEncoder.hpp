#pragma once

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace mimirmind::server {

/// Server-Sent-Events chunk builders and writers for
/// `/v1/chat/completions` in streaming mode.
///
/// Free functions in a class-scoped namespace — kept as a struct
/// with static methods so future stateful extensions (e.g. per-client
/// backpressure) have a natural home without touching call sites.
struct SseEncoder {
    /// Empty skeleton with the id/created/model triple shared by every
    /// chunk on the stream.
    [[nodiscard]] static nlohmann::json streamChunkSkeleton(
        const std::string& id,
        std::int64_t       created,
        const std::string& model);

    /// Initial delta chunk with `role: "assistant"`.
    [[nodiscard]] static nlohmann::json buildRoleChunk(
        const std::string& id,
        std::int64_t       created,
        const std::string& model);

    /// Delta chunk carrying a piece of assistant text.
    [[nodiscard]] static nlohmann::json buildContentChunk(
        const std::string& id,
        std::int64_t       created,
        const std::string& model,
        std::string_view   text);

    /// Terminal chunk with `finish_reason`.
    [[nodiscard]] static nlohmann::json buildFinishChunk(
        const std::string& id,
        std::int64_t       created,
        const std::string& model,
        std::string_view   finishReason);

    /// Format `payload` as one SSE event and push it onto `sink`.
    /// Returns false if the sink refused the write — caller should
    /// treat that as "client disconnected".
    [[nodiscard]] static bool writeSseEvent(httplib::DataSink& sink,
                                            const nlohmann::json& payload);

    /// Named SSE event: `event: <name>\ndata: <json>\n\n`. OpenAI-style
    /// stream consumers ignore named events, so this is a safe place
    /// to publish mimirmind-specific side-channel signals (e.g.
    /// `prefill_done`, `prefill_progress`).
    [[nodiscard]] static bool writeSseNamedEvent(httplib::DataSink& sink,
                                                  std::string_view    name,
                                                  const nlohmann::json& payload);

    /// Final `data: [DONE]\n\n` sentinel.
    [[nodiscard]] static bool writeSseDone(httplib::DataSink& sink);

    /// Index of the first byte of an incomplete UTF-8 codepoint at the
    /// end of `s`, or `s.size()` if the buffer ends on a codepoint
    /// boundary. Used to split a token-stream into "safe-to-emit"
    /// prefix + "buffer for next token" suffix — GPT-2 BPE can put a
    /// multi-byte UTF-8 char across two tokens, and SSE consumers
    /// choke on partial codepoints.
    [[nodiscard]] static std::size_t utf8IncompleteTailStart(const std::string& s);
};

} // namespace mimirmind::server