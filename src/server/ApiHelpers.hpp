#pragma once

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <cstdint>
#include <string>
#include <string_view>

namespace mimirmind::server {

/// Small helpers shared by every endpoint handler in `src/server/`.

/// Random `chatcmpl-<24 alnum>` id for chat/completions responses.
[[nodiscard]] std::string makeRequestId();

/// Seconds since Unix epoch as int64 — the OpenAI shape for
/// `created` in chat.completion and chat.completion.chunk.
[[nodiscard]] std::int64_t unixNow();

/// `res.status = status; res.set_content(body.dump(), "application/json");`
void sendJson(httplib::Response& res, int status, const nlohmann::json& body);

/// Emit an OpenAI-shaped `{error: {message, type, code}}` envelope.
void sendError(httplib::Response& res, int status,
                std::string_view type, std::string_view message);

} // namespace mimirmind::server