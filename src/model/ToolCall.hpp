// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <string>

namespace mimirmind::model {

/// A function tool the client offered in the request's `tools` array.
///   `name`     — the function name (for logging / validation).
///   `toolJson` — the full OpenAI tool object serialised verbatim,
///                `{"type":"function","function":{"name":…,"description":…,
///                "parameters":{…}}}`. Kept as text so the chat template
///                splices it into the model's `<tools>` block without any
///                re-escaping (the request parser already validated it).
struct ToolSpec {
    std::string name;
    std::string toolJson{"{}"};
};

/// A tool call the model emitted, normalised to the OpenAI shape so the
/// server can return it under `choices[].message.tool_calls[]`.
///
///   `id`            — server-synthesised correlation id (e.g. "call_0");
///                     most local models don't emit one, but OpenAI clients
///                     need it to match a later role:"tool" result.
///   `name`          — the function name.
///   `argumentsJson` — the arguments as a JSON *string* (OpenAI stringifies
///                     them), e.g. `{"query":"Rechnung"}`.
struct ToolCall {
    std::string id;
    std::string name;
    std::string argumentsJson{"{}"};
};

} // namespace mimirmind::model
