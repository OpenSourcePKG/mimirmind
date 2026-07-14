// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "model/ChatTemplate.hpp"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mimirmind::server {

/// Parsed OpenAI-style chat-completions request body.
///
/// The `has*` flags for penalties distinguish "client sent 0" from
/// "client sent nothing"; the latter picks up the server-side default
/// while the former stays at exactly 0.
struct ChatRequest {
    std::vector<model::ChatMessage> messages;
    std::size_t                     maxTokens{0};   // 0 => use server default
    float                           temperature{0.0F};
    bool                            hasTemperature{false};
    float                           topP{1.0F};
    std::size_t                     topK{0};
    std::uint64_t                   seed{0};
    std::vector<std::string>        stopStrings;
    bool                            stream{false};
    std::string                     model;

    // M7f — repetition-control penalties.
    float                           frequencyPenalty{0.0F};
    bool                            hasFrequencyPenalty{false};
    float                           presencePenalty{0.0F};
    bool                            hasPresencePenalty{false};
    float                           repetitionPenalty{1.0F};
    bool                            hasRepetitionPenalty{false};
};

/// Parse an OpenAI chat-completions request body. Throws std::runtime_error
/// on malformed input; the caller maps that to a 400 response.
[[nodiscard]] ChatRequest parseChatRequest(const nlohmann::json& body);

} // namespace mimirmind::server