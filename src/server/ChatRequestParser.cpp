#include "server/ChatRequestParser.hpp"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace mimirmind::server {

using nlohmann::json;

ChatRequest parseChatRequest(const json& body) {
    ChatRequest req;

    if (!body.is_object()) {
        throw std::runtime_error("request body must be a JSON object");
    }

    if (body.contains("model") && body["model"].is_string()) {
        req.model = body["model"].get<std::string>();
    }

    if (!body.contains("messages") || !body["messages"].is_array()) {
        throw std::runtime_error("messages: missing or not an array");
    }
    for (const auto& m : body["messages"]) {
        if (!m.is_object() || !m.contains("role") || !m.contains("content")) {
            throw std::runtime_error(
                "messages[]: each entry needs role + content");
        }
        const auto roleStr = m["role"].get<std::string>();
        model::ChatRole role;
        if (!model::parseChatRole(roleStr, role)) {
            throw std::runtime_error(
                "messages[].role: unsupported value '" + roleStr + "'");
        }
        std::string content;
        if (m["content"].is_string()) {
            content = m["content"].get<std::string>();
        } else if (m["content"].is_null()) {
            content = "";
        } else {
            // OpenAI also accepts content arrays (multimodal). Not supported.
            throw std::runtime_error(
                "messages[].content: only plain strings are supported");
        }
        req.messages.push_back({role, std::move(content)});
    }

    auto readSize = [&](const char* key, std::size_t& dst) {
        if (body.contains(key) && body[key].is_number_integer()) {
            const auto v = body[key].get<std::int64_t>();
            if (v > 0) {
                dst = static_cast<std::size_t>(v);
            }
        }
    };
    auto readFloat = [&](const char* key, float& dst, bool& has) {
        if (body.contains(key) && body[key].is_number()) {
            dst = body[key].get<float>();
            has = true;
        }
    };

    // OpenAI: max_completion_tokens (current) overrides max_tokens (legacy).
    readSize("max_tokens", req.maxTokens);
    readSize("max_completion_tokens", req.maxTokens);
    readSize("top_k", req.topK);

    bool hasTopP = false;
    readFloat("temperature", req.temperature, req.hasTemperature);
    readFloat("top_p", req.topP, hasTopP);
    (void)hasTopP;

    readFloat("frequency_penalty",  req.frequencyPenalty,  req.hasFrequencyPenalty);
    readFloat("presence_penalty",   req.presencePenalty,   req.hasPresencePenalty);
    readFloat("repetition_penalty", req.repetitionPenalty, req.hasRepetitionPenalty);

    if (body.contains("seed") && body["seed"].is_number_integer()) {
        req.seed = body["seed"].get<std::uint64_t>();
    }

    if (body.contains("stream") && body["stream"].is_boolean()) {
        req.stream = body["stream"].get<bool>();
    }

    if (body.contains("stop")) {
        const auto& s = body["stop"];
        if (s.is_string()) {
            req.stopStrings.push_back(s.get<std::string>());
        } else if (s.is_array()) {
            for (const auto& e : s) {
                if (e.is_string()) {
                    req.stopStrings.push_back(e.get<std::string>());
                }
            }
        }
    }

    return req;
}

} // namespace mimirmind::server