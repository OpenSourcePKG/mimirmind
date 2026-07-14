// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "server/SseEncoder.hpp"

#include <string>

namespace mimirmind::server {

using nlohmann::json;

json SseEncoder::streamChunkSkeleton(const std::string& id,
                                     std::int64_t       created,
                                     const std::string& model) {
    return json{
        {"id",      id},
        {"object",  "chat.completion.chunk"},
        {"created", created},
        {"model",   model},
    };
}

json SseEncoder::buildRoleChunk(const std::string& id, std::int64_t created,
                                 const std::string& model) {
    json out = streamChunkSkeleton(id, created, model);
    out["choices"] = json::array({
        json{
            {"index",         0},
            {"delta",         json{{"role", "assistant"}}},
            {"finish_reason", nullptr},
        },
    });
    return out;
}

json SseEncoder::buildContentChunk(const std::string& id, std::int64_t created,
                                    const std::string& model, std::string_view text) {
    json out = streamChunkSkeleton(id, created, model);
    out["choices"] = json::array({
        json{
            {"index",         0},
            {"delta",         json{{"content", std::string{text}}}},
            {"finish_reason", nullptr},
        },
    });
    return out;
}

json SseEncoder::buildFinishChunk(const std::string& id, std::int64_t created,
                                   const std::string& model, std::string_view finishReason) {
    json out = streamChunkSkeleton(id, created, model);
    out["choices"] = json::array({
        json{
            {"index",         0},
            {"delta",         json::object()},
            {"finish_reason", std::string{finishReason}},
        },
    });
    return out;
}

bool SseEncoder::writeSseEvent(httplib::DataSink& sink, const json& payload) {
    const std::string line = "data: " + payload.dump() + "\n\n";
    return sink.write(line.data(), line.size());
}

bool SseEncoder::writeSseNamedEvent(httplib::DataSink&     sink,
                                     std::string_view       name,
                                     const json&            payload) {
    std::string line;
    line.reserve(name.size() + payload.dump().size() + 16);
    line.append("event: ").append(name).append("\n");
    line.append("data: ").append(payload.dump()).append("\n\n");
    return sink.write(line.data(), line.size());
}

bool SseEncoder::writeSseDone(httplib::DataSink& sink) {
    static constexpr std::string_view kDone = "data: [DONE]\n\n";
    return sink.write(kDone.data(), kDone.size());
}

std::size_t SseEncoder::utf8IncompleteTailStart(const std::string& s) {
    if (s.empty()) return 0;

    // Walk back over continuation bytes (10xxxxxx). Max 3 trailing
    // continuations are legal in UTF-8 (4-byte codepoint).
    std::size_t i        = s.size();
    std::size_t contSeen = 0;
    while (i > 0 && (static_cast<unsigned char>(s[i - 1]) & 0xC0) == 0x80
                 && contSeen < 3) {
        --i;
        ++contSeen;
    }
    if (i == 0) {
        // Pathological: buffer is all continuation bytes. Emit and let
        // the receiver deal with it — we can't keep buffering forever.
        return s.size();
    }

    const auto lead = static_cast<unsigned char>(s[i - 1]);
    std::size_t expectedCont;
    if      ((lead & 0x80) == 0)    expectedCont = 0;  // ASCII
    else if ((lead & 0xE0) == 0xC0) expectedCont = 1;
    else if ((lead & 0xF0) == 0xE0) expectedCont = 2;
    else if ((lead & 0xF8) == 0xF0) expectedCont = 3;
    else                            return s.size();   // malformed lead

    if (contSeen >= expectedCont) {
        return s.size();        // last codepoint is complete
    }
    return i - 1;               // last codepoint starts here and is partial
}

} // namespace mimirmind::server