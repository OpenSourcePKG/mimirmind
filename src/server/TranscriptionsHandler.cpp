// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "server/TranscriptionsHandler.hpp"

#include "runtime/audio/AudioEngine.hpp"
#include "server/ApiHelpers.hpp"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace mimirmind::server {

using nlohmann::json;

TranscriptionsHandler::TranscriptionsHandler(
    std::vector<LoadedTranscriber> transcribers, const ServerConfig& cfg)
    : _cfg{cfg} {
    _slots.reserve(transcribers.size());
    for (auto& t : transcribers) {
        Slot s{};
        s.id     = std::move(t.id);
        s.title  = std::move(t.title);
        s.engine = t.engine;
        s.mutex  = std::make_unique<std::mutex>();
        _slots.push_back(std::move(s));
    }
}

std::vector<TranscriptionsHandler::ModelInfo>
TranscriptionsHandler::listModels() const {
    std::vector<ModelInfo> out;
    out.reserve(_slots.size());
    for (const auto& s : _slots) {
        out.push_back({s.id, s.title.empty() ? s.id : s.title});
    }
    return out;
}

TranscriptionsHandler::Slot*
TranscriptionsHandler::resolve(const std::string& model) {
    if (_slots.empty()) {
        return nullptr;
    }
    if (model.empty()) {
        return &_slots.front();
    }
    for (auto& s : _slots) {
        if (s.id == model) {
            return &s;
        }
    }
    return nullptr;
}

void TranscriptionsHandler::handle(const httplib::Request& req,
                                   httplib::Response& res) {
    if (_slots.empty()) {
        sendError(res, 404, "model_not_found",
                  "no transcription model is loaded (configure a model with "
                  "task=transcribe)");
        return;
    }

    // OpenAI /v1/audio/transcriptions is multipart/form-data. httplib parses
    // both file parts and plain text fields into the same file map, so the
    // scalar fields (model/language/response_format) are read the same way.
    if (!req.has_file("file")) {
        sendError(res, 400, "invalid_request_error",
                  "multipart form field 'file' (the audio upload) is required");
        return;
    }
    const auto fileField = req.get_file_value("file");
    if (fileField.content.empty()) {
        sendError(res, 400, "invalid_request_error", "'file' is empty");
        return;
    }

    const std::string model =
        req.has_file("model") ? req.get_file_value("model").content : std::string{};
    Slot* slot = resolve(model);
    if (slot == nullptr) {
        sendError(res, 400, "model_not_found",
                  "unknown transcription model '" + model + "'");
        return;
    }

    std::string language =
        req.has_file("language") ? req.get_file_value("language").content
                                 : std::string{"en"};
    if (language.empty()) {
        language = "en";
    }

    // Supported response formats: "json" (default) and "text". The richer
    // verbose_json (segments/timestamps), srt and vtt are later increments.
    std::string responseFormat =
        req.has_file("response_format")
            ? req.get_file_value("response_format").content
            : std::string{"json"};
    if (responseFormat.empty()) {
        responseFormat = "json";
    }
    if (responseFormat != "json" && responseFormat != "text") {
        sendError(res, 400, "invalid_request_error",
                  "unsupported response_format '" + responseFormat +
                      "' (only \"json\" and \"text\" are supported)");
        return;
    }

    std::string text;
    try {
        const std::lock_guard<std::mutex> lk{*slot->mutex};
        const auto* bytes =
            reinterpret_cast<const std::byte*>(fileField.content.data());
        text = slot->engine->transcribe(
            std::span<const std::byte>{bytes, fileField.content.size()}, language);
    } catch (const std::exception& e) {
        sendError(res, 500, "internal_error",
                  std::string{"transcription failed: "} + e.what());
        return;
    }

    if (responseFormat == "text") {
        res.status = 200;
        res.set_content(text, "text/plain; charset=utf-8");
        return;
    }
    sendJson(res, 200, json{{"text", text}});
}

} // namespace mimirmind::server
