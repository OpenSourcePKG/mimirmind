// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "server/SpeechHandler.hpp"

#include "runtime/audio/SpeakEngine.hpp"
#include "runtime/audio/WavWriter.hpp"
#include "server/ApiHelpers.hpp"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace mimirmind::server {

using nlohmann::json;

SpeechHandler::SpeechHandler(std::vector<LoadedSpeaker> speakers,
                             const ServerConfig& cfg)
    : _cfg{cfg} {
    _slots.reserve(speakers.size());
    for (auto& s : speakers) {
        Slot slot{};
        slot.id     = std::move(s.id);
        slot.title  = std::move(s.title);
        slot.engine = s.engine;
        slot.mutex  = std::make_unique<std::mutex>();
        _slots.push_back(std::move(slot));
    }
}

std::vector<SpeechHandler::ModelInfo> SpeechHandler::listModels() const {
    std::vector<ModelInfo> out;
    out.reserve(_slots.size());
    for (const auto& s : _slots) {
        out.push_back({s.id, s.title.empty() ? s.id : s.title});
    }
    return out;
}

SpeechHandler::Slot* SpeechHandler::resolve(const std::string& model) {
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
    // Lenient single-model fallback: an unmatched id resolves to the sole
    // loaded model. Only ambiguous with 2+ models, where the 404 stands.
    if (_slots.size() == 1) {
        return &_slots.front();
    }
    return nullptr;
}

void SpeechHandler::handle(const httplib::Request& req, httplib::Response& res) {
    if (_slots.empty()) {
        sendError(res, 404, "model_not_found",
                  "no speech model is loaded (configure a model with task=speak)");
        return;
    }

    json body;
    try {
        body = json::parse(req.body);
    } catch (const std::exception&) {
        sendError(res, 400, "invalid_request_error",
                  "request body must be valid JSON");
        return;
    }
    if (!body.is_object()) {
        sendError(res, 400, "invalid_request_error",
                  "request body must be a JSON object");
        return;
    }

    const std::string input =
        body.value("input", body.value("text", std::string{}));
    if (input.empty()) {
        sendError(res, 400, "invalid_request_error",
                  "'input' (the text to synthesize) is required");
        return;
    }

    const std::string model = body.value("model", std::string{});
    Slot* slot = resolve(model);
    if (slot == nullptr) {
        sendError(res, 400, "model_not_found",
                  "unknown speech model '" + model + "'");
        return;
    }

    const std::string voice = body.value("voice", std::string{});
    std::string responseFormat = body.value("response_format", std::string{"wav"});
    if (responseFormat.empty()) {
        responseFormat = "wav";
    }
    if (responseFormat != "wav" && responseFormat != "pcm") {
        sendError(res, 400, "invalid_request_error",
                  "unsupported response_format '" + responseFormat +
                      "' (only \"wav\" and \"pcm\" are supported)");
        return;
    }

    try {
        const std::lock_guard<std::mutex> lk{*slot->mutex};
        std::vector<std::byte> audio;
        std::string            contentType;
        if (responseFormat == "pcm") {
            const std::vector<float> pcm = slot->engine->synthesizePcm(input, voice);
            audio       = runtime::audio::WavWriter::encodeRawPcm16(pcm);
            contentType = "audio/L16; rate=24000; channels=1";
        } else {
            audio       = slot->engine->synthesizeWav(input, voice);
            contentType = "audio/wav";
        }
        res.status = 200;
        res.set_content(reinterpret_cast<const char*>(audio.data()), audio.size(),
                        contentType);
    } catch (const std::exception& e) {
        sendError(res, 500, "internal_error",
                  std::string{"speech synthesis failed: "} + e.what());
    }
}

} // namespace mimirmind::server
