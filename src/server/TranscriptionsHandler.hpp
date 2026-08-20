// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "server/ApiServer.hpp"

#include <httplib.h>

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace mimirmind::server {

/**
 * Handles POST /v1/audio/transcriptions — the OpenAI speech-to-text endpoint:
 *
 *   request  multipart/form-data:
 *              file            (required) — the audio upload (WAV for now)
 *              model           (optional) — selects the ASR model; empty picks
 *                                           the first transcribe entry
 *              language        (optional) — Whisper language code, default "en"
 *              response_format (optional) — "json" (default) | "text"
 *   response json  { "text": "<transcript>" }
 *            text  the raw transcript (text/plain)
 *
 * Each transcribe model gets its own serialisation mutex (AudioEngine is not
 * thread-safe), mirroring EmbeddingsHandler / RerankHandler. verbose_json
 * (segments/timestamps) and streaming are later increments.
 */
class TranscriptionsHandler {
public:
    TranscriptionsHandler(std::vector<LoadedTranscriber> transcribers,
                          const ServerConfig& cfg);

    TranscriptionsHandler(const TranscriptionsHandler&)            = delete;
    TranscriptionsHandler& operator=(const TranscriptionsHandler&) = delete;
    TranscriptionsHandler(TranscriptionsHandler&&)                 = delete;
    TranscriptionsHandler& operator=(TranscriptionsHandler&&)      = delete;

    void handle(const httplib::Request& req, httplib::Response& res);

    [[nodiscard]] bool empty() const noexcept { return _slots.empty(); }

    struct ModelInfo {
        std::string id;
        std::string title;
    };
    [[nodiscard]] std::vector<ModelInfo> listModels() const;

private:
    struct Slot {
        std::string                  id;
        std::string                  title;
        runtime::audio::AudioEngine* engine{nullptr};
        std::unique_ptr<std::mutex>  mutex;
    };

    [[nodiscard]] Slot* resolve(const std::string& model);

    std::vector<Slot>   _slots;
    const ServerConfig& _cfg;
};

} // namespace mimirmind::server
