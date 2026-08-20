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
 * Handles POST /v1/audio/speech — the OpenAI text-to-speech endpoint:
 *
 *   request  application/json:
 *              model           (optional) — selects the TTS model; empty picks
 *                                           the first speak entry
 *              input           (required) — the text to synthesize
 *              voice           (optional) — one of the Orpheus voices
 *                                           (tara/leah/...); empty -> "tara"
 *              response_format (optional) — "wav" (default) | "pcm"
 *              speed           (optional) — accepted, not yet applied
 *   response audio bytes with the matching Content-Type (audio/wav for wav,
 *            audio/L16 for raw 24 kHz PCM16).
 *
 * Each speak model gets its own serialisation mutex (the acoustic InferenceEngine
 * has one KV cache — one synthesize() at a time), mirroring the other handlers.
 * mp3/opus/aac/flac (needing an encoder) and streaming are later increments.
 */
class SpeechHandler {
public:
    SpeechHandler(std::vector<LoadedSpeaker> speakers, const ServerConfig& cfg);

    SpeechHandler(const SpeechHandler&)            = delete;
    SpeechHandler& operator=(const SpeechHandler&) = delete;
    SpeechHandler(SpeechHandler&&)                 = delete;
    SpeechHandler& operator=(SpeechHandler&&)      = delete;

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
        runtime::audio::SpeakEngine* engine{nullptr};
        std::unique_ptr<std::mutex>  mutex;
    };

    [[nodiscard]] Slot* resolve(const std::string& model);

    std::vector<Slot>   _slots;
    const ServerConfig& _cfg;
};

} // namespace mimirmind::server
