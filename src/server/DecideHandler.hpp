// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "server/ApiServer.hpp"
#include "server/ModelProvider.hpp"   // ResidentModelMemory

#include <httplib.h>

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace mimirmind::server {

/**
 * Handles POST /v1/decide — the "System-One" typed-decision endpoint (8.23).
 * A small linear-probe head on the resident bge-m3 encoder answers a typed
 * question (choice over a fixed label set) in one encoder forward + a host-side
 * matvec, instead of a full LLM turn. Pegenaut calls it at turn start and falls
 * back to the LLM when the head is absent or under-confident.
 *
 *   request  { "model"?,                       // decide model id (optional)
 *              "state":  { "user": "..." } | "...",   // the text to classify
 *              "questions": { "tool": {...}, "rag": {...} }? }  // keys select
 *                                                 // heads; omitted → all heads
 *   response { "model", "latency_ms",
 *              "answers": { "tool": { "choice", "p", "confident", "dist": {...} },
 *                           "route": { "error": "no_head" } } }
 *
 * Each decide model gets its own serialisation mutex (the shared EncoderRunner
 * is not thread-safe), mirroring EmbeddingsHandler. Resolution is by request
 * `model`; an empty/omitted model picks the first decide entry.
 */
class DecideHandler {
public:
    DecideHandler(std::vector<LoadedDecider> deciders, const ServerConfig& cfg);

    DecideHandler(const DecideHandler&)            = delete;
    DecideHandler& operator=(const DecideHandler&) = delete;
    DecideHandler(DecideHandler&&)                 = delete;
    DecideHandler& operator=(DecideHandler&&)      = delete;

    void handle(const httplib::Request& req, httplib::Response& res);

    [[nodiscard]] bool empty() const noexcept { return _slots.empty(); }

    struct ModelInfo {
        std::string id;
        std::string title;
    };
    [[nodiscard]] std::vector<ModelInfo> listModels() const;

    /// Per-model weight attribution for GET /v1/system/memory. Tagged
    /// role="decide", inGenerativePool=false, servingActive=false.
    [[nodiscard]] std::vector<ResidentModelMemory> residentModelsMemory() const;

private:
    struct Slot {
        std::string                     id;
        std::string                     title;
        runtime::encoder::DecideEngine* engine{nullptr};
        std::unique_ptr<std::mutex>     mutex;
    };

    [[nodiscard]] Slot* resolve(const std::string& model);

    std::vector<Slot>   _slots;
    const ServerConfig& _cfg;
};

} // namespace mimirmind::server
