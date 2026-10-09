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
 * Handles POST /v1/systemone — the native Laya typed-decision endpoint (8.27).
 * Jev-compatible: a `state` plus one or more typed `questions` (choice / score /
 * noul) answered in one backbone + head forward per question (LayaSystemOne).
 *
 *   request  { "model"?,                        // systemone model id (optional)
 *              "state": "..." | {...},           // the text/object to judge
 *              "questions": { "<id>": { "type": "choice|score|noul",
 *                                       "instructions": "...",
 *                                       "criteria": {..}|[..] } } }
 *   response { "model",
 *              "answers": { "<id>": { "type", "choice"|"score"|"noul",
 *                                     "probabilities": {..}, "confidence",
 *                                     "answer_confidence", "action": {act_probability} } },
 *              "routing": { "model" } }
 *
 * Each systemone model gets its own serialisation mutex (the compute stack is
 * not thread-safe), mirroring DecideHandler. Resolution is by request `model`;
 * an empty/omitted model picks the first systemone entry.
 */
class SystemOneHandler {
public:
    SystemOneHandler(std::vector<LoadedSystemOne> models, const ServerConfig& cfg);

    SystemOneHandler(const SystemOneHandler&)            = delete;
    SystemOneHandler& operator=(const SystemOneHandler&) = delete;
    SystemOneHandler(SystemOneHandler&&)                 = delete;
    SystemOneHandler& operator=(SystemOneHandler&&)      = delete;

    void handle(const httplib::Request& req, httplib::Response& res);

    [[nodiscard]] bool empty() const noexcept { return _slots.empty(); }

    struct ModelInfo {
        std::string id;
        std::string title;
    };
    [[nodiscard]] std::vector<ModelInfo> listModels() const;

private:
    struct Slot {
        std::string                         id;
        std::string                         title;
        runtime::encoder::laya::LayaSystemOne* engine{nullptr};
        std::unique_ptr<std::mutex>         mutex;
    };

    [[nodiscard]] Slot* resolve(const std::string& model);

    std::vector<Slot>   _slots;
    const ServerConfig& _cfg;
};

} // namespace mimirmind::server
