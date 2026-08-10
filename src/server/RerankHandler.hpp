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
 * Handles POST /v1/rerank — a Jina/Cohere-style cross-encoder rerank endpoint:
 *
 *   request  { "model", "query", "documents": [str...],
 *              "top_n"?, "return_documents"? }
 *   response { "model", "results": [ { "index", "relevance_score",
 *                                      "document"? } ]  }   (best-first)
 *
 * Each rerank model gets its own serialisation mutex (EncoderRunner is not
 * thread-safe), mirroring the per-engine locking of the chat path. Resolution
 * is by request `model`; an empty/omitted model picks the first rerank entry.
 */
class RerankHandler {
public:
    RerankHandler(std::vector<LoadedReranker> rerankers, const ServerConfig& cfg);

    RerankHandler(const RerankHandler&)            = delete;
    RerankHandler& operator=(const RerankHandler&) = delete;
    RerankHandler(RerankHandler&&)                 = delete;
    RerankHandler& operator=(RerankHandler&&)      = delete;

    void handle(const httplib::Request& req, httplib::Response& res);

    [[nodiscard]] bool empty() const noexcept { return _slots.empty(); }

    struct ModelInfo {
        std::string id;
        std::string title;
    };
    [[nodiscard]] std::vector<ModelInfo> listModels() const;

private:
    struct Slot {
        std::string                     id;
        std::string                     title;
        runtime::encoder::RerankEngine* engine{nullptr};
        std::unique_ptr<std::mutex>     mutex;
    };

    [[nodiscard]] Slot* resolve(const std::string& model);

    std::vector<Slot>   _slots;
    const ServerConfig& _cfg;
};

} // namespace mimirmind::server
