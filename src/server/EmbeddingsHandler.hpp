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
 * Handles POST /v1/embeddings — the OpenAI embeddings endpoint:
 *
 *   request  { "model", "input": str | [str...],
 *              "encoding_format"? ("float", the only supported value) }
 *   response { "object": "list", "model", "data": [ { "object": "embedding",
 *              "index", "embedding": [float...] } ], "usage": {...} }
 *
 * Each embed model gets its own serialisation mutex (EncoderRunner is not
 * thread-safe), mirroring RerankHandler. Resolution is by request `model`; an
 * empty/omitted model picks the first embed entry. Vectors are the bi-encoder
 * CLS-pooled, L2-normalized sentence embeddings from EmbedEngine.
 */
class EmbeddingsHandler {
public:
    EmbeddingsHandler(std::vector<LoadedEmbedder> embedders, const ServerConfig& cfg);

    EmbeddingsHandler(const EmbeddingsHandler&)            = delete;
    EmbeddingsHandler& operator=(const EmbeddingsHandler&) = delete;
    EmbeddingsHandler(EmbeddingsHandler&&)                 = delete;
    EmbeddingsHandler& operator=(EmbeddingsHandler&&)      = delete;

    void handle(const httplib::Request& req, httplib::Response& res);

    [[nodiscard]] bool empty() const noexcept { return _slots.empty(); }

    struct ModelInfo {
        std::string id;
        std::string title;
    };
    [[nodiscard]] std::vector<ModelInfo> listModels() const;

private:
    struct Slot {
        std::string                    id;
        std::string                    title;
        runtime::encoder::EmbedEngine* engine{nullptr};
        std::unique_ptr<std::mutex>    mutex;
    };

    [[nodiscard]] Slot* resolve(const std::string& model);

    std::vector<Slot>   _slots;
    const ServerConfig& _cfg;
};

} // namespace mimirmind::server
