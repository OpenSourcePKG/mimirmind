// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "server/EmbeddingsHandler.hpp"

#include "runtime/encoder/EmbedEngine.hpp"
#include "server/ApiHelpers.hpp"

#include <nlohmann/json.hpp>

#include <span>
#include <string>
#include <vector>

namespace mimirmind::server {

using nlohmann::json;

EmbeddingsHandler::EmbeddingsHandler(std::vector<LoadedEmbedder> embedders,
                                     const ServerConfig& cfg)
    : _cfg{cfg} {
    _slots.reserve(embedders.size());
    for (auto& e : embedders) {
        Slot s{};
        s.id     = std::move(e.id);
        s.title  = std::move(e.title);
        s.engine = e.engine;
        s.mutex  = std::make_unique<std::mutex>();
        _slots.push_back(std::move(s));
    }
}

std::vector<EmbeddingsHandler::ModelInfo> EmbeddingsHandler::listModels() const {
    std::vector<ModelInfo> out;
    out.reserve(_slots.size());
    for (const auto& s : _slots) {
        out.push_back({s.id, s.title.empty() ? s.id : s.title});
    }
    return out;
}

EmbeddingsHandler::Slot* EmbeddingsHandler::resolve(const std::string& model) {
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

void EmbeddingsHandler::handle(const httplib::Request& req, httplib::Response& res) {
    if (_slots.empty()) {
        sendError(res, 404, "model_not_found",
                  "no embedding model is loaded (configure a model with task=embed)");
        return;
    }

    json body;
    try {
        body = json::parse(req.body);
    } catch (const std::exception& e) {
        sendError(res, 400, "invalid_request_error",
                  std::string{"invalid JSON: "} + e.what());
        return;
    }
    if (!body.is_object()) {
        sendError(res, 400, "invalid_request_error", "body must be a JSON object");
        return;
    }

    const std::string model = body.value("model", std::string{});
    Slot* slot = resolve(model);
    if (slot == nullptr) {
        sendError(res, 400, "model_not_found",
                  "unknown embedding model '" + model + "'");
        return;
    }

    // `input` is a string or an array of strings (OpenAI accepts token-id
    // arrays too; we support text only).
    if (!body.contains("input")) {
        sendError(res, 400, "invalid_request_error",
                  "'input' (string or array of strings) is required");
        return;
    }
    std::vector<std::string> inputs;
    const auto& in = body["input"];
    if (in.is_string()) {
        inputs.push_back(in.get<std::string>());
    } else if (in.is_array()) {
        inputs.reserve(in.size());
        for (const auto& s : in) {
            if (!s.is_string()) {
                sendError(res, 400, "invalid_request_error",
                          "'input' array must contain strings");
                return;
            }
            inputs.push_back(s.get<std::string>());
        }
    } else {
        sendError(res, 400, "invalid_request_error",
                  "'input' must be a string or an array of strings");
        return;
    }
    if (inputs.empty()) {
        sendError(res, 400, "invalid_request_error", "'input' is empty");
        return;
    }

    // Only the default float encoding is supported (no base64).
    if (body.contains("encoding_format") && body["encoding_format"].is_string() &&
        body["encoding_format"].get<std::string>() != "float") {
        sendError(res, 400, "invalid_request_error",
                  "only encoding_format \"float\" is supported");
        return;
    }

    std::vector<std::vector<float>> vectors;
    try {
        const std::lock_guard<std::mutex> lk{*slot->mutex};
        vectors = slot->engine->embedBatch(std::span<const std::string>{inputs});
    } catch (const std::exception& e) {
        sendError(res, 500, "internal_error",
                  std::string{"embedding failed: "} + e.what());
        return;
    }

    json out;
    out["object"] = "list";
    out["model"]  = slot->id;
    json data = json::array();
    for (std::size_t i = 0; i < vectors.size(); ++i) {
        json item;
        item["object"]    = "embedding";
        item["index"]     = i;
        item["embedding"] = vectors[i];
        data.push_back(std::move(item));
    }
    out["data"] = std::move(data);
    // Token accounting is not tracked on the encoder path; report zeros so the
    // OpenAI response shape stays intact for clients that read `usage`.
    out["usage"] = json{{"prompt_tokens", 0}, {"total_tokens", 0}};
    sendJson(res, 200, out);
}

} // namespace mimirmind::server
