// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "server/RerankHandler.hpp"

#include "runtime/encoder/RerankEngine.hpp"
#include "server/ApiHelpers.hpp"

#include <nlohmann/json.hpp>

#include <optional>
#include <span>
#include <string>
#include <vector>

namespace mimirmind::server {

using nlohmann::json;

RerankHandler::RerankHandler(std::vector<LoadedReranker> rerankers,
                             const ServerConfig& cfg)
    : _cfg{cfg} {
    _slots.reserve(rerankers.size());
    for (auto& r : rerankers) {
        Slot s{};
        s.id     = std::move(r.id);
        s.title  = std::move(r.title);
        s.engine = r.engine;
        s.mutex  = std::make_unique<std::mutex>();
        _slots.push_back(std::move(s));
    }
}

std::vector<RerankHandler::ModelInfo> RerankHandler::listModels() const {
    std::vector<ModelInfo> out;
    out.reserve(_slots.size());
    for (const auto& s : _slots) {
        out.push_back({s.id, s.title.empty() ? s.id : s.title});
    }
    return out;
}

RerankHandler::Slot* RerankHandler::resolve(const std::string& model) {
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

void RerankHandler::handle(const httplib::Request& req, httplib::Response& res) {
    if (_slots.empty()) {
        sendError(res, 404, "model_not_found",
                  "no rerank model is loaded (configure a model with task=rerank)");
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
                  "unknown rerank model '" + model + "'");
        return;
    }

    if (!body.contains("query") || !body["query"].is_string()) {
        sendError(res, 400, "invalid_request_error",
                  "'query' (string) is required");
        return;
    }
    const std::string query = body["query"].get<std::string>();

    if (!body.contains("documents") || !body["documents"].is_array()) {
        sendError(res, 400, "invalid_request_error",
                  "'documents' (array of strings) is required");
        return;
    }
    std::vector<std::string> documents;
    documents.reserve(body["documents"].size());
    for (const auto& d : body["documents"]) {
        if (!d.is_string()) {
            sendError(res, 400, "invalid_request_error",
                      "'documents' must be an array of strings");
            return;
        }
        documents.push_back(d.get<std::string>());
    }
    if (documents.empty()) {
        sendError(res, 400, "invalid_request_error", "'documents' is empty");
        return;
    }

    std::optional<std::size_t> topN;
    if (body.contains("top_n") && body["top_n"].is_number_integer()) {
        const auto n = body["top_n"].get<long long>();
        if (n > 0) {
            topN = static_cast<std::size_t>(n);
        }
    }
    const bool returnDocuments = body.value("return_documents", false);

    std::vector<runtime::encoder::RerankEngine::Result> results;
    try {
        const std::lock_guard<std::mutex> lk{*slot->mutex};
        results = slot->engine->rerank(query, std::span<const std::string>{documents},
                                       /*sortDescending=*/true, topN);
    } catch (const std::exception& e) {
        sendError(res, 500, "internal_error",
                  std::string{"rerank failed: "} + e.what());
        return;
    }

    json out;
    out["model"] = slot->id;
    out["object"] = "list";
    json arr = json::array();
    for (const auto& r : results) {
        json item;
        item["index"] = r.index;
        item["relevance_score"] = r.score;
        if (returnDocuments && r.index < documents.size()) {
            item["document"] = json{{"text", documents[r.index]}};
        }
        arr.push_back(std::move(item));
    }
    out["results"] = std::move(arr);
    sendJson(res, 200, out);
}

} // namespace mimirmind::server
