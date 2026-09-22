// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "server/DecideHandler.hpp"

#include "runtime/encoder/DecideEngine.hpp"
#include "server/ApiHelpers.hpp"

#include <nlohmann/json.hpp>

#include <chrono>
#include <span>
#include <string>
#include <vector>

namespace mimirmind::server {

using nlohmann::json;

DecideHandler::DecideHandler(std::vector<LoadedDecider> deciders,
                             const ServerConfig& cfg)
    : _cfg{cfg} {
    _slots.reserve(deciders.size());
    for (auto& d : deciders) {
        Slot s{};
        s.id     = std::move(d.id);
        s.title  = std::move(d.title);
        s.engine = d.engine;
        s.mutex  = std::make_unique<std::mutex>();
        _slots.push_back(std::move(s));
    }
}

std::vector<DecideHandler::ModelInfo> DecideHandler::listModels() const {
    std::vector<ModelInfo> out;
    out.reserve(_slots.size());
    for (const auto& s : _slots) {
        out.push_back({s.id, s.title.empty() ? s.id : s.title});
    }
    return out;
}

std::vector<ResidentModelMemory> DecideHandler::residentModelsMemory() const {
    std::vector<ResidentModelMemory> out;
    out.reserve(_slots.size());
    for (const auto& s : _slots) {
        ResidentModelMemory m{};
        m.id               = s.id;
        m.title            = s.title.empty() ? s.id : s.title;
        m.isDefault        = false;
        m.weightBytes      = s.engine != nullptr ? s.engine->weightBytes() : 0;
        m.servingActive    = false;   // encoders hold no paged KV cache
        m.role             = "decide";
        m.inGenerativePool = false;
        out.push_back(std::move(m));
    }
    return out;
}

DecideHandler::Slot* DecideHandler::resolve(const std::string& model) {
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
    // Lenient single-model fallback, mirroring EmbeddingsHandler.
    if (_slots.size() == 1) {
        return &_slots.front();
    }
    return nullptr;
}

namespace {

// Liberal-accept extraction of the text to classify. Accepts, in order:
//   state.user (string) | state (string) | text (string) | input (string).
// Returns false when none is present/valid.
bool extractText(const json& body, std::string& out) {
    if (const auto it = body.find("state"); it != body.end()) {
        if (it->is_object()) {
            if (const auto u = it->find("user");
                u != it->end() && u->is_string()) {
                out = u->get<std::string>();
                return true;
            }
        } else if (it->is_string()) {
            out = it->get<std::string>();
            return true;
        }
    }
    for (const char* key : {"text", "input"}) {
        if (const auto it = body.find(key); it != body.end() && it->is_string()) {
            out = it->get<std::string>();
            return true;
        }
    }
    return false;
}

} // namespace

void DecideHandler::handle(const httplib::Request& req, httplib::Response& res) {
    if (_slots.empty()) {
        sendError(res, 404, "model_not_found",
                  "no decision model is loaded (configure a model with task=decide)");
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
                  "unknown decision model '" + model + "'");
        return;
    }

    std::string text;
    if (!extractText(body, text)) {
        sendError(res, 400, "invalid_request_error",
                  "'state.user' (or 'state'/'text'/'input') string is required");
        return;
    }

    // Requested questions = keys of the `questions` object. Absent → all heads.
    std::vector<std::string> questions;
    if (const auto it = body.find("questions"); it != body.end()) {
        if (!it->is_object()) {
            sendError(res, 400, "invalid_request_error",
                      "'questions' must be an object keyed by question name");
            return;
        }
        questions.reserve(it->size());
        for (const auto& [name, spec] : it->items()) {
            questions.push_back(name);
        }
    }

    std::vector<runtime::encoder::DecideEngine::Answer> answers;
    double latencyMs = 0.0;
    try {
        const std::lock_guard<std::mutex> lk{*slot->mutex};
        const auto t0 = std::chrono::steady_clock::now();
        answers = slot->engine->decide(
            text, std::span<const std::string>{questions});
        const auto t1 = std::chrono::steady_clock::now();
        latencyMs =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
    } catch (const std::exception& e) {
        sendError(res, 500, "internal_error",
                  std::string{"decision failed: "} + e.what());
        return;
    }

    json ans = json::object();
    for (const auto& a : answers) {
        if (!a.hasHead) {
            ans[a.question] = json{{"error", "no_head"}};
            continue;
        }
        json dist = json::object();
        for (const auto& p : a.result.dist) {
            dist[p.label] = p.p;
        }
        ans[a.question] = json{
            {"choice",    a.result.choice},
            {"p",         a.result.p},
            {"confident", a.result.confident},
            {"dist",      std::move(dist)},
        };
    }

    json out;
    out["model"]      = slot->id;
    out["latency_ms"] = latencyMs;
    out["answers"]    = std::move(ans);
    sendJson(res, 200, out);
}

void DecideHandler::handleUpload(const httplib::Request& req, httplib::Response& res) {
    if (_slots.empty()) {
        sendError(res, 404, "model_not_found",
                  "no decision model is loaded (configure a model with task=decide)");
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
                  "unknown decision model '" + model + "'");
        return;
    }
    if (slot->engine->headsDir().empty()) {
        sendError(res, 409, "not_configured",
                  "decision model '" + slot->id +
                  "' has no headsDir configured — head upload is disabled");
        return;
    }

    // Required fields.
    if (!body.contains("name") || !body["name"].is_string()) {
        sendError(res, 400, "invalid_request_error", "'name' (string) is required");
        return;
    }
    if (!body.contains("labels") || !body["labels"].is_array()) {
        sendError(res, 400, "invalid_request_error",
                  "'labels' (array of strings) is required");
        return;
    }
    if (!body.contains("hidden") || !body["hidden"].is_number_unsigned()) {
        sendError(res, 400, "invalid_request_error",
                  "'hidden' (positive integer) is required");
        return;
    }
    if (!body.contains("weight") || !body["weight"].is_array()) {
        sendError(res, 400, "invalid_request_error",
                  "'weight' (array of floats, labels*hidden row-major) is required");
        return;
    }
    if (!body.contains("bias") || !body["bias"].is_array()) {
        sendError(res, 400, "invalid_request_error",
                  "'bias' (array of floats, one per label) is required");
        return;
    }

    runtime::encoder::DecisionHead::Spec spec{};
    try {
        spec.name        = body["name"].get<std::string>();
        spec.labels      = body["labels"].get<std::vector<std::string>>();
        spec.hidden      = body["hidden"].get<std::size_t>();
        spec.temperature = body.value("temperature", 1.0F);
        spec.threshold   = body.value("threshold", 0.0F);
        spec.encoder     = body.value("encoder", std::string{});
        spec.weight      = body["weight"].get<std::vector<float>>();
        spec.bias        = body["bias"].get<std::vector<float>>();
    } catch (const std::exception& e) {
        sendError(res, 400, "invalid_request_error",
                  std::string{"malformed head fields: "} + e.what());
        return;
    }

    try {
        const std::lock_guard<std::mutex> lk{*slot->mutex};
        slot->engine->upsertHead(spec);
    } catch (const std::exception& e) {
        // Bad shape / hidden mismatch / unsafe name — the engine validated and
        // rejected it before touching the live head set.
        sendError(res, 400, "invalid_request_error",
                  std::string{"head rejected: "} + e.what());
        return;
    }

    json out;
    out["model"]  = slot->id;
    out["name"]   = spec.name;
    out["labels"] = spec.labels;
    out["heads"]  = slot->engine->headNames();
    out["status"] = "ok";
    sendJson(res, 200, out);
}

} // namespace mimirmind::server
