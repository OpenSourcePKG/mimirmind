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

} // namespace mimirmind::server
