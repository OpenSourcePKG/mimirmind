// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "server/SystemOneHandler.hpp"

#include "runtime/encoder/laya/LayaSystemOne.hpp"

#include <nlohmann/json.hpp>

#include <mutex>
#include <stdexcept>
#include <string>

namespace mimirmind::server {

namespace {

using LayaSystemOne = runtime::encoder::laya::LayaSystemOne;

std::string criterionText(const nlohmann::ordered_json& v) {
    return v.is_string() ? v.get<std::string>() : v.dump();
}

void sendError(httplib::Response& res, int status, const std::string& msg) {
    nlohmann::json e = {{"error", {{"message", msg}, {"type", "invalid_request_error"}}}};
    res.status = status;
    res.set_content(e.dump(), "application/json");
}

} // namespace

SystemOneHandler::SystemOneHandler(std::vector<LoadedSystemOne> models,
                                   const ServerConfig& cfg)
    : _cfg{cfg} {
    _slots.reserve(models.size());
    for (auto& m : models) {
        _slots.push_back(Slot{m.id, m.title, m.engine, std::make_unique<std::mutex>()});
    }
}

SystemOneHandler::Slot* SystemOneHandler::resolve(const std::string& model) {
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

std::vector<SystemOneHandler::ModelInfo> SystemOneHandler::listModels() const {
    std::vector<ModelInfo> out;
    out.reserve(_slots.size());
    for (const auto& s : _slots) {
        out.push_back({s.id, s.title});
    }
    return out;
}

void SystemOneHandler::handle(const httplib::Request& req, httplib::Response& res) {
    // ordered_json so a choice question's `criteria` keeps its request order —
    // the sequential option layout is order-sensitive (upstream parallel_layout note).
    nlohmann::ordered_json body;
    try {
        body = nlohmann::ordered_json::parse(req.body);
    } catch (const std::exception& e) {
        sendError(res, 400, std::string("invalid JSON body: ") + e.what());
        return;
    }
    if (!body.is_object() || !body.contains("questions") || !body["questions"].is_object()) {
        sendError(res, 400, "body must be an object with a 'questions' object");
        return;
    }

    const std::string model = body.value("model", std::string{});
    Slot* slot = resolve(model);
    if (slot == nullptr) {
        sendError(res, 404, model.empty() ? "no systemone model loaded"
                                          : ("unknown systemone model '" + model + "'"));
        return;
    }

    // state: a string, or any JSON value serialized compactly (serialize_state).
    std::string state;
    if (body.contains("state")) {
        const auto& s = body["state"];
        state = s.is_string() ? s.get<std::string>() : s.dump();
    }

    nlohmann::ordered_json answers = nlohmann::ordered_json::object();
    try {
        const std::lock_guard<std::mutex> lock(*slot->mutex);
        for (const auto& [qid, qdef] : body["questions"].items()) {
            if (!qdef.is_object()) {
                sendError(res, 400, "question '" + qid + "' must be an object");
                return;
            }
            const std::string type = qdef.value("type", std::string{"choice"});
            LayaSystemOne::Question q;
            q.instructions = qdef.value("instructions", std::string{});

            if (type == "choice") {
                q.type = LayaSystemOne::QType::Choice;
                if (!qdef.contains("criteria") || !qdef["criteria"].is_object()) {
                    sendError(res, 400, "choice question '" + qid + "' needs a 'criteria' object");
                    return;
                }
                for (const auto& [key, desc] : qdef["criteria"].items()) {
                    q.choice.emplace_back(key, desc.is_null() ? std::string{} : criterionText(desc));
                }
            } else if (type == "score") {
                q.type = LayaSystemOne::QType::Score;
                if (!qdef.contains("criteria") || !qdef["criteria"].is_array()) {
                    sendError(res, 400, "score question '" + qid + "' needs a 'criteria' array");
                    return;
                }
                for (const auto& c : qdef["criteria"]) {
                    q.levels.push_back(criterionText(c));
                }
            } else if (type == "noul") {
                q.type = LayaSystemOne::QType::Noul;
                const auto& crit = qdef["criteria"];
                if (crit.is_object()) {
                    if (crit.contains("false") && !crit["false"].is_null()) q.noulFalse = criterionText(crit["false"]);
                    if (crit.contains("true")  && !crit["true"].is_null())  q.noulTrue  = criterionText(crit["true"]);
                }
                if (qdef.contains("labels") && qdef["labels"].is_object()) {
                    q.noulFalseLabel = qdef["labels"].value("false", std::string{"false"});
                    q.noulTrueLabel  = qdef["labels"].value("true", std::string{"true"});
                }
            } else {
                sendError(res, 400, "question '" + qid + "': unknown type '" + type + "'");
                return;
            }

            const LayaSystemOne::Answer a = slot->engine->decide(state, q);
            nlohmann::ordered_json ans = nlohmann::ordered_json::object();
            nlohmann::ordered_json action = {{"act_probability", a.actProbability}};
            if (q.type == LayaSystemOne::QType::Choice) {
                ans["type"] = "choice";
                ans["choice"] = a.choice;
                nlohmann::ordered_json probs = nlohmann::ordered_json::object();
                for (const auto& [k, p] : a.probabilities) probs[k] = p;
                ans["probabilities"] = probs;
                ans["confidence"] = a.confidence;
                ans["answer_confidence"] = a.answerConfidence;
            } else if (q.type == LayaSystemOne::QType::Score) {
                ans["type"] = "score";
                ans["score"] = a.score;
                nlohmann::ordered_json probs = nlohmann::ordered_json::object();
                for (const auto& [k, p] : a.probabilities) probs[k] = p;
                ans["probabilities"] = probs;
                ans["confidence"] = a.confidence;
                ans["answer_confidence"] = a.answerConfidence;
            } else {
                ans["type"] = "noul";
                ans["noul"] = a.noul;
                ans["confidence"] = a.confidence;
                ans["answer_confidence"] = a.answerConfidence;
            }
            ans["action"] = action;
            answers[qid] = ans;
        }
    } catch (const std::exception& e) {
        sendError(res, 500, std::string("systemone decode failed: ") + e.what());
        return;
    }

    nlohmann::ordered_json out = {
        {"model", slot->id},
        {"answers", answers},
        {"routing", {{"model", slot->id}}},
    };
    res.status = 200;
    res.set_content(out.dump(), "application/json");
}

} // namespace mimirmind::server
