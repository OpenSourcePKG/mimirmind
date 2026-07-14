// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "server/RequestDispatcher.hpp"

#include "runtime/InferenceEngine.hpp"
#include "core/log/Log.hpp"

#include <nlohmann/json.hpp>

#include <stdexcept>

namespace mimirmind::server {

using nlohmann::json;

namespace {

runtime::InferenceEngine& pickDefault(const std::vector<LoadedEngine>& in,
                                       const std::string&                modelId) {
    if (in.empty()) {
        throw std::runtime_error("RequestDispatcher: engines list is empty");
    }
    if (!modelId.empty()) {
        for (const auto& le : in) {
            if (le.id == modelId) {
                return *le.engine;
            }
        }
        throw std::runtime_error(
            "RequestDispatcher: modelId '" + modelId +
            "' does not match any LoadedEngine.id");
    }
    return *in.front().engine;
}

} // namespace

RequestDispatcher::RequestDispatcher(
    std::vector<LoadedEngine>                  engines,
    const std::string&                          modelId,
    runtime::InferenceEngine*                   draftEngine,
    const std::string&                          speculativeTargetId,
    const runtime::SpeculativeDecoder::Config&  speculativeConfig)
    : _defaultEngine{&pickDefault(engines, modelId)},
      _draftEngine{draftEngine} {

    // Identify the default entry so its metadata is available for
    // listModels() without a second lookup.
    for (const auto& le : engines) {
        if ((!modelId.empty() && le.id == modelId) ||
            (modelId.empty() && le.engine == _defaultEngine)) {
            _defaultId    = le.id;
            _defaultTitle = le.title;
            break;
        }
    }

    // Non-default engines become ExtraHandle entries. Each carries its
    // own dispatch mutex; chat/completions requests targeting these
    // engines lock the handle mutex, not the default mutex.
    for (auto& le : engines) {
        if (le.id == _defaultId) continue;
        ExtraHandle h;
        h.id     = le.id;
        h.title  = le.title;
        h.engine = le.engine;
        _extraHandles.push_back(std::move(h));
    }

    // M9.11.4 — stand up the spec-dec orchestrator now that both
    // engines are ready. Spec-dec is currently wired only for the
    // default engine — if `speculativeTargetId` names an ExtraHandle
    // instead, spec-dec stays off with a warning so the operator can
    // fix the config.
    if (_draftEngine != nullptr) {
        if (speculativeTargetId.empty() || speculativeTargetId == _defaultId) {
            _speculativeDecoder =
                std::make_unique<runtime::SpeculativeDecoder>(
                    *_defaultEngine, *_draftEngine, speculativeConfig);
        } else {
            MM_LOG_WARN("server",
                        "speculative.target='{}' is not the default "
                        "engine (default='{}') — spec-dec disabled. "
                        "Move the target to the default modelId or "
                        "restructure your config.",
                        speculativeTargetId, _defaultId);
        }
    }
}

std::vector<RequestDispatcher::ModelEntry> RequestDispatcher::listModels() const {
    std::vector<ModelEntry> out;
    out.reserve(1 + _extraHandles.size());
    out.push_back({_defaultId, _defaultTitle});
    for (const auto& h : _extraHandles) {
        out.push_back({h.id, h.title});
    }
    return out;
}

std::optional<RequestDispatcher::Target> RequestDispatcher::resolveTarget(
    const std::string& model, httplib::Response& res) {
    if (model.empty() || model == _defaultId) {
        return Target{_defaultEngine, &_defaultMutex,
                      _speculativeDecoder.get(),
                      _defaultId, _defaultTitle};
    }
    for (auto& h : _extraHandles) {
        if (h.id == model) {
            return Target{h.engine, h.mutex.get(),
                          /*spec=*/nullptr,
                          h.id, h.title};
        }
    }
    json body = {
        {"error", {
            {"message", "no such loaded model: '" + model + "'"},
            {"type",    "model_not_found"},
            {"code",    nullptr},
        }},
    };
    res.status = 400;
    res.set_content(body.dump(), "application/json");
    return std::nullopt;
}

} // namespace mimirmind::server