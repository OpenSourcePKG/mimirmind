// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "server/RequestDispatcher.hpp"

#include "runtime/spec/Drafter.hpp"
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
    runtime::Drafter*                           drafter,
    const std::string&                          speculativeTargetId,
    const runtime::SpeculativeDecoder::Config&  speculativeConfig)
    : _defaultEngine{&pickDefault(engines, modelId)},
      _drafter{drafter} {

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

    // M9.11.4 — stand up the spec-dec orchestrator now that the
    // drafter is ready. Spec-dec is currently wired only for the
    // default engine — if `speculativeTargetId` names an ExtraHandle
    // instead, spec-dec stays off with a warning so the operator can
    // fix the config.
    if (_drafter != nullptr) {
        if (speculativeTargetId.empty() || speculativeTargetId == _defaultId) {
            _speculativeDecoder =
                std::make_unique<runtime::SpeculativeDecoder>(
                    *_defaultEngine, *_drafter, speculativeConfig);
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
    // M-Munin.3 (full): the default model stays eager even when a provider
    // is set (ServeMode keeps ONE anchor model resident for thermal/power/
    // fan/perf monitoring and the draft-model vocab check) — list it plus
    // any eager extras, THEN the provider's pool-managed models. A provider
    // model id colliding with the default is deduped (shouldn't happen —
    // ServeMode never registers the default with the pool — but listModels
    // must stay correct even if a config does something unexpected).
    std::vector<ModelEntry> out;
    out.reserve(1 + _extraHandles.size());
    out.push_back({_defaultId, _defaultTitle});
    for (const auto& h : _extraHandles) {
        out.push_back({h.id, h.title});
    }
    if (_provider != nullptr) {
        for (const auto& m : _provider->listModels()) {
            if (m.id == _defaultId) continue;
            out.push_back({m.id, m.title});
        }
    }
    return out;
}

std::vector<ResidentModelMemory> RequestDispatcher::residentModelsMemory() const {
    std::vector<ResidentModelMemory> out;
    // Eager anchor(s): the always-resident default engine + co-resident extras.
    // memoryTelemetry() is a fast counter/mem-info read; it takes no generate
    // lock, so this never blocks in-flight decoding.
    const auto addEng = [&out](const std::string& id, const std::string& title,
                               bool isDefault, runtime::InferenceEngine* eng) {
        if (eng == nullptr) {
            return;
        }
        const auto mt = eng->memoryTelemetry();
        out.push_back(ResidentModelMemory{id, title, isDefault, mt.weightBytes,
                                          mt.servingActive, mt.kvResidentBytes,
                                          mt.kvNumBlocks});
    };
    addEng(_defaultId, _defaultTitle, /*isDefault=*/true, _defaultEngine);
    for (const auto& h : _extraHandles) {
        addEng(h.id, h.title, /*isDefault=*/false, h.engine);
    }
    // Pool mode: append the provider's currently-materialized slots, deduped
    // against the eager default (mirrors listModels()).
    if (_provider != nullptr) {
        for (auto& r : _provider->residentModels()) {
            if (r.id == _defaultId) {
                continue;
            }
            out.push_back(std::move(r));
        }
    }
    return out;
}

std::size_t RequestDispatcher::poolCapacity() const {
    return _provider != nullptr ? _provider->poolCapacity() : std::size_t{1};
}

std::optional<RequestDispatcher::Target> RequestDispatcher::resolveTarget(
    const std::string& model, httplib::Response& res) {
    // M-Munin.3 (full): check the eager default + extras FIRST (unchanged
    // behavior, zero cost), THEN fall back to the provider (pool mode) for
    // anything else. ServeMode keeps the default model eager even when a
    // pool is active — only non-default chat models register with the
    // provider — so this ordering means "empty or default id" never touches
    // the provider at all, exactly like the no-provider case.
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

    // M-Munin.3 pool mode: resolve through the provider, which materializes
    // (and may evict the LRU slot) on a miss. The returned Target carries a
    // `pin` that keeps the slot alive for the request, plus this slot's own
    // batcher/spec-dec decoder (built per-slot by the provider's factory —
    // see AttachedModelProvider).
    if (_provider != nullptr && _provider->knows(model)) {
        try {
            auto acq = _provider->acquire(model);
            if (!acq) {
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
            return Target{acq->engine, acq->mutex, acq->spec,
                          acq->id, acq->title, acq->batcher,
                          std::move(acq->pin)};
        } catch (const std::exception& x) {
            MM_LOG_ERROR("server",
                         "model '{}' materialization failed: {}", model, x.what());
            json body = {
                {"error", {
                    {"message", std::string{"model unavailable: "} + x.what()},
                    {"type",    "model_unavailable"},
                    {"code",    nullptr},
                }},
            };
            res.status = 503;
            res.set_content(body.dump(), "application/json");
            return std::nullopt;
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