// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "server/ApiServer.hpp"

#include "runtime/spec/SpeculativeDecoder.hpp"

#include <httplib.h>

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace mimirmind::runtime {
class InferenceEngine;
}

namespace mimirmind::server {

/// Owns the loaded-engine pool, per-engine dispatch mutexes, and the
/// M9.11.4 spec-dec orchestrator (when applicable), and resolves a
/// chat request's `model` field to a concrete dispatch target.
///
/// The dispatcher is constructed by ApiServer::Impl at startup with
/// the engines the operator loaded. Each engine keeps its own mutex,
/// so requests to different models run in parallel; requests to the
/// same model still serialise on that engine's mutex.
class RequestDispatcher {
public:
    /// Resolved handle used by the chat handler to lock, dispatch,
    /// and optionally route through spec-dec.
    struct Target {
        runtime::InferenceEngine*     engine{nullptr};
        std::mutex*                   mutex{nullptr};
        runtime::SpeculativeDecoder*  spec{nullptr};  // null for extras
        std::string                   id;             // resolved model id
        std::string                   title;
    };

    struct ModelEntry {
        std::string id;
        std::string title;
    };

    /// `engines` must be non-empty. `modelId` (if set) picks the
    /// default engine — must match one of the entries. `draftEngine`
    /// enables spec-dec on the default engine when
    /// `speculativeTargetId` is empty or matches the default id;
    /// otherwise spec-dec stays off with a warning.
    RequestDispatcher(std::vector<LoadedEngine>                  engines,
                       const std::string&                          modelId,
                       runtime::InferenceEngine*                   draftEngine,
                       const std::string&                          speculativeTargetId,
                       const runtime::SpeculativeDecoder::Config&  speculativeConfig);

    RequestDispatcher(const RequestDispatcher&)            = delete;
    RequestDispatcher& operator=(const RequestDispatcher&) = delete;
    RequestDispatcher(RequestDispatcher&&)                 = delete;
    RequestDispatcher& operator=(RequestDispatcher&&)      = delete;

    [[nodiscard]] runtime::InferenceEngine&    defaultEngine() noexcept { return *_defaultEngine; }
    [[nodiscard]] const std::string&           defaultId()     const noexcept { return _defaultId; }
    [[nodiscard]] const std::string&           defaultTitle()  const noexcept { return _defaultTitle; }
    [[nodiscard]] std::mutex&                  defaultMutex()  noexcept { return _defaultMutex; }
    [[nodiscard]] runtime::SpeculativeDecoder* speculativeDecoder() noexcept { return _speculativeDecoder.get(); }
    [[nodiscard]] runtime::InferenceEngine*    draftEngine()   noexcept { return _draftEngine; }

    /// All loaded models in order: default first, then extras.
    [[nodiscard]] std::vector<ModelEntry> listModels() const;

    /// Resolve the request's `model` to a dispatch target. Empty or
    /// matching-default returns the default engine. Unknown model id
    /// sends 400 to `res` and returns nullopt.
    [[nodiscard]] std::optional<Target> resolveTarget(const std::string& model,
                                                       httplib::Response& res);

private:
    struct ExtraHandle {
        std::string                                  id;
        std::string                                  title;
        runtime::InferenceEngine*                    engine{nullptr};
        std::unique_ptr<std::mutex>                  mutex{std::make_unique<std::mutex>()};
    };

    runtime::InferenceEngine*                    _defaultEngine{nullptr};
    std::string                                  _defaultId;
    std::string                                  _defaultTitle;
    std::mutex                                   _defaultMutex;
    runtime::InferenceEngine*                    _draftEngine{nullptr};
    std::unique_ptr<runtime::SpeculativeDecoder> _speculativeDecoder;
    std::vector<ExtraHandle>                     _extraHandles;
};

} // namespace mimirmind::server