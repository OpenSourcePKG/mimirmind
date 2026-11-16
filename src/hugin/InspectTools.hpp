// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "hugin/McpServer.hpp"
#include "hugin/ModelView.hpp"

#include <memory>
#include <string>

namespace mimirmind::hugin {

/**
 * Hugin's weight-level inspection tools (roadmap 7.2 Inc-1 — no forward pass,
 * no GPU). Registers on an McpServer:
 *
 *  - open_model     load a GGUF for inspection (replaces the current one)
 *  - model_info     path, architecture, base fingerprint, tensor/type summary
 *  - list_tensors   names, types, dims, bytes, patchable-by-rune flag
 *  - tensor_stats   mean/std/min/max/abs-max/L2/NaN/Inf + per-row norms
 *  - tensor_hash    the content hash Runes bind each op to
 *  - rune_inspect   decoded manifest of a rune file
 *  - rune_check     Futhark dry run of rune files against the loaded model
 *                   (exactly the engine's checks; nothing is written)
 */
class InspectTools {
public:
    /// `initialModel` may be empty (then open_model must be called first).
    /// `initialModel` may be a GGUF file or an NVFP4 checkpoint directory.
    explicit InspectTools(const std::string& initialModel, std::string tokenizerGguf = {});

    void registerOn(McpServer& server);

    /// Path of the currently opened model ("" = none): a GGUF file or an
    /// NVFP4 checkpoint directory. The forward tools follow it, so open_model
    /// retargets them too.
    [[nodiscard]] std::string modelPath() const {
        return _model ? _model->path() : _checkpointDir;
    }
    /// Tokenizer GGUF for an NVFP4 checkpoint ("" = the checkpoint's own).
    [[nodiscard]] const std::string& tokenizerGguf() const noexcept { return _tokenizerGguf; }

private:
    [[nodiscard]] const ModelView& model() const;

    void open(const std::string& path, std::string tokenizerGguf);

    std::unique_ptr<ModelView> _model;
    std::string                _checkpointDir;   // NVFP4: forward tools only
    std::string                _tokenizerGguf;
};

} // namespace mimirmind::hugin
