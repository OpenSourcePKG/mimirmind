// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "hugin/McpServer.hpp"

#include <memory>
#include <string>

namespace mimirmind::hugin {

class ForwardTools;
class RuneStaging;
namespace train {
class TrainModel;
}

/**
 * Hugin training-core tools (roadmap 7.4). 7.4.2:
 *  - train_block_check  builds one dense block (llama / qwen2) from the train
 *    ops with the engine's own (dequantised) weights, compares its forward
 *    with the engine's residual after that block, and checks the recompute
 *    backward (input + LoRA gradients) with directional finite differences.
 * 7.4.3:
 *  - train_lora_sft     LoRA-SFT of the whole dense model (llama / qwen2) on
 *    prompt/response pairs (responses may be self-distilled from a teacher
 *    prompt by the base model), staged as a LowRank Rune, with a parity check
 *    of trainer logits vs the engine with that Rune applied by Futhark.
 */
class TrainTools {
public:
    TrainTools(ForwardTools& forward, RuneStaging& staging);
    ~TrainTools();

    TrainTools(const TrainTools&)            = delete;
    TrainTools& operator=(const TrainTools&) = delete;

    void registerOn(McpServer& server);

private:
    /// The whole model as train ops (dequantised once, cached per target).
    train::TrainModel& model();

    ForwardTools&                      _forward;
    RuneStaging&                       _staging;
    std::unique_ptr<train::TrainModel> _model;
    std::string                        _modelKey;
};

} // namespace mimirmind::hugin
