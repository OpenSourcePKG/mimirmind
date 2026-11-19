// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "hugin/McpServer.hpp"

namespace mimirmind::hugin {

class ForwardTools;

/**
 * Hugin training-core tools (roadmap 7.4). 7.4.2:
 *  - train_block_check  builds one dense block (llama / qwen2) from the train
 *    ops with the engine's own (dequantised) weights, compares its forward
 *    with the engine's residual after that block, and checks the recompute
 *    backward (input + LoRA gradients) with directional finite differences.
 */
class TrainTools {
public:
    explicit TrainTools(ForwardTools& forward) : _forward(forward) {}

    void registerOn(McpServer& server);

private:
    ForwardTools& _forward;
};

} // namespace mimirmind::hugin
