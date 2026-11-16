// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "hugin/McpServer.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace mimirmind::hugin {

class ForwardTools;

/**
 * Hugin's forward-only surgery tools, preview stage (roadmap 7.3 Inc-1).
 * Locate where a behaviour lives and preview an intervention on the engine's
 * own forward — nothing here writes a weight or a Rune; the base model and
 * the serving engine are untouched. Interventions use the engine's 7.3 write
 * hooks (InferenceEngine::ForwardProbe residual / stage edits), active for
 * one forward only.
 *
 *  - causal_trace    activation patching: clean vs corrupt prompt, restore the
 *                    clean residual at (block, position) in the corrupt run,
 *                    report how much of the clean answer comes back
 *  - steer_vector    build a steering direction at one block from contrastive
 *                    prompt sets (mean residual difference, last position)
 *  - steer_preview   next-token shift (KL, top-k) and optional greedy text,
 *                    base vs steered
 *  - ablate_preview  zero attention heads of one block (pre O-projection):
 *                    shift, or a per-head scan ranked by effect
 *  - generate        greedy continuation with optional steer / ablate
 *  - steer_list      steering vectors of this session
 *
 * Steering vectors live in the session and are dropped when open_model
 * switches the target.
 */
class SurgeryTools {
public:
    explicit SurgeryTools(ForwardTools& forward);

    void registerOn(McpServer& server);

    struct SteerVector {
        std::uint32_t      block{0};
        std::vector<float> values;
        double             norm{0};
        double             residualNorm{0};   // mean ||h|| at that block over the inputs
        std::size_t        positives{0};
        std::size_t        negatives{0};
    };

private:
    const SteerVector& vector(const std::string& name);
    void               dropStaleVectors();

    ForwardTools&                        _forward;
    std::string                          _vectorsKey;
    std::map<std::string, SteerVector>   _vectors;
};

} // namespace mimirmind::hugin
