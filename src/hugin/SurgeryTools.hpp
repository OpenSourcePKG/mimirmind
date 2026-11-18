// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "hugin/McpServer.hpp"
#include "core/rune/RuneApplier.hpp"
#include "core/rune/RuneFile.hpp"

#include <cstdint>
#include <map>
#include <utility>
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
 *  - ablate_preview  zero (or scale by 'scale') attention heads of one block
 *                    (pre O-projection): shift, or a per-head scan ranked by
 *                    effect
 *  - generate        greedy continuation with optional steer / head edits
 *  - eval_preview    goldset eval (as `eval`) with steer / head edits applied
 *                    to every forward, against the cached base result
 *  - steer_list      steering vectors of this session
 *  - condition_vector  (7.11) fit a linear probe on the normalised residual at
 *                    one block (positive vs negative prompts, at the last /
 *                    every / generated positions): a unit direction + cosine
 *                    threshold, the condition of a conditional steer
 *  - steer_if_preview  (7.11) a steer (add or projection), optionally gated
 *                    per token on a condition, installed through the SAME
 *                    engine overlay a steer_if / steer_project Rune uses;
 *                    base vs edited greedy text (KV-cached serving decode),
 *                    gate fire rates, optional goldset eval
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

    /// Session steering vector by name; throws when unknown / model changed.
    const SteerVector& steerVector(const std::string& name) { return vector(name); }

    /// 7.11 condition of a conditional steer: rows with cos(h_block, direction)
    /// > threshold are gated on.
    struct ConditionVector {
        std::uint32_t      block{0};
        std::vector<float> direction;   ///< unit [d_model]
        double             threshold{0};
        std::string        positions;   ///< "last" | "all" | "generated"
        std::size_t        positiveRows{0};
        std::size_t        negativeRows{0};
        double             trainAccuracy{0};
        double             valAccuracy{0};
    };
    const ConditionVector& conditionVector(const std::string& name);

    /// {vector, alpha, mode, condition?, threshold?} -> the Rune op + payloads
    /// of a projection / conditional steer. rune_stage_steer_if writes exactly
    /// this; steer_if_preview resolves it through the real RuneApplier
    /// (condSteerArg), so preview and staged Rune install the same overlay.
    struct CondSteerSpec {
        core::rune::RuneOp                         op;
        std::map<std::string, std::vector<float>>  payloads;
        nlohmann::json                             params;
    };
    CondSteerSpec             condSteerSpec(const nlohmann::json& a);
    core::rune::RuneCondSteer condSteerArg(const nlohmann::json& a);

private:
    void               registerConditionalOn(McpServer& server);
    const SteerVector& vector(const std::string& name);
    void               dropStaleVectors();

    ForwardTools&                        _forward;
    std::string                          _vectorsKey;
    std::map<std::string, SteerVector>   _vectors;
    std::map<std::string, ConditionVector> _conditions;
    /// eval_preview base results per goldset path (dropped with the vectors).
    std::map<std::string, std::pair<nlohmann::json, nlohmann::json>> _baseEvals;
};

} // namespace mimirmind::hugin
