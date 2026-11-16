// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "hugin/McpServer.hpp"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <string>

namespace mimirmind::hugin {

class ForwardTools;
class SurgeryTools;

/**
 * Hugin's Rune lifecycle (roadmap 7.3 Inc-2): turns a previewed intervention
 * into a Rune file and walks it through staged -> gated -> committed, with
 * guardrails at every step. The base model file is never touched; a Rune is
 * only ever activated by listing it in a model's `runes` config, which Hugin
 * does NOT do — commit only hands back the config line.
 *
 * Layout under the rune directory:
 *   staged/<name>.rune + .json     authored, not yet (or green-) gated
 *   committed/<name>.rune + .json  green gate, unchanged since the gate
 *   rejected/<name>.rune + .json   red gate (auto-rollback) or discarded
 *   audit.jsonl                    every lifecycle event, append-only
 * The .json sidecar holds status, gate result, the rune file hash the gate
 * ran on, and the event history. The .rune itself is immutable after
 * staging (its own manifest carries the authoring provenance + budget).
 *
 *  - rune_stage_steer   session steering vector -> Steer op (overlay after a
 *                       block); budget: |alpha|·||v|| / mean residual norm
 *  - rune_stage_ablate  zero attention heads via a 0/1 column Scale on
 *                       blk.N.attn_output.weight (F32/F16/BF16, exact Q8_0
 *                       masking); budget: fraction of the block's heads
 *  - rune_gate          goldset eval base vs rune on a target set and an
 *                       optional control set; green/red by thresholds; red
 *                       moves the rune to rejected/ (auto-rollback)
 *  - rune_commit        green + unchanged hash -> committed/
 *  - rune_rollback      committed -> staged (gate cleared), staged -> rejected
 *  - rune_list          every rune with status / gate summary
 */
class RuneStaging {
public:
    RuneStaging(ForwardTools& forward, SurgeryTools& surgery, std::filesystem::path dir,
                std::string huginVersion);

    void registerOn(McpServer& server);

    /// Hard ceilings the per-call budget arguments can never exceed.
    static constexpr double kMaxSteerRelativeNorm = 2.0;
    static constexpr double kMaxHeadFraction      = 0.5;

private:
    enum class State { Staged, Committed, Rejected };

    [[nodiscard]] std::filesystem::path dirOf(State s) const;
    [[nodiscard]] std::filesystem::path runePath(State s, const std::string& name) const;
    [[nodiscard]] std::filesystem::path sidecarPath(State s, const std::string& name) const;
    /// Where `name` currently lives; throws when it does not exist.
    [[nodiscard]] State locate(const std::string& name) const;
    [[nodiscard]] bool  exists(const std::string& name) const;

    nlohmann::json readSidecar(State s, const std::string& name) const;
    void           writeSidecar(State s, const std::string& name, const nlohmann::json& j) const;
    void           move(State from, State to, const std::string& name) const;
    void           audit(const std::string& event, const std::string& name,
                         nlohmann::json details) const;

    /// Base fingerprint of the current target (what the rune binds to).
    [[nodiscard]] std::string baseFingerprint() const;

    /// Writes staged/<name>.rune + sidecar; returns the summary.
    nlohmann::json stage(const std::string& name, const std::string& kind,
                         const std::vector<std::uint8_t>& bytes, nlohmann::json params) const;

    ForwardTools&          _forward;
    SurgeryTools&          _surgery;
    std::filesystem::path  _dir;
    std::string            _version;
};

} // namespace mimirmind::hugin
