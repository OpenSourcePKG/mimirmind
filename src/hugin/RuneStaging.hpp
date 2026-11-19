// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/gguf/GgufTypes.hpp"
#include "core/rune/RuneFormat.hpp"
#include "hugin/McpServer.hpp"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

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
 *                       blk.N.attn_output.weight; budget: fraction of the
 *                       block's heads
 *  - rune_stage_heads   scale attention heads of one or more blocks (0 =
 *                       ablate, > 1 = amplify) via column Scales on the
 *                       O-projections (F32/F16/BF16, exact block scales on
 *                       Q8_0 / FP8_E4M3 — GGUF and NVFP4 checkpoints);
 *                       budget: head fraction per block + max scale
 *  - rune_stage_experts disable MoE experts of one block via a router Mask op
 *                       (engine overlay: logit -> -inf before top-K, any
 *                       weight format); budget: fraction of the block's experts
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
    /// 7.12: steer dose above this needs an 'override_reason' (related work:
    /// every config that moved > ~50 % of a layer's residual broke the model).
    static constexpr double kSteerDoseReasonFloor = 0.5;
    static constexpr double kMaxHeadFraction      = 0.5;
    static constexpr double kMaxExpertFraction    = 0.25;
    static constexpr double kMaxHeadScale         = 16.0;

    /// 7.4.3: one trained LoRA adapter on projection `target`
    /// ("blk.<i>.<tensor>"): A [rank, in], B [out, rank], scale alpha.
    struct LowRankSpec {
        std::string        target;
        std::uint32_t      rank{0};
        double             alpha{1.0};
        std::vector<float> a, b;
    };
    /// Writes the adapters as LowRank overlay ops into staged/<name>.rune
    /// (bound to the base fingerprint); `params` = training provenance.
    nlohmann::json stageLowRank(const std::string& name, const std::vector<LowRankSpec>& specs,
                                nlohmann::json params) const;

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

    /// Scale `heads` of `block` by `scale` (0 = ablate).
    struct HeadScale {
        std::uint32_t            block{0};
        std::vector<std::size_t> heads;
        double                   scale{0};
    };
    /// A head-scale rune before naming / provenance (shared by staging and
    /// edit_search trial runes); validates budgets from `a`, sorts `specs`.
    struct HeadRune {
        core::rune::RuneManifest                   manifest;
        std::map<std::string, std::vector<float>>  payloads;
        nlohmann::json                             edits;
        std::size_t                                heads{0};
        double                                     maxFrac{0};
        double                                     maxScale{0};
    };
    HeadRune buildHeadRune(const nlohmann::json& a, std::vector<HeadScale>& specs) const;
    /// Writes <dir>/search/<tag>.rune (no gates, no staging); returns its path.
    std::string writeSearchHeadRune(std::vector<HeadScale> specs, const std::string& tag) const;
    struct HeadCache {
        struct Tensor {
            core::gguf::GgmlType type{};
            std::uint64_t  cols{0};
            std::string    hash;
        };
        std::string                             key;
        std::string                             fingerprint;
        std::size_t                             heads{0};
        std::uint32_t                           blocks{0};
        std::map<std::uint32_t, Tensor>         tensors;
    };
    mutable HeadCache _headCache;

    /// Column-Scale rune over blk.N.attn_output.weight for every spec, bound
    /// to the target bytes the engine actually holds; budgets from `a`.
    nlohmann::json stageHeadScales(const nlohmann::json& a, const std::string& name,
                                   const std::string& kind, std::vector<HeadScale> specs) const;

    /// Writes staged/<name>.rune + sidecar; returns the summary.
    nlohmann::json stage(const std::string& name, const std::string& kind,
                         const std::vector<std::uint8_t>& bytes, nlohmann::json params) const;

    ForwardTools&          _forward;
    SurgeryTools&          _surgery;
    std::filesystem::path  _dir;
    std::string            _version;
};

} // namespace mimirmind::hugin
