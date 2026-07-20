// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "runtime/thermal/ThermalProfile.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace mimirmind::core::config {

// Tri-state feature switch: bench decides, or force one path.
enum class TriState { Auto, Force, Disable };

/**
 * Runtime knobs that shape how a model loads and runs.
 *
 * All fields are `std::optional` at the model level so `models[].runtime`
 * can override just the fields it cares about; the top-level `runtime`
 * fills the gaps. `effective(base, override)` returns a fully-populated
 * struct with the merge applied.
 */
struct RuntimeSettings {
    // "" (empty) / "f32" / "q8_0"
    std::optional<std::string>  kvDtype{};
    std::optional<std::size_t>  maxContextTokens{};
    std::optional<int>          usmProbeTotalGib{};
    std::optional<bool>         preserveThinking{};
    // Where compiled .spv kernels live. Missing → wired-in default.
    std::optional<std::string>  spvDir{};
};

/**
 * One loadable model entry. Multiple entries with `loadOnStart:true` are
 * allowed — main() constructs one InferenceEngine per entry, and the
 * chat/completions dispatch routes on the request's `model` field.
 */
struct ModelEntry {
    std::string       id{};              // OpenAI-facing name (matched against request.model)
    std::string       title{};           // Optional human-readable name for UI dropdowns.
                                         // Falls back to `id` in /v1/models when empty.
    std::string       path{};
    bool              loadOnStart{true};
    RuntimeSettings   runtime{};         // per-model override, merged onto top-level

    /// Backend token — pool selector. Recognised values:
    ///   `""` / unset / `"auto"` — use `BackendPool::select(Auto)`
    ///   `"l0"` / `"l0:0"` / ...   — pin to a LevelZero device
    ///   `"hip"` / `"hip:0"` / ... — pin to a HIP device
    ///   `"cpu"`                   — force the reference CPU backend
    ///
    /// The token must resolve against the pool that
    /// `BackendPool::discoverAll()` produced at startup; unknown tokens
    /// fail loud in `ServeMode`. Enables dual-GPU deployments — e.g.
    /// target model on `"hip:0"` (dGPU) while a small draft sits on
    /// `"l0:0"` (iGPU) in the same process.
    std::string       backend{};
};

struct LogSettings {
    std::string file{};                  // empty → stderr
    std::string level{"info"};           // debug | info | warn | error
};

struct ServerSettings {
    int         port{8080};
    LogSettings log{};
};

struct FeatureSettings {
    // Command-list replay. Auto-off for MoE remains hardcoded regardless of this.
    bool                       clr{true};
    bool                       flashPrefill{true};
    // GQA-head-packed Q8_0 prefill kernel. Default false: A/B on both
    // prod models (2026-07-12) showed a monotonic regression scaling
    // with N_MAX register-waste ratio (26B-A4B nQPerKv=2 → 3.7 %, E4B
    // nQPerKv=4 → 1.8 %). The kernel statically dimensions register
    // arrays on ATTN_FLASH_PREFILL_N_Q_PER_KV_MAX=8, so any model with
    // nQPerKv < 8 pays for unused registers. A reconstructed v2 with
    // compile-time N_MAX specialisation would flip this back on — until
    // then, plain per-Q-head Q8_0 wins. Suffix ties the flag to the
    // Q8_0-specific kernel; future dtype variants get their own.
    bool                       flashPrefillGqaQ8{false};
    // K-tile size baked into the Q8_0 GQA prefill kernel. Compile-time
    // constant in the .cl source; a second SPV is built alongside the
    // default with `-D ATTN_FLASH_PREFILL_KTILE=64`. Runtime picks
    // between the two based on this value. Valid: {0, 64, 128}.
    //   0   — autotune (pick per host at startup). NOT YET RECONSTRUCTED
    //         — the plumbing exists but the bench loop is a follow-up.
    //         Falls back to 128 for now with a startup log warning.
    //   64  — pin to KTILE=64 variant.
    //   128 — pin to KTILE=128 variant (M5i.J default).
    std::size_t                flashPrefillKTileQ8{128};
    bool                       fusedQkv{true};
    bool                       moeGroup{true};
    TriState                   gemm{TriState::Auto};
    bool                       gemmV2{false};
    // When set: pin the crossover threshold, skip the bench.
    std::optional<std::size_t> gemmMinM{};
    // Default is Disable per lesson_dp4a_autotune_prod_hazard.
    TriState                   dp4a{TriState::Disable};
    // M8.K.Q8_0-Reorder — routes Q8_0 matvec through the reordered-
    // layout kernel (matmul_q8_0_vec_reorder) which reads scales and
    // quants as two contiguous regions instead of the native ggml
    // 34-byte-block interleave. Default is Disable — until the load-
    // time weight-reorder pass lands (Phase 4 of the M8.K track), the
    // reorder kernel has no reordered weights to consume so enabling
    // this alone is a no-op. Auto flips to the reorder path once the
    // weight preprocess is in place; Enable forces it (fails loud if
    // the weights weren't reordered). See kernels/matmul_q8_0_vec
    // _reorder.cl and Q8_0::reorderRow for the layout contract.
    TriState                   q8_0Reorder{TriState::Disable};
    // M-MoE.Fused-Decode prototype. Routes the T=1 MoE decode path
    // through the fused-K down-projection kernel
    // (`moe_down_fused_k_q6k`) instead of the K sequential
    // down-matmul + scaledAdd dispatches. Currently Q6_K-only —
    // silently disabled for any expert weight type != Q6_K, or when
    // the kernel didn't load. Disable-default until an A/B on
    // L0_TARGET_HOST confirms a signal.
    //   Disable — always sequential
    //   Auto    — use fused when kernel loaded and weight is Q6_K
    //   Force   — same as Auto today; a future gating condition can
    //             differ (kept distinct for symmetry with q8_0Reorder).
    TriState                   moeFusedDown{TriState::Disable};
};

struct SpeculativeSettings {
    /// Source of speculative draft tokens. `Model` uses a second loaded
    /// `InferenceEngine` (classic naive draft — needs `draft` set and a
    /// vocab-compatible model). `NGram` uses in-context Prompt-Lookup
    /// Decoding — no second model, no vocab check, works on any target.
    enum class Drafter { Model, NGram };

    bool        enabled{false};
    Drafter     drafter{Drafter::Model};
    std::string target{};                // model id in `models[]`
    std::string draft{};                 // model id in `models[]` (Drafter::Model only)
    int         n{4};

    /// Prompt-Lookup Decoding tuning knobs. Bounded to [1, 32] in the
    /// parser. Ignored when drafter != NGram.
    int         ngramMinK{2};
    int         ngramMaxK{3};
};

struct FanSettings {
    // Positive knob: true = install the FanController if the host
    // exposes writable PWM knobs; false = never install.
    bool                boost{true};
    std::optional<int>  pwmBoost{};
    std::optional<int>  pwmMin{};
};

struct GovernorSettings {
    // "rp0" | "rpn" | "<numeric MHz>" | "off" | "0"
    std::optional<std::string>  gpuClockPin{};
    // Enable per-tick NDJSON sink. When true, the governor appends one
    // JSON line per tick to `tickLogFile`. Kept as a separate boolean
    // (not implied by a non-empty path) so an operator can disable the
    // sink without deleting the archived file path from config.json.
    bool                        tickLog{false};
    // Path for the tick sink (M9.6.6.0). Falls back to
    // `diagnostics.traceDecodeFile` for one release with a deprecation
    // warning — that reuse conflates decode-trace and governor-tick
    // streams, which the M9.6.6.1 baseline runbook consumes separately.
    std::string                 tickLogFile{};
    FanSettings                 fan{};
    // Full thermal-profile struct inlined — no more separate JSON file.
    // Empty `name` means "no profile" and the guard runs unprotected.
    ::mimirmind::runtime::ThermalProfile thermal{};
};

struct DiagnosticsSettings {
    // Path prefix for parity-dump artifacts. Empty → disabled.
    // Warning: enabling in prod wrote 7 GiB per 3.4k-token prefill in the past.
    std::string parityDump{};
    bool        traceBlock0{false};
    std::string traceDecodeFile{};       // empty → off
    bool        traceOpTimes{false};
    bool        gpuBench{false};
    // Default on; explicit false disables the detector installer.
    bool        regressionAlert{true};
};

/**
 * Serving-class knobs for Bragi (Mimir-2.0). Controls whether
 * PagedAttention + Continuous Batching (M-Cuda.Batch) is activated at
 * startup, gated by the HW-capacity probe (M-Startup.CapacityProbe).
 *
 * Server-side decision only — no user-per-request toggle
 * (`feedback_no_user_toggles`). Rate-limits, fairness, adapter-switch
 * are Bragi follow-ups, not exposed here.
 */
struct ServingSettings {
    // How to decide whether PagedAttention + Continuous Batching runs
    // at this instance:
    //   Auto    — HW-capacity probe decides (sustainableBatch >= minBatchForEnable)
    //   Force   — always enable, warn on-startup if probe below minBatch
    //   Disable — never enable, keep single-session semantics
    // Auto is the default; single-session mimirmind consumers see no
    // behaviour change unless they explicitly opt in.
    TriState        enableBatching{TriState::Auto};

    // When enableBatching=Auto, the minimum sustainable batch (rounded
    // to a scheduler step by BatchCapacityProbe::roundToSchedulerStep)
    // at which serving-class features light up. Below this the
    // instance stays single-session even on capable HW — reflects
    // the "batching-overhead pays off around B=8" empirical threshold.
    std::size_t     minBatchForEnable{8};
};

/**
 * Root config, loaded once from `config.json` at startup.
 *
 * Precedence: CLI flags > config.json > compiled defaults.
 *
 * Plain-data value type. Owned by `main()` and passed by `const&` to every
 * consumer (InferenceEngine, GpuMatmul, KvCache, ...) — no global state,
 * no singleton. Testable end-to-end by constructing a Config on the stack.
 */
struct Config {
    std::string                defaultModel{};    // id; empty → models[0]
    std::vector<ModelEntry>    models{};
    ServerSettings             server{};
    RuntimeSettings            runtime{};
    FeatureSettings            features{};
    SpeculativeSettings        speculative{};
    GovernorSettings           governor{};
    DiagnosticsSettings        diagnostics{};
    ServingSettings            serving{};

    // Resolve the effective runtime for a given model id: top-level defaults,
    // then per-model overrides applied.
    [[nodiscard]] RuntimeSettings effectiveRuntime(std::string_view modelId) const;

    // Lookup a model by id. Throws std::runtime_error if not found.
    [[nodiscard]] const ModelEntry& model(std::string_view id) const;

    // Default model (per `defaultModel`, or first entry). Throws if `models` is empty.
    [[nodiscard]] const ModelEntry& defaultModelEntry() const;
};

/**
 * Load config.json. Fail-fast on:
 *   - missing / unreadable file
 *   - invalid JSON
 *   - unknown fields (typo protection)
 *   - missing required fields (`models` must be non-empty; each model
 *     needs `id` and `path`)
 *   - unresolved `speculative.target` / `.draft` (must reference existing model ids)
 */
[[nodiscard]] Config loadConfig(std::string_view path);

/**
 * CLI overrides applied on top of the loaded Config. Unset fields leave
 * the JSON value in place.
 */
struct CliOverrides {
    // Overrides `models[<defaultModel>].path`.
    std::optional<std::string>  modelPath{};
    std::optional<int>          port{};
    std::optional<std::string>  logFile{};
    std::optional<std::string>  logLevel{};
};

// Mutate cfg in place. If `modelPath` is set but `models` is empty, a synthetic
// single-entry list is created with id "primary". Errors on inconsistent state
// (e.g. --port <= 0).
void applyCliOverrides(Config& cfg, const CliOverrides& cli);

// Compute the effective runtime by merging `override` onto `base`. Fields set
// in `override` win; unset (nullopt) fields inherit from `base`.
[[nodiscard]] RuntimeSettings mergeRuntime(const RuntimeSettings& base,
                                           const RuntimeSettings& override_);

} // namespace mimirmind::core::config