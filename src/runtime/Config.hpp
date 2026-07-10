#pragma once

#include "runtime/ThermalProfile.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace mimirmind::runtime {

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
    // Independent rollback for the GQA-head-packed Q8_0 prefill kernel.
    // Set to false to fall back to the plain per-Q-head Q8_0 kernel when
    // the packed variant regresses on a given host.
    bool                       flashPrefillGqa{true};
    bool                       fusedQkv{true};
    bool                       moeGroup{true};
    TriState                   gemm{TriState::Auto};
    bool                       gemmV2{false};
    // When set: pin the crossover threshold, skip the bench.
    std::optional<std::size_t> gemmMinM{};
    // Default is Disable per lesson_dp4a_autotune_prod_hazard.
    TriState                   dp4a{TriState::Disable};
};

struct SpeculativeSettings {
    bool        enabled{false};
    std::string target{};                // model id in `models[]`
    std::string draft{};                 // model id in `models[]`
    int         n{4};
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
    ThermalProfile              thermal{};
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

} // namespace mimirmind::runtime