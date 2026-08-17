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
 * On-disk weight format for a model entry.
 *   Auto  — infer from `path`: a directory with a
 *           `model.safetensors.index.json` (or a single `model.safetensors`)
 *           is Nvfp4/ModelOpt; anything else is Gguf.
 *   Gguf  — a `*.gguf` file (the L0 / HIP / CPU path).
 *   Nvfp4 — a ModelOpt NVFP4 safetensors checkpoint directory (CUDA/Bragi).
 * The concrete filesystem probe for `Auto` lives in the load path, not the
 * config parser — Config stays filesystem-free.
 */
enum class ModelFormat { Auto, Gguf, Nvfp4 };

/// Parse a `format` string ("auto"|"gguf"|"nvfp4") or return nullopt.
[[nodiscard]] std::optional<ModelFormat> modelFormatFromString(std::string_view s) noexcept;

/// Canonical lower-case name for a format ("auto"|"gguf"|"nvfp4").
[[nodiscard]] std::string_view modelFormatName(ModelFormat f) noexcept;

/**
 * What a model entry serves. `Chat` = the autoregressive decoder behind
 * /v1/chat/completions (the default). `Rerank` = a bidirectional cross-encoder
 * (EncoderRunner) behind /v1/rerank — no KV cache, no sampler, one score per
 * (query, document) pair. `Embed` = a bi-encoder embedding model (same encoder,
 * no classifier head) behind /v1/embeddings — CLS-pooled, L2-normalized vector
 * per input text.
 */
enum class ModelTask { Chat, Rerank, Embed };

/// Parse a `task` string ("chat"|"rerank"|"embed") or return nullopt.
[[nodiscard]] std::optional<ModelTask> modelTaskFromString(std::string_view s) noexcept;

/// Canonical lower-case name for a task ("chat"|"rerank"|"embed").
[[nodiscard]] std::string_view modelTaskName(ModelTask t) noexcept;

/**
 * One loadable model entry. Multiple entries with `loadOnStart:true` are
 * allowed — main() constructs one InferenceEngine per entry, and the
 * chat/completions dispatch routes on the request's `model` field.
 */
struct ModelEntry {
    std::string       id{};              // OpenAI-facing name (matched against request.model)
    std::string       title{};           // Optional human-readable name for UI dropdowns.
                                         // Falls back to `id` in /v1/models when empty.
    std::string       path{};            // GGUF file, or NVFP4 checkpoint directory
    ModelFormat       format{ModelFormat::Auto};
    // What this entry serves: chat (default, decoder) or rerank (cross-encoder).
    // A rerank entry loads a dense F32 XLM-R safetensors dir (EncoderModel) and
    // is exposed under /v1/rerank instead of /v1/chat/completions.
    ModelTask         task{ModelTask::Chat};
    // For `format: nvfp4`: path to a GGUF whose tokenizer to reuse (the NVFP4
    // checkpoint ships only an HF tokenizer.json, which we don't parse yet).
    // Ignored for GGUF models (their tokenizer comes from the file itself).
    std::string       tokenizerGguf{};
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

/// In-process TLS termination (OpenSSL via httplib SSLServer).
struct TlsSettings {
    /// `std::nullopt` → apply the secure-by-default (true). Explicit value
    /// wins.
    std::optional<bool> enabled{};
    /// PEM certificate path. Empty → auto self-signed next to the config.
    std::string         certFile{};
    /// PEM private-key path. Empty → auto self-signed next to the config.
    std::string         keyFile{};
};

/// Bearer-token API-key auth.
struct AuthSettings {
    /// `std::nullopt` → bind-address-aware default (off on loopback, on when
    /// bound to a non-loopback interface). Explicit value wins.
    std::optional<bool>      enabled{};
    /// Auto-generate + persist a key when auth is on and none is configured.
    bool                     autoGenerateKey{true};
    /// Explicit keys. Each entry is `key` or `name:key[:tenantId]`.
    std::vector<std::string> keys{};
    /// Keyfile path. Empty → default `<config-dir>/mimir-apikeys.txt`.
    std::string              keyFile{};
};

/// Per-tenant (per-API-key) usage metrics for the admin routes.
struct MetricsSettings {
    /// Aggregate per-tenant request/token counters and expose them on the
    /// operator-only routes (`/v1/admin/tenants`, `/metrics`). On by default;
    /// the recording is a cheap map update off the request path.
    bool        enabled{true};
    /// Persistence file so totals survive a restart. Empty → default
    /// `<config-dir>/mimir-tenant-metrics.json`. Set to override the location.
    std::string path{};
};

struct ServerSettings {
    int             port{8080};
    LogSettings     log{};
    TlsSettings     tls{};
    AuthSettings    auth{};
    MetricsSettings metrics{};
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

    // ---- M-Cuda.Batch Phase C knobs ---------------------------------
    //
    // Consumed by RequestScheduler / ChunkedPrefillScheduler /
    // PreemptionPolicy / PagedKvBlockAllocator at Phase-D construction
    // time. Defaults match the corresponding class-level defaults so a
    // config with only `enableBatching: "force"` produces the same
    // behaviour as constructing the classes with their vanilla defaults.

    // Tokens per iteration (Sarathi-Serve style). Drives
    // `ChunkedPrefillScheduler::tokenBudget`. Balance-point: too small
    // → too many scheduler-cycles; too large → decode-first-priority
    // loses effect because prefill fills the batch before decode ever
    // gets a look-in. 512 is vLLM V1 default. Range 1..8192.
    std::size_t     tokenBudget{512};

    // Concurrent (Prefilling + Decoding) request cap. Admissions
    // beyond this stay in the Waiting queue until an existing request
    // Completes or Preempts. Drives `RequestScheduler::maxActiveRequests`.
    // 32 matches the Bragi-v1 target (per M-Cuda.Batch note) —
    // realistic Spark load at 200-500 tok/s. Range 1..256.
    std::size_t     maxActiveRequests{32};

    // Per-tenant slice of the accepted-but-unfinished budget: the most
    // concurrent (running + queued) requests a SINGLE API-key tenant may
    // hold at once in the ContinuousBatcher. Stops one caller from
    // monopolising the whole `maxActiveRequests` pool and starving
    // co-tenants under load; a submit past it is shed with a 429 (tenant
    // quota) rather than a 503 (whole-server overload). Only meaningful
    // when auth is enabled (tenant = key `tenantId`); requests with an
    // empty tenant label (auth off) are never per-tenant limited and share
    // one bucket. 0 = off (default) so single-tenant / Pegenaut deployments
    // keep the full budget and see no behaviour change. Range 0..256.
    std::size_t     maxActiveRequestsPerTenant{0};

    // Paged-KV free-block ratio below which the preemption policy
    // triggers `RequestScheduler::preemptOne()`. Drives
    // `PreemptionPolicy::freeBlockThreshold`. 0.05 = 5% headroom
    // before pressure — matches vLLM's `gpu_memory_utilization=0.95`.
    // Range [0.0, 1.0]; 0.0 disables preemption, 1.0 is maximally-
    // eager. Real tuning happens with Spark load-test numbers.
    double          preemptFreeBlockThreshold{0.05};

    // Tokens per paged KV block. Drives
    // `PagedKvBlockAllocator::blockSize` and paged-attention kernel
    // `block_size` argument. vLLM-consistent default 16; also legal:
    // 8, 32. Larger blocks reduce table-overhead but increase
    // fragmentation on short sequences. Must be a power of 2 for
    // future prefix-cache alignment work.
    std::size_t     blockSize{16};
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