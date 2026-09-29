// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace mimirmind::runtime {
class InferenceEngine;
class Drafter;
class SystemMonitor;
class ThermalGuard;
class PowerMonitor;
class PerfRegressionDetector;
struct ComputeStack;
} // namespace mimirmind::runtime

namespace mimirmind::runtime::serving {
class ContinuousBatcher;
} // namespace mimirmind::runtime::serving

namespace mimirmind::runtime::encoder {
class RerankEngine;
class EmbedEngine;
class DecideEngine;
} // namespace mimirmind::runtime::encoder

namespace mimirmind::runtime::audio {
class AudioEngine;
class SpeakEngine;
} // namespace mimirmind::runtime::audio

namespace mimirmind::server {
struct LoadedEngine;
struct LoadedReranker;
struct LoadedEmbedder;
struct LoadedDecider;
struct LoadedTranscriber;
struct LoadedSpeaker;
struct ServerConfig;
class AttachedModelProvider;
} // namespace mimirmind::server

namespace mimirmind::core::config {
struct Config;
struct ModelEntry;
} // namespace mimirmind::core::config

namespace mimirmind::core::backend {
class BackendPool;
} // namespace mimirmind::core::backend

namespace mimirmind::cli {

struct CliArgs;

/**
 * Boot-time construction helpers for `mimirmind serve`. Each method builds
 * one cohesive slice of the server's long-lived state and RETURNS the owning
 * objects, so `runServe` keeps them as locals and their RAII lifetime /
 * destruction order stay anchored in the serve frame (the monitors and the
 * drafter must outlive `ApiServer::run`). This keeps `runServe` a readable
 * orchestrator instead of a multi-thousand-line boot body.
 */
class ServerBootstrap {
public:
    /**
     * Optional speculative-decoding drafter, built from `cfg.speculative`
     * against the already-loaded default/target engine.
     *
     * `draftEngine` MUST outlive `drafter` — a `ModelDrafter` holds a
     * reference into `*draftEngine`, so the member order here is load-bearing:
     * on destruction `drafter` (declared last) is torn down first, then
     * `draftEngine`. Both are null when speculative decoding is disabled or a
     * drafter could not be constructed (vocab mismatch / load failure); the
     * method logs the reason and degrades to no speculation.
     */
    struct SpeculativeSetup {
        std::unique_ptr<runtime::InferenceEngine> draftEngine;
        std::unique_ptr<runtime::Drafter>         drafter;
    };

    [[nodiscard]] static SpeculativeSetup buildSpeculative(
        const core::config::Config&    cfg,
        const runtime::InferenceEngine& targetEngine);

    /**
     * Process-wide thermal / power / fan ancillaries, built from
     * `cfg.governor` + `cfg.diagnostics` and wired into `engine`.
     *
     * Returns the owning objects that `runServe` must keep alive past
     * `ApiServer::run` (they are consulted by every engine's generate()
     * through the non-owning setters below). Member order is load-bearing:
     * `monitor` is declared before `guard` because `ThermalGuard` holds a
     * `SystemMonitor&`, so `monitor` must outlive `guard` (declared-first =
     * destroyed-last). The GpuClockGovernor and FanController are kept as
     * function-local statics inside the method (program lifetime, as before)
     * and are not returned — only their raw pointers are wired into `engine`.
     *
     * `attachedMode` (Munin worker) skips the sysfs-writing regulators
     * (governor / fan) per the M-Munin "Governor-Sonderregel"; the read-only
     * SystemMonitor + ThermalGuard are still installed. `fatalExitCode`, when
     * set, means a required sensor was missing in standalone mode and
     * `runServe` must abort with that code.
     */
    struct Ancillaries {
        std::unique_ptr<runtime::SystemMonitor>          monitor;
        std::unique_ptr<runtime::ThermalGuard>           guard;
        std::unique_ptr<runtime::PowerMonitor>           powerMonitor;
        std::unique_ptr<runtime::PerfRegressionDetector> perfDetector;
        std::optional<int>                               fatalExitCode;
    };

    [[nodiscard]] static Ancillaries wireThermalGovernorFan(
        runtime::InferenceEngine&   engine,
        const core::config::Config& cfg,
        bool                        attachedMode);

    /**
     * Continuous batcher for the default/eager engine, or null when the engine
     * is not serving-class-eligible or the batcher failed to initialise (the
     * caller then falls back to single-session generate()). Caller wires
     * `scfg.batcher = result.get()` and keeps the returned owner alive for the
     * server's lifetime — the batcher's dtor joins its worker on shutdown.
     * `defaultId` is used only for log lines.
     */
    [[nodiscard]] static std::unique_ptr<runtime::serving::ContinuousBatcher>
    buildDefaultBatcher(runtime::InferenceEngine&    engine,
                        const core::config::Config&  cfg,
                        const std::string&           defaultId);

    /**
     * Every `loadOnStart:true` model, loaded into its own engine/stack. Filled
     * by `loadModels` via out-parameter (NOT returned by value) because the
     * loaded Rerank/Embed/Decide/Transcribe engines hold references into the
     * ComputeStack vectors that precede them here — a by-value return would
     * move those vectors and dangle the references. Member order is
     * destruction-order-critical: each `owned*Stacks` vector is declared BEFORE
     * the engines it backs, so the engines (which free USM through the stacks'
     * ops) are destroyed first. `runServe` holds this for the whole serve
     * lifetime; `loadedEngines` / `loaded*` are std::move'd into the ApiServer,
     * the `owned*` vectors keep the concrete objects alive behind them.
     *
     * `exitCode`, when set, means loading hit a fatal config/attach error (or a
     * dev/bench env-mode ran and produced its own exit code); `runServe`
     * returns it instead of booting the HTTP server.
     */
    struct LoadedModels {
        std::vector<std::unique_ptr<runtime::InferenceEngine>>        ownedEngines;
        std::vector<server::LoadedEngine>                            loadedEngines;
        std::vector<runtime::ComputeStack>                           ownedRerankStacks;
        std::vector<std::unique_ptr<runtime::encoder::RerankEngine>> ownedRerankers;
        std::vector<server::LoadedReranker>                          loadedRerankers;
        std::vector<runtime::ComputeStack>                           ownedEmbedStacks;
        std::vector<std::unique_ptr<runtime::encoder::EmbedEngine>>  ownedEmbedders;
        std::vector<server::LoadedEmbedder>                          loadedEmbedders;
        std::vector<runtime::ComputeStack>                           ownedDecideStacks;
        std::vector<std::unique_ptr<runtime::encoder::DecideEngine>> ownedDeciders;
        std::vector<server::LoadedDecider>                           loadedDeciders;
        std::vector<runtime::ComputeStack>                           ownedTranscribeStacks;
        std::vector<std::unique_ptr<runtime::audio::AudioEngine>>    ownedTranscribers;
        std::vector<server::LoadedTranscriber>                       loadedTranscribers;
        std::vector<std::unique_ptr<runtime::InferenceEngine>>       ownedSpeakBackbones;
        std::vector<std::unique_ptr<runtime::audio::SpeakEngine>>    ownedSpeakers;
        std::vector<server::LoadedSpeaker>                           loadedSpeakers;
        std::vector<std::shared_ptr<void>>                           attachedKeepAlive;
        std::vector<core::config::ModelEntry>                        poolChatModels;
        std::optional<int>                                           exitCode;
    };

    /// Attach-to-Munin callback: materialises engine `e` for model `m` and
    /// returns the (importer, client) keep-alive pair, or nullopt on failure.
    /// Kept as a `runServe` local (the M-Munin.3 pool factory captures it too);
    /// passed in here rather than owned so both attach sites share one impl.
    using AttachKeepAlive =
        std::pair<std::shared_ptr<void>, std::shared_ptr<void>>;
    using AttachEngineFn = std::function<std::optional<AttachKeepAlive>(
        runtime::InferenceEngine&, const core::config::ModelEntry&)>;

    /**
     * Load every `loadOnStart:true` model into `out`. Attached-mode workers
     * first probe Munin's healthz. `backendPool` and `attachEngine` stay owned
     * by `runServe` (the pool factory reuses them), so they are passed in by
     * reference. Sets `out.exitCode` on any fatal boot error or when a dev/bench
     * env-mode handled the run.
     */
    static void loadModels(LoadedModels&                 out,
                           const core::config::Config&   cfg,
                           const cli::CliArgs&           args,
                           core::backend::BackendPool&   backendPool,
                           const AttachEngineFn&         attachEngine,
                           const std::string&            defaultId,
                           bool                          attachedMode);

    /**
     * Attach engine `e` to Munin for model `m` over this build's IPC transport
     * (L0 IPC handles on Xe-LPG, POSIX-shm on CUDA) and materialise its weights
     * via loadModelAttached / loadModelAttachedNvfp4. Returns the (importer,
     * client) keep-alive pair, or nullopt (and logs) on failure. This is the
     * single shared implementation behind the `AttachEngineFn` that both
     * `loadModels` and `buildModelProvider` consume; the caller decides where
     * the keep-alive lives.
     */
    [[nodiscard]] static std::optional<AttachKeepAlive>
    attachEngineToMunin(runtime::InferenceEngine&       e,
                        const core::config::ModelEntry& m,
                        const std::string&              attachSocket);

    /**
     * 5.27.10: force one tiny main-thread generate() so process-global lazy
     * CUDA state (in particular the CUTLASS NVFP4-TC grouped GEMM,
     * "nvfp4-tc-banks") initialises on the context-owning main thread before
     * any batcher / httplib worker touches it — a first worker-thread touch
     * poisons the context and crashes the first request. Guarded so a warmup
     * failure never blocks serving; a no-op for archs that don't drive the
     * TC-banks path or engines that are not serving-class. `defaultId` is used
     * only for log lines.
     */
    static void warmupDefaultEngine(runtime::InferenceEngine& engine,
                                    const std::string&        defaultId);

    /**
     * M-Munin.3 (full): build the worker-side materialise/evict pool provider
     * for the non-default chat models (`poolChatModels`). Its factory attaches
     * to Munin, materialises, and — mirroring the default engine's setup —
     * wires each slot's OWN continuous batcher and, if it is the configured
     * `speculative.target`, its OWN spec-dec decoder sharing the default's
     * drafter. Returns nullptr when there are no pool models. The caller wires
     * `scfg.modelProvider = result.get()` and keeps the owner alive for the
     * server's lifetime. `backendPool` / `attachEngine` outlive the provider
     * (owned by `runServe`); the ancillary monitors are snapshotted from
     * `defaultEngine`.
     */
    [[nodiscard]] static std::unique_ptr<server::AttachedModelProvider>
    buildModelProvider(runtime::InferenceEngine&                    defaultEngine,
                       const core::config::Config&                  cfg,
                       core::backend::BackendPool&                  backendPool,
                       const AttachEngineFn&                        attachEngine,
                       const server::ServerConfig&                  scfg,
                       runtime::Drafter*                            drafter,
                       const std::vector<core::config::ModelEntry>& poolChatModels,
                       const std::string&                           defaultId);
};

} // namespace mimirmind::cli
