// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <memory>
#include <optional>
#include <string>

namespace mimirmind::runtime {
class InferenceEngine;
class Drafter;
class SystemMonitor;
class ThermalGuard;
class PowerMonitor;
class PerfRegressionDetector;
} // namespace mimirmind::runtime

namespace mimirmind::runtime::serving {
class ContinuousBatcher;
} // namespace mimirmind::runtime::serving

namespace mimirmind::core::config {
struct Config;
} // namespace mimirmind::core::config

namespace mimirmind::cli {

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
};

} // namespace mimirmind::cli
