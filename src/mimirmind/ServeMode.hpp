// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

namespace mimirmind::core::config { struct Config; }
namespace mimirmind::cli { struct CliArgs; }

namespace mimirmind::cli {

/**
 * `mimirmind serve` mode. Boots the OpenAI-compatible HTTP server:
 *   - acquires the governor flock (standalone mode) or probes
 *     Munin's healthz (attached mode via `--attach unix:PATH`)
 *   - loads every `loadOnStart:true` model into its own
 *     InferenceEngine
 *   - installs process-wide ancillaries (SystemMonitor,
 *     ThermalGuard, PowerMonitor, GpuClockGovernor, FanController,
 *     PerfRegressionDetector) with attached-mode carve-outs where
 *     Munin owns the sysfs writes
 *   - registers SIGINT/SIGTERM → server.stop
 *   - hands off to `ApiServer::run` (blocks until Ctrl-C or fatal)
 *
 * Returns 0 on clean shutdown, 1 on server error, 2 on
 * configuration / boot-time failure.
 *
 * The function owns non-trivial state — SIGINT handler needs the
 * server pointer, so a `g_runningServer` atomic lives in the .cpp.
 * `signalStop` is exposed as extern "C" for `std::signal`.
 */
[[nodiscard]] int runServe(const CliArgs& args,
                           const ::mimirmind::core::config::Config& cfg);

} // namespace mimirmind::cli

extern "C" void signalStop(int sig);