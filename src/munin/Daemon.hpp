// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/config/Config.hpp"

#include <string>

namespace mimirmind::munin {

/**
 * CLI options for `munin`. Parsed from argv in main.cpp; passed as a
 * value to `Daemon::run`. Missing / defaults resolve against config.json.
 */
struct CliOptions {
    std::string configPath{};   // required
    std::string socketPath{};   // default /var/run/munin/munin.sock
    std::string logFile{};      // overrides config.server.log.file
    std::string logLevel{};     // overrides config.server.log.level
};

/**
 * Daemon lifecycle: parse config, build UsmAllocator, bring the
 * ModelStore up, install signal wiring, run the SocketServer accept
 * loop, and drive a clean shutdown when SIGINT/SIGTERM arrive.
 *
 * Stateless class; `run()` owns all the resources on the stack so
 * teardown is deterministic.
 */
class Daemon {
public:
    /// Blocks until shutdown. Returns the intended process exit code.
    [[nodiscard]] static int run(const CliOptions& opts) noexcept;

private:
    Daemon() = default;
};

} // namespace mimirmind::munin