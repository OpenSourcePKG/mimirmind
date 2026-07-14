// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "mimirmind/CliArgs.hpp"
#include "mimirmind/CliParser.hpp"
#include "mimirmind/ParityMode.hpp"
#include "mimirmind/ServeMode.hpp"
#include "mimirmind/SmokeMode.hpp"

#include "core/config/Config.hpp"
#include "core/l0/L0Context.hpp"
#include "core/log/Log.hpp"

#include <cstdint>
#include <exception>
#include <iostream>

int main(int argc, char** argv) {
    using mimirmind::cli::CliArgs;
    using mimirmind::cli::Mode;
    using mimirmind::cli::parseArgs;

    CliArgs args;
    if (!parseArgs(argc, argv, args)) {
        return 2;
    }

    // Load config.json — hard error if missing or malformed. `--config` (or
    // default `./config.json`) is the single source of truth for every
    // knob that used to live in `MIMIRMIND_*` env vars.
    mimirmind::core::config::Config cfg;
    try {
        cfg = mimirmind::core::config::loadConfig(args.configPath);
    } catch (const std::exception& e) {
        std::cerr << "config: " << e.what() << "\n";
        return 2;
    }

    // Apply CLI overrides (higher precedence than config.json).
    mimirmind::core::config::CliOverrides ovr{};
    if (!args.modelPath.empty()) ovr.modelPath = args.modelPath;
    if (args.port.has_value())   ovr.port      = static_cast<int>(*args.port);
    if (!args.logLevel.empty())  ovr.logLevel  = args.logLevel;
    if (!args.logFile.empty())   ovr.logFile   = args.logFile;
    if (!args.dumpDir.empty())   cfg.diagnostics.parityDump = args.dumpDir;
    try {
        mimirmind::core::config::applyCliOverrides(cfg, ovr);
    } catch (const std::exception& e) {
        std::cerr << "config: " << e.what() << "\n";
        return 2;
    }

    // Log has to be initialised AFTER config load — the level/file live
    // in the resolved server.log section.
    mimirmind::core::log::Log::initFromConfig(cfg.server.log);

    // Reflect the resolved-model path into CliArgs.modelPath so the many
    // downstream subcommand paths that consult it keep working without
    // being rewritten to reach into Config themselves.
    if (args.modelPath.empty() && !cfg.models.empty()) {
        args.modelPath = cfg.defaultModelEntry().path;
    }
    if (!args.port.has_value()) {
        args.port = static_cast<std::uint16_t>(cfg.server.port);
    }

    try {
        switch (args.mode) {
            case Mode::Smoke:  return mimirmind::cli::runSmoke(args, cfg);
            case Mode::Serve:  return mimirmind::cli::runServe(args, cfg);
            case Mode::Parity: return mimirmind::cli::runParity(args, cfg);
        }
        return 0;
    } catch (const mimirmind::core::l0::L0Error& e) {
        MM_LOG_ERROR("main", "Level Zero error: {}", e.what());
        std::cerr << "Level Zero error: " << e.what() << "\n";
        return 2;
    } catch (const std::exception& e) {
        MM_LOG_ERROR("main", "fatal: {}", e.what());
        std::cerr << "Fatal: " << e.what() << "\n";
        return 1;
    }
}