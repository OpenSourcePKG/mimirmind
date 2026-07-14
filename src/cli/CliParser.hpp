// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "cli/CliArgs.hpp"

namespace mimirmind::cli {

/**
 * The startup banner printed once before mode-dispatch. Kept as a
 * public constant so mode handlers can reproduce it in their own
 * boot logs if needed (`serve` does).
 */
extern const char* const kBanner;

/**
 * Multi-line usage text printed by `--help` and on unknown-argument
 * error. Kept as a constant so the message text has one source of
 * truth and can be diffed cleanly when flags change.
 */
extern const char* const kUsage;

/**
 * Parse `argv` into `out`. Returns `false` on any parse error (bad
 * flag, missing value, unknown mode) — the caller should return 2
 * (usage-error) after failure.
 *
 * `--help` / `-h` calls `std::exit(0)` after printing `kUsage` — the
 * function does not return in that case. Every other exit is via
 * return-value so the caller keeps control of cleanup and status
 * codes.
 *
 * All error diagnostics go to `std::cerr`. The parser does not
 * initialise the Log subsystem — logging is a later step (needs
 * config.json to have been loaded first).
 */
[[nodiscard]] bool parseArgs(int argc, char** argv, CliArgs& out);

} // namespace mimirmind::cli