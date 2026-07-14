// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

namespace mimirmind::core::config { struct Config; }
namespace mimirmind::cli { struct CliArgs; }

namespace mimirmind::cli {

/**
 * `mimirmind parity` mode. Runs the tensor-parity workflow used
 * during architecture bringup (M8.* Gemma-4 track): shells out to
 * `llama-parity-dump` on the same prompt to get reference oracle
 * dumps, runs mimirmind with `diagnostics.parityDump` set so the
 * engine writes per-stage bin files, then shells out to
 * `parity-diff` to compare the two dump trees.
 *
 * Dumps live under `/tmp/dumps/{llama,mimir}` (wiped + recreated).
 * Requires `--model PATH`; returns 2 if not provided. Returns the
 * exit code of `parity-diff` (0 = all stages match within tolerance,
 * non-zero = at least one stage diverged).
 */
[[nodiscard]] int runParity(const CliArgs& args,
                            const ::mimirmind::core::config::Config& cfg);

} // namespace mimirmind::cli