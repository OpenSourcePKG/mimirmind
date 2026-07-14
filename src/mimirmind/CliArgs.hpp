// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace mimirmind::cli {

/**
 * Top-level dispatch mode picked by the first positional argument.
 *
 *  - `Smoke`  — default. Runs the M1-M5 diagnostic suite and an
 *               end-to-end generate against the resolved model. Used
 *               during Bringup + as a post-deploy sanity check.
 *  - `Serve`  — starts the OpenAI-compatible HTTP server. This is
 *               what production runs.
 *  - `Parity` — spawns llama.cpp on the same prompt, dumps per-block
 *               hidden state, diffs against mimirmind's own dumps.
 *               Used for numerical parity verification during
 *               architecture bringup (M8.* Gemma-4 work).
 */
enum class Mode {
    Smoke,
    Serve,
    Parity,
};

/**
 * Parsed CLI arguments. Populated by `parseArgs` from `argv`; then
 * merged with `config.json` in `main()` before mode-dispatch. Fields
 * that stay at their default value mean "not set on the CLI" — the
 * CLI-override precedence lives in `main()`, not here.
 *
 * When adding a new flag: add the field with a sensible default, add
 * the corresponding branch in `parseArgs`, and describe it in
 * `kUsage` (both in `CliParser.cpp`). Keep the struct plain-data so
 * it can be passed by const-ref to every mode handler without ADL
 * concerns.
 */
struct CliArgs {
    Mode          mode{Mode::Smoke};
    std::string   configPath{"./config.json"};
    // Optional overrides — empty / sentinel means "not set on CLI".
    std::string   modelPath;
    std::string   prompt{"Hello, world!"};
    std::size_t   maxNew{20};
    std::optional<std::uint16_t> port;
    float         temperature{0.0F};
    std::size_t   topK{0};
    float         topP{1.0F};
    std::uint64_t seed{0};
    bool          chat{false};
    std::string   systemMessage{};
    std::string   logLevel{};
    std::string   logFile{};
    std::string   dumpDir{};
    // Attached mode for M-Munin. `unix:/path` means: skip local
    // tensor load, connect to Munin over the Unix socket, import
    // weights via SCM_RIGHTS. Governor is owned by Munin in this
    // mode — the worker does not acquire the governor flock and
    // does not install its own thermal/governor/fan monitors.
    std::string   attachSocket{};
};

} // namespace mimirmind::cli