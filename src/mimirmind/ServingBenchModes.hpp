// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <optional>
#include <string>

namespace mimirmind::runtime { class InferenceEngine; }
namespace mimirmind::core::config { struct Config; }

namespace mimirmind::cli {

/**
 * Developer/benchmark harnesses that `mimirmind serve` runs in place of
 * booting the HTTP server when the matching `MIMIRMIND_*` environment
 * variable is set. Each harness times or parity-checks a batched /
 * speculative decode path on the freshly loaded engine and then makes the
 * process exit (returns an exit code); none of them fall through to normal
 * serving. They live here, out of `runServe`'s boot path, so the serve
 * orchestration stays readable. All CUDA/qwen35moe-gated at runtime; the
 * translation unit compiles under every backend.
 */
class ServingBenchModes {
public:
    /**
     * If exactly one bench-mode env var is set (and the loaded arch
     * supports it), run that harness and return its process exit code;
     * otherwise return std::nullopt so `runServe` continues to boot.
     */
    [[nodiscard]] static std::optional<int> maybeRun(
        runtime::InferenceEngine& engine,
        const std::string& arch,
        const core::config::Config& cfg);

private:
    static int runBatchBench(runtime::InferenceEngine& engine,
        const std::string& arch, const core::config::Config& cfg);
    static int runServingParity(runtime::InferenceEngine& engine,
        const std::string& arch, const core::config::Config& cfg);
    static int runL0Batch(runtime::InferenceEngine& engine,
        const std::string& arch, const core::config::Config& cfg);
    static int runServingLoop(runtime::InferenceEngine& engine,
        const std::string& arch, const core::config::Config& cfg);
    static int runBatcherTest(runtime::InferenceEngine& engine,
        const std::string& arch, const core::config::Config& cfg);
    static int runMtpTest(runtime::InferenceEngine& engine,
        const std::string& arch, const core::config::Config& cfg);
    static int runDflashTest(runtime::InferenceEngine& engine,
        const std::string& arch, const core::config::Config& cfg);
    static int runDflashBatch(runtime::InferenceEngine& engine,
        const std::string& arch, const core::config::Config& cfg);
    static int runMtpVerifyTest(runtime::InferenceEngine& engine,
        const std::string& arch, const core::config::Config& cfg);
    static int runMtpDraftTest(runtime::InferenceEngine& engine,
        const std::string& arch, const core::config::Config& cfg);
    static int runMtpBatchTest(runtime::InferenceEngine& engine,
        const std::string& arch, const core::config::Config& cfg);
    static int runMtpAccept(runtime::InferenceEngine& engine,
        const std::string& arch, const core::config::Config& cfg);
    static int runMtpPerf(runtime::InferenceEngine& engine,
        const std::string& arch, const core::config::Config& cfg);
    static int runMtpMultiTest(runtime::InferenceEngine& engine,
        const std::string& arch, const core::config::Config& cfg);
    static int runMtpPerfTest(runtime::InferenceEngine& engine,
        const std::string& arch, const core::config::Config& cfg);
};

} // namespace mimirmind::cli
