// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/ProbeConfig.hpp"

#include "core/log/Log.hpp"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>

namespace mimirmind::runtime {

std::optional<ProbePicks>
loadProbePicks(const std::string& dir, const std::string& fingerprint) {
    namespace fs = std::filesystem;
    const fs::path path = fs::path{dir} / "hw" / fingerprint / "probe-result.json";

    std::error_code ec;
    if (!fs::exists(path, ec)) {
        return std::nullopt;   // no artefact for this HW — the common case
    }

    std::ifstream f{path};
    if (!f) {
        MM_LOG_WARN("probe", "probe artefact {} exists but is unreadable", path.string());
        return std::nullopt;
    }

    nlohmann::json j;
    try {
        f >> j;
    } catch (const std::exception& e) {
        MM_LOG_WARN("probe", "probe artefact {} failed to parse: {}", path.string(), e.what());
        return std::nullopt;
    }

    ProbePicks picks;
    picks.fingerprint = j.value("fingerprint", std::string{});
    picks.modelId     = j.value("model_id", std::string{});
    picks.probeStatus = j.value("probe_status", std::string{});

    // Defensive: the fingerprint lives in the path AND the file; a mismatch
    // means a stale/corrupt artefact — refuse it rather than apply the wrong
    // hardware's picks.
    if (picks.fingerprint != fingerprint) {
        MM_LOG_WARN("probe",
                    "probe artefact {} fingerprint '{}' != expected '{}' — ignoring",
                    path.string(), picks.fingerprint, fingerprint);
        return std::nullopt;
    }

    const auto autotuneIt = j.find("autotune");
    if (autotuneIt != j.end() && autotuneIt->is_array() && !autotuneIt->empty()) {
        picks.hasAutotune = true;
        bool anyGemmPicked = false;
        for (const auto& a : *autotuneIt) {
            if (a.value("gemm_available", false) && a.value("gemm_picked", false)) {
                anyGemmPicked = true;
                break;
            }
        }
        picks.gemmNeverForAllMatmul = !anyGemmPicked;
    }

    return picks;
}

} // namespace mimirmind::runtime
