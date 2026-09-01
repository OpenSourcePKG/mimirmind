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

    // Layer-2 flag intents. A flag drives the runtime only if its entry opts in
    // with "apply": true; otherwise it stays advisory (record-only) and the
    // runtime keeps its own value. This is the quality guardrail — lossy /
    // precision-unvalidated flags are authored apply:false until goldsetted.
    const auto flagsIt = j.find("flags");
    if (flagsIt != j.end() && flagsIt->is_object()) {
        const auto readFlag =
            [&flagsIt](const char* key) -> std::optional<bool> {
            const auto it = flagsIt->find(key);
            if (it == flagsIt->end() || !it->is_object()) return std::nullopt;
            if (!it->value("apply", false)) return std::nullopt;
            return it->value("value", 0) != 0;
        };
        picks.applyPrefillCudnn     = readFlag("MIMIRMIND_ATTN_CUDNN");
        picks.applyF32TcPrefill     = readFlag("MIMIRMIND_F32_TC_PREFILL");
        picks.applyCublasFp8Prefill = readFlag("MIMIRMIND_CUBLAS_FP8_PREFILL");
        picks.applyMmq              = readFlag("MIMIRMIND_MMQ");
        picks.applyMmqTc            = readFlag("MIMIRMIND_MMQ_TC");
        picks.applyAttnCudnnPaged   = readFlag("MIMIRMIND_ATTN_CUDNN_PAGED");
        picks.applyMoeSiluFuse      = readFlag("MIMIRMIND_MOE_SILU_FUSE");
    }

    return picks;
}

} // namespace mimirmind::runtime
