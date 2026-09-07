// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/ProbeConfig.hpp"

#include "core/log/Log.hpp"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>

namespace mimirmind::runtime {

namespace {

// Parse a `flags` object (shared by the HW profile and the per-model overlay)
// into `picks`. An entry drives its flag only with "apply": true; otherwise it
// stays advisory and the corresponding optional is left untouched.
void parseFlagsBlock(const nlohmann::json& flags, ProbePicks& picks) {
    if (!flags.is_object()) {
        return;
    }
    const auto readFlag = [&flags](const char* key) -> std::optional<bool> {
        const auto it = flags.find(key);
        if (it == flags.end() || !it->is_object()) return std::nullopt;
        if (!it->value("apply", false)) return std::nullopt;
        return it->value("value", 0) != 0;
    };
    const auto readIntFlag = [&flags](const char* key) -> std::optional<int> {
        const auto it = flags.find(key);
        if (it == flags.end() || !it->is_object()) return std::nullopt;
        if (!it->value("apply", false)) return std::nullopt;
        return it->value("value", 0);
    };
    // Only overwrite when the flag is present-and-applied, so an overlay that
    // omits a key leaves the base value in place (see mergeOverlay).
    if (auto v = readFlag("MIMIRMIND_ATTN_CUDNN"))        picks.applyPrefillCudnn = v;
    if (auto v = readFlag("MIMIRMIND_F32_TC_PREFILL"))    picks.applyF32TcPrefill = v;
    if (auto v = readFlag("MIMIRMIND_CUBLAS_FP8_PREFILL")) picks.applyCublasFp8Prefill = v;
    if (auto v = readFlag("MIMIRMIND_MMQ"))               picks.applyMmq = v;
    if (auto v = readFlag("MIMIRMIND_MMQ_TC"))            picks.applyMmqTc = v;
    if (auto v = readFlag("MIMIRMIND_ATTN_CUDNN_PAGED"))  picks.applyAttnCudnnPaged = v;
    if (auto v = readFlag("MIMIRMIND_MOE_SILU_FUSE"))     picks.applyMoeSiluFuse = v;
    if (auto v = readIntFlag("MIMIRMIND_MOE_DECODE_REG")) picks.applyMoeDecodeReg = v;
    if (auto v = readIntFlag("MIMIRMIND_GROUPED_MOE"))    picks.applyGroupedMoe = v;
}

} // namespace

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
    if (flagsIt != j.end()) {
        parseFlagsBlock(*flagsIt, picks);
    }

    return picks;
}

std::optional<ProbePicks>
loadModelOverlay(const std::string& dir, const std::string& fingerprint,
                 const std::string& modelId) {
    namespace fs = std::filesystem;
    if (modelId.empty()) {
        return std::nullopt;
    }
    // Sanitise the model id to a bare filename — it comes from serve config and
    // must not escape the profile dir (no '/', '\\' or '..').
    if (modelId.find('/') != std::string::npos
        || modelId.find('\\') != std::string::npos
        || modelId.find("..") != std::string::npos) {
        MM_LOG_WARN("probe", "model overlay: refusing unsafe model id '{}'", modelId);
        return std::nullopt;
    }
    // Dir name is "model-overlays", NOT "models" — the latter is .gitignore'd
    // (model weight directories), which would leave the overlay untracked.
    const fs::path path =
        fs::path{dir} / "hw" / fingerprint / "model-overlays" / (modelId + ".json");

    std::error_code ec;
    if (!fs::exists(path, ec)) {
        return std::nullopt;   // no per-model overlay — the common case
    }
    std::ifstream f{path};
    if (!f) {
        MM_LOG_WARN("probe", "model overlay {} exists but is unreadable", path.string());
        return std::nullopt;
    }
    nlohmann::json j;
    try {
        f >> j;
    } catch (const std::exception& e) {
        MM_LOG_WARN("probe", "model overlay {} failed to parse: {}", path.string(), e.what());
        return std::nullopt;
    }

    ProbePicks over;
    over.modelId = j.value("model_id", modelId);
    const auto flagsIt = j.find("flags");
    if (flagsIt != j.end()) {
        parseFlagsBlock(*flagsIt, over);
    }
    return over;
}

void mergeOverlay(ProbePicks& base, const ProbePicks& over) {
    if (over.applyPrefillCudnn)     base.applyPrefillCudnn     = over.applyPrefillCudnn;
    if (over.applyF32TcPrefill)     base.applyF32TcPrefill     = over.applyF32TcPrefill;
    if (over.applyCublasFp8Prefill) base.applyCublasFp8Prefill = over.applyCublasFp8Prefill;
    if (over.applyMmq)              base.applyMmq              = over.applyMmq;
    if (over.applyMmqTc)            base.applyMmqTc            = over.applyMmqTc;
    if (over.applyAttnCudnnPaged)   base.applyAttnCudnnPaged   = over.applyAttnCudnnPaged;
    if (over.applyMoeSiluFuse)      base.applyMoeSiluFuse      = over.applyMoeSiluFuse;
    if (over.applyMoeDecodeReg)     base.applyMoeDecodeReg     = over.applyMoeDecodeReg;
    if (over.applyGroupedMoe)       base.applyGroupedMoe       = over.applyGroupedMoe;
}

} // namespace mimirmind::runtime
