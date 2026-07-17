// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/backend/BackendRegistry.hpp"

#include "core/cpu/CpuContext.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// Forward declarations of the per-backend probe + createContext
// functions live at file scope — declaring
// `mimirmind::core::l0::probeBackend()` from inside
// `namespace mimirmind::core::backend` is not permitted.
//
// Definitions of these symbols live in the respective backend's
// translation unit (mimirmind_core_l0 / mimirmind_core_hip) so this
// file — part of mimirmind_core_common — never pulls in any backend-
// specific header or link dep. The MIMIRMIND_HAVE_* defines are set
// by CMake as PUBLIC compile-definitions on mimirmind_core_common
// whenever the corresponding MIMIRMIND_ENABLE_* option is on.

#ifdef MIMIRMIND_HAVE_L0
namespace mimirmind::core::l0 {
    ::mimirmind::core::backend::BackendProbe probeBackend() noexcept;
    std::unique_ptr<::mimirmind::core::backend::ComputeContext>
        createComputeContext();
}
#endif

#ifdef MIMIRMIND_HAVE_HIP
namespace mimirmind::core::hip {
    ::mimirmind::core::backend::BackendProbe probeBackend() noexcept;
    std::unique_ptr<::mimirmind::core::backend::ComputeContext>
        createComputeContext();
}
#endif

namespace mimirmind::core::backend {

namespace {

BackendProbe missing(BackendKind kind, const char* why) {
    return BackendProbe{kind, /*compiledIn=*/false, /*available=*/false, why};
}

BackendProbe cudaPlaceholder() {
    return missing(BackendKind::Cuda,
                   "not compiled in — CUDA backend not committed (no DGX-class target)");
}

} // namespace

std::vector<BackendProbe> BackendRegistry::probeAll() noexcept {
    std::vector<BackendProbe> out;
    out.reserve(4);

#ifdef MIMIRMIND_HAVE_L0
    out.push_back(::mimirmind::core::l0::probeBackend());
#else
    out.push_back(missing(BackendKind::LevelZero,
                          "not compiled in (MIMIRMIND_ENABLE_L0=OFF at build time)"));
#endif

#ifdef MIMIRMIND_HAVE_HIP
    out.push_back(::mimirmind::core::hip::probeBackend());
#else
    out.push_back(missing(BackendKind::Hip,
                          "not compiled in (MIMIRMIND_ENABLE_HIP=OFF at build time)"));
#endif

    out.push_back(cudaPlaceholder());
    out.push_back(::mimirmind::core::cpu::probeBackend());

    return out;
}

const char* BackendRegistry::name(BackendKind k) noexcept {
    switch (k) {
        case BackendKind::LevelZero: return "LevelZero";
        case BackendKind::Hip:       return "Hip";
        case BackendKind::Cuda:      return "Cuda";
        case BackendKind::Cpu:       return "Cpu";
        case BackendKind::Unknown:   return "Unknown";
    }
    return "Unknown";
}

std::optional<BackendKind>
BackendRegistry::parseKind(std::string_view s) noexcept {
    std::string lower{s};
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    if (lower == "l0" || lower == "levelzero" || lower == "level_zero") {
        return BackendKind::LevelZero;
    }
    if (lower == "hip" || lower == "rocm" || lower == "amd") {
        return BackendKind::Hip;
    }
    if (lower == "cuda" || lower == "nvidia") {
        return BackendKind::Cuda;
    }
    if (lower == "cpu") {
        return BackendKind::Cpu;
    }
    return std::nullopt;
}

BackendKind BackendRegistry::resolveKind(BackendKind defaultKind) noexcept {
    if (const char* env = std::getenv("MIMIRMIND_BACKEND")) {
        if (auto parsed = parseKind(env)) {
            return *parsed;
        }
        // Silently fall through to default on unrecognised value — the
        // signature is noexcept. Callers can compare the returned kind
        // against the env value themselves if they want to warn.
    }
    return defaultKind;
}

BackendKind BackendRegistry::autoSelect(BackendKind fallback) noexcept {
    // Explicit env-var override wins over the probe-based pick — same
    // precedence as `resolveKind`.
    if (const char* env = std::getenv("MIMIRMIND_BACKEND")) {
        if (auto parsed = parseKind(env)) {
            return *parsed;
        }
    }

    // Walk the probe table and take the first backend that is both
    // compiled-in AND runtime-available. `probeAll()` returns entries
    // in `BackendKind` enum order (LevelZero, Hip, Cuda, Cpu), which
    // acts as the priority policy without extra machinery. On a box
    // with only an AMD dGPU LevelZero probes as available=no, so HIP
    // wins; on a Meteor Lake laptop with no AMD driver, LevelZero wins.
    const auto probes = probeAll();
    for (const auto& p : probes) {
        if (p.compiledIn && p.available) {
            return p.kind;
        }
    }
    return fallback;
}

std::unique_ptr<ComputeContext>
BackendRegistry::createContext(BackendKind kind) {
    switch (kind) {
        case BackendKind::LevelZero:
#ifdef MIMIRMIND_HAVE_L0
            return ::mimirmind::core::l0::createComputeContext();
#else
            throw std::runtime_error{
                "BackendKind::LevelZero not compiled in "
                "(MIMIRMIND_ENABLE_L0=OFF at build time)"};
#endif

        case BackendKind::Hip:
#ifdef MIMIRMIND_HAVE_HIP
            return ::mimirmind::core::hip::createComputeContext();
#else
            throw std::runtime_error{
                "BackendKind::Hip not compiled in "
                "(MIMIRMIND_ENABLE_HIP=OFF at build time)"};
#endif

        case BackendKind::Cuda:
            throw std::runtime_error{
                "BackendKind::Cuda has no compiled backend "
                "(no DGX-class target committed)"};

        case BackendKind::Cpu:
            return ::mimirmind::core::cpu::createComputeContext();

        case BackendKind::Unknown:
            throw std::runtime_error{
                "BackendKind::Unknown — cannot construct a ComputeContext"};
    }
    throw std::runtime_error{"BackendRegistry::createContext: unreachable"};
}

} // namespace mimirmind::core::backend