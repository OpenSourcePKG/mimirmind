// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/backend/BackendRegistry.hpp"

#include <string>
#include <vector>

// Forward declarations of the per-backend probe functions live at
// file scope — declaring `mimirmind::core::l0::probeBackend()` from
// inside `namespace mimirmind::core::backend` is not permitted.
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
}
#endif

#ifdef MIMIRMIND_HAVE_HIP
namespace mimirmind::core::hip {
    ::mimirmind::core::backend::BackendProbe probeBackend() noexcept;
}
#endif

namespace mimirmind::core::backend {

namespace {

BackendProbe missing(BackendKind kind, const char* why) {
    return BackendProbe{kind, /*compiledIn=*/false, /*available=*/false, why};
}

BackendProbe cpuPlaceholder() {
    // No real Cpu backend today — announce it as not-compiled-in so
    // consumers understand the story is symmetric.
    return missing(BackendKind::Cpu,
                   "no CPU compute backend — reference paths live inline in compute/");
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
    out.push_back(cpuPlaceholder());

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

} // namespace mimirmind::core::backend