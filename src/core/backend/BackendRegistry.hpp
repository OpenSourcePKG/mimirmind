// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/backend/ComputeBackend.hpp"
#include "core/backend/ComputeContext.hpp"

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace mimirmind::core::backend {

/**
 * Snapshot of one backend's probe at binary-start time. `compiledIn`
 * reflects the CMake `MIMIRMIND_ENABLE_<backend>` flag at build time;
 * `available` is a lightweight runtime check (driver reachable, at
 * least one device visible). Neither call constructs a full backend
 * — that's the caller's job, and may still fail even when
 * `available` was true (bad permissions on `/dev/dri`, half-installed
 * driver, etc).
 *
 * Order in the vector returned by `probeAll()` matches the
 * `BackendKind` enum order (LevelZero, Hip, Cuda, Cpu) so callers can
 * apply a stable priority policy on top.
 */
struct BackendProbe {
    BackendKind  kind{BackendKind::Unknown};
    bool         compiledIn{false};
    bool         available{false};
    std::string  detail;    // one-line human-readable summary
};

/**
 * Compile-time enumeration of every backend linked into the binary,
 * with a runtime availability probe per entry. Foundation for later
 * auto-selection + routing — this step just exposes the picture; the
 * `mimirmind` binary today still hard-binds to Level Zero because the
 * compute stack (GpuOps, GpuMatmul, CommandQueue) is L0-native.
 *
 * The registry is purely static — no state, no init order dance,
 * probes are re-runnable without side effects.
 *
 * See
 * [[MimirMind — HW-Abstraktions-Strategie für Multi-Backend-Support]]
 * Schicht 1 / 6 in Synaipse for the broader plan.
 */
class BackendRegistry {
public:
    /// One entry per BackendKind that mimirmind knows about, whether
    /// or not it was compiled into this build. Never throws.
    [[nodiscard]] static std::vector<BackendProbe> probeAll() noexcept;

    /// Human-readable name for logging / tools. Never throws.
    [[nodiscard]] static const char* name(BackendKind k) noexcept;

    /// Parse a backend name ("l0" / "levelzero", "hip", "cuda", "cpu")
    /// to a `BackendKind`. Case-insensitive. Returns `std::nullopt` for
    /// unrecognised values. Never throws.
    [[nodiscard]] static std::optional<BackendKind>
        parseKind(std::string_view s) noexcept;

    /// Resolve the backend the process should use, honouring the
    /// `MIMIRMIND_BACKEND` env var if set (values as per `parseKind`),
    /// otherwise falling back to `defaultKind`. The env-var takes
    /// precedence; a compiled-out backend is not caught here — the
    /// caller must check `probeAll()` afterwards if they want to know
    /// whether the resolved backend is actually available. Never throws.
    [[nodiscard]] static BackendKind
        resolveKind(BackendKind defaultKind = BackendKind::LevelZero) noexcept;

    /// Construct a fully-initialised `ComputeContext` for the given
    /// backend. The single entry point where runtime backend-selection
    /// materialises. Throws the backend's concrete error type on driver /
    /// device init failure (`L0Error`, `HipError`, ...).
    ///
    /// `kind` must correspond to a backend that was compiled in. Passing
    /// a compiled-out backend throws `std::runtime_error` with a message
    /// that names the missing CMake flag.
    [[nodiscard]] static std::unique_ptr<ComputeContext>
        createContext(BackendKind kind);
};

} // namespace mimirmind::core::backend