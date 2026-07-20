// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/backend/ComputeBackend.hpp"
#include "core/backend/ComputeContext.hpp"
#include "core/backend/SelectionMode.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace mimirmind::core::backend {

/**
 * One selectable device in the `BackendPool`. `kind` + `deviceIx`
 * uniquely identify the device inside the process; `token` is the
 * config-facing string (`"hip:0"`, `"l0:0"`, `"cpu"`).
 *
 * `ctx` is lazy — probing the driver is cheap, constructing a full
 * `ComputeContext` (autotune, module load) is not. Pool consumers call
 * `context()` when they actually need to run compute.
 *
 * Non-copyable (owns `unique_ptr`); movable so `std::vector` works.
 */
struct PoolEntry {
    BackendKind                       kind{BackendKind::Unknown};
    std::size_t                       deviceIx{0};
    std::string                       name;    // "AMD Radeon RX 7800 XT"
    std::string                       token;   // "hip:0"
    std::string                       detail;  // free-form for /system/info

    PoolEntry() = default;
    PoolEntry(const PoolEntry&)            = delete;
    PoolEntry& operator=(const PoolEntry&) = delete;
    PoolEntry(PoolEntry&&) noexcept        = default;
    PoolEntry& operator=(PoolEntry&&) noexcept = default;

    /// Lazy accessor — constructs `_ctx` on first call via
    /// `BackendRegistry::createContext(kind)`. Repeated calls return
    /// the same context. Throws on driver / device init failure.
    [[nodiscard]] ComputeContext& context();

    /// True iff `context()` has been called at least once and did not
    /// throw. Used by status endpoints to report "loaded" vs
    /// "reserved" state without triggering a lazy init.
    [[nodiscard]] bool            hasContext() const noexcept { return _ctx != nullptr; }

private:
    std::unique_ptr<ComputeContext>   _ctx;
};

/**
 * Startup-time discovery + selection for every compiled-in and
 * runtime-available backend + device.
 *
 * The pool replaces the single-shot `BackendRegistry::autoSelect` +
 * `createContext` flow with a persistent view: probe once, keep the
 * `PoolEntry` list around for the process lifetime, hand out
 * references to individual entries by mode or by token.
 *
 * On a dual-GPU host (e.g. Meteor Lake iGPU + discrete Radeon), the
 * pool holds one entry per device; per-model config in
 * `models[].backend` picks which device each `InferenceEngine` binds
 * to. That way a small draft can sit on the iGPU while the target
 * model runs on the dGPU.
 *
 * Multi-device-per-kind (two AMD dGPUs in one box) is not modelled yet
 * — `discoverAll` reports one entry per available `BackendKind`
 * today. The `deviceIx` field on `PoolEntry` is reserved so the API
 * stays stable when the follow-up lands.
 *
 * Not thread-safe.
 */
class BackendPool {
public:
    BackendPool() = default;
    ~BackendPool() = default;

    BackendPool(const BackendPool&)            = delete;
    BackendPool& operator=(const BackendPool&) = delete;
    BackendPool(BackendPool&&) noexcept        = default;
    BackendPool& operator=(BackendPool&&) noexcept = default;

    /// Enumerate compiled-in + runtime-available backends via
    /// `BackendRegistry::probeAll()`, plus the always-available Cpu
    /// entry. Idempotent — a second call clears + re-fills. Never throws.
    void discoverAll();

    /// Read-only view of the discovered entries.
    [[nodiscard]] const std::vector<PoolEntry>& entries() const noexcept { return _entries; }

    /// Mutable view for callers that need `context()` (which mutates
    /// the entry's lazy `_ctx`). Callers should hold `PoolEntry&`
    /// references only for the lifetime of the pool.
    [[nodiscard]] std::vector<PoolEntry>&       entries()       noexcept { return _entries; }

    /// Pick an entry by mode. `Auto` walks `entries()` in
    /// discovery order and returns the first one. Throws
    /// `std::runtime_error` when the pool is empty.
    [[nodiscard]] PoolEntry& select(SelectionMode mode);

    /// Look up by config-facing token. Recognised forms:
    ///   `"auto"`       -> delegates to `select(Auto)`
    ///   `"cpu"`        -> the Cpu entry
    ///   `"l0"`         -> shorthand for `"l0:0"`
    ///   `"l0:<n>"`     -> LevelZero device index n
    ///   `"hip"`        -> shorthand for `"hip:0"`
    ///   `"hip:<n>"`    -> Hip device index n
    ///   `"cuda"`       -> shorthand for `"cuda:0"`
    ///   `"cuda:<n>"`   -> Cuda device index n
    /// Throws `std::runtime_error` when the token names a backend/
    /// device that is not in `entries()`.
    [[nodiscard]] PoolEntry& selectByToken(std::string_view token);

private:
    std::vector<PoolEntry> _entries;
};

/// Free-fn helper mirrored from `BackendRegistry::name` for the token
/// side. Given a kind and device index, returns the canonical token
/// (`"hip:0"`). `Cpu` collapses to `"cpu"` regardless of `deviceIx`.
[[nodiscard]] std::string tokenFor(BackendKind kind, std::size_t deviceIx) noexcept;

} // namespace mimirmind::core::backend
