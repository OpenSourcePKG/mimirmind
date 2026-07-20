// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/backend/ComputeContext.hpp"
#include "core/gpu/l0/CommandQueue.hpp"
#include "core/gpu/l0/L0Context.hpp"
#include "core/gpu/l0/UsmAllocator.hpp"

#include <optional>
#include <string>

namespace mimirmind::core::l0 {

/**
 * Level-Zero implementation of the neutral `ComputeContext` interface.
 * Owns the full L0 runtime state: driver+context+device via
 * `L0Context`, USM allocations via `UsmAllocator`, kernel launches via
 * `runtime::CommandQueue`. The three used to live as separate members
 * on `InferenceEngine`; this class packages them as a single owned
 * unit so backend selection (`BackendRegistry::createContext(kind)`)
 * has something to return.
 *
 * Consumers that touch L0-native handles (`GpuOps`, `GpuMatmul`, ...)
 * still ask for `L0Context&` / `UsmAllocator&` / `runtime::CommandQueue&`
 * explicitly — they downcast `ComputeContext&` to this concrete type
 * and pull the getters. Making those launch APIs backend-neutral is a
 * follow-up refactor (Schicht 4 in the HW-abstraction strategy). Until
 * then, `L0ComputeContext` gives us the ownership + selection story
 * without pretending the launch API is unified.
 *
 * Construction order matches the pre-existing `InferenceEngine`
 * layout: `L0Context` first (initialises the driver + picks the
 * device), then `UsmAllocator` (needs the context to allocate from),
 * then `CommandQueue` (needs the context for the queue handle). Errors
 * from any stage bubble as `L0Error`.
 */
class L0ComputeContext : public ::mimirmind::core::backend::ComputeContext {
public:
    /// Grouped construction options — mirror the config-json knobs
    /// that the current `InferenceEngine` pulls from `Config::runtime`.
    /// Passed as a struct so the ctor signature stays stable when a
    /// new knob is added (config.example.json evolves faster than the
    /// engine's ctor list should).
    struct Options {
        std::string                 spvDirOverride;      // runtime.spvDir
        std::optional<int>          usmProbeTotalGiB;    // runtime.usmProbeTotalGib
        /// When `nullopt`, the ctor calls `selectUsmAllocKind(l0Context)`
        /// which honours the `MIMIRMIND_USM_KIND` env var and falls back
        /// to Host on integrated GPUs (Munin-IPC-safe). Explicit values
        /// bypass the auto-pick and are honoured verbatim — mostly useful
        /// for bench tools that want deterministic behaviour.
        std::optional<UsmAllocKind> usmKindOverride;
    };

    L0ComputeContext() : L0ComputeContext(Options{}) {}
    explicit L0ComputeContext(Options opts);
    ~L0ComputeContext() override;

    L0ComputeContext(const L0ComputeContext&)            = delete;
    L0ComputeContext& operator=(const L0ComputeContext&) = delete;
    L0ComputeContext(L0ComputeContext&&)                 = delete;
    L0ComputeContext& operator=(L0ComputeContext&&)      = delete;

    // ---- ComputeContext interface ------------------------------------

    [[nodiscard]] ::mimirmind::core::backend::ComputeBackend&
        backend() noexcept override { return _ctx; }
    [[nodiscard]] const ::mimirmind::core::backend::ComputeBackend&
        backend() const noexcept override { return _ctx; }

    /// Sustained memory bandwidth heuristic for Xe-LPG family:
    ///   integrated iGPU (Meteor Lake / Arrow Lake / Lunar Lake /
    ///     Panther Lake) — ~70 GB/s sustained per pegenaut-skynet
    ///     measurements (dual-channel DDR5-5600 IMC, 89.6 theor).
    ///   discrete GPU (Arc B70 / A770 / Battlemage) — ~450 GB/s.
    /// Rough per-family heuristic; a real per-device probe is a
    /// follow-up. Never throws.
    [[nodiscard]] std::size_t bandwidthGBps() const noexcept override;

    // ---- L0-native accessors (backend-specific consumers only) -------
    //
    // These are the escape hatches. Any code that has to touch a raw
    // L0 handle (GpuOps, GpuMatmul, munin IPC) downcasts to
    // `L0ComputeContext&` and pulls the concrete-typed reference.
    // A future backend-neutral `launch_kernel` API on `ComputeContext`
    // would replace most of them; that refactor is not on this
    // commit's diff.

    [[nodiscard]] L0Context&              l0Context() noexcept       { return _ctx; }
    [[nodiscard]] const L0Context&        l0Context() const noexcept { return _ctx; }

    [[nodiscard]] UsmAllocator&           allocator() noexcept       { return _alloc; }
    [[nodiscard]] const UsmAllocator&     allocator() const noexcept { return _alloc; }

    [[nodiscard]] ::mimirmind::runtime::CommandQueue& queue() noexcept { return _queue; }

private:
    L0Context                             _ctx;
    UsmAllocator                          _alloc;
    ::mimirmind::runtime::CommandQueue    _queue;
};

} // namespace mimirmind::core::l0