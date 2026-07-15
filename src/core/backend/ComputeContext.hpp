// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/backend/ComputeBackend.hpp"

namespace mimirmind::core::backend {

/**
 * RAII owner of a fully-initialised backend runtime — the concrete
 * backend context plus whatever runtime state (allocator, queue/stream,
 * module cache) that context needs to launch kernels.
 *
 * Sits one layer above `ComputeBackend`. `ComputeBackend` is the
 * device-info + feature-flag interface — it says WHICH backend and
 * WHAT it can do. `ComputeContext` is the "backend is initialised
 * and ready to compute" marker — it owns the resources and exposes
 * the concrete backend via `backend()`.
 *
 * Concrete subclasses (`L0ComputeContext`, `HipComputeContext`, ...)
 * add backend-specific typed getters (`l0Context()`, `queue()`,
 * `hipStream()` etc.) that consumers with backend-specific kernel
 * launches downcast to. That downcast is intentional: `compute::GpuOps`
 * and friends today take `L0Context&` / `CommandQueue&` /
 * `UsmAllocator&` directly, and porting them to a truly-neutral
 * kernel-launch API is a follow-up refactor (Schicht 4 of the HW-
 * abstraction strategy). Until then, `ComputeContext` gives us the
 * ownership + backend-selection story without pretending the launch
 * API is unified.
 *
 * Ownership contract:
 *   - Constructor initialises everything or throws (the backend's
 *     concrete error type — `L0Error`, `HipError`, ...).
 *   - Destructor tears everything down in reverse order.
 *   - Move/copy deleted — consumers pass `ComputeContext&` or
 *     `std::unique_ptr<ComputeContext>`.
 *
 * Instantiate via `BackendRegistry::createContext(BackendKind)` —
 * that's the single entry point where runtime backend-selection
 * happens.
 */
class ComputeContext {
public:
    virtual ~ComputeContext() = default;

    ComputeContext(const ComputeContext&)            = delete;
    ComputeContext& operator=(const ComputeContext&) = delete;
    ComputeContext(ComputeContext&&)                 = delete;
    ComputeContext& operator=(ComputeContext&&)      = delete;

    /// The concrete backend the runtime is bound to. Cheap accessor —
    /// no probing, just returns the backend the ctor already selected.
    [[nodiscard]] virtual ComputeBackend& backend() noexcept       = 0;
    [[nodiscard]] virtual const ComputeBackend& backend() const noexcept = 0;

    /// Convenience — same as `backend().kind()`, just less typing at
    /// the call site.
    [[nodiscard]] BackendKind kind() const noexcept { return backend().kind(); }

protected:
    ComputeContext() = default;
};

} // namespace mimirmind::core::backend
