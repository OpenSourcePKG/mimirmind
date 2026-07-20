// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/backend/ComputeContext.hpp"
#include "core/gpu/hip/HipContext.hpp"
#include "core/gpu/hip/HipMemoryAllocator.hpp"
#include "core/gpu/hip/HipStream.hpp"

namespace mimirmind::core::hip {

/**
 * HIP implementation of the neutral `ComputeContext` interface. Owns
 * the full HIP runtime state: device selection via `HipContext`, VRAM
 * + host-pinned allocations via `HipMemoryAllocator`, kernel launches
 * via `HipStream`.
 *
 * Parallel to `L0ComputeContext` on the Level Zero side. Backend
 * selection (`BackendRegistry::createContext(BackendKind::Hip)`)
 * returns one of these as a `std::unique_ptr<ComputeContext>`;
 * consumers that need HIP-native handles downcast and pull the
 * typed getters (`hipContext()`, `stream()`, `allocator()`).
 *
 * Not-yet-here on this class: `HipModule` cache, and the
 * `HipGpuOps` / `HipGpuMatmul` compute-op wrappers that would consume
 * this context the way `GpuOps` / `GpuMatmul` today consume the L0
 * trio. Those land as separate commits once the compute-stack
 * refactor (Schicht 4) is in.
 *
 * Construction order matches the L0 side: context first, then
 * allocator, then stream. Errors bubble as `HipError`.
 */
class HipComputeContext : public ::mimirmind::core::backend::ComputeContext {
public:
    /// Grouped construction options. `deviceIndex = -1` means auto-
    /// select: prefer the first non-integrated GPU, fall back to the
    /// integrated one if no dGPU is present. Same policy as
    /// `HipContext::HipContext(-1)`.
    struct Options {
        int           deviceIndex = -1;
        HipStreamKind streamKind  = HipStreamKind::BlockingDefault;
    };

    HipComputeContext() : HipComputeContext(Options{}) {}
    explicit HipComputeContext(Options opts);
    ~HipComputeContext() override;

    HipComputeContext(const HipComputeContext&)            = delete;
    HipComputeContext& operator=(const HipComputeContext&) = delete;
    HipComputeContext(HipComputeContext&&)                 = delete;
    HipComputeContext& operator=(HipComputeContext&&)      = delete;

    // ---- ComputeContext interface ------------------------------------

    [[nodiscard]] ::mimirmind::core::backend::ComputeBackend&
        backend() noexcept override { return _ctx; }
    [[nodiscard]] const ::mimirmind::core::backend::ComputeBackend&
        backend() const noexcept override { return _ctx; }

    /// Sustained memory bandwidth heuristic for AMD family:
    ///   integrated APU (Phoenix / Hawk Point / Strix Halo) —
    ///     ~100 GB/s (LPDDR5x-8000/8533 class).
    ///   discrete GPU (RDNA3 RX 7800 XT / 7900 XTX etc.) —
    ///     ~500 GB/s (real gfx1101 = 624, gfx1030 = 512, average).
    /// Rough per-family heuristic; a real per-device probe is a
    /// follow-up. Never throws.
    [[nodiscard]] std::size_t bandwidthGBps() const noexcept override;

    // ---- HIP-native accessors ----------------------------------------

    [[nodiscard]] HipContext&              hipContext() noexcept       { return _ctx; }
    [[nodiscard]] const HipContext&        hipContext() const noexcept { return _ctx; }

    [[nodiscard]] HipMemoryAllocator&      allocator() noexcept        { return _alloc; }
    [[nodiscard]] const HipMemoryAllocator& allocator() const noexcept { return _alloc; }

    [[nodiscard]] HipStream&               stream() noexcept           { return _stream; }

private:
    HipContext           _ctx;
    HipMemoryAllocator   _alloc;
    HipStream            _stream;
};

} // namespace mimirmind::core::hip