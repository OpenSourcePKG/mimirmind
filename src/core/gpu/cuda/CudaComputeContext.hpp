// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/backend/ComputeContext.hpp"
#include "core/gpu/cuda/CudaContext.hpp"
#include "core/gpu/cuda/CudaMemoryAllocator.hpp"
#include "core/gpu/cuda/CudaStream.hpp"

namespace mimirmind::core::cuda {

/**
 * CUDA implementation of the neutral `ComputeContext` interface. Owns
 * the full CUDA runtime state: device selection via `CudaContext`,
 * device/pinned/managed allocations via `CudaMemoryAllocator`, kernel
 * launches via `CudaStream`.
 *
 * Parallel to `L0ComputeContext` on the Level Zero side and
 * `HipComputeContext` on the HIP side. Backend selection
 * (`BackendRegistry::createContext(BackendKind::Cuda)`) returns one of
 * these as a `std::unique_ptr<ComputeContext>`; consumers that need
 * CUDA-native handles downcast and pull the typed getters
 * (`cudaContext()`, `stream()`, `allocator()`).
 *
 * Not-yet-here on this class: `CudaModule` cache, and the
 * `CudaGpuOps` / `CudaGpuMatmul` compute-op wrappers that would
 * consume this context the way `GpuOps` / `GpuMatmul` today consume
 * the L0 / HIP trios. Those land as separate commits under Track 3
 * (compute layer) on this same branch tree.
 *
 * Construction order matches the L0 / HIP sides: context first, then
 * allocator, then stream. Errors bubble as `CudaError`.
 */
class CudaComputeContext : public ::mimirmind::core::backend::ComputeContext {
public:
    /// Grouped construction options. `deviceIndex = -1` means auto-
    /// select: prefer the first non-integrated GPU, fall back to the
    /// integrated one if no dGPU is present. Same policy as
    /// `CudaContext::CudaContext(-1)`. On DGX Spark the fallback fires
    /// (integrated=true is the only device).
    struct Options {
        int            deviceIndex = -1;
        CudaStreamKind streamKind  = CudaStreamKind::BlockingDefault;
    };

    CudaComputeContext() : CudaComputeContext(Options{}) {}
    explicit CudaComputeContext(Options opts);
    ~CudaComputeContext() override;

    CudaComputeContext(const CudaComputeContext&)            = delete;
    CudaComputeContext& operator=(const CudaComputeContext&) = delete;
    CudaComputeContext(CudaComputeContext&&)                 = delete;
    CudaComputeContext& operator=(CudaComputeContext&&)      = delete;

    // ---- ComputeContext interface ------------------------------------

    [[nodiscard]] ::mimirmind::core::backend::ComputeBackend&
        backend() noexcept override { return _ctx; }
    [[nodiscard]] const ::mimirmind::core::backend::ComputeBackend&
        backend() const noexcept override { return _ctx; }

    /// Sustained memory bandwidth heuristic for NVIDIA family:
    ///   integrated (Grace ARM + GB10 = DGX Spark, or Jetson) —
    ///     ~273 GB/s (Spark LPDDR5X-8533, matches Bragi-anchor demo).
    ///   discrete (Blackwell / Ada / Ampere consumer / RTX 5070 etc.)
    ///     — ~500 GB/s (RTX 5070 = 672, RTX 5060 = 448, average).
    /// Rough per-family heuristic; a real per-device probe is a
    /// follow-up. Never throws.
    [[nodiscard]] std::size_t bandwidthGBps() const noexcept override;

    // ---- CUDA-native accessors ---------------------------------------

    [[nodiscard]] CudaContext&              cudaContext() noexcept       { return _ctx; }
    [[nodiscard]] const CudaContext&        cudaContext() const noexcept { return _ctx; }

    [[nodiscard]] CudaMemoryAllocator&      allocator() noexcept         { return _alloc; }
    [[nodiscard]] const CudaMemoryAllocator& allocator() const noexcept  { return _alloc; }

    [[nodiscard]] CudaStream&               stream() noexcept            { return _stream; }

private:
    CudaContext           _ctx;
    CudaMemoryAllocator   _alloc;
    CudaStream            _stream;
};

} // namespace mimirmind::core::cuda