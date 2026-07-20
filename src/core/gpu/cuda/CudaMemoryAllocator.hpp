// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/gpu/cuda/CudaContext.hpp"
#include "core/log/Log.hpp"

#include <cstddef>
#include <cstdint>

namespace mimirmind::core::cuda {

/**
 * Which `cudaMalloc*` variant backs an allocation.
 *
 *   Device      — cudaMalloc. VRAM on a dGPU; on Grace-ARM DGX Spark
 *                 the driver backs it from shared LPDDR5x but callers
 *                 still cannot dereference the pointer from host code.
 *   HostPinned  — cudaMallocHost. Page-locked host memory, mapped so
 *                 the GPU can DMA into/out-of it. Fast staging buffer
 *                 for H↔D transfers.
 *   Managed     — cudaMallocManaged. Driver-migrated between host and
 *                 device page-by-page. Cheap on UMA integrated parts
 *                 (single physical pool — Jetson, DGX Spark), real
 *                 PCIe migration on discrete. Kept as a fallback,
 *                 NOT the default — per Synaipse
 *                 `lesson_hip_managed_unusable_rdna_consumer` we
 *                 stay explicit about which pointers live where.
 */
enum class CudaAllocKind {
    Device,
    HostPinned,
    Managed,
};

/**
 * Snapshot of per-process CudaMemoryAllocator bookkeeping. Copyable
 * value type — not a live reference.
 */
struct CudaMemStats {
    std::uint64_t liveBytes           {0};
    std::uint64_t peakBytes           {0};
    std::uint64_t liveAllocations     {0};
    std::uint64_t totalAllocations    {0};
    std::uint64_t totalDeallocations  {0};
    std::uint64_t bytesCopiedH2D      {0};
    std::uint64_t bytesCopiedD2H      {0};
    std::uint64_t bytesCopiedD2D      {0};
    std::uint64_t failedAllocs        {0};
    std::uint64_t managedFallbackHits {0};   // DGX-Spark KV-managed count
};

/**
 * Thin per-process CUDA memory manager. Direct pass-through to
 * `cudaMalloc` / `cudaMallocHost` / `cudaMallocManaged` for allocation
 * and to synchronous `cudaMemcpy` for transfers — no pool, no
 * free-list, no stream binding. Mirrors the HIP allocator design.
 *
 * Adds two CUDA-specific helpers for the DGX Spark UMA-pressure story
 * (per Synaipse `reference_dgx_spark_engine_fit.md` and the ds4/DwarfStar
 * scan): `shouldUseManagedForKvCache` and `adviseReadMostly`.
 *
 * Not thread-safe by contract; wrap in a mutex at the caller level if
 * two threads must share the same allocator instance.
 *
 * Every cuda* failure throws `CudaError`. `deallocate` and stat
 * queries never throw.
 */
class CudaMemoryAllocator {
public:
    explicit CudaMemoryAllocator(CudaContext& ctx);
    ~CudaMemoryAllocator();

    CudaMemoryAllocator(const CudaMemoryAllocator&)            = delete;
    CudaMemoryAllocator& operator=(const CudaMemoryAllocator&) = delete;
    CudaMemoryAllocator(CudaMemoryAllocator&&)                 = delete;
    CudaMemoryAllocator& operator=(CudaMemoryAllocator&&)      = delete;

    /**
     * Allocate `bytes` of memory of the given kind. Returns a pointer
     * suitable for the kind's semantics (device VA for Device / Managed,
     * host VA for HostPinned). Throws `CudaError` on driver failure.
     * A zero-byte request returns nullptr without touching the driver.
     */
    [[nodiscard]] void* allocate(std::size_t bytes, CudaAllocKind kind = CudaAllocKind::Device);

    /**
     * Return a pointer previously obtained from `allocate`. Caller
     * passes back the original `kind` — CUDA has separate `cudaFree`
     * and `cudaFreeHost` entry points. `bytes` is stats-only. nullptr
     * is a no-op.
     */
    void deallocate(void* ptr, std::size_t bytes, CudaAllocKind kind) noexcept;

    // --- Synchronous transfers ------------------------------------------
    // Blocking `cudaMemcpy` on the current device's default (null) stream.

    void copyH2D(void* dst, const void* src, std::size_t bytes);
    void copyD2H(void* dst, const void* src, std::size_t bytes);
    void copyD2D(void* dst, const void* src, std::size_t bytes);

    // --- DGX Spark UMA-pressure helpers ----------------------------------
    // KV-managed fallback pattern adapted from antirez/ds4 (MIT).
    // Original: ds4_cuda.cu:2374-2408. See Synaipse
    // `reference_dgx_spark_engine_fit.md` for the design rationale.

    /**
     * Policy decision for a KV-cache allocation on unified-memory
     * NVIDIA parts (DGX Spark, Jetson). Returns `true` when the
     * requested KV-cache size is large enough (>= 8 GiB) AND the
     * free device memory would drop below a 2 GiB safety headroom,
     * signalling the caller should switch from `Device` to `Managed`
     * to let the driver page-fault rather than OOM the whole box.
     *
     * On discrete dGPUs this returns false — the policy only matters
     * for UMA parts where CPU+GPU share one physical pool.
     *
     * Safe to call before every KV-cache allocation; single
     * `cudaMemGetInfo` call, sub-microsecond.
     */
    [[nodiscard]] bool shouldUseManagedForKvCache(std::size_t kvBytes) noexcept;

    /**
     * Weight-tensor placement hint for UMA parts. Calls
     * `cudaMemAdvise(SetReadMostly)` + `cudaMemAdvise(SetPreferredLocation)`
     * on the given range so the driver's page-fault handler keeps
     * weight pages on the GPU side after first touch. No-op on
     * discrete GPUs (the advise APIs still accept but do nothing
     * useful there).
     *
     * Silently swallows non-fatal advise failures — the hint is
     * best-effort. Real driver errors are still logged at WARN.
     */
    void adviseReadMostly(void* ptr, std::size_t bytes) noexcept;

    // --- Introspection --------------------------------------------------

    [[nodiscard]] CudaContext&        context() noexcept { return _ctx; }
    [[nodiscard]] const CudaMemStats& stats() const noexcept { return _stats; }

    void logStats(::mimirmind::core::log::LogLevel lvl) const;

private:
    CudaContext&  _ctx;
    CudaMemStats  _stats{};

    void recordAlloc(std::size_t bytes) noexcept;
    void recordFree(std::size_t bytes) noexcept;
};

/**
 * RAII owner for a CUDA allocation. Frees on destruction, movable but
 * not copyable, same-thread-only. Same pattern as `HipBuffer`.
 */
class CudaBuffer {
public:
    CudaBuffer() noexcept = default;

    CudaBuffer(CudaMemoryAllocator& alloc,
               std::size_t          bytes,
               CudaAllocKind        kind = CudaAllocKind::Device);

    ~CudaBuffer() noexcept { reset(); }

    CudaBuffer(const CudaBuffer&)            = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    CudaBuffer(CudaBuffer&& other) noexcept;
    CudaBuffer& operator=(CudaBuffer&& other) noexcept;

    void reset() noexcept;

    [[nodiscard]] void*         data()  const noexcept { return _ptr; }
    [[nodiscard]] std::size_t   bytes() const noexcept { return _bytes; }
    [[nodiscard]] CudaAllocKind kind()  const noexcept { return _kind; }
    [[nodiscard]] bool          empty() const noexcept { return _ptr == nullptr; }

private:
    CudaMemoryAllocator* _alloc{nullptr};
    void*                _ptr{nullptr};
    std::size_t          _bytes{0};
    CudaAllocKind        _kind{CudaAllocKind::Device};
};

} // namespace mimirmind::core::cuda