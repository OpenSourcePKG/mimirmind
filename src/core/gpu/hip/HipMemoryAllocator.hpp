// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/gpu/hip/HipContext.hpp"
#include "core/log/Log.hpp"

#include <cstddef>
#include <cstdint>

namespace mimirmind::core::hip {

/**
 * Which `hipMalloc*` variant backs an allocation.
 *
 *   Device      — hipMalloc. VRAM on a dGPU; host-side reads require
 *                 an explicit hipMemcpy. On integrated APUs the driver
 *                 backs it from shared RAM but callers still cannot
 *                 dereference the pointer from host code.
 *   HostPinned  — hipHostMalloc. Page-locked host memory, mapped so
 *                 the GPU can DMA into/out-of it. Fast staging buffer
 *                 for H↔D transfers. Not device-native — kernel arg
 *                 must be the *device pointer* obtained via
 *                 `hipHostGetDevicePointer` if the kernel needs to
 *                 read from it directly. For our staging use-case
 *                 (host writes, device consumes via memcpy) that
 *                 lookup is unneeded.
 *   Managed     — hipMallocManaged. Driver-migrated between host and
 *                 device page-by-page. Cheap on UMA APUs (single
 *                 physical pool), expensive on dGPU (real PCIe
 *                 migration). Skeleton exposes it for symmetry with
 *                 the L0 `UsmAllocKind::Shared` — actual perf story
 *                 for dGPU is measure-first.
 */
enum class HipAllocKind {
    Device,
    HostPinned,
    Managed,
};

/**
 * Snapshot of per-process HipMemoryAllocator bookkeeping. Copyable
 * value type — not a live reference.
 */
struct HipMemStats {
    std::uint64_t liveBytes           {0};
    std::uint64_t peakBytes           {0};
    std::uint64_t liveAllocations     {0};
    std::uint64_t totalAllocations    {0};
    std::uint64_t totalDeallocations  {0};
    std::uint64_t bytesCopiedH2D      {0};
    std::uint64_t bytesCopiedD2H      {0};
    std::uint64_t bytesCopiedD2D      {0};
    std::uint64_t failedAllocs        {0};
};

/**
 * Thin per-process HIP memory manager. Direct pass-through to
 * `hipMalloc` / `hipHostMalloc` / `hipMallocManaged` for allocation
 * and to synchronous `hipMemcpy` for transfers — no pool, no
 * free-list, no stream binding. The bucket-pool + limits-probe layer
 * from the L0 `UsmAllocator` is deliberately out of scope: we add
 * that once the kernel-port hits allocation-churn perf on
 * HIP_TARGET_HOST.
 *
 * Not thread-safe by contract; the underlying HIP runtime allows
 * concurrent calls on different streams, but this class's stats
 * bookkeeping isn't guarded. Wrap in a mutex at the caller level if
 * two threads must share the same allocator instance.
 *
 * Every hip* failure throws `HipError`. `deallocate` and stat
 * queries never throw.
 */
class HipMemoryAllocator {
public:
    explicit HipMemoryAllocator(HipContext& ctx);
    ~HipMemoryAllocator();

    HipMemoryAllocator(const HipMemoryAllocator&)            = delete;
    HipMemoryAllocator& operator=(const HipMemoryAllocator&) = delete;
    HipMemoryAllocator(HipMemoryAllocator&&)                 = delete;
    HipMemoryAllocator& operator=(HipMemoryAllocator&&)      = delete;

    /**
     * Allocate `bytes` of memory of the given kind. Returns a pointer
     * suitable for the kind's semantics (device VA for Device / Managed,
     * host VA for HostPinned). Throws `HipError` on driver failure.
     * A zero-byte request returns nullptr without touching the driver.
     */
    [[nodiscard]] void* allocate(std::size_t bytes, HipAllocKind kind = HipAllocKind::Device);

    /**
     * Return a pointer previously obtained from `allocate`. The caller
     * must pass back the original `kind` — HIP has separate `hipFree`
     * and `hipHostFree` entry points that would mis-account otherwise.
     * `bytes` is used for the stats update only (there is no pool to
     * decide bucket for). nullptr is a no-op.
     */
    void deallocate(void* ptr, std::size_t bytes, HipAllocKind kind) noexcept;

    // --- Synchronous transfers ------------------------------------------
    // Blocking `hipMemcpy` on the current device's default (null) stream.
    // Async variants come with HipStream in a follow-up commit.

    void copyH2D(void* dst, const void* src, std::size_t bytes);
    void copyD2H(void* dst, const void* src, std::size_t bytes);
    void copyD2D(void* dst, const void* src, std::size_t bytes);

    // --- Introspection --------------------------------------------------

    [[nodiscard]] HipContext&        context() noexcept { return _ctx; }
    [[nodiscard]] const HipMemStats& stats() const noexcept { return _stats; }

    /// Emit a one-line stats summary at the given log level. Cheap.
    void logStats(::mimirmind::core::log::LogLevel lvl) const;

private:
    HipContext&  _ctx;
    HipMemStats  _stats{};

    void recordAlloc(std::size_t bytes) noexcept;
    void recordFree(std::size_t bytes) noexcept;
};

/**
 * RAII owner for a HIP allocation. Frees on destruction, movable but
 * not copyable, same-thread-only. The typical use pattern is
 *
 *     HipBuffer buf{alloc, 4 << 20};          // 4 MiB device alloc
 *     alloc.copyH2D(buf.data(), hostSrc, buf.bytes());
 *     kernel<<<...>>>(buf.data(), ...);
 *     alloc.copyD2H(hostDst, buf.data(), buf.bytes());
 *
 * Move-only. Assignment to an already-owning buffer releases the
 * previous allocation first.
 */
class HipBuffer {
public:
    HipBuffer() noexcept = default;

    /// Allocate `bytes` from `alloc` immediately. Throws `HipError` on
    /// driver failure — no owning state constructed in that case.
    HipBuffer(HipMemoryAllocator& alloc,
              std::size_t         bytes,
              HipAllocKind        kind = HipAllocKind::Device);

    ~HipBuffer() noexcept { reset(); }

    HipBuffer(const HipBuffer&)            = delete;
    HipBuffer& operator=(const HipBuffer&) = delete;

    HipBuffer(HipBuffer&& other) noexcept;
    HipBuffer& operator=(HipBuffer&& other) noexcept;

    /// Release the current allocation, if any. Idempotent.
    void reset() noexcept;

    [[nodiscard]] void*        data()  const noexcept { return _ptr; }
    [[nodiscard]] std::size_t  bytes() const noexcept { return _bytes; }
    [[nodiscard]] HipAllocKind kind()  const noexcept { return _kind; }
    [[nodiscard]] bool         empty() const noexcept { return _ptr == nullptr; }

private:
    HipMemoryAllocator* _alloc{nullptr};
    void*               _ptr{nullptr};
    std::size_t         _bytes{0};
    HipAllocKind        _kind{HipAllocKind::Device};
};

} // namespace mimirmind::core::hip