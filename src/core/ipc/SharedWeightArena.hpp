// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <map>
#include <memory>
#include <mutex>
#include <string>

namespace mimirmind::core::ipc {

/**
 * 7.2.3 — a memfd-backed arena for materialized model weights that a second
 * process (Hugin) can map without a second copy.
 *
 * Owner mode (`create`): one sparse memfd of `reserveBytes`, mapped once
 * MAP_SHARED read/write at a stable base. `allocate` hands out page-aligned
 * ranges first-fit from a free list; `deallocate` returns a range and punches
 * it out of the memfd (FALLOC_FL_PUNCH_HOLE), so freed load-time slabs give
 * their physical pages back to the OS instead of to a driver pool. Pages are
 * pre-faulted on allocate (MADV_POPULATE_WRITE, best effort) so GPU kernels
 * writing a fresh range do not fault page by page.
 *
 * Attach mode (`attach`): maps a received fd read-only (or read/write) for
 * the attacher. It never allocates.
 *
 * GPU use requires a device with pageableMemoryAccess (GB10): kernels read the
 * host pointer directly, validated by tools/hugin-coexist-rig (7.0).
 *
 * Every live arena registers itself in a process-wide registry so a central
 * allocator can route a free of any arena pointer back here (`owning`).
 * Thread-safe.
 */
class SharedWeightArena {
public:
    static constexpr std::size_t kPageBytes = 4096;

    [[nodiscard]] static std::expected<std::unique_ptr<SharedWeightArena>, std::string>
    create(const std::string& name, std::uint64_t reserveBytes) noexcept;

    /// Map a received arena fd. Takes ownership of `fd` (closed on destruction).
    [[nodiscard]] static std::expected<std::unique_ptr<SharedWeightArena>, std::string>
    attach(int fd, std::uint64_t mapBytes, bool readOnly) noexcept;

    ~SharedWeightArena();

    SharedWeightArena(const SharedWeightArena&)            = delete;
    SharedWeightArena& operator=(const SharedWeightArena&) = delete;
    SharedWeightArena(SharedWeightArena&&)                 = delete;
    SharedWeightArena& operator=(SharedWeightArena&&)      = delete;

    /// Page-aligned range of at least `bytes`, or nullptr if the reserve is
    /// exhausted (or this is an attach-mode arena). Zero bytes -> nullptr.
    [[nodiscard]] void* allocate(std::size_t bytes) noexcept;

    /// Return a range from `allocate` and release its pages. Unknown pointers
    /// are ignored (logged by the caller if it cares).
    void deallocate(void* ptr) noexcept;

    [[nodiscard]] bool contains(const void* ptr) const noexcept;
    /// Byte offset of `ptr` from the arena base. Precondition: contains(ptr).
    [[nodiscard]] std::uint64_t offsetOf(const void* ptr) const noexcept;
    /// Size of the live allocation starting at `ptr`, 0 if none starts there.
    [[nodiscard]] std::size_t allocationBytes(const void* ptr) const noexcept;

    [[nodiscard]] void*         base() const noexcept { return _base; }
    [[nodiscard]] int           fd() const noexcept { return _fd; }
    [[nodiscard]] std::uint64_t mapBytes() const noexcept { return _mapBytes; }
    [[nodiscard]] bool          owner() const noexcept { return _owner; }
    /// End of the highest live allocation (the extent an attacher must map).
    [[nodiscard]] std::uint64_t highWaterBytes() const noexcept;
    [[nodiscard]] std::uint64_t liveBytes() const noexcept;
    [[nodiscard]] std::uint64_t peakBytes() const noexcept;
    [[nodiscard]] std::size_t   liveAllocations() const noexcept;

    /// The registered arena whose mapping contains `ptr`, or nullptr.
    [[nodiscard]] static SharedWeightArena* owning(const void* ptr) noexcept;

    /// True while any arena is alive in this process. Lock-free; lets hot
    /// paths keep their exact pre-arena behaviour when no arena exists.
    [[nodiscard]] static bool anyLive() noexcept;

private:
    SharedWeightArena(int fd, void* base, std::uint64_t mapBytes, bool owner) noexcept;

    int           _fd{-1};
    void*         _base{nullptr};
    std::uint64_t _mapBytes{0};
    bool          _owner{false};

    mutable std::mutex                    _mtx;
    std::map<std::uint64_t, std::uint64_t> _free;   ///< offset -> bytes, coalesced
    std::map<std::uint64_t, std::uint64_t> _live;   ///< offset -> bytes
    std::uint64_t                          _liveBytes{0};
    std::uint64_t                          _peakBytes{0};
};

/**
 * Thread-local routing guard: while alive, a central allocator that honours
 * it (CudaMemoryAllocator) serves weight allocations of this thread from
 * `arena`. Nested guards stack; nullptr disables routing for the scope.
 */
class ScopedWeightArena {
public:
    explicit ScopedWeightArena(SharedWeightArena* arena) noexcept;
    ~ScopedWeightArena();

    ScopedWeightArena(const ScopedWeightArena&)            = delete;
    ScopedWeightArena& operator=(const ScopedWeightArena&) = delete;

    [[nodiscard]] static SharedWeightArena* current() noexcept;

private:
    SharedWeightArena* _previous;
};

} // namespace mimirmind::core::ipc
