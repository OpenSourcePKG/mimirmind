// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/gpu/l0/L0Context.hpp"
#include "core/log/Log.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace mimirmind::core::l0 {

/**
 * Which zeMemAlloc* variant backs the free-list rawAlloc path.
 *
 *   Shared — zeMemAllocShared. Baseline behaviour: driver decides
 *            placement (host-RAM vs device-RAM) and may migrate.
 *            On UMA-iGPU (Meteor Lake, Lunar Lake, Alder/Raptor Lake)
 *            the hint is inert because there is no discrete VRAM.
 *
 *   Host   — zeMemAllocHost. Always host-RAM allocation; the iGPU
 *            still reads it via UMA-DMA on Meteor Lake, so decode
 *            perf is identical to Shared on this class of hardware.
 *            Required for L0 IPC on Meteor Lake — the Intel L0
 *            driver's zeMemOpenIpcHandle path returns unusable
 *            kernel-VA pointers for Shared and Device allocations
 *            (verified 2026-07-13 via tools/l0-ipc-testrig; see
 *            research/l0-ipc-host-only-meteor-lake.md).
 *
 * The Munin daemon must use Host so it can export model tensors
 * via SCM_RIGHTS to attached mimirmind workers. Standalone
 * mimirmind can pick either; Shared is the historical default.
 */
enum class UsmAllocKind {
    Shared,
    Host,
};

/// Parse a kind name ("shared" / "host") to UsmAllocKind. Returns
/// std::nullopt for unrecognised values.
[[nodiscard]] std::optional<UsmAllocKind>
parseUsmAllocKind(std::string_view s) noexcept;

/// Pick the kind the process should use.
///
/// Precedence:
///   1. `MIMIRMIND_USM_KIND` env var (values "shared" or "host"). Only
///      set this when comparing the two kinds during a perf-bench —
///      it is intentionally undocumented in config.example.json so
///      operators can't fat-finger it.
///   2. Hardware auto-detect via `ZE_DEVICE_PROPERTY_FLAG_INTEGRATED`:
///      integrated iGPU → `Host` (unblocks M-Munin IPC without operator
///      action — perf-parity to Shared verified 2026-07-13 on
///      L0_TARGET_HOST, ledger entry #31), discrete GPU → `Shared`
///      (Shared→Device migration remains a real perf lever on dGPU).
///   3. `fallback` — only used when `zeDeviceGetProperties` itself
///      fails, which should be near-impossible after L0Context is up.
[[nodiscard]] UsmAllocKind
selectUsmAllocKind(L0Context&   ctx,
                   UsmAllocKind fallback = UsmAllocKind::Shared) noexcept;

/// Round-trip name of a kind for logs and diagnostics.
[[nodiscard]] std::string_view toString(UsmAllocKind k) noexcept;

/**
 * Empirically-discovered allocation limits on the current device/loader.
 * Populated by UsmAllocator::probeLimits(); zero-initialised otherwise.
 *
 * `maxMemAllocSize` from ze_device_properties_t is what the loader
 * declares. The real ceiling on Meteor Lake (and on older Intel iGPUs)
 * routinely differs — sometimes lower than declared due to driver
 * fragmentation, sometimes higher when the relaxed-allocation ENV vars
 * are set. We trust only what we measure.
 */
struct UsmLimits {
    std::size_t   perAllocMaxBytes{0};
    std::size_t   totalAllocatableBytes{0};
    std::uint32_t probeBlocksGranted{0};
    bool          probed{false};
};

/**
 * Process-wide allocator stats. Snapshot, not a live reference.
 */
struct UsmStats {
    std::uint64_t liveBytes          {0};
    std::uint64_t peakBytes          {0};
    std::uint64_t liveAllocations    {0};
    std::uint64_t totalAllocations   {0};
    std::uint64_t totalDeallocations {0};
    std::uint64_t zeAllocCalls       {0};
    std::uint64_t zeFreeCalls        {0};
    std::uint64_t freeListHits       {0};
    std::uint64_t freeListMisses     {0};
    std::uint64_t bytesRequested     {0};   // sum across alloc() calls
    std::uint64_t bytesServed        {0};   // sum of bucket sizes handed out
    std::uint64_t oversizedAllocs    {0};   // > kMaxBucketBytes (no recycling)
    std::uint64_t failedAllocs       {0};
};

/**
 * Per-tensor `zeMemAllocShared` allocator with a segregated power-of-two
 * free list and bookkeeping.
 *
 * Design notes:
 *   - Buckets are sized 4 KiB, 8 KiB, ..., 256 MiB (17 buckets).
 *   - Requests > 256 MiB are passed through (exact-size alloc, no caching).
 *   - The free list caches up to `kFreeListCapPerBucket` chunks per
 *     bucket; surplus frees are returned to the driver immediately.
 *   - Caller must pass the original requested size back to deallocate().
 *     Round-trip is deterministic because bucketSizeFor() is pure.
 *
 * See [[ADR — Memory Strategy: Per-Tensor USM]].
 */
class UsmAllocator {
public:
    /// `probeTotalGiB`: from `runtime.usmProbeTotalGib` in config.json. Caps
    /// how much host RAM the Phase 2 sweep is allowed to touch. `0` skips
    /// Phase 2 entirely. `std::nullopt` uses the compiled default (4 GiB).
    ///
    /// `kind` selects the underlying zeMemAlloc* variant. Default `Shared`
    /// preserves pre-M-Munin behaviour; Munin and its attached workers
    /// pick `Host` (see UsmAllocKind docs above).
    explicit UsmAllocator(L0Context&                ctx,
                          std::optional<int>        probeTotalGiB = {},
                          UsmAllocKind              kind          = UsmAllocKind::Shared);

    [[nodiscard]] UsmAllocKind kind() const noexcept { return _kind; }
    ~UsmAllocator();

    UsmAllocator(const UsmAllocator&)            = delete;
    UsmAllocator& operator=(const UsmAllocator&) = delete;
    UsmAllocator(UsmAllocator&&)                 = delete;
    UsmAllocator& operator=(UsmAllocator&&)      = delete;

    /**
     * Walk the per-allocation ceiling (doubling + bisection) and the
     * total-allocatable ceiling (256-MiB-block sweep). Idempotent — a
     * second call is a no-op. May spend several seconds on devices with
     * lots of addressable shared memory.
     */
    void probeLimits();

    [[nodiscard]] const UsmLimits& limits() const noexcept { return _limits; }

    /**
     * Allocate `bytes` of USM-shared memory. Rounds up to the smallest
     * power-of-two bucket >= bytes, serves from the free-list when warm,
     * falls through to zeMemAllocShared otherwise. Throws L0Error on
     * driver failure.
     */
    [[nodiscard]] void* allocate(std::size_t bytes);

    /**
     * Return a chunk previously obtained from allocate(). `bytes` must be
     * the size originally requested — the round-trip through bucketSizeFor
     * recovers the bucket. Caller is responsible for matching sizes; never
     * throws. nullptr is a no-op.
     */
    void deallocate(void* ptr, std::size_t bytes) noexcept;

    [[nodiscard]] UsmStats stats() const noexcept;

    /// Emit a stats summary at `lvl`, plus per-bucket detail at DEBUG.
    void logStats(::mimirmind::core::log::LogLevel lvl) const;

    /// Drop all cached free-list chunks back to the driver. Idempotent.
    void shrinkPool() noexcept;

    // --- Bucket layout helpers (exposed for tests and diagnostics) -------

    static constexpr std::size_t kAlignment              = 4096;
    static constexpr std::size_t kMinBucketBytes         = 4096;       // 4 KiB
    static constexpr std::size_t kMaxBucketBytes         = 1ULL << 20; // 1 MiB
    static constexpr std::size_t kBucketCount            = 9;          // 4K..1M inclusive
    // Anything > kMaxBucketBytes is allocated at exact size (no rounding,
    // no free-list reuse). Power-of-two rounding above 1 MiB wastes up to
    // 50% — fatal for FFN weights (Qwen2.5-7B: 38 MiB ffn rows in 64 MiB
    // buckets cost ~2 GiB of phantom resident).
    static constexpr std::size_t kFreeListCapPerBucket   = 4;

    [[nodiscard]] static std::size_t bucketIndexFor(std::size_t bytes) noexcept;
    [[nodiscard]] static std::size_t bucketSizeFor(std::size_t bytes) noexcept;

private:
    struct Bucket {
        std::vector<void*> cache;
        std::uint64_t      allocs    {0};
        std::uint64_t      frees     {0};
        std::uint64_t      hits      {0};
        std::uint64_t      misses    {0};
        std::uint64_t      liveBytes {0};
        std::uint64_t      peakBytes {0};
    };

    void* rawAlloc(std::size_t bytes);
    void  rawFree(void* ptr) noexcept;

    L0Context&                       _ctx;
    std::optional<int>               _probeTotalGiB{};
    UsmAllocKind                     _kind{UsmAllocKind::Shared};
    UsmLimits                        _limits{};
    mutable std::mutex               _mutex;
    std::array<Bucket, kBucketCount> _buckets{};
    UsmStats                         _stats{};
};

} // namespace mimirmind::core::l0