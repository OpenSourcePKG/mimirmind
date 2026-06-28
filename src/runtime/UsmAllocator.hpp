#pragma once

#include "runtime/L0Context.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace mimirmind::runtime {

enum class LogLevel : int;

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
    explicit UsmAllocator(L0Context& ctx);
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
    void logStats(LogLevel lvl) const;

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
    UsmLimits                        _limits{};
    mutable std::mutex               _mutex;
    std::array<Bucket, kBucketCount> _buckets{};
    UsmStats                         _stats{};
};

} // namespace mimirmind::runtime