// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/l0/UsmAllocator.hpp"

#include "core/log/Log.hpp"

#include <bit>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

namespace mimirmind::core::l0 {

namespace {

constexpr std::size_t   kProbeStartBytes      = 256ULL << 20;   // 256 MiB
constexpr std::size_t   kProbeBlockBytes      = 256ULL << 20;   // 256 MiB
constexpr std::uint32_t kProbeMaxBlocks       = 256;            // 256 × 256 MiB = 64 GiB ceiling
constexpr std::size_t   kProbeBisectStep      = 64ULL << 20;    // 64 MiB resolution
constexpr std::size_t   kProbeFallbackCeiling = 32ULL << 30;    // 32 GiB if device reports 0
// Default cap for Phase 2 (total-allocatable sweep). The result is only
// diagnostic; running the sweep to the full device-reported ceiling can
// occupy *all* of host RAM on UMA systems for a few seconds, which is
// risky on shared hosts. Override with `runtime.usmProbeTotalGib` in
// config.json. Set to 0 to skip Phase 2 entirely.
constexpr std::size_t   kProbeTotalDefaultGiB = 4;

std::size_t probeTotalCapBytes(std::optional<int> overrideGiB,
                               std::size_t        deviceCeiling) {
    if (overrideGiB.has_value()) {
        const int v = *overrideGiB;
        if (v <= 0) {
            return 0; // explicitly skip phase 2
        }
        const std::size_t bytes =
            static_cast<std::size_t>(v) * (1ULL << 30);
        return std::min(bytes, deviceCeiling);
    }
    return std::min(
        static_cast<std::size_t>(kProbeTotalDefaultGiB * (1ULL << 30)),
        deviceCeiling);
}

double bytesToMiB(std::size_t b) noexcept {
    return static_cast<double>(b) / (1024.0 * 1024.0);
}
double bytesToGiB(std::size_t b) noexcept {
    return static_cast<double>(b) / (1024.0 * 1024.0 * 1024.0);
}

} // namespace

// --- UsmAllocKind free helpers ----------------------------------------------

std::optional<UsmAllocKind> parseUsmAllocKind(std::string_view s) noexcept {
    if (s == "shared") return UsmAllocKind::Shared;
    if (s == "host")   return UsmAllocKind::Host;
    return std::nullopt;
}

UsmAllocKind selectUsmAllocKind(L0Context&   ctx,
                                UsmAllocKind fallback) noexcept {
    if (const char* env = std::getenv("MIMIRMIND_USM_KIND"); env != nullptr) {
        if (const auto k = parseUsmAllocKind(std::string_view{env}); k.has_value()) {
            MM_LOG_INFO("usm",
                        "MIMIRMIND_USM_KIND='{}' — bench override picks {}",
                        env, toString(*k));
            return *k;
        }
        MM_LOG_WARN("usm",
                    "MIMIRMIND_USM_KIND='{}' — unknown value, ignoring "
                    "(accepted: \"shared\", \"host\")",
                    env);
    }
    // ctx.info().isIntegrated was already probed at L0Context ctor time
    // via ZE_DEVICE_PROPERTY_FLAG_INTEGRATED — no need to re-query the
    // device properties here. Equivalent to
    // `ctx.hasFeature(BackendFeature::UnifiedMemoryHost)` but expressed
    // via the L0-native accessor since we're inside the L0 backend.
    if (ctx.info().isIntegrated) {
        MM_LOG_INFO("usm",
                    "integrated iGPU detected — picking host "
                    "(IPC-safe on Meteor Lake, perf-neutral to shared on UMA)");
        return UsmAllocKind::Host;
    }
    MM_LOG_INFO("usm",
                "discrete GPU detected — picking shared "
                "(placement hint drives VRAM/system-RAM migration)");
    return UsmAllocKind::Shared;
}

std::string_view toString(UsmAllocKind k) noexcept {
    switch (k) {
        case UsmAllocKind::Shared: return "shared";
        case UsmAllocKind::Host:   return "host";
    }
    return "?";
}

// --- Bucket sizing ----------------------------------------------------------

std::size_t UsmAllocator::bucketIndexFor(std::size_t bytes) noexcept {
    if (bytes <= kMinBucketBytes) {
        return 0;
    }
    // Smallest power-of-two >= bytes. bit_width(n-1) gives ceil(log2(n)).
    const auto width = std::bit_width(bytes - 1);
    // kMinBucketBytes == 2^12 -> index 0
    constexpr int kMinBucketShift = 12;
    const auto idx = static_cast<int>(width) - kMinBucketShift;
    if (idx >= static_cast<int>(kBucketCount)) {
        return kBucketCount;   // signals "oversized"
    }
    return static_cast<std::size_t>(idx);
}

std::size_t UsmAllocator::bucketSizeFor(std::size_t bytes) noexcept {
    const std::size_t idx = bucketIndexFor(bytes);
    if (idx >= kBucketCount) {
        return bytes;          // oversized: exact size, no rounding
    }
    return kMinBucketBytes << idx;
}

// --- ctor/dtor --------------------------------------------------------------

UsmAllocator::UsmAllocator(L0Context&          ctx,
                           std::optional<int>  probeTotalGiB,
                           UsmAllocKind        kind)
    : _ctx{ctx}, _probeTotalGiB{probeTotalGiB}, _kind{kind} {
    MM_LOG_INFO("usm", "alloc-kind={}", toString(_kind));
}

UsmAllocator::~UsmAllocator() {
    shrinkPool();
    if (_stats.liveAllocations > 0 || _stats.liveBytes > 0) {
        MM_LOG_WARN("usm",
                    "destructor with leaks — live={} allocs, {} bytes "
                    "({:.2f} MiB). Caller forgot to deallocate.",
                    _stats.liveAllocations,
                    _stats.liveBytes,
                    static_cast<double>(_stats.liveBytes) / (1024.0 * 1024.0));
    }
}

// --- raw alloc / free (no bookkeeping) --------------------------------------

void* UsmAllocator::rawAlloc(std::size_t bytes) {
    ze_device_mem_alloc_desc_t deviceDesc{};
    deviceDesc.stype   = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    deviceDesc.flags   = 0;
    deviceDesc.ordinal = 0;

    ze_host_mem_alloc_desc_t hostDesc{};
    hostDesc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
    hostDesc.flags = 0;

    void*             localPtr = nullptr;
    ze_result_t       r        = ZE_RESULT_SUCCESS;
    const char*       apiName  = "zeMemAllocShared";
    switch (_kind) {
        case UsmAllocKind::Shared:
            apiName = "zeMemAllocShared";
            r = zeMemAllocShared(
                _ctx.context(), &deviceDesc, &hostDesc,
                bytes, kAlignment, _ctx.device(), &localPtr);
            break;
        case UsmAllocKind::Host:
            apiName = "zeMemAllocHost";
            r = zeMemAllocHost(
                _ctx.context(), &hostDesc,
                bytes, kAlignment, &localPtr);
            break;
    }

    if (r == ZE_RESULT_SUCCESS && localPtr != nullptr) {
        MM_LOG_TRACE("usm",
                     "{} OK — size={} bytes ({:.2f} MiB) ptr={}",
                     apiName, bytes, bytesToMiB(bytes), localPtr);
        return localPtr;
    }
    MM_LOG_TRACE("usm",
                 "{} FAIL — size={} bytes ({:.2f} MiB) "
                 "result={} (0x{:x})",
                 apiName, bytes, bytesToMiB(bytes),
                 L0Context::resultToString(r),
                 static_cast<unsigned>(r));
    return nullptr;
}

void UsmAllocator::rawFree(void* ptr) noexcept {
    if (ptr == nullptr) {
        return;
    }
    const ze_result_t r = zeMemFree(_ctx.context(), ptr);
    if (r != ZE_RESULT_SUCCESS) {
        MM_LOG_WARN("usm",
                    "zeMemFree returned {} (0x{:x}) for ptr={}",
                    L0Context::resultToString(r),
                    static_cast<unsigned>(r),
                    ptr);
    }
}

// --- public allocate / deallocate -------------------------------------------

void* UsmAllocator::allocate(std::size_t bytes) {
    if (bytes == 0) {
        MM_LOG_WARN("usm", "allocate(0) — rounding up to {} bytes",
                    kMinBucketBytes);
        bytes = 1;
    }

    const std::size_t idx     = bucketIndexFor(bytes);
    const std::size_t bktSize = bucketSizeFor(bytes);
    const bool oversized      = (idx >= kBucketCount);

    void* ptr = nullptr;
    bool  hit = false;

    {
        std::scoped_lock lock{_mutex};

        if (!oversized) {
            auto& bkt = _buckets[idx];
            if (!bkt.cache.empty()) {
                ptr = bkt.cache.back();
                bkt.cache.pop_back();
                hit = true;
                ++bkt.hits;
                ++_stats.freeListHits;
            } else {
                ++bkt.misses;
                ++_stats.freeListMisses;
            }
        } else {
            ++_stats.oversizedAllocs;
        }
    }

    if (ptr == nullptr) {
        ptr = rawAlloc(bktSize);
        if (ptr == nullptr) {
            std::scoped_lock lock{_mutex};
            ++_stats.failedAllocs;
            MM_LOG_ERROR("usm",
                         "allocate FAIL — requested={} bytes ({:.2f} MiB) "
                         "bucket={} bytes ({:.2f} MiB) oversized={}",
                         bytes, bytesToMiB(bytes),
                         bktSize, bytesToMiB(bktSize),
                         oversized);
            throw L0Error("UsmAllocator::allocate",
                          ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY);
        }
        std::scoped_lock lock{_mutex};
        ++_stats.zeAllocCalls;
    }

    {
        std::scoped_lock lock{_mutex};

        ++_stats.totalAllocations;
        ++_stats.liveAllocations;
        _stats.bytesRequested += bytes;
        _stats.bytesServed    += bktSize;
        _stats.liveBytes      += bktSize;
        if (_stats.liveBytes > _stats.peakBytes) {
            _stats.peakBytes = _stats.liveBytes;
        }

        if (!oversized) {
            auto& bkt = _buckets[idx];
            ++bkt.allocs;
            bkt.liveBytes += bktSize;
            if (bkt.liveBytes > bkt.peakBytes) {
                bkt.peakBytes = bkt.liveBytes;
            }
        }
    }

    MM_LOG_DEBUG("usm",
                 "allocate — req={} bytes bucket={} bytes ({:.2f} MiB) "
                 "ptr={} src={}",
                 bytes, bktSize, bytesToMiB(bktSize), ptr,
                 hit ? "freelist" : "ze");
    return ptr;
}

void UsmAllocator::deallocate(void* ptr, std::size_t bytes) noexcept {
    if (ptr == nullptr) {
        return;
    }

    const std::size_t idx     = bucketIndexFor(bytes == 0 ? 1 : bytes);
    const std::size_t bktSize = bucketSizeFor(bytes == 0 ? 1 : bytes);
    const bool oversized      = (idx >= kBucketCount);

    bool cached = false;
    bool wasLive = true;

    {
        std::scoped_lock lock{_mutex};

        if (_stats.liveAllocations == 0 || _stats.liveBytes < bktSize) {
            wasLive = false;
            MM_LOG_WARN("usm",
                        "deallocate underflow — ptr={} bytes={} liveAllocs={} "
                        "liveBytes={} — likely double-free or wrong size",
                        ptr, bytes,
                        _stats.liveAllocations,
                        _stats.liveBytes);
        } else {
            --_stats.liveAllocations;
            _stats.liveBytes -= bktSize;
            ++_stats.totalDeallocations;
        }

        if (!oversized && wasLive) {
            auto& bkt = _buckets[idx];
            ++bkt.frees;
            if (bkt.liveBytes >= bktSize) {
                bkt.liveBytes -= bktSize;
            }
            if (bkt.cache.size() < kFreeListCapPerBucket) {
                bkt.cache.push_back(ptr);
                cached = true;
            }
        }
    }

    if (!cached) {
        rawFree(ptr);
        std::scoped_lock lock{_mutex};
        ++_stats.zeFreeCalls;
    }

    MM_LOG_DEBUG("usm",
                 "deallocate — req={} bytes bucket={} bytes ({:.2f} MiB) "
                 "ptr={} sink={}",
                 bytes, bktSize, bytesToMiB(bktSize), ptr,
                 cached ? "freelist" : "ze");
}

// --- stats / pool management ------------------------------------------------

UsmStats UsmAllocator::stats() const noexcept {
    std::scoped_lock lock{_mutex};
    return _stats;
}

void UsmAllocator::logStats(::mimirmind::core::log::LogLevel lvl) const {
    UsmStats s;
    std::array<Bucket, kBucketCount> bktSnap{};
    {
        std::scoped_lock lock{_mutex};
        s = _stats;
        for (std::size_t i = 0; i < kBucketCount; ++i) {
            bktSnap[i].allocs    = _buckets[i].allocs;
            bktSnap[i].frees     = _buckets[i].frees;
            bktSnap[i].hits      = _buckets[i].hits;
            bktSnap[i].misses    = _buckets[i].misses;
            bktSnap[i].liveBytes = _buckets[i].liveBytes;
            bktSnap[i].peakBytes = _buckets[i].peakBytes;
            bktSnap[i].cache     = _buckets[i].cache;   // copy of pointers
        }
    }

    const auto hitRatePct =
        (s.totalAllocations > 0)
        ? (100.0 * static_cast<double>(s.freeListHits) /
                   static_cast<double>(s.totalAllocations))
        : 0.0;

    ::mimirmind::core::log::detail::logFormat(
        lvl, "usm", std::source_location::current(),
        "stats — live: {} allocs / {} bytes ({:.2f} MiB), peak {} bytes "
        "({:.2f} MiB), total {} allocs / {} frees, ze {} alloc / {} free, "
        "free-list {} hits / {} misses ({:.1f}% hit rate), oversized={}, "
        "failed={}",
        s.liveAllocations, s.liveBytes, bytesToMiB(s.liveBytes),
        s.peakBytes, bytesToMiB(s.peakBytes),
        s.totalAllocations, s.totalDeallocations,
        s.zeAllocCalls, s.zeFreeCalls,
        s.freeListHits, s.freeListMisses, hitRatePct,
        s.oversizedAllocs, s.failedAllocs);

    for (std::size_t i = 0; i < kBucketCount; ++i) {
        const auto& bkt = bktSnap[i];
        if (bkt.allocs == 0 && bkt.frees == 0 && bkt.cache.empty()) {
            continue;
        }
        const std::size_t bktSize = kMinBucketBytes << i;
        MM_LOG_DEBUG("usm",
                     "bucket[{:>2}] size={:>10} ({:>8.2f} MiB) "
                     "allocs={} frees={} hits={} misses={} "
                     "cached={} live={} bytes peak={} bytes",
                     i, bktSize, bytesToMiB(bktSize),
                     bkt.allocs, bkt.frees, bkt.hits, bkt.misses,
                     bkt.cache.size(), bkt.liveBytes, bkt.peakBytes);
    }
}

void UsmAllocator::shrinkPool() noexcept {
    std::vector<void*> toFree;
    {
        std::scoped_lock lock{_mutex};
        for (auto& bkt : _buckets) {
            for (void* p : bkt.cache) {
                toFree.push_back(p);
            }
            bkt.cache.clear();
        }
    }
    if (toFree.empty()) {
        return;
    }
    MM_LOG_INFO("usm", "shrinkPool — releasing {} cached chunk(s)",
                toFree.size());
    for (void* p : toFree) {
        rawFree(p);
        std::scoped_lock lock{_mutex};
        ++_stats.zeFreeCalls;
    }
}

// --- probeLimits ------------------------------------------------------------

void UsmAllocator::probeLimits() {
    if (_limits.probed) {
        MM_LOG_DEBUG("usm", "probeLimits — already probed, no-op");
        return;
    }

    const std::size_t reported    = _ctx.info().totalLocalMem;
    const std::size_t hardCeiling = reported > 0 ? reported : kProbeFallbackCeiling;

    if (reported == 0) {
        MM_LOG_WARN("usm",
                    "device reports totalLocalMem=0 — falling back to "
                    "kProbeFallbackCeiling={} bytes ({:.2f} GiB)",
                    kProbeFallbackCeiling, bytesToGiB(kProbeFallbackCeiling));
    } else {
        MM_LOG_INFO("usm",
                    "probe ceiling = device-reported totalLocalMem = {} bytes "
                    "({:.2f} GiB)",
                    reported, bytesToGiB(reported));
    }

    // --- Phase 1 — per-allocation maximum ------------------------------------

    MM_LOG_INFO("usm",
                "phase 1 — per-alloc max: doubling from {} MiB up to {} MiB",
                kProbeStartBytes >> 20,
                hardCeiling >> 20);

    std::size_t lastGood = 0;
    std::size_t firstBad = 0;

    {
        std::size_t size = kProbeStartBytes;
        while (size <= hardCeiling) {
            void* ptr = rawAlloc(size);
            rawFree(ptr);
            if (ptr != nullptr) {
                lastGood = size;
                if (size == hardCeiling) {
                    break;
                }
                const std::size_t doubled = size << 1;
                size = doubled > hardCeiling ? hardCeiling : doubled;
            } else {
                firstBad = size;
                MM_LOG_DEBUG("usm",
                             "phase 1 — doubling stopped: lastGood={} MiB, "
                             "firstBad={} MiB",
                             lastGood >> 20, firstBad >> 20);
                break;
            }
        }
    }

    if (firstBad != 0 && firstBad - lastGood > kProbeBisectStep) {
        MM_LOG_DEBUG("usm",
                     "phase 1 — bisecting [{} MiB .. {} MiB], step={} MiB",
                     lastGood >> 20, firstBad >> 20, kProbeBisectStep >> 20);
        std::size_t lo = lastGood;
        std::size_t hi = firstBad;
        while (hi - lo > kProbeBisectStep) {
            const std::size_t mid = lo + ((hi - lo) >> 1);
            void* ptr = rawAlloc(mid);
            rawFree(ptr);
            if (ptr != nullptr) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lastGood = lo;
    }

    _limits.perAllocMaxBytes = lastGood;

    MM_LOG_INFO("usm",
                "phase 1 result — perAllocMax = {} bytes ({:.2f} GiB)",
                lastGood, bytesToGiB(lastGood));

    // --- Phase 2 — total allocatable -----------------------------------------

    const std::size_t phase2Cap = probeTotalCapBytes(_probeTotalGiB, hardCeiling);
    if (phase2Cap == 0) {
        MM_LOG_INFO("usm",
                    "phase 2 — skipped (runtime.usmProbeTotalGib=0); "
                    "totalAllocatable left at 0");
        _limits.probed = true;
        return;
    }

    MM_LOG_INFO("usm",
                "phase 2 — total allocatable: {} MiB blocks, cap={} MiB "
                "(runtime.usmProbeTotalGib overrides; "
                "device ceiling is {} MiB)",
                kProbeBlockBytes >> 20, phase2Cap >> 20, hardCeiling >> 20);

    const std::uint32_t maxBlocks = std::min<std::uint32_t>(
        kProbeMaxBlocks,
        static_cast<std::uint32_t>(phase2Cap / kProbeBlockBytes));

    std::vector<void*> blocks;
    blocks.reserve(maxBlocks);

    while (blocks.size() < maxBlocks) {
        const std::size_t soFar = blocks.size() * kProbeBlockBytes;
        if (soFar + kProbeBlockBytes > phase2Cap) {
            MM_LOG_DEBUG("usm",
                         "phase 2 — stopped at cap: {} blocks held, "
                         "next would exceed {} bytes",
                         blocks.size(), phase2Cap);
            break;
        }
        void* ptr = rawAlloc(kProbeBlockBytes);
        if (ptr == nullptr) {
            MM_LOG_DEBUG("usm",
                         "phase 2 — alloc failed at block #{} ({} MiB held)",
                         blocks.size(), soFar >> 20);
            break;
        }
        blocks.push_back(ptr);
        if (blocks.size() % 8 == 0) {
            MM_LOG_DEBUG("usm",
                         "phase 2 — progress: {} blocks held ({} MiB)",
                         blocks.size(),
                         (blocks.size() * kProbeBlockBytes) >> 20);
        }
    }

    _limits.probeBlocksGranted    = static_cast<std::uint32_t>(blocks.size());
    _limits.totalAllocatableBytes = blocks.size() * kProbeBlockBytes;

    MM_LOG_INFO("usm",
                "phase 2 result — totalAllocatable = {} bytes ({:.2f} GiB) "
                "across {} block(s) of {} MiB",
                _limits.totalAllocatableBytes,
                bytesToGiB(_limits.totalAllocatableBytes),
                blocks.size(), kProbeBlockBytes >> 20);

    MM_LOG_DEBUG("usm", "freeing {} probe blocks", blocks.size());
    for (void* p : blocks) {
        rawFree(p);
    }

    _limits.probed = true;
    MM_LOG_INFO("usm", "probeLimits done");
}

} // namespace mimirmind::core::l0