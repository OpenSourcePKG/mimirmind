// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <utility>

namespace mimirmind::core::gpu {

/**
 * Coarse ownership category for a device allocation, used by the central
 * allocators (CudaMemoryAllocator / UsmAllocator) to book each `allocate()`
 * into a bucket so `GET /v1/system/memory` (roadmap 8.16 Stage B) can report
 * where the device RAM actually went — instead of a single "external" lump.
 *
 * The active category is a thread_local slot set by an RAII `ScopedAllocCategory`
 * guard around the owning call-site (weight load, KV-pool ctor, session-buffer
 * setup). Any allocation made with no guard active is `Unknown` — the honest
 * bucket for untagged-but-central-allocator traffic (allocator scratch etc.).
 * Allocations that bypass the central allocator entirely (GpuMatmul's direct
 * cudaMalloc, cuBLAS/cuDNN internal workspaces) are not seen here at all and
 * remain in the route's `external` residual (deviceUsed - Σcategories).
 */
enum class AllocCategory : std::uint8_t {
    Unknown = 0,
    Weights,
    KvCache,
    Session,
    Scratch,
};

/// Number of AllocCategory values — the width of the per-category counter
/// arrays carried in the allocator stats + surfaced by ComputeOps.
inline constexpr std::size_t kAllocCategoryCount = 5;

/// Stable lower-case label for JSON keys / logs. Index-aligned with the enum.
[[nodiscard]] inline std::string_view allocCategoryName(AllocCategory c) noexcept {
    switch (c) {
        case AllocCategory::Weights: return "weights";
        case AllocCategory::KvCache: return "kv_cache";
        case AllocCategory::Session: return "session";
        case AllocCategory::Scratch: return "scratch";
        case AllocCategory::Unknown: return "unknown";
    }
    return "unknown";
}

[[nodiscard]] inline std::string_view allocCategoryName(std::size_t idx) noexcept {
    return allocCategoryName(static_cast<AllocCategory>(idx));
}

/**
 * RAII guard that sets the current thread's allocation category for its
 * lifetime and restores the previous value on destruction — the exact
 * stash/restore idiom of `core::security::ScopedTenant`, so nested guards
 * and reused pool threads never inherit a stale category. Wrap the body of an
 * owning call-site:
 *
 *   ScopedAllocCategory g{AllocCategory::Weights};
 *   loader.loadTensors(ops);   // every allocate() inside is tagged Weights
 *
 * Read side (inside the allocator's allocate()): `ScopedAllocCategory::current()`.
 */
class ScopedAllocCategory {
public:
    explicit ScopedAllocCategory(AllocCategory category) noexcept
        : _previous{slot()} {
        slot() = category;
    }

    ~ScopedAllocCategory() { slot() = _previous; }

    ScopedAllocCategory(const ScopedAllocCategory&)            = delete;
    ScopedAllocCategory& operator=(const ScopedAllocCategory&) = delete;
    ScopedAllocCategory(ScopedAllocCategory&&)                 = delete;
    ScopedAllocCategory& operator=(ScopedAllocCategory&&)      = delete;

    /// The active category for the calling thread (Unknown when no guard).
    [[nodiscard]] static AllocCategory current() noexcept { return slot(); }

private:
    static AllocCategory& slot() noexcept {
        static thread_local AllocCategory cat{AllocCategory::Unknown};
        return cat;
    }

    AllocCategory _previous;
};

} // namespace mimirmind::core::gpu
