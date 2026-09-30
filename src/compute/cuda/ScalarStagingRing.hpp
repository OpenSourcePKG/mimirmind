// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/gpu/cuda/CudaMemoryAllocator.hpp"

#include <cstddef>
#include <cstdint>

namespace mimirmind::compute::cuda {

/**
 * Pinned-host ring for scalar (int32) H2D staging on the decode hot path
 * (8.30.11.4, carved out of GpuOps). The hot path updates a device scalar slot
 * (`_curLenSlotUsm`) ~3× per block × N blocks per token; a sync copy from a
 * stack local forced the host to wait for the compute stream to drain each time
 * (profiling showed the GPU ~96% idle in decode). A page-locked ring lets
 * `cudaMemcpyAsync` truly enqueue without stalling; the ring cycles cleanly
 * (kSize slots >> max in-flight copies per decode step).
 *
 * RAII: `init()` allocates the pinned ring from the given allocator; the dtor
 * frees it. The allocator must outlive the ring (GpuOps holds it via its
 * CudaComputeContext, destroyed after this member). CUDA-only.
 */
class ScalarStagingRing {
public:
    ScalarStagingRing() = default;
    ScalarStagingRing(const ScalarStagingRing&)            = delete;
    ScalarStagingRing& operator=(const ScalarStagingRing&) = delete;

    ~ScalarStagingRing() {
        if (_ring != nullptr) {
            _alloc->deallocate(_ring, kSize * sizeof(std::int32_t),
                               core::cuda::CudaAllocKind::HostPinned);
        }
    }

    /// Allocate the pinned ring. Call once, at owner construction.
    void init(core::cuda::CudaMemoryAllocator& alloc) {
        _alloc = &alloc;
        _ring  = static_cast<std::int32_t*>(
            alloc.allocate(kSize * sizeof(std::int32_t),
                           core::cuda::CudaAllocKind::HostPinned));
        _idx = 0;
    }

    /// Stage `value` into the next pinned slot, advance the ring, and return the
    /// slot pointer for an async H2D copy. The slot stays valid until the ring
    /// wraps back to it (kSize copies later).
    [[nodiscard]] std::int32_t* stage(std::int32_t value) {
        std::int32_t* slot = &_ring[_idx];
        *slot = value;
        _idx  = (_idx + 1) & (kSize - 1);
        return slot;
    }

private:
    // kSize is a power of two so the advance is a mask, not a modulo. Generous
    // vs. the max in-flight copies in one decode step (< 100 per token).
    static constexpr std::size_t kSize = 256;

    core::cuda::CudaMemoryAllocator* _alloc{nullptr};
    std::int32_t*                    _ring{nullptr};
    std::size_t                      _idx{0};
};

} // namespace mimirmind::compute::cuda
