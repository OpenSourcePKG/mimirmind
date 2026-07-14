// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/gpu/hip/HipMemoryAllocator.hpp"

#include "core/log/Log.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <string>

namespace mimirmind::core::hip {

namespace {

// Wrap a hip* call — throw HipError on non-Success. Kept local to this
// TU rather than promoted to a header so HipContext.hpp doesn't need
// to expose it. Same idiom the HipContext ctor uses.
inline void hipCheck(const char* call, hipError_t code) {
    if (code != hipSuccess) {
        throw HipError(call, code);
    }
}

const char* kindName(HipAllocKind k) noexcept {
    switch (k) {
        case HipAllocKind::Device:     return "Device";
        case HipAllocKind::HostPinned: return "HostPinned";
        case HipAllocKind::Managed:    return "Managed";
    }
    return "Unknown";
}

} // namespace

// ---------- HipMemoryAllocator --------------------------------------------

HipMemoryAllocator::HipMemoryAllocator(HipContext& ctx) : _ctx(ctx) {
    MM_LOG_INFO("HipMem", "allocator bound to device #{} ({})",
                _ctx.hipDeviceIndex(), _ctx.hipDeviceInfo().name);
}

HipMemoryAllocator::~HipMemoryAllocator() {
    if (_stats.liveAllocations > 0 || _stats.liveBytes > 0) {
        MM_LOG_WARN("HipMem",
                    "dtor with {} live allocation(s) / {} live bytes — leak or "
                    "caller forgot to deallocate",
                    _stats.liveAllocations, _stats.liveBytes);
    }
}

void HipMemoryAllocator::recordAlloc(std::size_t bytes) noexcept {
    _stats.totalAllocations += 1;
    _stats.liveAllocations  += 1;
    _stats.liveBytes        += bytes;
    _stats.peakBytes = std::max(_stats.peakBytes, _stats.liveBytes);
}

void HipMemoryAllocator::recordFree(std::size_t bytes) noexcept {
    _stats.totalDeallocations += 1;
    if (_stats.liveAllocations > 0) _stats.liveAllocations -= 1;
    _stats.liveBytes = (_stats.liveBytes >= bytes) ? _stats.liveBytes - bytes : 0;
}

void* HipMemoryAllocator::allocate(std::size_t bytes, HipAllocKind kind) {
    if (bytes == 0) return nullptr;

    void*      ptr = nullptr;
    hipError_t rc  = hipSuccess;

    switch (kind) {
        case HipAllocKind::Device:
            rc = hipMalloc(&ptr, bytes);
            break;
        case HipAllocKind::HostPinned:
            // hipHostMallocDefault flag: don't request mapped or
            // write-combined (mapped is only needed if the kernel reads
            // the host pointer directly, which we don't do; write-combined
            // is a niche perf trade-off for write-only staging that we
            // don't need in the skeleton).
            rc = hipHostMalloc(&ptr, bytes, hipHostMallocDefault);
            break;
        case HipAllocKind::Managed:
            rc = hipMallocManaged(&ptr, bytes, hipMemAttachGlobal);
            break;
    }

    if (rc != hipSuccess) {
        _stats.failedAllocs += 1;
        throw HipError(std::string("hip alloc (") + kindName(kind) + ")", rc);
    }
    recordAlloc(bytes);
    return ptr;
}

void HipMemoryAllocator::deallocate(void*        ptr,
                                    std::size_t  bytes,
                                    HipAllocKind kind) noexcept {
    if (ptr == nullptr) return;

    hipError_t rc = hipSuccess;
    switch (kind) {
        case HipAllocKind::Device:
        case HipAllocKind::Managed:
            rc = hipFree(ptr);
            break;
        case HipAllocKind::HostPinned:
            rc = hipHostFree(ptr);
            break;
    }
    if (rc != hipSuccess) {
        // Never throws — swallow and log. Frees during shutdown are
        // routine even when the runtime has already been torn down; a
        // WARN is honest but not actionable.
        MM_LOG_WARN("HipMem", "free ({}) failed: {}",
                    kindName(kind), hipGetErrorString(rc));
    }
    recordFree(bytes);
}

void HipMemoryAllocator::copyH2D(void* dst, const void* src, std::size_t bytes) {
    if (bytes == 0) return;
    hipCheck("hipMemcpyHostToDevice",
             hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice));
    _stats.bytesCopiedH2D += bytes;
}

void HipMemoryAllocator::copyD2H(void* dst, const void* src, std::size_t bytes) {
    if (bytes == 0) return;
    hipCheck("hipMemcpyDeviceToHost",
             hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost));
    _stats.bytesCopiedD2H += bytes;
}

void HipMemoryAllocator::copyD2D(void* dst, const void* src, std::size_t bytes) {
    if (bytes == 0) return;
    hipCheck("hipMemcpyDeviceToDevice",
             hipMemcpy(dst, src, bytes, hipMemcpyDeviceToDevice));
    _stats.bytesCopiedD2D += bytes;
}

void HipMemoryAllocator::logStats(::mimirmind::core::log::LogLevel lvl) const {
    ::mimirmind::core::log::detail::logFormat(
        lvl, "HipMem", std::source_location::current(),
        "live={} allocs={} peak={} MiB total_alloc={} total_free={} "
        "H2D={} MiB D2H={} MiB D2D={} MiB fails={}",
        _stats.liveAllocations,
        _stats.liveBytes,
        _stats.peakBytes >> 20,
        _stats.totalAllocations,
        _stats.totalDeallocations,
        _stats.bytesCopiedH2D >> 20,
        _stats.bytesCopiedD2H >> 20,
        _stats.bytesCopiedD2D >> 20,
        _stats.failedAllocs);
}

// ---------- HipBuffer ------------------------------------------------------

HipBuffer::HipBuffer(HipMemoryAllocator& alloc,
                     std::size_t         bytes,
                     HipAllocKind        kind)
    : _alloc(&alloc), _bytes(bytes), _kind(kind) {
    // If allocate throws, _alloc has been set but _ptr stays nullptr, and
    // reset() is a no-op — no double-free risk.
    _ptr = alloc.allocate(bytes, kind);
}

HipBuffer::HipBuffer(HipBuffer&& other) noexcept
    : _alloc(other._alloc),
      _ptr  (other._ptr),
      _bytes(other._bytes),
      _kind (other._kind) {
    other._alloc = nullptr;
    other._ptr   = nullptr;
    other._bytes = 0;
}

HipBuffer& HipBuffer::operator=(HipBuffer&& other) noexcept {
    if (this != &other) {
        reset();
        _alloc = other._alloc;
        _ptr   = other._ptr;
        _bytes = other._bytes;
        _kind  = other._kind;
        other._alloc = nullptr;
        other._ptr   = nullptr;
        other._bytes = 0;
    }
    return *this;
}

void HipBuffer::reset() noexcept {
    if (_ptr != nullptr && _alloc != nullptr) {
        _alloc->deallocate(_ptr, _bytes, _kind);
    }
    _alloc = nullptr;
    _ptr   = nullptr;
    _bytes = 0;
}

} // namespace mimirmind::core::hip