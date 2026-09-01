// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/gpu/cuda/CudaMemoryAllocator.hpp"

#include "core/log/Log.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#ifdef __linux__
#include <execinfo.h>   // 8.16 (a) untagged-alloc backtrace finder (temporary)
#endif

namespace mimirmind::core::cuda {

namespace {

inline void cudaCheck(const char* call, cudaError_t code) {
    if (code != cudaSuccess) {
        throw CudaError(call, code);
    }
}

const char* kindName(CudaAllocKind k) noexcept {
    switch (k) {
        case CudaAllocKind::Device:     return "Device";
        case CudaAllocKind::HostPinned: return "HostPinned";
        case CudaAllocKind::Managed:    return "Managed";
    }
    return "Unknown";
}

// KV-managed fallback thresholds — adapted from antirez/ds4
// ds4_cuda.cu:2374-2408. Named constants so tomorrow's Spark tuning
// touches one place instead of magic numbers scattered across the
// allocator. If the KV-cache exceeds kKvManagedMinBytes AND free
// device memory would drop below kKvManagedFreeHeadroom, switch to
// cudaMallocManaged so the driver page-faults rather than OOMs.
constexpr std::size_t kKvManagedMinBytes      = std::size_t{8} * 1024 * 1024 * 1024;   // 8 GiB
constexpr std::size_t kKvManagedFreeHeadroom  = std::size_t{2} * 1024 * 1024 * 1024;   // 2 GiB

} // namespace

// ---------- CudaMemoryAllocator --------------------------------------------

CudaMemoryAllocator::CudaMemoryAllocator(CudaContext& ctx) : _ctx(ctx) {
    MM_LOG_INFO("CudaMem", "allocator bound to device #{} ({})",
                _ctx.cudaDeviceIndex(), _ctx.cudaDeviceInfo().name);
}

CudaMemoryAllocator::~CudaMemoryAllocator() {
    if (_stats.liveAllocations > 0 || _stats.liveBytes > 0) {
        MM_LOG_WARN("CudaMem",
                    "dtor with {} live allocation(s) / {} live bytes — leak or "
                    "caller forgot to deallocate",
                    _stats.liveAllocations, _stats.liveBytes);
    }
}

void CudaMemoryAllocator::recordAlloc(std::size_t bytes) noexcept {
    _stats.totalAllocations += 1;
    _stats.liveAllocations  += 1;
    _stats.liveBytes        += bytes;
    _stats.peakBytes = std::max(_stats.peakBytes, _stats.liveBytes);
}

void CudaMemoryAllocator::recordFree(std::size_t bytes) noexcept {
    _stats.totalDeallocations += 1;
    if (_stats.liveAllocations > 0) _stats.liveAllocations -= 1;
    _stats.liveBytes = (_stats.liveBytes >= bytes) ? _stats.liveBytes - bytes : 0;
}

void CudaMemoryAllocator::chargeCategory(void* ptr, std::size_t bytes) noexcept {
    const auto        cat = gpu::ScopedAllocCategory::current();
    const std::size_t idx = static_cast<std::size_t>(cat) < gpu::kAllocCategoryCount
                          ? static_cast<std::size_t>(cat) : 0;
    // 8.16 Stage B (a) — temporary finder: log a backtrace for large UNTAGGED
    // allocations so the untagged call-sites can be wrapped with a guard.
    // Env-gated (MIMIRMIND_ALLOC_TRACE=<min-MiB>), off by default -> zero cost.
    if (cat == gpu::AllocCategory::Unknown) {
        static const std::size_t traceMinBytes = [] {
            const char* e = std::getenv("MIMIRMIND_ALLOC_TRACE");
            const long  mib = (e != nullptr) ? std::strtol(e, nullptr, 10) : 0;
            return mib > 0 ? static_cast<std::size_t>(mib) * (1ULL << 20) : 0;
        }();
        if (traceMinBytes > 0 && bytes >= traceMinBytes) {
            std::string bt;
#ifdef __linux__
            void*  frames[12];
            const int n = ::backtrace(frames, 12);
            char** syms = ::backtrace_symbols(frames, n);
            for (int i = 2; i < n && i < 8; ++i) {   // skip this fn + allocate()
                bt += "\n    ";
                bt += (syms != nullptr && syms[i] != nullptr) ? syms[i] : "?";
            }
            if (syms != nullptr) std::free(syms);
#endif
            MM_LOG_INFO("CudaMem", "ALLOC_TRACE untagged {} MiB:{}",
                        bytes >> 20, bt);
        }
    }
    std::lock_guard<std::mutex> lk{_catMutex};
    _ptrCategory[ptr] = cat;
    _stats.liveBytesByCategory[idx] += bytes;
    _stats.peakBytesByCategory[idx] =
        std::max(_stats.peakBytesByCategory[idx], _stats.liveBytesByCategory[idx]);
}

void CudaMemoryAllocator::unchargeCategory(void* ptr, std::size_t bytes) noexcept {
    std::lock_guard<std::mutex> lk{_catMutex};
    const auto it = _ptrCategory.find(ptr);
    const auto cat = (it != _ptrCategory.end()) ? it->second
                                                : gpu::AllocCategory::Unknown;
    const std::size_t idx = static_cast<std::size_t>(cat) < gpu::kAllocCategoryCount
                          ? static_cast<std::size_t>(cat) : 0;
    _stats.liveBytesByCategory[idx] =
        (_stats.liveBytesByCategory[idx] >= bytes)
            ? _stats.liveBytesByCategory[idx] - bytes : 0;
    if (it != _ptrCategory.end()) _ptrCategory.erase(it);
}

void* CudaMemoryAllocator::allocate(std::size_t bytes, CudaAllocKind kind) {
    if (bytes == 0) return nullptr;

    void*       ptr = nullptr;
    cudaError_t rc  = cudaSuccess;

    switch (kind) {
        case CudaAllocKind::Device:
            rc = cudaMalloc(&ptr, bytes);
            break;
        case CudaAllocKind::HostPinned:
            // cudaHostAllocDefault: don't request mapped or write-combined
            // — matches the HIP allocator default.
            rc = cudaMallocHost(&ptr, bytes);
            break;
        case CudaAllocKind::Managed:
            rc = cudaMallocManaged(&ptr, bytes, cudaMemAttachGlobal);
            break;
    }

    if (rc != cudaSuccess) {
        _stats.failedAllocs += 1;
        throw CudaError(std::string("cuda alloc (") + kindName(kind) + ")", rc);
    }
    recordAlloc(bytes);
    chargeCategory(ptr, bytes);
    return ptr;
}

void CudaMemoryAllocator::deallocate(void*         ptr,
                                     std::size_t   bytes,
                                     CudaAllocKind kind) noexcept {
    if (ptr == nullptr) return;

    cudaError_t rc = cudaSuccess;
    switch (kind) {
        case CudaAllocKind::Device:
        case CudaAllocKind::Managed:
            rc = cudaFree(ptr);
            break;
        case CudaAllocKind::HostPinned:
            rc = cudaFreeHost(ptr);
            break;
    }
    if (rc != cudaSuccess) {
        MM_LOG_WARN("CudaMem", "free ({}) failed: {}",
                    kindName(kind), cudaGetErrorString(rc));
    }
    unchargeCategory(ptr, bytes);
    recordFree(bytes);
}

void CudaMemoryAllocator::copyH2D(void* dst, const void* src, std::size_t bytes) {
    if (bytes == 0) return;
    cudaCheck("cudaMemcpyHostToDevice",
              cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
    _stats.bytesCopiedH2D += bytes;
}

void CudaMemoryAllocator::copyD2H(void* dst, const void* src, std::size_t bytes) {
    if (bytes == 0) return;
    cudaCheck("cudaMemcpyDeviceToHost",
              cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
    _stats.bytesCopiedD2H += bytes;
}

void CudaMemoryAllocator::copyD2D(void* dst, const void* src, std::size_t bytes) {
    if (bytes == 0) return;
    cudaCheck("cudaMemcpyDeviceToDevice",
              cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice));
    _stats.bytesCopiedD2D += bytes;
}

// KV-managed fallback pattern adapted from antirez/ds4 (MIT).
// Original: ds4_cuda.cu:2374-2408.
bool CudaMemoryAllocator::shouldUseManagedForKvCache(std::size_t kvBytes) noexcept {
    // Discrete dGPUs: never fall back — cudaMallocManaged there costs
    // real PCIe migration. Only integrated (UMA) parts benefit from the
    // page-fault mode.
    if (!_ctx.cudaDeviceInfo().isIntegrated) {
        return false;
    }
    if (kvBytes < kKvManagedMinBytes) {
        return false;
    }

    std::size_t freeMem = 0;
    std::size_t totalMem = 0;
    const cudaError_t rc = cudaMemGetInfo(&freeMem, &totalMem);
    if (rc != cudaSuccess) {
        // Best-effort — if we can't ask the driver, do NOT fall back.
        // A wrong managed choice is worse than a wrong device choice.
        (void)cudaGetLastError();
        MM_LOG_WARN("CudaMem", "cudaMemGetInfo failed: {} — skipping KV-managed check",
                    cudaGetErrorString(rc));
        return false;
    }

    // Fall back to managed only if the KV would leave less than the
    // headroom in device memory.
    const bool tight = freeMem < (kvBytes + kKvManagedFreeHeadroom);
    if (tight) {
        _stats.managedFallbackHits += 1;
        MM_LOG_INFO("CudaMem",
                    "KV-managed fallback: kv={} MiB, free={} MiB, total={} MiB — using cudaMallocManaged",
                    kvBytes >> 20, freeMem >> 20, totalMem >> 20);
    }
    return tight;
}

// Weight-tensor placement hint for UMA parts.
// Same source: antirez/ds4 (MIT), ds4_cuda.cu:2374-2408.
void CudaMemoryAllocator::adviseReadMostly(void* ptr, std::size_t bytes) noexcept {
    if (ptr == nullptr || bytes == 0) return;
    if (!_ctx.cudaDeviceInfo().isIntegrated) {
        // Discrete cards don't benefit from these hints meaningfully;
        // save the driver call.
        return;
    }

    const int dev = _ctx.cudaDeviceIndex();
    cudaMemLocation loc{};
    loc.type = cudaMemLocationTypeDevice;
    loc.id   = dev;

    cudaError_t rc = cudaMemAdvise(ptr, bytes, cudaMemAdviseSetReadMostly, loc);
    if (rc != cudaSuccess) {
        MM_LOG_WARN("CudaMem", "cudaMemAdvise(SetReadMostly) failed: {}",
                    cudaGetErrorString(rc));
        (void)cudaGetLastError();
        return;
    }
    rc = cudaMemAdvise(ptr, bytes, cudaMemAdviseSetPreferredLocation, loc);
    if (rc != cudaSuccess) {
        MM_LOG_WARN("CudaMem", "cudaMemAdvise(SetPreferredLocation) failed: {}",
                    cudaGetErrorString(rc));
        (void)cudaGetLastError();
    }
}

void CudaMemoryAllocator::logStats(::mimirmind::core::log::LogLevel lvl) const {
    ::mimirmind::core::log::detail::logFormat(
        lvl, "CudaMem", std::source_location::current(),
        "live={} allocs={} peak={} MiB total_alloc={} total_free={} "
        "H2D={} MiB D2H={} MiB D2D={} MiB fails={} kv_managed_fallbacks={}",
        _stats.liveAllocations,
        _stats.liveBytes,
        _stats.peakBytes >> 20,
        _stats.totalAllocations,
        _stats.totalDeallocations,
        _stats.bytesCopiedH2D >> 20,
        _stats.bytesCopiedD2H >> 20,
        _stats.bytesCopiedD2D >> 20,
        _stats.failedAllocs,
        _stats.managedFallbackHits);
}

// ---------- CudaBuffer -----------------------------------------------------

CudaBuffer::CudaBuffer(CudaMemoryAllocator& alloc,
                       std::size_t          bytes,
                       CudaAllocKind        kind)
    : _alloc(&alloc), _bytes(bytes), _kind(kind) {
    _ptr = alloc.allocate(bytes, kind);
}

CudaBuffer::CudaBuffer(CudaBuffer&& other) noexcept
    : _alloc(other._alloc),
      _ptr  (other._ptr),
      _bytes(other._bytes),
      _kind (other._kind) {
    other._alloc = nullptr;
    other._ptr   = nullptr;
    other._bytes = 0;
}

CudaBuffer& CudaBuffer::operator=(CudaBuffer&& other) noexcept {
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

void CudaBuffer::reset() noexcept {
    if (_ptr != nullptr && _alloc != nullptr) {
        _alloc->deallocate(_ptr, _bytes, _kind);
    }
    _alloc = nullptr;
    _ptr   = nullptr;
    _bytes = 0;
}

} // namespace mimirmind::core::cuda