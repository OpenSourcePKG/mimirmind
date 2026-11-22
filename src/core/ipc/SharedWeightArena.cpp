// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/ipc/SharedWeightArena.hpp"

#include <fcntl.h>
#include <linux/falloc.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <vector>

#ifndef MFD_CLOEXEC
#define MFD_CLOEXEC 0x0001U
#endif
#ifndef MADV_POPULATE_WRITE
#define MADV_POPULATE_WRITE 23
#endif

namespace mimirmind::core::ipc {

namespace {

std::mutex& registryMutex() {
    static std::mutex m;
    return m;
}

std::vector<SharedWeightArena*>& registry() {
    static std::vector<SharedWeightArena*> r;
    return r;
}

thread_local SharedWeightArena* tlsArena = nullptr;

std::atomic<int> liveArenas{0};

std::uint64_t roundUpPage(std::uint64_t b) noexcept {
    const std::uint64_t p = SharedWeightArena::kPageBytes;
    return (b + p - 1) / p * p;
}

std::string errnoText(const char* what) {
    return std::string{what} + ": " + std::strerror(errno);
}

} // namespace

SharedWeightArena::SharedWeightArena(int fd, void* base, std::uint64_t mapBytes,
                                     bool owner) noexcept
    : _fd(fd), _base(base), _mapBytes(mapBytes), _owner(owner) {
    if (_owner) {
        _free.emplace(0, _mapBytes);
    }
    std::lock_guard<std::mutex> g(registryMutex());
    registry().push_back(this);
    liveArenas.fetch_add(1);
}

std::expected<std::unique_ptr<SharedWeightArena>, std::string>
SharedWeightArena::create(const std::string& name, std::uint64_t reserveBytes) noexcept {
    reserveBytes = roundUpPage(reserveBytes);
    if (reserveBytes == 0) {
        return std::unexpected("SharedWeightArena: reserve must be > 0");
    }
    const long fd = ::syscall(SYS_memfd_create, name.c_str(), MFD_CLOEXEC);
    if (fd < 0) {
        return std::unexpected(errnoText("SharedWeightArena: memfd_create"));
    }
    // Sparse: ftruncate allocates nothing; pages appear on first touch.
    if (::ftruncate(static_cast<int>(fd), static_cast<off_t>(reserveBytes)) != 0) {
        std::string e = errnoText("SharedWeightArena: ftruncate");
        ::close(static_cast<int>(fd));
        return std::unexpected(std::move(e));
    }
    void* base = ::mmap(nullptr, reserveBytes, PROT_READ | PROT_WRITE,
                        MAP_SHARED | MAP_NORESERVE, static_cast<int>(fd), 0);
    if (base == MAP_FAILED) {
        std::string e = errnoText("SharedWeightArena: mmap");
        ::close(static_cast<int>(fd));
        return std::unexpected(std::move(e));
    }
    return std::unique_ptr<SharedWeightArena>(
        new SharedWeightArena(static_cast<int>(fd), base, reserveBytes, true));
}

std::expected<std::unique_ptr<SharedWeightArena>, std::string>
SharedWeightArena::attach(int fd, std::uint64_t mapBytes, bool readOnly) noexcept {
    if (fd < 0 || mapBytes == 0) {
        if (fd >= 0) ::close(fd);
        return std::unexpected("SharedWeightArena::attach: bad fd or zero size");
    }
    mapBytes = roundUpPage(mapBytes);
    struct stat st {};
    if (::fstat(fd, &st) != 0 || static_cast<std::uint64_t>(st.st_size) < mapBytes) {
        ::close(fd);
        return std::unexpected("SharedWeightArena::attach: fd is smaller than the "
                               "advertised extent");
    }
    const int prot = readOnly ? PROT_READ : (PROT_READ | PROT_WRITE);
    void* base = ::mmap(nullptr, mapBytes, prot, MAP_SHARED | MAP_NORESERVE, fd, 0);
    if (base == MAP_FAILED) {
        std::string e = errnoText("SharedWeightArena::attach: mmap");
        ::close(fd);
        return std::unexpected(std::move(e));
    }
    return std::unique_ptr<SharedWeightArena>(
        new SharedWeightArena(fd, base, mapBytes, false));
}

SharedWeightArena::~SharedWeightArena() {
    {
        std::lock_guard<std::mutex> g(registryMutex());
        auto& r = registry();
        r.erase(std::remove(r.begin(), r.end(), this), r.end());
        liveArenas.fetch_sub(1);
    }
    if (_base != nullptr) {
        ::munmap(_base, _mapBytes);
    }
    if (_fd >= 0) {
        ::close(_fd);
    }
}

void* SharedWeightArena::allocate(std::size_t bytes) noexcept {
    if (!_owner || bytes == 0) {
        return nullptr;
    }
    const std::uint64_t need = roundUpPage(bytes);
    std::uint64_t off = 0;
    {
        std::lock_guard<std::mutex> g(_mtx);
        auto it = std::find_if(_free.begin(), _free.end(),
                               [need](const auto& kv) { return kv.second >= need; });
        if (it == _free.end()) {
            return nullptr;
        }
        off = it->first;
        const std::uint64_t rest = it->second - need;
        _free.erase(it);
        if (rest > 0) {
            _free.emplace(off + need, rest);
        }
        _live.emplace(off, need);
        _liveBytes += need;
        _peakBytes = std::max(_peakBytes, _liveBytes);
    }
    auto* p = static_cast<std::uint8_t*>(_base) + off;
    // Best effort: fault the range in from the CPU so the first GPU write
    // does not take one ATS fault per page. Old kernels return EINVAL.
    (void)::madvise(p, need, MADV_POPULATE_WRITE);
    return p;
}

void SharedWeightArena::deallocate(void* ptr) noexcept {
    if (!_owner || !contains(ptr)) {
        return;
    }
    const std::uint64_t off = offsetOf(ptr);
    std::uint64_t len = 0;
    {
        std::lock_guard<std::mutex> g(_mtx);
        auto it = _live.find(off);
        if (it == _live.end()) {
            return;
        }
        len = it->second;
        _live.erase(it);
        _liveBytes -= len;
        // Insert and coalesce with the neighbours.
        auto [ins, ok] = _free.emplace(off, len);
        (void)ok;
        if (ins != _free.begin()) {
            auto prev = std::prev(ins);
            if (prev->first + prev->second == ins->first) {
                prev->second += ins->second;
                _free.erase(ins);
                ins = prev;
            }
        }
        auto next = std::next(ins);
        if (next != _free.end() && ins->first + ins->second == next->first) {
            ins->second += next->second;
            _free.erase(next);
        }
    }
    // Give the pages back to the OS; the range reads as zero afterwards.
    (void)::fallocate(_fd, FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE,
                      static_cast<off_t>(off), static_cast<off_t>(len));
}

bool SharedWeightArena::contains(const void* ptr) const noexcept {
    const auto* p = static_cast<const std::uint8_t*>(ptr);
    const auto* b = static_cast<const std::uint8_t*>(_base);
    return p >= b && p < b + _mapBytes;
}

std::uint64_t SharedWeightArena::offsetOf(const void* ptr) const noexcept {
    return static_cast<std::uint64_t>(static_cast<const std::uint8_t*>(ptr) -
                                      static_cast<const std::uint8_t*>(_base));
}

std::size_t SharedWeightArena::allocationBytes(const void* ptr) const noexcept {
    if (!contains(ptr)) {
        return 0;
    }
    std::lock_guard<std::mutex> g(_mtx);
    auto it = _live.find(offsetOf(ptr));
    return it == _live.end() ? 0 : static_cast<std::size_t>(it->second);
}

std::uint64_t SharedWeightArena::highWaterBytes() const noexcept {
    std::lock_guard<std::mutex> g(_mtx);
    if (_live.empty()) {
        return 0;
    }
    const auto& last = *_live.rbegin();
    return last.first + last.second;
}

std::uint64_t SharedWeightArena::liveBytes() const noexcept {
    std::lock_guard<std::mutex> g(_mtx);
    return _liveBytes;
}

std::uint64_t SharedWeightArena::peakBytes() const noexcept {
    std::lock_guard<std::mutex> g(_mtx);
    return _peakBytes;
}

std::size_t SharedWeightArena::liveAllocations() const noexcept {
    std::lock_guard<std::mutex> g(_mtx);
    return _live.size();
}

bool SharedWeightArena::anyLive() noexcept {
    return liveArenas.load(std::memory_order_relaxed) > 0;
}

SharedWeightArena* SharedWeightArena::owning(const void* ptr) noexcept {
    if (ptr == nullptr || !anyLive()) {
        return nullptr;
    }
    std::lock_guard<std::mutex> g(registryMutex());
    for (SharedWeightArena* a : registry()) {
        if (a->contains(ptr)) {
            return a;
        }
    }
    return nullptr;
}

ScopedWeightArena::ScopedWeightArena(SharedWeightArena* arena) noexcept
    : _previous(tlsArena) {
    tlsArena = arena;
}

ScopedWeightArena::~ScopedWeightArena() {
    tlsArena = _previous;
}

SharedWeightArena* ScopedWeightArena::current() noexcept {
    return tlsArena;
}

} // namespace mimirmind::core::ipc
