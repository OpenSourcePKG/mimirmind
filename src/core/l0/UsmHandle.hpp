#pragma once

#include "core/l0/UsmAllocator.hpp"

#include <cstddef>

namespace mimirmind::runtime {

/**
 * Scope-bound USM allocation. Frees on destruction so an exception
 * thrown mid-pipeline does not leak.
 */
class UsmHandle {
public:
    UsmHandle() = default;

    UsmHandle(UsmAllocator& alloc, std::size_t bytes)
        : _alloc{&alloc}, _ptr{alloc.allocate(bytes)}, _bytes{bytes} {}

    ~UsmHandle() {
        if (_alloc != nullptr && _ptr != nullptr) {
            _alloc->deallocate(_ptr, _bytes);
        }
    }

    UsmHandle(const UsmHandle&)            = delete;
    UsmHandle& operator=(const UsmHandle&) = delete;

    UsmHandle(UsmHandle&& other) noexcept
        : _alloc{other._alloc}, _ptr{other._ptr}, _bytes{other._bytes} {
        other._alloc = nullptr;
        other._ptr   = nullptr;
        other._bytes = 0;
    }

    UsmHandle& operator=(UsmHandle&& other) noexcept {
        if (this != &other) {
            if (_alloc != nullptr && _ptr != nullptr) {
                _alloc->deallocate(_ptr, _bytes);
            }
            _alloc = other._alloc;
            _ptr   = other._ptr;
            _bytes = other._bytes;
            other._alloc = nullptr;
            other._ptr   = nullptr;
            other._bytes = 0;
        }
        return *this;
    }

    [[nodiscard]] void*       get()       noexcept { return _ptr; }
    [[nodiscard]] const void* get() const noexcept { return _ptr; }

    template <typename T> [[nodiscard]] T* as() noexcept {
        return static_cast<T*>(_ptr);
    }
    template <typename T> [[nodiscard]] const T* as() const noexcept {
        return static_cast<const T*>(_ptr);
    }

    [[nodiscard]] std::size_t bytes() const noexcept { return _bytes; }

private:
    UsmAllocator* _alloc{nullptr};
    void*         _ptr{nullptr};
    std::size_t   _bytes{0};
};

} // namespace mimirmind::runtime