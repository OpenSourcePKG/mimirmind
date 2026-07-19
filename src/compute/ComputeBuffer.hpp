// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstddef>

namespace mimirmind::compute {

/**
 * Backend-neutral scope-bound device buffer. Move-only RAII wrapper
 * around a raw device pointer plus the deleter closure the allocating
 * backend installed at `ComputeOps::allocate()` time.
 *
 * Same shape and semantics as `core::l0::UsmHandle` (single ptr +
 * bytes + free-on-destruction), extended by two extra pointers — a
 * type-erased deleter function and its opaque context — so the buffer
 * knows how to release itself without depending on any backend type.
 * This is idiomatically the same trick `std::unique_ptr<T, Deleter>`
 * uses; the concrete wrapper just packages the `void* + size` payload
 * that every GPU allocator surfaces.
 *
 * Value semantics on purpose: `BlockBuffers` and other scratch structs
 * hold `ComputeBuffer` as inline members, not as heap-allocated
 * `unique_ptr<ComputeBuffer>`. Keeps cache locality of the scratch
 * struct high and avoids ~20 extra small allocations per model init.
 *
 * Not thread-safe. The deleter is invoked exactly once, in the dtor
 * or move-assign, and the deleter itself must be noexcept (backend
 * allocators surface their release paths as `noexcept` already —
 * `UsmAllocator::deallocate`, `HipMemoryAllocator::deallocate`).
 */
class ComputeBuffer {
public:
    /// Deleter signature. `ctx` is the backend-provided opaque payload
    /// (typically a pointer to the allocator instance that produced
    /// the buffer). Must be noexcept — invoked from the dtor.
    using DeleterFn = void (*)(void* ptr, std::size_t bytes, void* ctx) noexcept;

    ComputeBuffer() = default;

    ComputeBuffer(void*       ptr,
                  std::size_t bytes,
                  DeleterFn   deleter,
                  void*       ctx) noexcept
        : _ptr{ptr}, _bytes{bytes}, _deleter{deleter}, _ctx{ctx} {}

    ~ComputeBuffer() noexcept {
        if (_deleter != nullptr && _ptr != nullptr) {
            _deleter(_ptr, _bytes, _ctx);
        }
    }

    ComputeBuffer(const ComputeBuffer&)            = delete;
    ComputeBuffer& operator=(const ComputeBuffer&) = delete;

    ComputeBuffer(ComputeBuffer&& other) noexcept
        : _ptr{other._ptr},
          _bytes{other._bytes},
          _deleter{other._deleter},
          _ctx{other._ctx} {
        other._ptr     = nullptr;
        other._bytes   = 0;
        other._deleter = nullptr;
        other._ctx     = nullptr;
    }

    ComputeBuffer& operator=(ComputeBuffer&& other) noexcept {
        if (this != &other) {
            if (_deleter != nullptr && _ptr != nullptr) {
                _deleter(_ptr, _bytes, _ctx);
            }
            _ptr     = other._ptr;
            _bytes   = other._bytes;
            _deleter = other._deleter;
            _ctx     = other._ctx;
            other._ptr     = nullptr;
            other._bytes   = 0;
            other._deleter = nullptr;
            other._ctx     = nullptr;
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

    [[nodiscard]] explicit operator bool() const noexcept {
        return _ptr != nullptr;
    }

private:
    void*       _ptr{nullptr};
    std::size_t _bytes{0};
    DeleterFn   _deleter{nullptr};
    void*       _ctx{nullptr};
};

} // namespace mimirmind::compute