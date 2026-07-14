// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <level_zero/ze_api.h>

#include <cstddef>
#include <cstdint>

namespace mimirmind::runtime {

class GpuModule;

/**
 * Thin RAII-less wrapper around a ze_kernel_handle_t. The handle's
 * lifetime is owned by the GpuModule that created it; this class just
 * gives us a typed builder for setArgument + setGroupSize.
 *
 * Construction from a raw `ze_kernel_handle_t` is intentionally private
 * — the only legitimate producer is `GpuModule::kernel()`. This
 * prevents backend-specific handles from leaking into consumer code
 * that should stay backend-agnostic. `handle()` remains public because
 * `CommandQueue` needs it to submit to L0; when a second backend lands
 * in Phase 1c that accessor will be replaced with a backend-opaque
 * launch path.
 *
 * Not thread-safe — argument binding lives on the kernel handle itself,
 * so two threads launching with different args would race.
 */
class GpuKernel {
public:
    GpuKernel(const GpuKernel&)                = default;
    GpuKernel(GpuKernel&&) noexcept            = default;
    GpuKernel& operator=(const GpuKernel&)     = default;
    GpuKernel& operator=(GpuKernel&&) noexcept = default;
    ~GpuKernel()                               = default;

    [[nodiscard]] ze_kernel_handle_t handle() const noexcept { return _h; }

    /// Bind one pointer argument (e.g., USM pointer).
    void setPtr(std::uint32_t index, const void* ptr);

    /// Bind one POD argument (int, float, etc.).
    template <typename T>
    void setValue(std::uint32_t index, const T& v) {
        setRaw(index, sizeof(T), &v);
    }

    /// Set the per-workgroup thread layout. Must be called before launch.
    void setGroupSize(std::uint32_t x, std::uint32_t y = 1, std::uint32_t z = 1);

private:
    friend class GpuModule;

    explicit GpuKernel(ze_kernel_handle_t handle) noexcept : _h{handle} {}

    void setRaw(std::uint32_t index, std::size_t bytes, const void* data);

    ze_kernel_handle_t _h;
};

} // namespace mimirmind::runtime