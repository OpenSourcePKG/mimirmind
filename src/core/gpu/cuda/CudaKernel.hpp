// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/gpu/cuda/CudaStream.hpp"

// CUDA has TWO APIs. Runtime API (cuda_runtime.h) uses cudaError_t;
// Driver API (cuda.h) uses CUresult. Module loading + kernel launching
// live in the driver API — the runtime API has no equivalent to
// hipModuleGetFunction / hipModuleLaunchKernel that takes a code-object
// blob at runtime. We use the driver API for CudaModule / CudaKernel
// and the runtime API for streams / memory / events.
#include <cuda.h>
#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace mimirmind::core::cuda {

class CudaModule;

/**
 * Exception peer to `CudaError` but for the CUDA driver API. `CUresult`
 * has a different value space than `cudaError_t` and the decode
 * function is `cuGetErrorString`, not `cudaGetErrorString`.
 */
class CudaDriverError : public std::runtime_error {
public:
    CudaDriverError(const std::string& call, CUresult code);
    [[nodiscard]] CUresult code() const noexcept { return _code; }

private:
    CUresult _code;
};

/**
 * Wrapper around a `CUfunction`. Lifetime is owned by the parent
 * `CudaModule` — this class does not destroy the function handle
 * (CUDA has no per-function destroy).
 *
 * `cuLaunchKernel` takes kernel arguments as `void**` (array of
 * pointers to values) at launch time. Same stage-then-launch pattern
 * as `HipKernel`: `setValue(idx, ...)` copies bytes into slot `idx`,
 * `launch(...)` builds the `void**` array from the slots and calls
 * `cuLaunchKernel`.
 *
 * Move-only; not thread-safe (arg buffer is shared state).
 */
class CudaKernel {
public:
    // Constructed only by CudaModule::getFunction — public ctor for
    // temporary return but callers must not construct manually.
    CudaKernel(CudaModule& parent, CUfunction fn, std::string name);

    ~CudaKernel() = default;

    CudaKernel(const CudaKernel&)            = delete;
    CudaKernel& operator=(const CudaKernel&) = delete;
    CudaKernel(CudaKernel&&) noexcept        = default;
    CudaKernel& operator=(CudaKernel&&) noexcept = default;

    // ---- Argument staging ------------------------------------------------

    /// Bind one pointer argument (a device pointer).
    void setPtr(std::uint32_t index, const void* devicePtr);

    /// Bind one POD-by-value argument (int, float, struct...).
    template <typename T>
    void setValue(std::uint32_t index, const T& v) {
        setRaw(index, sizeof(T), &v);
    }

    /// Copy `bytes` from `value` into slot `index`. Grows the internal
    /// arg storage if needed.
    void setRaw(std::uint32_t index, std::size_t bytes, const void* value);

    /// Clear all staged args.
    void clearArgs() noexcept;

    // ---- Launch ---------------------------------------------------------

    /// Launch on `stream` with the given grid/block dimensions.
    /// `sharedMemBytes` is the dynamic shared memory request in bytes.
    /// All staged arguments must be set — an unset slot is a driver
    /// error. Throws `CudaDriverError` on non-Success.
    void launch(CudaStream&   stream,
                std::uint32_t gridX,
                std::uint32_t gridY,
                std::uint32_t gridZ,
                std::uint32_t blockX,
                std::uint32_t blockY,
                std::uint32_t blockZ,
                std::size_t   sharedMemBytes = 0);

    // ---- Introspection --------------------------------------------------

    [[nodiscard]] CUfunction handle() const noexcept { return _fn; }
    [[nodiscard]] const std::string& name() const noexcept { return _name; }

private:
    // Fixed-capacity arg storage — matches HipKernel to avoid the
    // per-launch heap allocation that would burn thousands of tiny
    // mallocs on ~10k dispatches per decode. 16 slots × 16 B covers
    // every kernel we launch (largest ~12 args, none wider than a
    // pointer).
    static constexpr std::size_t kMaxArgs      = 16;
    static constexpr std::size_t kMaxArgBytes  = 16;

    CUfunction                          _fn{nullptr};
    std::string                         _name;
    std::size_t                         _argCount{0};
    alignas(std::uint64_t) std::array<
        std::array<std::uint8_t, kMaxArgBytes>, kMaxArgs> _argStorage{};
    std::array<void*, kMaxArgs>         _argPtrs{};
};

} // namespace mimirmind::core::cuda