// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/gpu/hip/HipStream.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mimirmind::core::hip {

class HipModule;

/**
 * Wrapper around a `hipFunction_t`. Lifetime is owned by the parent
 * `HipModule` — this class does not call `hipFuncSet*`-family destroy
 * (HIP has no per-function destroy; the module owns everything).
 *
 * HIP takes kernel arguments as `void**` (array of pointers to values)
 * at launch time, not staged on the handle like L0. To keep the API
 * shape parallel to `runtime::GpuKernel` (stage-then-launch), this
 * class buffers argument values internally: `setArgumentValue(idx, ...)`
 * copies bytes into slot `idx`, `launch(...)` builds the `void**`
 * array from the slots and calls `hipModuleLaunchKernel`.
 *
 * Move-only; not thread-safe (arg buffer is shared state).
 */
class HipKernel {
public:
    // Constructed only by HipModule::getKernel — the ctor is public so
    // HipModule::getKernel can return a temporary, but callers must
    // not construct manually. Enforced by convention (no factory
    // gymnastics, mirrors GpuKernel policy).
    HipKernel(HipModule& parent, hipFunction_t fn, std::string name);

    ~HipKernel() = default;

    HipKernel(const HipKernel&)            = delete;
    HipKernel& operator=(const HipKernel&) = delete;
    HipKernel(HipKernel&&) noexcept        = default;
    HipKernel& operator=(HipKernel&&) noexcept = default;

    // ---- Argument staging ------------------------------------------------

    /// Bind one pointer argument (a device pointer). Convenience wrapper
    /// around setRaw.
    void setPtr(std::uint32_t index, const void* devicePtr);

    /// Bind one POD-by-value argument (int, float, struct...).
    template <typename T>
    void setValue(std::uint32_t index, const T& v) {
        setRaw(index, sizeof(T), &v);
    }

    /// Copy `bytes` from `value` into slot `index`. Grows the internal
    /// arg storage if needed. Same shape as
    /// `zeKernelSetArgumentValue` on the L0 side.
    void setRaw(std::uint32_t index, std::size_t bytes, const void* value);

    /// Clear all staged args. Call between reuse of the same kernel
    /// object with different argument sets.
    void clearArgs() noexcept;

    // ---- Launch ---------------------------------------------------------

    /// Launch on `stream` with the given grid/block dimensions. `sharedMem`
    /// is the dynamic LDS request in bytes (0 if the kernel declares its
    /// LDS statically). All staged arguments must be set — an unset slot
    /// is a runtime driver error.
    void launch(HipStream&    stream,
                std::uint32_t gridX,
                std::uint32_t gridY,
                std::uint32_t gridZ,
                std::uint32_t blockX,
                std::uint32_t blockY,
                std::uint32_t blockZ,
                std::size_t   sharedMemBytes = 0);

    // ---- Introspection --------------------------------------------------

    [[nodiscard]] hipFunction_t handle() const noexcept { return _fn; }
    [[nodiscard]] const std::string& name() const noexcept { return _name; }

private:
    hipFunction_t                       _fn{nullptr};
    std::string                         _name;
    std::vector<std::vector<std::byte>> _argStorage;
};

} // namespace mimirmind::core::hip