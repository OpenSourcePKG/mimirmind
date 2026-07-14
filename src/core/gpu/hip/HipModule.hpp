// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/gpu/hip/HipContext.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <span>
#include <string>
#include <string_view>

namespace mimirmind::core::hip {

class HipKernel;

/**
 * RAII wrapper around a `hipModule_t`. Loads a code object (`.hsaco`
 * produced by `hipcc --genco`) from either an in-memory blob or a
 * filesystem path. Kernels are retrieved by symbol name via
 * `getKernel(...)` — the returned `HipKernel` is owned by this
 * module and must not outlive it.
 *
 * Parallel to `runtime::GpuModule` on the Level Zero side (which wraps
 * `ze_module_handle_t` + `zeModuleCreate` from SPV). Same
 * "one module per .cl / .hip file" pattern.
 *
 * Move-only, single-threaded by contract, throws `HipError` on
 * driver failure.
 */
class HipModule {
public:
    /// Construct from an in-memory code-object blob (as produced by
    /// `hipcc --genco`). The blob is copied by the driver — caller may
    /// free after ctor returns.
    HipModule(HipContext& ctx, std::span<const std::byte> hsacoData);

    /// Convenience: read the file into memory and forward to the
    /// blob constructor.
    static HipModule fromFile(HipContext& ctx, const std::string& path);

    ~HipModule() noexcept;

    HipModule(const HipModule&)            = delete;
    HipModule& operator=(const HipModule&) = delete;
    HipModule(HipModule&& other) noexcept;
    HipModule& operator=(HipModule&& other) noexcept;

    /// Look up a kernel by its extern "C" symbol name. Throws
    /// `HipError` (typically `hipErrorNotFound`) if the symbol is
    /// missing from this module.
    [[nodiscard]] HipKernel getKernel(std::string_view name);

    [[nodiscard]] hipModule_t handle()  const noexcept { return _module; }
    [[nodiscard]] HipContext& context() const noexcept { return *_ctx; }

private:
    HipContext* _ctx{nullptr};
    hipModule_t _module{nullptr};

    void destroy() noexcept;
};

} // namespace mimirmind::core::hip