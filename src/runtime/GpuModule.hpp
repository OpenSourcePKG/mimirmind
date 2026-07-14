// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "runtime/GpuKernel.hpp"

#include <level_zero/ze_api.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace mimirmind::core::l0 { class L0Context; }

namespace mimirmind::runtime {

using ::mimirmind::core::l0::L0Context;

/**
 * A single SPIR-V module loaded into a Level Zero context. Holds the
 * ze_module_handle_t plus a cache of `ze_kernel_handle_t` looked up by
 * kernel name.
 *
 * Loaded from a `.spv` file on disk (produced by ocloc at build time).
 * The path is resolved against `L0Context::spvDirOverride()` (from
 * `runtime.spvDir` in config.json) if non-empty, else
 * `/usr/local/share/mimirmind/spv`, else the build-tree fallback.
 *
 * **Backend surface:** L0-backend implementation of the compute-module
 * concept. Return type of `kernel()` is opaque `GpuKernel` — consumers
 * do not see `ze_kernel_handle_t` in the API. Storage still uses
 * L0-typed members (`ze_module_handle_t`), so this header does pull in
 * `<level_zero/ze_api.h>`; that's intentional and mirrors CommandQueue.
 */
class GpuModule {
public:
    /// Load `<spvDirOrEnv>/<name>.spv` into the context's L0 module.
    GpuModule(L0Context& ctx, std::string_view name);
    ~GpuModule();

    GpuModule(const GpuModule&)            = delete;
    GpuModule& operator=(const GpuModule&) = delete;
    GpuModule(GpuModule&&)                 = delete;
    GpuModule& operator=(GpuModule&&)      = delete;

    /// Look up (or first-time-create) a kernel by its `__kernel` symbol
    /// name. Returns a `GpuKernel` (opaque wrapper) by value — the
    /// underlying `ze_kernel_handle_t` stays owned by this module and
    /// is destroyed in the module dtor. Throws L0Error if the symbol
    /// isn't in this module.
    [[nodiscard]] GpuKernel kernel(const char* kernelName);

    [[nodiscard]] std::string_view name() const noexcept { return _name; }

private:
    static std::vector<std::uint8_t> readSpv(std::string_view spvDirOverride,
                                             std::string_view name);

    L0Context&                                       _ctx;
    std::string                                      _name;
    ze_module_handle_t                               _module{nullptr};
    std::unordered_map<std::string, ze_kernel_handle_t> _kernels;
};

} // namespace mimirmind::runtime