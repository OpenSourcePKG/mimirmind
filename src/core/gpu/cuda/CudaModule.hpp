// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/gpu/cuda/CudaContext.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <span>
#include <string>
#include <string_view>

namespace mimirmind::core::cuda {

class CudaKernel;

/**
 * RAII wrapper around a `CUmodule`. Loads a code object (PTX or cubin
 * produced by `nvcc --ptx` / `nvcc --cubin`) from either an in-memory
 * blob or a filesystem path. Kernels are retrieved by symbol name via
 * `getFunction(...)` — the returned `CudaKernel` is owned by this
 * module and must not outlive it.
 *
 * Parallel to `HipModule` on the AMD side (which wraps `hipModule_t` +
 * `hipModuleLoadData` from .hsaco). Same "one module per .cu file"
 * pattern.
 *
 * Move-only, single-threaded by contract, throws `CudaDriverError` on
 * driver failure (module load / unload / function lookup are all
 * driver-API operations).
 */
class CudaModule {
public:
    /// Construct from an in-memory code-object blob. Accepts PTX
    /// (null-terminated text) or cubin (binary — the driver
    /// disambiguates on the header). The blob is copied by the driver
    /// — caller may free after ctor returns.
    CudaModule(CudaContext& ctx, std::span<const std::byte> blob);

    /// Convenience: read the file into memory and forward to the
    /// blob constructor.
    static CudaModule fromFile(CudaContext& ctx, const std::string& path);

    ~CudaModule() noexcept;

    CudaModule(const CudaModule&)            = delete;
    CudaModule& operator=(const CudaModule&) = delete;
    CudaModule(CudaModule&& other) noexcept;
    CudaModule& operator=(CudaModule&& other) noexcept;

    /// Look up a kernel by its extern "C" symbol name. Throws
    /// `CudaDriverError` (typically `CUDA_ERROR_NOT_FOUND`) if the
    /// symbol is missing from this module.
    [[nodiscard]] CudaKernel getFunction(std::string_view name);

    [[nodiscard]] CUmodule    handle()  const noexcept { return _module; }
    [[nodiscard]] CudaContext& context() const noexcept { return *_ctx; }

private:
    CudaContext* _ctx{nullptr};
    CUmodule     _module{nullptr};

    void destroy() noexcept;
};

} // namespace mimirmind::core::cuda