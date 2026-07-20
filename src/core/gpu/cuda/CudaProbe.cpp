// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

// Track 1 shipped this as a skeleton: `probeBackend()` real,
// `createComputeContext()` throwing "primitives not yet linked" so
// operators could distinguish "CUDA compiled out" from "CUDA compiled
// in, no impl yet". Track 2 (this branch) replaces the throw with a
// real `CudaComputeContext` instantiation now that the seven core
// primitives (`CudaContext` / `CudaStream` / `CudaMemoryAllocator` /
// `CudaKernel` / `CudaModule` / `CudaEvent` / `CudaComputeContext`)
// all exist under `src/core/gpu/cuda/`.
//
// Lives in `mimirmind_core_cuda` — the pure-CPU `mimirmind_core_common`
// library never pulls in <cuda_runtime.h>.

#include "core/backend/BackendRegistry.hpp"
#include "core/backend/ComputeBackend.hpp"
#include "core/backend/ComputeContext.hpp"

#include "core/gpu/cuda/CudaComputeContext.hpp"

#include <cuda_runtime.h>

#include <memory>
#include <string>

namespace mimirmind::core::cuda {

::mimirmind::core::backend::BackendProbe probeBackend() noexcept {
    using ::mimirmind::core::backend::BackendKind;
    using ::mimirmind::core::backend::BackendProbe;

    BackendProbe p{ BackendKind::Cuda, /*compiledIn=*/true, /*available=*/false, {} };

    int count = 0;
    cudaError_t rc = cudaGetDeviceCount(&count);
    if (rc != cudaSuccess) {
        p.detail = std::string("cudaGetDeviceCount failed: ")
                 + cudaGetErrorString(rc);
        // Clear the sticky error so subsequent CUDA calls in the same
        // process aren't misattributed. cudaGetLastError resets it.
        (void)cudaGetLastError();
        return p;
    }
    if (count <= 0) {
        p.detail = "no CUDA devices visible";
        return p;
    }

    // Grab device 0's name for the human-readable one-liner. Non-fatal
    // if it fails — the count is the important signal.
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        p.detail = std::to_string(count) + " device(s), first: " + prop.name
                 + " (sm_" + std::to_string(prop.major)
                 + std::to_string(prop.minor) + ")";
    } else {
        p.detail = std::to_string(count) + " device(s)";
        (void)cudaGetLastError();
    }
    p.available = true;
    return p;
}

// Factory hook consumed by BackendRegistry::createContext(). Lives in
// this TU (rather than being merged into CudaComputeContext.cpp) so
// the forward-decl in BackendRegistry.cpp resolves against the exact
// file Track 1 named. Uses default Options: auto-select device,
// BlockingDefault stream.
std::unique_ptr<::mimirmind::core::backend::ComputeContext>
createComputeContext() {
    return std::make_unique<CudaComputeContext>();
}

} // namespace mimirmind::core::cuda
