// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

// Track 1 skeleton for the CUDA backend. Provides the two symbols
// `BackendRegistry` forward-declares behind `MIMIRMIND_HAVE_CUDA`:
//
//   - `probeBackend()`         — real: counts visible CUDA devices via
//                                cudaGetDeviceCount, populates the
//                                one-line detail with the first
//                                device's name so /v1/system/info shows
//                                what the process saw.
//   - `createComputeContext()` — placeholder: throws a distinct message
//                                so operators can tell "backend not
//                                compiled in" from "backend compiled
//                                in, primitives not yet wired". Real
//                                impl lands with Track 2
//                                (`CudaContext` + `CudaStream` +
//                                `CudaMemoryAllocator` + friends).
//
// Lives in `mimirmind_core_cuda` — the pure-CPU `mimirmind_core_common`
// library never pulls in <cuda_runtime.h>.

#include "core/backend/BackendRegistry.hpp"
#include "core/backend/ComputeBackend.hpp"
#include "core/backend/ComputeContext.hpp"

#include <cuda_runtime.h>

#include <memory>
#include <stdexcept>
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

std::unique_ptr<::mimirmind::core::backend::ComputeContext>
createComputeContext() {
    // Track 2 fills this in with a real CudaComputeContext that owns
    // CudaContext + CudaMemoryAllocator + CudaStream. Until then, throw
    // a message that distinguishes "CUDA not compiled in" (handled in
    // BackendRegistry.cpp) from "CUDA compiled in but primitives layer
    // not yet linked" — the two operator experiences are different.
    throw std::runtime_error{
        "mimirmind::core::cuda::createComputeContext: skeleton only — "
        "CUDA primitives (CudaContext / CudaStream / CudaMemoryAllocator) "
        "land with Track 2 on feat/cuda-backend-skeleton"};
}

} // namespace mimirmind::core::cuda
