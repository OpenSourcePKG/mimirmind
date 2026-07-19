// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Tiny bringup kernel for hip_module_probe. Not part of the mimirmind
// runtime — its only job is to give HipModule + HipKernel a concrete
// symbol to load and launch during the module-layer sanity test. Each
// thread writes a deterministic pattern into its slot of `out` so the
// probe can byte-compare the round-trip.
//
// Compiled by hipcc --genco --offload-arch=<arch> into a .hsaco code
// object at CMake time. Kept intentionally trivial (no LDS, no
// wave-intrinsics, no matmul dispatch) so a build/run failure isolates
// to the module-loading path, not to kernel-side issues.

#include <hip/hip_runtime.h>

extern "C" __global__
void write_pattern(unsigned int* __restrict__ out, unsigned int n)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Same shape as the reference on the host side (see
        // hip_module_probe.cpp): deterministic per-index value.
        out[i] = i * 7u + 3u;
    }
}