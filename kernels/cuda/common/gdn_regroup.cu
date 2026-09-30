// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// GatedDeltaNet value-head regroup gathers, run on-device to replace the
// per-tensor readbackToHost -> host permute -> uploadHostBytes round-trip the
// NVFP4 loader used to do for every GDN tensor of every layer.
//
// Both kernels are pure byte-granular gathers driven by a device-resident
// index array, so their output is BIT-IDENTICAL to the host memcpy loop they
// replace (same permutation, same bytes) — the loader validates this against
// the coherent GGUF weights via the HTTP anchor (bit-parity vs the host path).
//
// The element type (BF16 = 2 bytes, F32 = 4 bytes) is folded into the byte
// strides by the caller, so one pair of kernels covers every GDN tensor.
//
// Launch (both): 1-D grid-stride over the total byte count, block 256.

#include <cuda_runtime.h>

#ifndef GDN_REGROUP_LOCAL
#define GDN_REGROUP_LOCAL 256
#endif

// Row gather: dst row r is source row perm[r], each row `rowBytes` bytes.
//   dst[r*rowBytes + b] = src[perm[r]*rowBytes + b]
// Covers attn_qkv / ssm_conv1d (value rows), attn_gate (all rows), and the
// per-head projections/biases (rowBytes == inCols*elemBytes, or == elemBytes
// for a 1-D per-head vector). src and dst must not alias (caller uses scratch).
extern "C" __global__ __launch_bounds__(GDN_REGROUP_LOCAL)
void gdn_gather_rows(
    const unsigned char* __restrict__ src,
    const int*           __restrict__ perm,
    unsigned char*       __restrict__ dst,
    const long long                   nRows,
    const long long                   rowBytes)
{
    const long long total  = nRows * rowBytes;
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += stride) {
        const long long r = i / rowBytes;
        const long long b = i - r * rowBytes;
        dst[i] = src[static_cast<long long>(perm[r]) * rowBytes + b];
    }
}

// Column gather: within each row, column c is source column perm[c].
//   dst[(r*cols + c)*elemBytes + e] = src[(r*cols + perm[c])*elemBytes + e]
// Covers ssm_out (value columns). src and dst must not alias.
extern "C" __global__ __launch_bounds__(GDN_REGROUP_LOCAL)
void gdn_gather_cols(
    const unsigned char* __restrict__ src,
    const int*           __restrict__ perm,
    unsigned char*       __restrict__ dst,
    const long long                   rows,
    const long long                   cols,
    const long long                   elemBytes)
{
    const long long total  = rows * cols * elemBytes;
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += stride) {
        const long long e   = i % elemBytes;
        const long long rc  = i / elemBytes;      // r*cols + c
        const long long c   = rc % cols;
        const long long r   = rc / cols;
        const long long src_elem = r * cols + static_cast<long long>(perm[c]);
        dst[i] = src[src_elem * elemBytes + e];
    }
}
