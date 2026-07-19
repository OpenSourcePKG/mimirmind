// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Scatter a fused QKV matmul output into separate Q / K / V buffers.
//
// HIP port of kernels/qkv_split.cl. First scatter/layout-transform
// port — no reduction, no LDS, one input read + one output write per
// thread. The interesting piece is that it services three destination
// buffers with different offset schemes:
//   • Q workspace: natural row offset m * Nq
//   • K cache:     (curLen + m) * Nkv, base = layer's K cache
//   • V cache:     (curLen + m) * Nkv, base = layer's V cache (optional)
//
// Fused input layout, row-major:  [M, Nfused]  where
//   Nfused = Nq + Nkv * (1 + hasV)
//
// Per row m:
//   [m, 0        .. Nq-1                  ] → Yq[m*Nq + i]
//   [m, Nq       .. Nq+Nkv-1              ] → Yk[(curLen+m)*Nkv + (i - Nq)]
//   [m, Nq+Nkv   .. Nq+2*Nkv-1  (if hasV) ] → Yv[(curLen+m)*Nkv + (i - Nq - Nkv)]
//
// `curLenPtr` is a device pointer to a single int (RoPE-style runtime
// slot). Yk / Yv are the layer's cache BASE, not the per-token write
// slot — the kernel does the offset arithmetic per launch so recorded
// command lists (once we bring up hipGraph) stay valid as the cache
// grows between replays. `Nkv` doubles as the row stride inside the
// cache (== kvDim), no extra scalar needed.
//
// Yv may be any valid pointer when hasV == 0 — the kernel never
// dereferences it because Nfused is Nq + Nkv in that case, so the
// V branch is never entered.
//
// Launch:
//   dim3 grid ( ceil(M*Nfused / QKV_SPLIT_LOCAL), 1, 1 )
//   dim3 block( QKV_SPLIT_LOCAL, 1, 1 )

#include <hip/hip_runtime.h>

#ifndef QKV_SPLIT_LOCAL
#define QKV_SPLIT_LOCAL 256
#endif

extern "C" __global__ __launch_bounds__(QKV_SPLIT_LOCAL)
void qkv_split(
    const float* __restrict__ fused,
          float* __restrict__ Yq,
          float* __restrict__ Yk,
          float* __restrict__ Yv,
    const int                 M,
    const int                 Nq,
    const int                 Nkv,
    const int                 hasV,
    const int                 Nfused,
    const int* __restrict__   curLenPtr)
{
    const int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = M * Nfused;
    if (idx >= total) return;

    const int m      = idx / Nfused;
    const int n      = idx - m * Nfused;
    const int curLen = curLenPtr[0];

    const float v = fused[idx];

    if (n < Nq) {
        Yq[m * Nq + n] = v;
    } else if (n < Nq + Nkv) {
        Yk[(curLen + m) * Nkv + (n - Nq)] = v;
    } else if (hasV != 0) {
        Yv[(curLen + m) * Nkv + (n - Nq - Nkv)] = v;
    }
}