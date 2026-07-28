// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

// Scatter a fused QKV matmul output into separate Q / K / V buffers.
//
// Fused output layout, row-major: [M, Nq + Nkv * (1 + hasV)]
// Per row m:
//   [m, 0        .. Nq-1                    ] → Y_q[m*Nq + i]
//   [m, Nq       .. Nq+Nkv-1                ] → Y_k[m*Nkv + (i - Nq)]
//   [m, Nq+Nkv   .. Nq+2*Nkv-1  (if hasV)   ] → Y_v[m*Nkv + (i - Nq - Nkv)]
//
// The kernel is dispatched with global_size = M * Nfused (1D). Y_v may
// be any valid pointer when hasV == 0 — the kernel never dereferences
// it because Nfused is set to Nq + Nkv in that case.

// M-CLR.2 Wave 3: `Yk` and `Yv` are the layer's K/V cache BASE, not
// the per-token write slot. The row offset picks up the current
// `curLen` from the shared USM slot so a recorded command list stays
// valid as the cache grows between replays. `Nkv` doubles as the row
// stride inside the cache (== kvDim) so no extra scalar is needed.
// `Yq` still points at a stable workspace buffer.
__kernel void qkv_split(
    __global const float* fused,
    __global       float* Yq,
    __global       float* Yk,
    __global       float* Yv,
    const int             M,
    const int             Nq,
    const int             Nkv,
    const int             hasV,
    const int             Nfused,
    __global const int*   curLenPtr)
{
    const int idx = (int)get_global_id(0);
    const int total = M * Nfused;
    if (idx >= total) return;

    const int m = idx / Nfused;
    const int n = idx - m * Nfused;
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