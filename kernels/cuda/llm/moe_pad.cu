// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// M-Cuda.MoeGroup Sub-Step E-d.4b — padding infrastructure for the FP4-TC
// grouped-MoE prefill path. The CUTLASS block-scaled grouped GEMM needs each
// expert's activation SFA to be tile-aligned (128-row atom), so the per-expert
// contiguous rows are spread into a buffer where expert e starts at a 128-padded
// offset. These small generic kernels build the padded row map and move rows /
// indices; nothing crosses to the host (M per expert stays on device).

#include <cuda_runtime.h>

// padOffset[e] = sum_{i<e} round_up(count_i, 128), count_i = expOffset[i+1]-
// expOffset[i]; padOffset[nExperts] = totalPad. Single-thread prefix (nExperts
// is small, ~128) — one launch per MoE layer, negligible next to the GEMMs.
extern "C" __global__
void moe_pad_offsets(const int* __restrict__ expOffset,
                           int* __restrict__ padOffset,
                     int nExperts) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int acc = 0;
    padOffset[0] = 0;
    for (int e = 0; e < nExperts; ++e) {
        const int c = expOffset[e + 1] - expOffset[e];
        acc += ((c + 127) / 128) * 128;
        padOffset[e + 1] = acc;
    }
}

// contigToPad[r] = padOffset[e] + (r - expOffset[e]) for the expert e whose
// contiguous range [expOffset[e], expOffset[e+1]) contains row r. One thread
// per contiguous row; e is found by binary search over expOffset.
extern "C" __global__
void moe_contig_to_pad(const int* __restrict__ expOffset,
                       const int* __restrict__ padOffset,
                             int* __restrict__ contigToPad,
                       int nExperts, int R) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= R) return;
    int lo = 0, hi = nExperts;                  // expOffset[lo] <= r < expOffset[hi]
    while (lo < hi - 1) {
        const int mid = (lo + hi) >> 1;
        if (expOffset[mid] <= r) lo = mid; else hi = mid;
    }
    contigToPad[r] = padOffset[lo] + (r - expOffset[lo]);
}

// dst[idxMap[r]*dim + d] = src[r*dim + d]  (spread contiguous rows to padded
// rows). `dst` padding rows are left untouched — the caller zeroes them first.
extern "C" __global__
void moe_rows_scatter_f32(const float* __restrict__ src,
                          const int*   __restrict__ idxMap,
                                float* __restrict__ dst,
                          int nRows, int dim) {
    const int r = blockIdx.x;
    if (r >= nRows) return;
    const int  dstRow = idxMap[r];
    const float* __restrict__ s = src + static_cast<long>(r) * dim;
    float* __restrict__ d = dst + static_cast<long>(dstRow) * dim;
    for (int c = blockIdx.y * blockDim.x + threadIdx.x; c < dim;
         c += gridDim.y * blockDim.x) {
        d[c] = s[c];
    }
}

// dst[i] = (src[i] < 0) ? -1 : idxMap[src[i]]   (remap an index array through
// contigToPad — e.g. asnToRow -> padded rows for the scatter).
extern "C" __global__
void moe_index_gather_i32(const int* __restrict__ src,
                          const int* __restrict__ idxMap,
                                int* __restrict__ dst,
                          int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int v = src[i];
    dst[i] = (v < 0) ? -1 : idxMap[v];
}
