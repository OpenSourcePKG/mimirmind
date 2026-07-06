// Fused Q + K + V RMSNorm — one dispatch instead of three.
//
// Q and K normalize with per-head_dim weights (plain rmsnorm semantics,
// matching kernels/rmsnorm.cl). V normalizes without a learned weight,
// matching kernels/rmsnorm_no_weight.cl.
//
// The buffers and their weights all share the same K = head_dim, so a
// single kernel can service all three streams with only the row index
// changing what it reads and writes. Workgroup grid layout:
//
//   groups [0,             T*nHeads)               → Q rows
//   groups [T*nHeads,      T*nHeads +   T*nKvHeads) → K rows
//   groups [T*(nHeads+nKvHeads), T*(nHeads+2*nKvHeads)) → V rows
//
// Total groups = T * (nHeads + 2 * nKvHeads). Each workgroup does the
// same tree reduction as rmsnorm.cl.

#ifndef RMSNORM_QKV_LOCAL_SIZE
#define RMSNORM_QKV_LOCAL_SIZE 128
#endif

// M-CLR.2 Wave 3: `k_x`/`k_y` and `v_x`/`v_y` now point at the layer's
// K/V cache BASE, and the kernel adds `curLen * kvDim` to reach the
// current write slot. Q buffers stay stable (workspace). Same USM slot
// as rope/attention — one int, host-writable.
__attribute__((reqd_work_group_size(RMSNORM_QKV_LOCAL_SIZE, 1, 1)))
__kernel void rmsnorm_qkv(
    __global const float* q_x,      // [T*nHeads,   head_dim]
    __global const float* q_w,      // [head_dim]
    __global       float* q_y,      // in-place OK (== q_x)
    __global const float* k_x,      // K cache base for this layer
    __global const float* k_w,      // [head_dim]
    __global       float* k_y,      // K cache base for this layer
    __global const float* v_x,      // V cache base for this layer
    __global       float* v_y,      // V cache base for this layer
    const int             qRows,    // T * nHeads
    const int             kRows,    // T * nKvHeads   (K and V share this)
    const int             K,        // head_dim
    const float           eps,
    __global const int*   curLenPtr,
    const int             kvDim)    // nKvHeads * head_dim  (row stride in cache)
{
    __local float scratch[RMSNORM_QKV_LOCAL_SIZE];

    const int gid   = (int)get_group_id(0);
    const int tid   = (int)get_local_id(0);
    const int lsize = (int)get_local_size(0);

    // Route this workgroup to Q / K / V and pick its row within that
    // stream. The three branches diverge across workgroups, not threads,
    // so there's no lane divergence. K/V paths add `curLen * kvDim` to
    // land on the current write slot inside the layer's cache.
    __global const float* xr;
    __global const float* wr;
    __global       float* yr;
    const size_t kvBase = (size_t)curLenPtr[0] * (size_t)kvDim;
    if (gid < qRows) {
        const int row = gid;
        xr = q_x + (size_t)row * (size_t)K;
        wr = q_w;
        yr = q_y + (size_t)row * (size_t)K;
    } else if (gid < qRows + kRows) {
        const int row = gid - qRows;
        xr = k_x + kvBase + (size_t)row * (size_t)K;
        wr = k_w;
        yr = k_y + kvBase + (size_t)row * (size_t)K;
    } else {
        const int row = gid - qRows - kRows;
        xr = v_x + kvBase + (size_t)row * (size_t)K;
        wr = (__global const float*)0;   // V: no learned weight
        yr = v_y + kvBase + (size_t)row * (size_t)K;
    }

    // Per-thread partial sum of squares.
    float acc = 0.0f;
    for (int k = tid; k < K; k += lsize) {
        const float v = xr[k];
        acc = mad(v, v, acc);
    }
    scratch[tid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Power-of-two tree reduction in shared memory.
    for (int stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const float mean   = scratch[0] / (float)K;
    const float invRms = rsqrt(mean + eps);

    if (wr != (__global const float*)0) {
        for (int k = tid; k < K; k += lsize) {
            yr[k] = xr[k] * wr[k] * invRms;
        }
    } else {
        for (int k = tid; k < K; k += lsize) {
            yr[k] = xr[k] * invRms;
        }
    }
}