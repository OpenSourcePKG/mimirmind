// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

// FP16-KV variant of rmsnorm_qkv.cl. Q stays fp32 (workspace), K and V
// live in the fp16 KV cache. Weights (`k_w`) stay fp32; the sum of
// squares and the invRms scalar are computed in fp32 regardless of
// storage dtype (vload_half promotes each read to fp32 before the
// mad). Writes use vstore_half so the fp16 cache gets round-trip
// normalised values.
//
// The Q, K and V branches diverge across workgroups only — every
// thread inside a workgroup takes the same branch and hits the same
// barriers (per-branch), which is legal under OpenCL WG semantics.
//
// M10.2 Phase 0 Commit 4 — invoked only when
// `KvCache::dtype() == FP16`; the f32 dispatch stays on rmsnorm_qkv.cl,
// preserving bit-parity against pre-M10.2 behaviour.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifndef RMSNORM_QKV_LOCAL_SIZE
#define RMSNORM_QKV_LOCAL_SIZE 128
#endif

__attribute__((reqd_work_group_size(RMSNORM_QKV_LOCAL_SIZE, 1, 1)))
__kernel void rmsnorm_qkv_fp16(
    __global const float* q_x,      // [T*nHeads,   head_dim] fp32 workspace
    __global const float* q_w,      // [head_dim] fp32
    __global       float* q_y,      // in-place OK (== q_x)
    __global const half*  k_x,      // K cache base for this layer (fp16)
    __global const float* k_w,      // [head_dim] fp32
    __global       half*  k_y,      // K cache base for this layer (fp16)
    __global const half*  v_x,      // V cache base for this layer (fp16)
    __global       half*  v_y,      // V cache base for this layer (fp16)
    const int             qRows,    // T * nHeads
    const int             kRows,    // T * nKvHeads
    const int             K,        // head_dim
    const float           eps,
    __global const int*   curLenPtr,
    const int             kvDim)    // nKvHeads * head_dim (row stride in cache)
{
    __local float scratch[RMSNORM_QKV_LOCAL_SIZE];

    const int gid   = (int)get_group_id(0);
    const int tid   = (int)get_local_id(0);
    const int lsize = (int)get_local_size(0);
    const size_t kvBase = (size_t)curLenPtr[0] * (size_t)kvDim;

    // Branch on stream (Q / K / V). Uniform across the workgroup.
    if (gid < qRows) {
        // ---- Q branch: fp32 workspace, identical to rmsnorm_qkv.cl. ---
        const int row = gid;
        __global const float* xr = q_x + (size_t)row * (size_t)K;
        __global const float* wr = q_w;
        __global       float* yr = q_y + (size_t)row * (size_t)K;

        float acc = 0.0f;
        for (int k = tid; k < K; k += lsize) {
            const float v = xr[k];
            acc = mad(v, v, acc);
        }
        scratch[tid] = acc;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int stride = lsize >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) scratch[tid] += scratch[tid + stride];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        const float invRms = rsqrt(scratch[0] / (float)K + eps);
        for (int k = tid; k < K; k += lsize) {
            yr[k] = xr[k] * wr[k] * invRms;
        }
    } else if (gid < qRows + kRows) {
        // ---- K branch: fp16 cache, weighted rmsnorm. ------------------
        const int row = gid - qRows;
        __global const half* xr =
            k_x + kvBase + (size_t)row * (size_t)K;
        __global const float* wr = k_w;
        __global       half* yr =
            k_y + kvBase + (size_t)row * (size_t)K;

        float acc = 0.0f;
        for (int k = tid; k < K; k += lsize) {
            const float v = vload_half(k, xr);
            acc = mad(v, v, acc);
        }
        scratch[tid] = acc;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int stride = lsize >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) scratch[tid] += scratch[tid + stride];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        const float invRms = rsqrt(scratch[0] / (float)K + eps);
        for (int k = tid; k < K; k += lsize) {
            const float xv = vload_half(k, xr);
            vstore_half(xv * wr[k] * invRms, k, yr);
        }
    } else {
        // ---- V branch: fp16 cache, no learned weight. ----------------
        const int row = gid - qRows - kRows;
        __global const half* xr =
            v_x + kvBase + (size_t)row * (size_t)K;
        __global       half* yr =
            v_y + kvBase + (size_t)row * (size_t)K;

        float acc = 0.0f;
        for (int k = tid; k < K; k += lsize) {
            const float v = vload_half(k, xr);
            acc = mad(v, v, acc);
        }
        scratch[tid] = acc;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int stride = lsize >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) scratch[tid] += scratch[tid + stride];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        const float invRms = rsqrt(scratch[0] / (float)K + eps);
        for (int k = tid; k < K; k += lsize) {
            const float xv = vload_half(k, xr);
            vstore_half(xv * invRms, k, yr);
        }
    }
}