// RMSNorm: y[m, k] = x[m, k] * weight[k] / sqrt(mean(x[m, :]^2) + eps)
//
// Launch: one workgroup per row m (global = (M * LOCAL_SIZE), local = LOCAL_SIZE).
// Inside the workgroup, threads cooperate on a tree reduction to compute
// the mean-of-squares for the row, then each thread applies the scale to
// its strided slice of K elements.

#ifndef RMSNORM_LOCAL_SIZE
#define RMSNORM_LOCAL_SIZE 128
#endif

__attribute__((reqd_work_group_size(RMSNORM_LOCAL_SIZE, 1, 1)))
__kernel void rmsnorm(
    __global const float* x,       // [M, K]
    __global const float* weight,  // [K]
    __global       float* y,       // [M, K]
    const float           eps,
    const int             K)
{
    __local float scratch[RMSNORM_LOCAL_SIZE];

    const int m         = (int)get_group_id(0);
    const int tid       = (int)get_local_id(0);
    const int lsize     = (int)get_local_size(0);

    __global const float* xr = x + (size_t)m * (size_t)K;
    __global       float* yr = y + (size_t)m * (size_t)K;

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

    const float mean    = scratch[0] / (float)K;
    const float invRms  = rsqrt(mean + eps);

    // Apply scale.
    for (int k = tid; k < K; k += lsize) {
        yr[k] = xr[k] * weight[k] * invRms;
    }
}