// Fused residual-add + RMSNorm.
//   x[m, k] += delta[m, k]     (in-place)
//   y[m, k] = x[m, k] * weight[k] / sqrt(mean(x[m, :]^2) + eps)
//
// Saves one dispatch per transformer block boundary where the plain
// pattern `_ops.addResidualAsync(x, delta, M*K); _ops.rmsNormAsync(x, M, K, w, ...)`
// runs. Every Gemma/Qwen backend hits this pattern at the end of the
// attention section (`x += attn_out; ffn_norm(x)`), so the fusion is
// architecture-agnostic.
//
// Launch: one workgroup per row m (global = M * LOCAL_SIZE,
// local = LOCAL_SIZE). Same tree-reduction body as rmsnorm.cl.

#ifndef ADD_RMSNORM_LOCAL_SIZE
#define ADD_RMSNORM_LOCAL_SIZE 128
#endif

__attribute__((reqd_work_group_size(ADD_RMSNORM_LOCAL_SIZE, 1, 1)))
__kernel void add_rmsnorm(
    __global       float* x,       // [M, K]  in-place accumulator
    __global const float* delta,   // [M, K]
    __global const float* weight,  // [K]
    __global       float* y,       // [M, K]  (may alias x)
    const float           eps,
    const int             K)
{
    __local float scratch[ADD_RMSNORM_LOCAL_SIZE];

    const int m     = (int)get_group_id(0);
    const int tid   = (int)get_local_id(0);
    const int lsize = (int)get_local_size(0);

    __global       float* xr = x     + (size_t)m * (size_t)K;
    __global const float* dr = delta + (size_t)m * (size_t)K;
    __global       float* yr = y     + (size_t)m * (size_t)K;

    // Phase 1: add-in-place + per-thread partial sum of squares. Each
    // thread walks its stride so the loop over K happens exactly once.
    float acc = 0.0f;
    for (int k = tid; k < K; k += lsize) {
        const float v = xr[k] + dr[k];
        xr[k] = v;
        acc = mad(v, v, acc);
    }
    scratch[tid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Power-of-two tree reduction.
    for (int stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const float mean   = scratch[0] / (float)K;
    const float invRms = rsqrt(mean + eps);

    // Phase 2: apply weight * invRms.
    for (int k = tid; k < K; k += lsize) {
        yr[k] = xr[k] * weight[k] * invRms;
    }
}