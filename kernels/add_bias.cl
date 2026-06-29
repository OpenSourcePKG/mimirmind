// In-place broadcast bias add:
//   y[m, k] += bias[k]   for m in [0, M), k in [0, K)
//
// Launch: 1D global = M * K work-items in groups of ADD_BIAS_LOCAL.

#ifndef ADD_BIAS_LOCAL
#define ADD_BIAS_LOCAL 256
#endif

__attribute__((reqd_work_group_size(ADD_BIAS_LOCAL, 1, 1)))
__kernel void add_bias(
    __global       float* y,
    __global const float* bias,
    const int             M,
    const int             K)
{
    const int gid = (int)get_global_id(0);
    const int total = M * K;
    if (gid >= total) {
        return;
    }
    const int k = gid % K;
    y[gid] += bias[k];
}