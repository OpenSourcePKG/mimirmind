// Fused scaled-accumulate: dst[i] += scale * src[i]
//
// Replaces what the Gemma 4 MoE per-expert loop did as two passes:
//   mulScalarAsync(expertOutBuf, combined)   // expertOutBuf *= combined
//   addResidualAsync(accumT, expertOutBuf)   // accumT += expertOutBuf
// — into one kernel that does both in a single read/write per element.
//
// The intermediate "scaled expertOutBuf" was never read by anything
// downstream (the next expert iteration's down-projection overwrites
// expertOutBuf anyway), so the fusion has no correctness implication
// beyond saving one launch per expert per layer.
//
// Per token on Gemma 4 26B-A4B: 8 active experts × 30 layers = 240
// fewer kernel launches. Tiny, but adds up.
//
// Uses mad() so the compiler can pick the fused multiply-add path
// natively on Xe-LPG.

#ifndef SCALED_ADD_RESIDUAL_LOCAL
#define SCALED_ADD_RESIDUAL_LOCAL 256
#endif

__attribute__((reqd_work_group_size(SCALED_ADD_RESIDUAL_LOCAL, 1, 1)))
__kernel void scaled_add_residual(
    __global       float* dst,
    __global const float* src,
    const float           scale,
    const int             n)
{
    const int gid = (int)get_global_id(0);
    if (gid >= n) {
        return;
    }
    dst[gid] = mad(scale, src[gid], dst[gid]);
}
