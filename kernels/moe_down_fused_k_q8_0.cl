// Fused MoE down-projection for T=1 decode — Q8_0 variant.
//
// Companion to moe_down_fused_k_q6k.cl; identical dispatch surface, only
// the per-block dequant math differs. Q8_0 = 34-byte blocks (fp16 scale
// + 32 int8 quants), no sub-scales, no bit packing. Straight FP32
// accumulation is fine — per lane touches ~K/16 terms (~132 for
// ffPer=2112) which stays inside FP32's representable range with X
// as the dominant noise source.
//
//   accum[n] += sum_{k=0..K-1} kw[k] * sum_{l=0..ffPer-1}
//                                dequant_q8_0(W[expIdx[k]], n, l)
//                                * gateAct[k, l]
//
// See moe_down_fused_k_q6k.cl for the full argument contract and layout
// rationale. `expertBytes = dModel * (ffPer/32) * 34` for Q8_0.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

#ifndef MOE_DOWN_LOCAL
#define MOE_DOWN_LOCAL 64
#endif

#ifndef MOE_DOWN_SG
#define MOE_DOWN_SG 16
#endif

#define MOE_DOWN_OUTPUTS_PER_GROUP (MOE_DOWN_LOCAL / MOE_DOWN_SG)

#define Q8_0_BLOCK_ELEMENTS 32
#define Q8_0_BLOCK_BYTES    34

// 1024 elements = 32 blocks = 4 KiB SLM per workgroup — matches
// matmul_q8_0_vec.cl and the Q6_K fused variant.
#define X_TILE_ELEMENTS 1024

__attribute__((reqd_work_group_size(MOE_DOWN_LOCAL, 1, 1)))
__attribute__((intel_reqd_sub_group_size(MOE_DOWN_SG)))
__kernel void moe_down_fused_k_q8_0(
    __global const float* X,          // [K, ffPer]
    __global const uchar* W,          // Q8_0 expert bank base
    __global const int*   expIdx,     // [K]
    __global const float* kw,         // [K] combined router × down-scale weight
    __global       float* accum,      // [dModel] read-modify-write
    const int             ffPer,
    const int             dModel,
    const int             kActive,
    const int             expertBytes)
{
    __local float xTile[X_TILE_ELEMENTS];

    const int  wg      = (int)get_group_id(0);
    const int  sgInWg  = (int)get_sub_group_id();
    const int  sgLocal = (int)get_sub_group_local_id();
    const int  tid     = (int)get_local_id(0);
    const int  lsize   = (int)get_local_size(0);
    const int  n       = wg * MOE_DOWN_OUTPUTS_PER_GROUP + sgInWg;
    const bool active  = (n < dModel);
    const int  nBlocks = ffPer / Q8_0_BLOCK_ELEMENTS;
    const int  rowBytes = nBlocks * Q8_0_BLOCK_BYTES;

    // Outer accumulator (across K experts). Same lane-partial semantics
    // as the Q6_K variant — each lane sums its quant-strided partials,
    // scales by kw[k], folds into `accumSum`, then a single
    // sub_group_reduce_add at the end sums across the 16 sub-group lanes.
    float accumSum = 0.0f;

    for (int k = 0; k < kActive; ++k) {
        const int e   = expIdx[k];
        const float ekw = kw[k];

        __global const uchar* Wexpert = W + (size_t)e * (size_t)expertBytes;
        __global const float* Xk      = X + (size_t)k * (size_t)ffPer;

        float sum = 0.0f;

        for (int tile = 0; tile < ffPer; tile += X_TILE_ELEMENTS) {
            const int tileK = min(X_TILE_ELEMENTS, ffPer - tile);
            for (int i = tid; i < tileK; i += lsize) {
                xTile[i] = Xk[tile + i];
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            if (active) {
                __global const uchar* row =
                    Wexpert + (size_t)n * (size_t)rowBytes;

                const int blockStart   = tile / Q8_0_BLOCK_ELEMENTS;
                const int blocksInTile = X_TILE_ELEMENTS / Q8_0_BLOCK_ELEMENTS;
                const int blockEnd     = min(blockStart + blocksInTile, nBlocks);

                for (int b = blockStart; b < blockEnd; ++b) {
                    __global const uchar* block = row + b * Q8_0_BLOCK_BYTES;
                    const float d =
                        vload_half(0, (__global const half*)(block));
                    __global const char* qs =
                        (__global const char*)(block + 2);

                    const int xLocalBase = (b - blockStart) * Q8_0_BLOCK_ELEMENTS;

                    // 16 lanes split the 32-element block: each lane handles
                    // 2 quants strided by MOE_DOWN_SG.
                    for (int l = sgLocal; l < Q8_0_BLOCK_ELEMENTS; l += MOE_DOWN_SG) {
                        sum = mad(xTile[xLocalBase + l],
                                  d * (float)qs[l],
                                  sum);
                    }
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // Scale this expert's lane-partial by its combined router×down-scale
        // weight and fold into the outer accumulator.
        accumSum += sum * ekw;
    }

    accumSum = sub_group_reduce_add(accumSum);

    if (active && sgLocal == 0) {
        accum[n] += accumSum;
    }
}