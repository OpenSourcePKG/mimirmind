// Fused MoE down-projection for T=1 decode. Collapses K sequential
// (down-matmul + scaledAddResidual) launches into a single dispatch.
//
//   accum[n] += sum_{k=0..K-1} kw[k] * sum_{l=0..ffPer-1}
//                                dequant_q6k(W[expIdx[k]], n, l)
//                                * gateAct[k, l]
//
//   X:     [K, ffPer]        F32, gate * up(gelu) per expert
//   W:     Q6_K expert bank  N-rows per expert × K-cols per row
//   accum: [dModel]          F32 read-modify-write
//   expIdx[K]      int32     routing indices into the expert bank
//   kw[K]          F32       topKWeight[k] * expDownScales[expIdx[k]]
//
// The routing arrays live in host-visible USM; the caller (Gemma4MoeBackend)
// writes them per layer before dispatch. All experts see the same output
// vector accum[dModel]; contributions are summed per output row.
//
// Layout of one expert weight bank (in bytes): dModel rows × nSuper Q6_K
// super-blocks × 210 bytes/super-block. Stride between experts is
// expertBytes = dModel * nSuper * 210.
//
// Launch geometry (mirrors matmul_q6k_vec.cl):
//   local_size_x          = MOE_DOWN_LOCAL (64)
//   sub_group_size        = MOE_DOWN_SG (16)
//   outputs per workgroup = MOE_DOWN_LOCAL / MOE_DOWN_SG (= 4)
//   global_size_x         = ceil(dModel / 4) * 64
//
// The outer K loop lives inside the workgroup — no extra dispatches.
// Each K iteration reloads the ffPer-wide X slice into SLM (K = 8 typically,
// ffPer = 2048; K × ffPer bandwidth is small compared to weight traffic).

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

#ifndef MOE_DOWN_LOCAL
#define MOE_DOWN_LOCAL 64
#endif

#ifndef MOE_DOWN_SG
#define MOE_DOWN_SG 16
#endif

#define MOE_DOWN_OUTPUTS_PER_GROUP (MOE_DOWN_LOCAL / MOE_DOWN_SG)

#define Q6K_BLOCK_ELEMENTS 256
#define Q6K_BLOCK_BYTES    210

// 1024 elements = 4 super-blocks = 4 KiB SLM per workgroup — matches
// matmul_q6k_vec.cl so occupancy characteristics stay comparable.
#define X_TILE_ELEMENTS 1024

__attribute__((reqd_work_group_size(MOE_DOWN_LOCAL, 1, 1)))
__attribute__((intel_reqd_sub_group_size(MOE_DOWN_SG)))
__kernel void moe_down_fused_k_q6k(
    __global const float* X,          // [K, ffPer]
    __global const uchar* W,          // Q6_K expert bank base
    __global const int*   expIdx,     // [K] active expert indices
    __global const float* kw,         // [K] combined router × down-scale weight
    __global       float* accum,      // [dModel] read-modify-write
    const int             ffPer,      // K-dim of each per-expert matmul
    const int             dModel,     // N-dim of each per-expert matmul
    const int             kActive,    // active expert count
    const int             expertBytes) // stride from expert e to e+1 in W
{
    __local float xTile[X_TILE_ELEMENTS];

    const int  wg      = (int)get_group_id(0);
    const int  sgInWg  = (int)get_sub_group_id();
    const int  sgLocal = (int)get_sub_group_local_id();
    const int  tid     = (int)get_local_id(0);
    const int  lsize   = (int)get_local_size(0);
    const int  n       = wg * MOE_DOWN_OUTPUTS_PER_GROUP + sgInWg;
    const bool active  = (n < dModel);
    const int  nSuper  = ffPer / Q6K_BLOCK_ELEMENTS;
    const int  rowBytes = nSuper * Q6K_BLOCK_BYTES;

    // Running accumulator across K experts. Kahan-compensated FP32 like
    // matmul_q6k_vec.cl — the terms per lane multiply out to ~K * ffPer/16
    // additions which can chew 4-5 bits of precision without compensation.
    float          accumSum = 0.0f;
    volatile float accumKc  = 0.0f;

    for (int k = 0; k < kActive; ++k) {
        const int e = expIdx[k];
        const float ekw = kw[k];

        __global const uchar* Wexpert = W + (size_t)e * (size_t)expertBytes;
        __global const float* Xk      = X + (size_t)k * (size_t)ffPer;

        // Per-expert per-row dot product; running sum lives in `sum` for
        // this K iteration only. After the tile loop we fold into
        // accumSum via kw[k].
        float          sum = 0.0f;
        volatile float kc  = 0.0f;

        for (int tile = 0; tile < ffPer; tile += X_TILE_ELEMENTS) {
            const int tileK = min(X_TILE_ELEMENTS, ffPer - tile);
            for (int i = tid; i < tileK; i += lsize) {
                xTile[i] = Xk[tile + i];
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            if (active) {
                __global const uchar* row = Wexpert + (size_t)n * (size_t)rowBytes;

                const int sbStart  = tile / Q6K_BLOCK_ELEMENTS;
                const int sbInTile = X_TILE_ELEMENTS / Q6K_BLOCK_ELEMENTS;
                const int sbEnd    = min(sbStart + sbInTile, nSuper);

                for (int sb = sbStart; sb < sbEnd; ++sb) {
                    __global const uchar* block = row + sb * Q6K_BLOCK_BYTES;

                    __global const uchar* ql = block;
                    __global const uchar* qh = block + 128;
                    __global const char*  sc =
                        (__global const char*)(block + 192);
                    const float d =
                        vload_half(0, (__global const half*)(block + 208));

                    const int xLocalBase = (sb - sbStart) * Q6K_BLOCK_ELEMENTS;

                    for (int hIdx = 0; hIdx < 2; ++hIdx) {
                        const int xHalfBase = xLocalBase + hIdx * 128;
                        __global const uchar* qlp = ql + hIdx * 64;
                        __global const uchar* qhp = qh + hIdx * 32;
                        __global const char*  scp = sc + hIdx * 8;

                        for (int l = sgLocal; l < 32; l += MOE_DOWN_SG) {
                            const int is = l / 16;

                            const char q1 = (char)((qlp[l +  0] & 0x0F) |
                                                   (((qhp[l] >> 0) & 0x03) << 4)) - 32;
                            const char q2 = (char)((qlp[l + 32] & 0x0F) |
                                                   (((qhp[l] >> 2) & 0x03) << 4)) - 32;
                            const char q3 = (char)((qlp[l +  0] >> 4) |
                                                   (((qhp[l] >> 4) & 0x03) << 4)) - 32;
                            const char q4 = (char)((qlp[l + 32] >> 4) |
                                                   (((qhp[l] >> 6) & 0x03) << 4)) - 32;

                            const float s0 = d * (float)scp[is + 0];
                            const float s2 = d * (float)scp[is + 2];
                            const float s4 = d * (float)scp[is + 4];
                            const float s6 = d * (float)scp[is + 6];

                            #define KAHAN_ADD(dest, comp, term)                 \
                                do {                                             \
                                    const float _y = (term) - (comp);            \
                                    const float _t = (dest) + _y;                \
                                    (comp) = (_t - (dest)) - _y;                 \
                                    (dest) = _t;                                 \
                                } while (0)

                            KAHAN_ADD(sum, kc, xTile[xHalfBase + l +  0] * (s0 * (float)q1));
                            KAHAN_ADD(sum, kc, xTile[xHalfBase + l + 32] * (s2 * (float)q2));
                            KAHAN_ADD(sum, kc, xTile[xHalfBase + l + 64] * (s4 * (float)q3));
                            KAHAN_ADD(sum, kc, xTile[xHalfBase + l + 96] * (s6 * (float)q4));

                            #undef KAHAN_ADD
                        }
                    }
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // Fold this expert's contribution into the outer accumulator.
        // Each lane's partial `sum` is only a fraction of the row dot;
        // we can't reduce yet because the next K iter still uses the
        // same lane split. So scale the lane partial by kw[k] and add
        // to the running outer sum; the final sub_group_reduce covers
        // all K contributions in one shot.
        const float lanePartial = (sum + kc) * ekw;
        const float _y = lanePartial - accumKc;
        const float _t = accumSum + _y;
        accumKc  = (_t - accumSum) - _y;
        accumSum = _t;
    }

    accumSum += accumKc;
    accumSum = sub_group_reduce_add(accumSum);

    if (active && sgLocal == 0) {
        accum[n] += accumSum;
    }
}