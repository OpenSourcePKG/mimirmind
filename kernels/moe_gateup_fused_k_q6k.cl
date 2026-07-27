// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

// Fused, device-indexed MoE gate+up projection for T=1 decode — Q6_K
// variant (Gemma 4 fused gate_up expert bank + GELU SwiGLU).
//
// M-CLR.MoE Increment 2: the decode-path enabler for Command-List-Replay.
// The pre-Increment-2 host loop read the router pick `e = topKIdx[k]` on
// the CPU to compute a per-expert weight base pointer, then dispatched one
// gate_up GEMV + gelu_mul + copy per k. That host read bakes a stale
// expert selection into any recorded command list. This kernel reads
// `expIdx[k]` on the device instead, so nothing on the host touches the
// routing between the device top-K and the expert matmuls.
//
//   gateActOut[k, f] = gelu_tanh( gate_f . x ) * ( up_f . x ) * downScale[e]
//
//     gate_f = row  f            of expert e's fused gate_up bank
//     up_f   = row (nFf + f)      of expert e's fused gate_up bank
//     e      = expIdx[k]
//
// downScale[e] (the ffn_down_exps.scale entry) is folded into the output
// here so the companion fused-K down kernel can take the raw router weight
// (topKWeight[k]) as `kw` — no host-side kw scratch write, one less CLR
// landmine. Scaling gateAct by downScale[e] scales the whole expert-down
// output by the same factor, so this is algebraically the host's
// `kw = topKWeight * downScale` applied one step earlier.
//
// Arguments
//   X          [dModel]              F32   shared token input (path-B norm)
//   W          Q6_K fused gate_up bank base — 2*nFf rows × dModel per expert
//   expIdx     [K]                   int32 routing indices (device top-K)
//   downScale  [nExperts]            F32   ffn_down_exps.scale
//   gateActOut [K, nFf]              F32   fused activation (down input)
//   dModel                          reduction dim of each row dot
//   nFf                             per-expert half-width (ffPerExpert)
//   kActive                         active expert count (K)
//   expertBytes                     stride from expert e to e+1 in W
//
// The dequant + Kahan accumulation mirror matmul_q6k_vec.cl exactly so the
// per-row dot is bit-for-bit the same as the host GEMV path it replaces;
// only the fused activation + downScale multiply are new. Parity target is
// the sequential (gate GEMV, up GEMV, gelu_mul) path in Gemma4MoeBackend.
//
// v2 (perf): the gate-row dot and the up-row dot share ONE super-block
// traversal — the two rows multiply the same x, so each x super-block is
// read from SLM once and used for both dequant-mads. Separate Kahan
// accumulators keep each dot's add-order identical to the two-pass v1, so
// the result stays bit-for-bit the same; it just halves the loop control
// and SLM traffic.
//
// The whole X vector lives in SLM (it is identical across all K experts,
// unlike the down kernel where each k has its own gate-activation slice).
// Capped by MOE_GU_XMAX; the launcher falls back to the host path when
// dModel exceeds it.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

#ifndef MOE_GU_LOCAL
#define MOE_GU_LOCAL 64
#endif

#ifndef MOE_GU_SG
#define MOE_GU_SG 16
#endif

#define MOE_GU_OUTPUTS_PER_GROUP (MOE_GU_LOCAL / MOE_GU_SG)

// Max reduction width held in SLM. Gemma 4 26B-A4B has dModel = 2560; leave
// headroom for larger siblings. The launcher guards this ceiling and falls
// back to the host path above it.
#ifndef MOE_GU_XMAX
#define MOE_GU_XMAX 4096
#endif

#define Q6K_BLOCK_ELEMENTS 256
#define Q6K_BLOCK_BYTES    210

// Fused Kahan-compensated Q6_K dot of BOTH the gate row and the up row for
// one sub-group lane, sharing the super-block traversal + x-SLM reads. Each
// accumulator keeps the exact add order of the single-row matmul_q6k_vec so
// `outG`/`outU` are bit-identical to two separate q6k row dots. Reads
// nothing when !active. A sub_group_reduce_add over the 16 lanes gives each
// full dot.
static inline void q6k_two_row_lane_dot(__local const float* xTile,
                                        __global const uchar* gRow,
                                        __global const uchar* uRow,
                                        int nSuper,
                                        int sgLocal,
                                        bool active,
                                        float* outG,
                                        float* outU)
{
    if (!active) {
        *outG = 0.0f;
        *outU = 0.0f;
        return;
    }

    float          sumG = 0.0f, sumU = 0.0f;
    volatile float kcG  = 0.0f, kcU  = 0.0f;

    for (int sb = 0; sb < nSuper; ++sb) {
        __global const uchar* gblock = gRow + sb * Q6K_BLOCK_BYTES;
        __global const uchar* ublock = uRow + sb * Q6K_BLOCK_BYTES;

        __global const uchar* gql = gblock;                 // 128 bytes
        __global const uchar* gqh = gblock + 128;           // 64 bytes
        __global const char*  gsc = (__global const char*)(gblock + 192);
        const float gd = vload_half(0, (__global const half*)(gblock + 208));

        __global const uchar* uql = ublock;
        __global const uchar* uqh = ublock + 128;
        __global const char*  usc = (__global const char*)(ublock + 192);
        const float ud = vload_half(0, (__global const half*)(ublock + 208));

        const int xBase = sb * Q6K_BLOCK_ELEMENTS;

        for (int hIdx = 0; hIdx < 2; ++hIdx) {
            const int xHalfBase = xBase + hIdx * 128;
            __global const uchar* gqlp = gql + hIdx * 64;
            __global const uchar* gqhp = gqh + hIdx * 32;
            __global const char*  gscp = gsc + hIdx * 8;
            __global const uchar* uqlp = uql + hIdx * 64;
            __global const uchar* uqhp = uqh + hIdx * 32;
            __global const char*  uscp = usc + hIdx * 8;

            for (int l = sgLocal; l < 32; l += MOE_GU_SG) {
                const int is = l / 16;

                // Shared x reads (same for gate + up).
                const float x0  = xTile[xHalfBase + l +  0];
                const float x32 = xTile[xHalfBase + l + 32];
                const float x64 = xTile[xHalfBase + l + 64];
                const float x96 = xTile[xHalfBase + l + 96];

                #define KAHAN(dst, comp, term)                                \
                    do {                                                      \
                        const float _y = (term) - (comp);                     \
                        const float _t = (dst) + _y;                          \
                        (comp) = (_t - (dst)) - _y;                           \
                        (dst)  = _t;                                          \
                    } while (0)

                // --- gate row (same add order as v1's first pass) ---
                {
                    const char q1 = (char)((gqlp[l +  0] & 0x0F) |
                                           (((gqhp[l] >> 0) & 0x03) << 4)) - 32;
                    const char q2 = (char)((gqlp[l + 32] & 0x0F) |
                                           (((gqhp[l] >> 2) & 0x03) << 4)) - 32;
                    const char q3 = (char)((gqlp[l +  0] >> 4) |
                                           (((gqhp[l] >> 4) & 0x03) << 4)) - 32;
                    const char q4 = (char)((gqlp[l + 32] >> 4) |
                                           (((gqhp[l] >> 6) & 0x03) << 4)) - 32;
                    const float s0 = gd * (float)gscp[is + 0];
                    const float s2 = gd * (float)gscp[is + 2];
                    const float s4 = gd * (float)gscp[is + 4];
                    const float s6 = gd * (float)gscp[is + 6];
                    KAHAN(sumG, kcG, x0  * (s0 * (float)q1));
                    KAHAN(sumG, kcG, x32 * (s2 * (float)q2));
                    KAHAN(sumG, kcG, x64 * (s4 * (float)q3));
                    KAHAN(sumG, kcG, x96 * (s6 * (float)q4));
                }
                // --- up row (same add order as v1's second pass) ---
                {
                    const char q1 = (char)((uqlp[l +  0] & 0x0F) |
                                           (((uqhp[l] >> 0) & 0x03) << 4)) - 32;
                    const char q2 = (char)((uqlp[l + 32] & 0x0F) |
                                           (((uqhp[l] >> 2) & 0x03) << 4)) - 32;
                    const char q3 = (char)((uqlp[l +  0] >> 4) |
                                           (((uqhp[l] >> 4) & 0x03) << 4)) - 32;
                    const char q4 = (char)((uqlp[l + 32] >> 4) |
                                           (((uqhp[l] >> 6) & 0x03) << 4)) - 32;
                    const float s0 = ud * (float)uscp[is + 0];
                    const float s2 = ud * (float)uscp[is + 2];
                    const float s4 = ud * (float)uscp[is + 4];
                    const float s6 = ud * (float)uscp[is + 6];
                    KAHAN(sumU, kcU, x0  * (s0 * (float)q1));
                    KAHAN(sumU, kcU, x32 * (s2 * (float)q2));
                    KAHAN(sumU, kcU, x64 * (s4 * (float)q3));
                    KAHAN(sumU, kcU, x96 * (s6 * (float)q4));
                }

                #undef KAHAN
            }
        }
    }

    *outG = sumG + kcG;
    *outU = sumU + kcU;
}

__attribute__((reqd_work_group_size(MOE_GU_LOCAL, 1, 1)))
__attribute__((intel_reqd_sub_group_size(MOE_GU_SG)))
__kernel void moe_gateup_fused_k_q6k(
    __global const float* X,          // [dModel] shared token input
    __global const uchar* W,          // Q6_K fused gate_up expert bank
    __global const int*   expIdx,     // [K]
    __global const float* downScale,  // [nExperts]
    __global       float* gateActOut, // [K, nFf]
    const int             dModel,
    const int             nFf,
    const int             kActive,
    const int             expertBytes)
{
    __local float xTile[MOE_GU_XMAX];

    const int  wg      = (int)get_group_id(0);
    const int  sgInWg  = (int)get_sub_group_id();
    const int  sgLocal = (int)get_sub_group_local_id();
    const int  tid     = (int)get_local_id(0);
    const int  lsize   = (int)get_local_size(0);
    const int  f       = wg * MOE_GU_OUTPUTS_PER_GROUP + sgInWg;
    const bool active  = (f < nFf);

    // Q6_K super-blocks along the reduction (dModel) dimension. Each row
    // of the fused bank (gate or up) is nSuper * 210 bytes.
    const int nSuper   = dModel / Q6K_BLOCK_ELEMENTS;
    const int rowBytes = nSuper * Q6K_BLOCK_BYTES;

    // Load the full token vector into SLM once — it is identical across all
    // K experts, so unlike the down kernel there is no per-k reload.
    for (int i = tid; i < dModel; i += lsize) {
        xTile[i] = X[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k = 0; k < kActive; ++k) {
        const int e = expIdx[k];
        __global const uchar* Wexpert = W + (size_t)e * (size_t)expertBytes;

        // gate = row f, up = row (nFf + f) of this expert's fused bank.
        __global const uchar* gateRow =
            Wexpert + (size_t)f * (size_t)rowBytes;
        __global const uchar* upRow =
            Wexpert + (size_t)(nFf + f) * (size_t)rowBytes;

        float laneG, laneU;
        q6k_two_row_lane_dot(xTile, gateRow, upRow, nSuper,
                             sgLocal, active, &laneG, &laneU);

        const float g = sub_group_reduce_add(laneG);
        const float u = sub_group_reduce_add(laneU);

        if (active && sgLocal == 0) {
            // gelu_tanh(g) * u, matching gelu_mul.cl, then fold downScale.
            const float g3 = g * g * g;
            const float t  = tanh(0.7978845608f * (g + 0.044715f * g3));
            const float ge = 0.5f * g * (1.0f + t);
            gateActOut[(size_t)k * (size_t)nFf + f] = ge * u * downScale[e];
        }
    }
}
