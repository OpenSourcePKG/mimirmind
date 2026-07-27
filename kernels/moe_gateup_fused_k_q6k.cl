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

// Max reduction width held in SLM (4 KiB * 4 = 16 KiB). Gemma 4 26B-A4B
// has dModel = 2560; leave headroom for larger d_model siblings. The
// launcher guards this ceiling and falls back to the host path above it.
#ifndef MOE_GU_XMAX
#define MOE_GU_XMAX 4096
#endif

#define Q6K_BLOCK_ELEMENTS 256
#define Q6K_BLOCK_BYTES    210

// Kahan-compensated Q6_K row dot for one sub-group lane. Returns this
// lane's partial (a sub_group_reduce_add across the 16 lanes gives the
// full dot). Reads nothing when !active so out-of-range rows are never
// touched. Byte layout + arithmetic copied verbatim from matmul_q6k_vec.cl.
static inline float q6k_row_lane_dot(__local const float* xTile,
                                     __global const uchar* row,
                                     int nSuper,
                                     int sgLocal,
                                     bool active)
{
    if (!active) {
        return 0.0f;
    }

    float          sum = 0.0f;
    volatile float kc  = 0.0f;

    for (int sb = 0; sb < nSuper; ++sb) {
        __global const uchar* block = row + sb * Q6K_BLOCK_BYTES;

        __global const uchar* ql = block;                 // 128 bytes
        __global const uchar* qh = block + 128;           // 64 bytes
        __global const char*  sc = (__global const char*)(block + 192);
        const float d = vload_half(0, (__global const half*)(block + 208));

        const int xBase = sb * Q6K_BLOCK_ELEMENTS;

        for (int hIdx = 0; hIdx < 2; ++hIdx) {
            const int xHalfBase = xBase + hIdx * 128;
            __global const uchar* qlp = ql + hIdx * 64;
            __global const uchar* qhp = qh + hIdx * 32;
            __global const char*  scp = sc + hIdx * 8;

            for (int l = sgLocal; l < 32; l += MOE_GU_SG) {
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

                #define KAHAN_ADD(term)                                       \
                    do {                                                      \
                        const float _y = (term) - kc;                         \
                        const float _t = sum + _y;                            \
                        kc  = (_t - sum) - _y;                                \
                        sum = _t;                                             \
                    } while (0)

                KAHAN_ADD(xTile[xHalfBase + l +  0] * (s0 * (float)q1));
                KAHAN_ADD(xTile[xHalfBase + l + 32] * (s2 * (float)q2));
                KAHAN_ADD(xTile[xHalfBase + l + 64] * (s4 * (float)q3));
                KAHAN_ADD(xTile[xHalfBase + l + 96] * (s6 * (float)q4));

                #undef KAHAN_ADD
            }
        }
    }

    sum += kc;
    return sum;
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

        const float laneG = q6k_row_lane_dot(xTile, gateRow, nSuper,
                                             sgLocal, active);
        const float laneU = q6k_row_lane_dot(xTile, upRow, nSuper,
                                             sgLocal, active);

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
