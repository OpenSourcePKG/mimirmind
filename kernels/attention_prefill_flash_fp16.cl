// FP16-KV variant of attention_prefill_flash.cl. Single-workgroup
// streaming FlashAttention for T_q > 1 (M5i.J), reading K/V as fp16
// instead of fp32. Loads promote to fp32 in registers via vload_half,
// so the online-softmax (m, l, o) state stays fully fp32-precision.
//
// Everything else — SLM layout, ktile geometry, Pass A/B/C structure,
// SWA masking, curLenPtr wiring — is bit-for-bit identical to the f32
// variant so the KV-dtype swap stays isolated to load-sites.
//
// M10.2 Phase 0 Commit 3 — this kernel is only invoked when
// `KvCache::dtype() == FP16`; the f32 path stays on
// attention_prefill_flash.cl, preserving bit parity against pre-M10.2
// behaviour.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

#ifndef ATTN_FLASH_PREFILL_LOCAL
#define ATTN_FLASH_PREFILL_LOCAL 16
#endif

#ifndef ATTN_FLASH_PREFILL_SG
#define ATTN_FLASH_PREFILL_SG 16
#endif

#ifndef ATTN_FLASH_PREFILL_KTILE
#define ATTN_FLASH_PREFILL_KTILE 128
#endif

#ifndef ATTN_FLASH_PREFILL_MAX_HEADDIM
#define ATTN_FLASH_PREFILL_MAX_HEADDIM 512
#endif

__attribute__((reqd_work_group_size(ATTN_FLASH_PREFILL_LOCAL, 1, 1)))
__attribute__((intel_reqd_sub_group_size(ATTN_FLASH_PREFILL_SG)))
__kernel void attention_prefill_flash_fp16(
    __global const float* q,
    __global const half*  k,
    __global const half*  v,
    __global       float* out,
    const int             T_q,
    const int             nHeads,
    const int             nKvHeads,
    const int             headDim,
    __global const int*   curLenPtr,
    const float           scale,
    const int             slidingWindow)
{
    (void)T_q;

    const int hq  = (int)get_group_id(0);
    const int pq  = (int)get_group_id(1);
    const int lid = (int)get_local_id(0);

    const int qStride        = nHeads   * headDim;
    const int kvStride       = nKvHeads * headDim;
    const int hkv            = (hq * nKvHeads) / nHeads;
    const int positionOffset = curLenPtr[0];
    const int absPos         = positionOffset + pq;
    const int kMax           = absPos + 1;
    const int kMin           = (slidingWindow > 0 && kMax > slidingWindow)
                                 ? (kMax - slidingWindow) : 0;
    const int ktStart        = kMin / ATTN_FLASH_PREFILL_KTILE;
    const int nKTiles        = (kMax + ATTN_FLASH_PREFILL_KTILE - 1)
                               / ATTN_FLASH_PREFILL_KTILE;

    __global const float* qVec = q   + (size_t)pq * (size_t)qStride
                                     + (size_t)hq * (size_t)headDim;
    __global       float* oVec = out + (size_t)pq * (size_t)qStride
                                     + (size_t)hq * (size_t)headDim;

    __local float scores[ATTN_FLASH_PREFILL_KTILE];
    __local float oRun  [ATTN_FLASH_PREFILL_MAX_HEADDIM];

    float m = -INFINITY;
    float l = 0.0f;
    for (int d = lid; d < headDim; d += ATTN_FLASH_PREFILL_LOCAL) {
        oRun[d] = 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int kt = ktStart; kt < nKTiles; ++kt) {
        const int kStartRaw = kt * ATTN_FLASH_PREFILL_KTILE;
        const int kStart    = (kStartRaw > kMin) ? kStartRaw : kMin;
        const int kEndRaw   = kStartRaw + ATTN_FLASH_PREFILL_KTILE;
        const int kEnd      = (kEndRaw < kMax) ? kEndRaw : kMax;
        const int tileLen   = kEnd - kStart;

        // -- Pass A — Q·K scaled scores for this K-tile. --------------
        for (int kk = lid; kk < tileLen; kk += ATTN_FLASH_PREFILL_LOCAL) {
            __global const half* kVec =
                k + (size_t)(kStart + kk) * (size_t)kvStride
                  + (size_t)hkv * (size_t)headDim;
            float acc = 0.0f;
            for (int d = 0; d < headDim; ++d) {
                acc += qVec[d] * vload_half(d, kVec);
            }
            scores[kk] = acc * scale;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // -- Pass B — Online-softmax rescale. -------------------------
        float mTilePart = -INFINITY;
        for (int kk = lid; kk < tileLen; kk += ATTN_FLASH_PREFILL_LOCAL) {
            const float s = scores[kk];
            if (s > mTilePart) mTilePart = s;
        }
        const float mTile = sub_group_reduce_max(mTilePart);
        const float mNew  = (m > mTile) ? m : mTile;
        const float alpha = exp(m - mNew);

        float lTilePart = 0.0f;
        for (int kk = lid; kk < tileLen; kk += ATTN_FLASH_PREFILL_LOCAL) {
            const float e = exp(scores[kk] - mNew);
            scores[kk] = e;
            lTilePart += e;
        }
        const float lTile = sub_group_reduce_add(lTilePart);
        l = alpha * l + lTile;
        m = mNew;
        barrier(CLK_LOCAL_MEM_FENCE);

        // -- Pass C — scale-and-accumulate V into oRun. ---------------
        for (int d = lid; d < headDim; d += ATTN_FLASH_PREFILL_LOCAL) {
            float acc = alpha * oRun[d];
            for (int kk = 0; kk < tileLen; ++kk) {
                __global const half* vVec =
                    v + (size_t)(kStart + kk) * (size_t)kvStride
                      + (size_t)hkv * (size_t)headDim;
                acc += scores[kk] * vload_half(d, vVec);
            }
            oRun[d] = acc;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const float invL = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (int d = lid; d < headDim; d += ATTN_FLASH_PREFILL_LOCAL) {
        oVec[d] = oRun[d] * invL;
    }
}