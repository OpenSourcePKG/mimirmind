// FP16-KV variant of attention_flash_partial.cl. Decode-mode
// FlashAttention pass 1 (T_q == 1), reading K/V as fp16 instead of
// fp32. Loads promote to fp32 in registers via vload_half; the per-tile
// (m, l, o_unnorm) partials written to `partialMlo` stay fp32 so the
// existing attention_flash_merge kernel (which reads and combines
// partials in fp32) is unchanged and works for both KV dtypes.
//
// Everything else — tile geometry, SLM layout, SWA masking, Pass 1/2/3
// structure, curLenPtr wiring — is bit-for-bit identical to the f32
// variant so the KV-dtype swap stays isolated to load-sites.
//
// M10.2 Phase 0 Commit 3 — this kernel is only invoked when
// `KvCache::dtype() == FP16`; the f32 path stays on
// attention_flash_partial.cl, preserving bit parity against pre-M10.2
// behaviour.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

#ifndef ATTN_FLASH_LOCAL
#define ATTN_FLASH_LOCAL 16
#endif

#ifndef ATTN_FLASH_SG
#define ATTN_FLASH_SG 16
#endif

#ifndef ATTN_FLASH_KTILE
#define ATTN_FLASH_KTILE 64
#endif

__attribute__((reqd_work_group_size(ATTN_FLASH_LOCAL, 1, 1)))
__attribute__((intel_reqd_sub_group_size(ATTN_FLASH_SG)))
__kernel void attention_flash_partial_fp16(
    __global const float* q,
    __global const half*  k,
    __global const half*  v,
    __global       float* partialMlo,
    const int             nHeads,
    const int             nKvHeads,
    const int             headDim,
    __global const int*   curLenPtr,
    const float           scale,
    const int             slidingWindow)
{
    const int hq  = (int)get_group_id(0);
    const int kt  = (int)get_group_id(1);
    const int lid = (int)get_local_id(0);

    const int kvStride       = nKvHeads * headDim;
    const int hkv            = (hq * nKvHeads) / nHeads;
    const int positionOffset = curLenPtr[0];

    const int kMax    = positionOffset + 1;
    const int kMin    = (slidingWindow > 0 && kMax > slidingWindow)
                          ? (kMax - slidingWindow) : 0;
    const int nKTiles = (kMax + ATTN_FLASH_KTILE - 1) / ATTN_FLASH_KTILE;
    const int kStartRaw = kt * ATTN_FLASH_KTILE;
    const int kStart  = (kStartRaw > kMin) ? kStartRaw : kMin;
    const int kEndRaw = kStartRaw + ATTN_FLASH_KTILE;
    const int kEnd    = (kEndRaw < kMax) ? kEndRaw : kMax;

    __global float* mloPtr =
        partialMlo + ((size_t)hq * (size_t)nKTiles + (size_t)kt) *
                     (size_t)(2 + headDim);

    // Past the causal mask OR entirely below the sliding-window low
    // bound — emit a neutral partial so the merge sees no contribution.
    if (kStart >= kEnd) {
        if (lid == 0) {
            mloPtr[0] = -INFINITY;
            mloPtr[1] = 0.0f;
        }
        for (int d = lid; d < headDim; d += ATTN_FLASH_LOCAL) {
            mloPtr[2 + d] = 0.0f;
        }
        return;
    }

    __global const float* qVec = q + (size_t)hq * (size_t)headDim;

    __local float scores[ATTN_FLASH_KTILE];

    const int tileLen = kEnd - kStart;

    // Pass 1 — Q·K scores, scaled.
    for (int kk = lid; kk < tileLen; kk += ATTN_FLASH_LOCAL) {
        const int absKk = kStart + kk;
        __global const half* kVec =
            k + (size_t)absKk * (size_t)kvStride + (size_t)hkv * (size_t)headDim;
        float acc = 0.0f;
        for (int d = 0; d < headDim; ++d) {
            acc += qVec[d] * vload_half(d, kVec);
        }
        scores[kk] = acc * scale;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Pass 2 — stable softmax on the tile.
    float mPart = -INFINITY;
    for (int kk = lid; kk < tileLen; kk += ATTN_FLASH_LOCAL) {
        const float s = scores[kk];
        if (s > mPart) mPart = s;
    }
    const float mLocal = sub_group_reduce_max(mPart);

    float lPart = 0.0f;
    for (int kk = lid; kk < tileLen; kk += ATTN_FLASH_LOCAL) {
        const float e = exp(scores[kk] - mLocal);
        scores[kk] = e;
        lPart += e;
    }
    const float lLocal = sub_group_reduce_add(lPart);
    barrier(CLK_LOCAL_MEM_FENCE);

    // Pass 3 — unnormalized V-weighted sum. Stripe d across threads.
    for (int d = lid; d < headDim; d += ATTN_FLASH_LOCAL) {
        float acc = 0.0f;
        for (int kk = 0; kk < tileLen; ++kk) {
            __global const half* vVec =
                v + (size_t)(kStart + kk) * (size_t)kvStride +
                    (size_t)hkv * (size_t)headDim;
            acc += scores[kk] * vload_half(d, vVec);
        }
        mloPtr[2 + d] = acc;
    }

    if (lid == 0) {
        mloPtr[0] = mLocal;
        mloPtr[1] = lLocal;
    }
}