// FlashAttention partial-tile kernel — decode mode (T_q == 1).
//
// Each workgroup processes ONE K-tile of ONE head: produces a partial
// (m, l, o) triple in global scratch. The companion attention_flash_merge
// kernel then folds the per-tile partials into a final output using
// the online-softmax identity (Dao et al, 2022):
//
//   m_new = max(m_a, m_b)
//   l_new = exp(m_a - m_new) * l_a + exp(m_b - m_new) * l_b
//   o_new = exp(m_a - m_new) * o_a + exp(m_b - m_new) * o_b
//
// where (m_t, l_t, o_t) is the (max, sum_exp, sum(softmax_unnorm * V))
// computed over tile t. Across tiles this is associative — the merge
// kernel does it sequentially.
//
// Why this kernel exists: the M5f.3 variant-a kernel launches only
// nHeads workgroups in decode mode, leaving the iGPU under-saturated.
// With K-tiles, decode launches nHeads * K_TILES workgroups (= 28*32
// for Qwen2.5 at max context, 32*32 for Gemma 4) — close to the
// ~1000-WG saturation threshold from M5e.
//
// Layouts:
//   q          [1, nHeads,    headDim]
//   k          [T_k, nKvHeads, headDim]
//   v          [T_k, nKvHeads, headDim]
//   partialMlo [nHeads, K_TILES, (2 + headDim)]   global scratch
//
// For tile t of head hq, the slot is:
//   partialMlo + (hq * K_TILES + t) * (2 + headDim)
//   - [0]            : m (local max of scaled Q·K within tile)
//   - [1]            : l (sum of exp(score - m) within tile)
//   - [2 .. 2+headDim) : o (sum of exp(score - m) * v[k][d] within tile,
//                          UNNORMALIZED — merge kernel divides by l_total)
//
// Launch: global = (nHeads, K_TILES, 1), local = (16, 1, 1) — one
//   subgroup per workgroup, same geometry idea as variant (a) but
//   tiled over K.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

#ifndef ATTN_FLASH_LOCAL
#define ATTN_FLASH_LOCAL 16
#endif

#ifndef ATTN_FLASH_SG
#define ATTN_FLASH_SG 16
#endif

// M5f.3.2 — shrunk from 256 to 64. At E4B decode with curLen ~500 the
// old geometry launched only nHeads * ceil(500/256) = 8*2 = 16
// workgroups on a 64-VE Xe-LPG (25 % occupancy). K_TILE=64 quadruples
// concurrent workgroups at typical decode lengths (8*8=64) — sweet
// spot for the Xe-Core scheduler. MAX_KTILES bumped to 256 to keep
// the 16384-token compile-time context envelope (256*64 = 16384).
#ifndef ATTN_FLASH_KTILE
#define ATTN_FLASH_KTILE 64
#endif

// M-CLR.2: positionOffset comes from a __global int-slot (curLenPtr).
// T_k was previously an unused arg (kernel body never read it); it's
// dropped. nKTiles is derived from positionOffset inside the kernel so
// the argument list carries just one varying value. See M-CLR inventory.
__attribute__((reqd_work_group_size(ATTN_FLASH_LOCAL, 1, 1)))
__attribute__((intel_reqd_sub_group_size(ATTN_FLASH_SG)))
__kernel void attention_flash_partial(
    __global const float* q,
    __global const float* k,
    __global const float* v,
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

    // Decode mode: T_q == 1. The single query position is at absPos =
    // positionOffset, attending to keys [kMin, positionOffset + 1) where
    // kMin = max(0, positionOffset + 1 - slidingWindow) for SWA layers
    // (0 for Full-Attention / non-SWA).
    const int kMax    = positionOffset + 1;
    const int kMin    = (slidingWindow > 0 && kMax > slidingWindow)
                          ? (kMax - slidingWindow) : 0;
    const int nKTiles = (kMax + ATTN_FLASH_KTILE - 1) / ATTN_FLASH_KTILE;
    const int kStartRaw = kt * ATTN_FLASH_KTILE;
    // Clamp low boundary against sliding-window.
    const int kStart  = (kStartRaw > kMin) ? kStartRaw : kMin;
    const int kEndRaw = kStartRaw + ATTN_FLASH_KTILE;
    const int kEnd    = (kEndRaw < kMax) ? kEndRaw : kMax;

    __global float* mloPtr =
        partialMlo + ((size_t)hq * (size_t)nKTiles + (size_t)kt) *
                     (size_t)(2 + headDim);

    // Past the causal mask OR entirely below the sliding-window low
    // bound — emit a neutral partial so the merge sees no contribution.
    // exp(-inf - mFinal) = 0 → β_t * l_t and β_t * o_t both vanish.
    // l = 0 and o = 0 are written for hygiene.
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
        __global const float* kVec =
            k + (size_t)absKk * (size_t)kvStride + (size_t)hkv * (size_t)headDim;
        float acc = 0.0f;
        for (int d = 0; d < headDim; ++d) {
            acc += qVec[d] * kVec[d];
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

    // Pass 3 — unnormalized V-weighted sum: o_partial[d] = sum_kk
    // exp(scores[kk] - mLocal) * v[kStart+kk][hkv][d]. Stripe d across
    // threads. The merge kernel divides by l_total at the end.
    for (int d = lid; d < headDim; d += ATTN_FLASH_LOCAL) {
        float acc = 0.0f;
        for (int kk = 0; kk < tileLen; ++kk) {
            __global const float* vVec =
                v + (size_t)(kStart + kk) * (size_t)kvStride +
                    (size_t)hkv * (size_t)headDim;
            acc += scores[kk] * vVec[d];
        }
        mloPtr[2 + d] = acc;
    }

    if (lid == 0) {
        mloPtr[0] = mLocal;
        mloPtr[1] = lLocal;
    }
}