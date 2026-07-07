// FlashAttention prefill kernel — single-workgroup streaming online-softmax
// per Dao et al. 2022 (original formulation, not the 2-kernel partial/merge
// variant used for decode).
//
// One workgroup owns exactly one (hq, pq) — the same launch geometry as
// the plain attention.cl kernel — and streams K tile-by-tile through
// local memory. `(m, l, o)` live intra-WG so there is NO global scratch
// buffer, which is what lets this scale to arbitrary T_k without the
// O(T_q * K_TILES * headDim) blow-up that blocks the decode-flash pattern
// from being reused for prefill.
//
// Layouts (all row-major f32 USM, identical to attention.cl):
//   q   [T_q, nHeads,    headDim]
//   k   [T_k, nKvHeads,  headDim]
//   v   [T_k, nKvHeads,  headDim]
//   out [T_q, nHeads,    headDim]
//
// Causal mask + sliding window: query position pq attends to keys
//   [kMin, kMax) with
//     kMax = positionOffset + pq + 1
//     kMin = (slidingWindow > 0 && kMax > slidingWindow)
//              ? (kMax - slidingWindow) : 0
// so `slidingWindow == 0` degenerates to pure causal. `slidingWindow > 0`
// matches Gemma-family SWA layers.
//
// GQA: query head hq reads from KV head hkv = (hq * nKvHeads) / nHeads.
//
// Launch: global = (nHeads, T_q, 1), local = (16, 1, 1). Same as the
// plain kernel — every (hq, pq) pair independently produces its
// out-row. Subgroup == workgroup at LOCAL=SG=16, so sub_group_reduce_*
// gives the workgroup-wide value without an extra SLM broadcast.
//
// SLM per workgroup:
//   scores [K_TILE]                   ~=  0.5 KiB at K_TILE=128
//   oRun   [MAX_HEADDIM]              ~=  2.0 KiB at headDim=512
//                                     ~=  2.5 KiB total
// (Compare against attention.cl variant (a) which reserves 64 KiB via
// scores[ATTN_MAX_TK=16384] per workgroup — the occupancy killer this
// kernel is here to fix.)
//
// M-CLR.2 compatibility: positionOffset comes from a __global int-slot
// (curLenPtr) so recorded command lists stay valid across a decode.
// nKTiles is derived from curLen inside the kernel.
//
// Rollback env: MIMIRMIND_DISABLE_FLASH_PREFILL=1 falls back to
// attention.cl variant (a). See GpuOps::attentionAsync dispatcher.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

#ifndef ATTN_FLASH_PREFILL_LOCAL
#define ATTN_FLASH_PREFILL_LOCAL 16
#endif

#ifndef ATTN_FLASH_PREFILL_SG
#define ATTN_FLASH_PREFILL_SG 16
#endif

// Streaming K-tile width. 128 keeps per-tile softmax rescale amortised
// (each Pass B/C pair pays one barrier + one alpha-rescale) while the
// SLM cost stays trivial (128 * 4 B = 512 B). Xe-LPG occupancy is not a
// launch-count concern here — prefill already launches nHeads * T_q
// workgroups, which is thousands at RAG-typical T_q. K-tiles only
// affect memory streaming and per-tile overhead amortisation.
#ifndef ATTN_FLASH_PREFILL_KTILE
#define ATTN_FLASH_PREFILL_KTILE 128
#endif

// Compile-time upper bound on headDim, matching GpuOps::kFlashMaxHeadDim.
// Gemma 4 full-attention layers have headDim=512, SWA layers 256, Qwen2.5
// 128. Bumping requires updating GpuOps.hpp's constant together.
#ifndef ATTN_FLASH_PREFILL_MAX_HEADDIM
#define ATTN_FLASH_PREFILL_MAX_HEADDIM 512
#endif

__attribute__((reqd_work_group_size(ATTN_FLASH_PREFILL_LOCAL, 1, 1)))
__attribute__((intel_reqd_sub_group_size(ATTN_FLASH_PREFILL_SG)))
__kernel void attention_prefill_flash(
    __global const float* q,
    __global const float* k,
    __global const float* v,
    __global       float* out,
    const int             T_q,
    const int             nHeads,
    const int             nKvHeads,
    const int             headDim,
    __global const int*   curLenPtr,
    const float           scale,
    const int             slidingWindow)
{
    (void)T_q;  // Reserved for future launch-geometry validation; the
                // kernel body derives everything it needs from lid/group.

    const int hq  = (int)get_group_id(0);
    const int pq  = (int)get_group_id(1);
    const int lid = (int)get_local_id(0);

    const int qStride        = nHeads   * headDim;
    const int kvStride       = nKvHeads * headDim;
    const int hkv            = (hq * nKvHeads) / nHeads;
    const int positionOffset = curLenPtr[0];
    const int absPos         = positionOffset + pq;
    const int kMax           = absPos + 1;
    // Sliding-window low bound. 0 → pure causal (Full-Attention / Qwen).
    const int kMin           = (slidingWindow > 0 && kMax > slidingWindow)
                                 ? (kMax - slidingWindow) : 0;
    // Start tile: first tile whose end crosses kMin. All-below-window
    // tiles are skipped entirely so the K-loop body only sees tiles
    // that hold at least one live key.
    const int ktStart        = kMin / ATTN_FLASH_PREFILL_KTILE;
    const int nKTiles        = (kMax + ATTN_FLASH_PREFILL_KTILE - 1)
                               / ATTN_FLASH_PREFILL_KTILE;

    __global const float* qVec = q   + (size_t)pq * (size_t)qStride
                                     + (size_t)hq * (size_t)headDim;
    __global       float* oVec = out + (size_t)pq * (size_t)qStride
                                     + (size_t)hq * (size_t)headDim;

    __local float scores[ATTN_FLASH_PREFILL_KTILE];
    __local float oRun  [ATTN_FLASH_PREFILL_MAX_HEADDIM];

    // Running (m, l) live in registers, identically replicated across
    // all 16 subgroup threads via sub_group_reduce_*. oRun lives in
    // SLM; each thread only writes its own stride of d's during init
    // and Pass C, then all threads read all d's during Pass C's V loop
    // (no — actually each thread owns its d's throughout, no read of
    // foreign d's; still needs SLM because private array indexing on
    // headDim/16 = 32 elements with variable index would spill).
    float m = -INFINITY;
    float l = 0.0f;
    for (int d = lid; d < headDim; d += ATTN_FLASH_PREFILL_LOCAL) {
        oRun[d] = 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int kt = ktStart; kt < nKTiles; ++kt) {
        const int kStartRaw = kt * ATTN_FLASH_PREFILL_KTILE;
        // Clamp low boundary against sliding-window: the first live tile
        // may have its early entries below kMin.
        const int kStart    = (kStartRaw > kMin) ? kStartRaw : kMin;
        const int kEndRaw   = kStartRaw + ATTN_FLASH_PREFILL_KTILE;
        const int kEnd      = (kEndRaw < kMax) ? kEndRaw : kMax;
        const int tileLen   = kEnd - kStart;
        // ktStart ensures kEndRaw > kMin, and nKTiles ensures kStartRaw <
        // kMax, so tileLen >= 1 on every iter — no empty-tile guard needed.

        // -- Pass A — Q·K scaled scores for this K-tile. --------------
        for (int kk = lid; kk < tileLen; kk += ATTN_FLASH_PREFILL_LOCAL) {
            __global const float* kVec =
                k + (size_t)(kStart + kk) * (size_t)kvStride
                  + (size_t)hkv * (size_t)headDim;
            float acc = 0.0f;
            for (int d = 0; d < headDim; ++d) {
                acc += qVec[d] * kVec[d];
            }
            scores[kk] = acc * scale;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // -- Pass B — Online-softmax rescale. -------------------------
        // The Dao 2022 identity: with (m_prev, l_prev, o_prev) already
        // representing the streamed prefix and (m_tile, l_tile, o_tile)
        // the current tile, the update is
        //   m_new  = max(m_prev, m_tile)
        //   alpha  = exp(m_prev - m_new)
        //   l_new  = alpha * l_prev + l_tile'      (l_tile using m_new)
        //   o_new  = alpha * o_prev + sum(beta_kk * v)  (beta_kk using m_new)
        // where beta_kk = exp(scores_kk - m_new). Here we fold m_new
        // into the beta write so Pass C can just multiply-accumulate.
        float mTilePart = -INFINITY;
        for (int kk = lid; kk < tileLen; kk += ATTN_FLASH_PREFILL_LOCAL) {
            const float s = scores[kk];
            if (s > mTilePart) mTilePart = s;
        }
        const float mTile = sub_group_reduce_max(mTilePart);
        const float mNew  = (m > mTile) ? m : mTile;
        // First iter: m == -INFINITY, mNew is finite from mTile, so
        // alpha = exp(-INFINITY - mNew) = 0. IEEE-correct, so the
        // prior-l and prior-o contributions vanish exactly as required.
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
                __global const float* vVec =
                    v + (size_t)(kStart + kk) * (size_t)kvStride
                      + (size_t)hkv * (size_t)headDim;
                acc += scores[kk] * vVec[d];
            }
            oRun[d] = acc;
        }
        // Barrier before the next tile's Pass A overwrites scores[].
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Normalise. l == 0 only in the degenerate all-masked case which
    // cannot happen for pq >= 0 (kMax >= 1). Guard emits 0 for hygiene.
    const float invL = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (int d = lid; d < headDim; d += ATTN_FLASH_PREFILL_LOCAL) {
        oVec[d] = oRun[d] * invL;
    }
}