// FlashAttention merge kernel — combine per-tile partials into the
// final attention output for decode mode (T_q == 1).
//
// Reads (m_t, l_t, o_t_unnorm) per tile, computes the global softmax
// normaliser, and folds the partials into the final out[hq, d].
//
// Identity (per Dao et al, applied across K_TILES tiles):
//
//   m_final = max over t of m_t
//   beta_t  = exp(m_t - m_final)
//   l_final = sum_t (beta_t * l_t)
//   out[d]  = sum_t (beta_t * o_t_unnorm[d]) / l_final
//
// The partial kernel emits o_t_unnorm = sum_kk exp(score_kk - m_t) *
// v[kk][d], i.e. it already used m_t as the local subtractive bias.
// Multiplying by beta_t = exp(m_t - m_final) re-aligns it to the
// global bias m_final, so the global softmax probabilities are
// (exp(score - m_final) / l_final) = (beta_t * exp(score - m_t) /
// l_final), exactly matching the variant-a kernel's output.
//
// Layouts:
//   partialMlo [nHeads, K_TILES, (2 + headDim)]
//   out        [1, nHeads, headDim]
//
// Launch: global = (nHeads, 1, 1), local = (16, 1, 1) — one subgroup
//   per head. The single-threaded prelude (m_final + l_final + alphas)
//   is cheap because K_TILES is small (≤ 32 at max context = 8192).

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

#ifndef ATTN_FLASH_LOCAL
#define ATTN_FLASH_LOCAL 16
#endif

#ifndef ATTN_FLASH_SG
#define ATTN_FLASH_SG 16
#endif

#ifndef ATTN_FLASH_MAX_KTILES
#define ATTN_FLASH_MAX_KTILES 64
#endif

#ifndef ATTN_FLASH_KTILE
#define ATTN_FLASH_KTILE 256
#endif

// M-CLR.2: nKTiles is derived from positionOffset (curLenPtr[0]) so the
// argument list is replay-friendly. Formula matches the partial kernel's
// derivation: nKTiles = ceil((positionOffset + 1) / ATTN_FLASH_KTILE).
__attribute__((reqd_work_group_size(ATTN_FLASH_LOCAL, 1, 1)))
__attribute__((intel_reqd_sub_group_size(ATTN_FLASH_SG)))
__kernel void attention_flash_merge(
    __global const float* partialMlo,
    __global       float* out,
    const int             nHeads,
    const int             headDim,
    __global const int*   curLenPtr)
{
    const int hq  = (int)get_group_id(0);
    const int lid = (int)get_local_id(0);

    const int positionOffset = curLenPtr[0];
    const int kMax           = positionOffset + 1;
    const int nKTiles        = (kMax + ATTN_FLASH_KTILE - 1) / ATTN_FLASH_KTILE;

    const int stride = 2 + headDim;
    __global const float* baseMlo =
        partialMlo + (size_t)hq * (size_t)nKTiles * (size_t)stride;

    __local float alphas[ATTN_FLASH_MAX_KTILES];
    __local float mFinalSlm;
    __local float lFinalSlm;

    if (lid == 0) {
        float mFinal = -INFINITY;
        for (int t = 0; t < nKTiles; ++t) {
            const float mt = baseMlo[(size_t)t * (size_t)stride + 0];
            if (mt > mFinal) mFinal = mt;
        }
        // mFinal can still be -INFINITY if every tile is past kMax.
        // That only happens when there's no valid K position at all,
        // which the wrapper rejects. Treat as 0 here to avoid NaN.
        if (mFinal == -INFINITY) mFinal = 0.0f;

        float lFinal = 0.0f;
        for (int t = 0; t < nKTiles; ++t) {
            const float mt = baseMlo[(size_t)t * (size_t)stride + 0];
            const float lt = baseMlo[(size_t)t * (size_t)stride + 1];
            const float beta = (mt == -INFINITY) ? 0.0f : exp(mt - mFinal);
            alphas[t] = beta;
            lFinal  += beta * lt;
        }

        mFinalSlm = mFinal;
        lFinalSlm = lFinal;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // lFinal == 0 only in the all-empty edge case — guard against
    // division by zero. Output stays at 0 which is the reasonable
    // "no information" answer.
    const float lFinal = lFinalSlm;
    const float invL   = (lFinal > 0.0f) ? (1.0f / lFinal) : 0.0f;

    __global float* oOut =
        out + (size_t)hq * (size_t)headDim;

    for (int d = lid; d < headDim; d += ATTN_FLASH_LOCAL) {
        float acc = 0.0f;
        for (int t = 0; t < nKTiles; ++t) {
            const float ot_d =
                baseMlo[(size_t)t * (size_t)stride + 2 + d];
            acc += alphas[t] * ot_d;
        }
        oOut[d] = acc * invL;
    }
}