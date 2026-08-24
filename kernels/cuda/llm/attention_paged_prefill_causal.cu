// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// paged_attention_prefill_causal — Tier 2 Increment II (roadmap 5.21).
//
// Paged, batched, CAUSAL, T>1 prefill attention: the missing piece for the
// mixed prefill+decode continuous-batching forward. Each slot `seq` carries
// seqT[seq] query tokens (ragged, laid out at token offset queryOff[seq]); query
// position pq attends causally over KV positions [0, startPos[seq]+pq] read from
// the paged pool via block-table indirection (same layout/addressing as
// paged_attention_v1). startPos[seq] = the slot's prior KV length before this
// chunk (its chunk's KV is already written into the pool at [startPos, startPos+
// seqT)). Streaming-softmax + GQA are byte-identical to paged_attention_v1, so
// pq's output equals a paged_attention_v1 decode run with seq_len=startPos+pq+1.
//
// query/out : ragged [totalTok, num_heads, head_size]  row-major fp32
// key/value : paged pool [num_blocks, block_size, num_kv_heads, head_size] fp32
// block_tables [num_seqs, max_num_blocks_per_seq]; seqT/queryOff/startPos [num_seqs]
// Launch: grid = (num_heads, num_seqs, maxT), block = PAGED_ATTN_PREFILL_LOCAL,
//         smem = (2*head_size + block) * sizeof(float).

#include <cuda_runtime.h>

#ifndef PAGED_ATTN_PREFILL_LOCAL
#define PAGED_ATTN_PREFILL_LOCAL 128
#endif

#define PAGED_ATTN_KV_DTYPE_FP32  0

extern "C" __global__ __launch_bounds__(PAGED_ATTN_PREFILL_LOCAL)
void paged_attention_prefill_causal(
          float* __restrict__ out,           // [totalTok, num_heads, head_size]
    const float* __restrict__ query,         // [totalTok, num_heads, head_size]
    const void*  __restrict__ key_cache,     // paged pool
    const void*  __restrict__ value_cache,   // paged pool
    const int*   __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int*   __restrict__ seqT,          // [num_seqs] query count per slot
    const int*   __restrict__ queryOff,      // [num_seqs] first-query token offset
    const int*   __restrict__ startPos,      // [num_seqs] prior KV length
    const int                 num_seqs,
    const int                 num_heads,
    const int                 num_kv_heads,
    const int                 head_size,
    const int                 block_size,
    const int                 max_num_blocks_per_seq,
    const float               scale,
    const float               softcap)   // F32 KV only (CudaKernel kMaxArgs=16)
{
    const int hq  = static_cast<int>(blockIdx.x);   // query head
    const int seq = static_cast<int>(blockIdx.y);   // slot
    const int pq  = static_cast<int>(blockIdx.z);   // query position within chunk
    if (hq >= num_heads || seq >= num_seqs) {
        return;
    }
    if (pq >= seqT[seq]) {
        return;   // ragged: this slot has fewer query tokens than the grid max
    }

    const int tid  = static_cast<int>(threadIdx.x);
    const int nthr = static_cast<int>(blockDim.x);

    const int hkv     = (hq * num_kv_heads) / num_heads;
    // Causal continuation: query pq attends keys [0, startPos+pq] inclusive.
    const int seq_len = startPos[seq] + pq + 1;
    const long tokIdx = static_cast<long>(queryOff[seq]) + pq;

    const float* __restrict__ q_row =
        query + (tokIdx * num_heads + hq) * head_size;
    float* __restrict__ o_row =
        out + (tokIdx * num_heads + hq) * head_size;

    extern __shared__ float smem[];
    float* __restrict__ sq  = smem;                     // [head_size]
    float* __restrict__ acc = smem + head_size;         // [head_size]
    float* __restrict__ red = smem + 2 * head_size;     // [nthr]

    for (int d = tid; d < head_size; d += nthr) {
        sq[d]  = q_row[d];
        acc[d] = 0.0f;
    }
    __syncthreads();

    if (seq_len <= 0) {
        for (int d = tid; d < head_size; d += nthr) {
            o_row[d] = 0.0f;
        }
        return;
    }

    __shared__ float s_m, s_l, s_score;
    if (tid == 0) { s_m = -1.0e30f; s_l = 0.0f; }
    __syncthreads();

    const float* __restrict__ kbase = static_cast<const float*>(key_cache);
    const float* __restrict__ vbase = static_cast<const float*>(value_cache);
    const int*   __restrict__ bt =
        block_tables + static_cast<long>(seq) * max_num_blocks_per_seq;

    for (int p = 0; p < seq_len; ++p) {
        const int  blk    = bt[p / block_size];
        const int  slot   = p % block_size;
        const long kv_off =
            ((static_cast<long>(blk) * block_size + slot) * num_kv_heads + hkv)
            * head_size;
        const float* __restrict__ k_p = kbase + kv_off;

        float partial = 0.0f;
        for (int d = tid; d < head_size; d += nthr) {
            partial += sq[d] * k_p[d];
        }
        red[tid] = partial;
        __syncthreads();
        for (int stride = nthr >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) { red[tid] += red[tid + stride]; }
            __syncthreads();
        }
        if (tid == 0) {
            float logit = red[0] * scale;
            if (softcap > 0.0f) { logit = softcap * tanhf(logit / softcap); }
            s_score = logit;
        }
        __syncthreads();

        const float score   = s_score;
        const float m_new   = fmaxf(s_m, score);
        const float rescale = __expf(s_m - m_new);
        const float p_exp   = __expf(score - m_new);
        const float* __restrict__ v_p = vbase + kv_off;
        for (int d = tid; d < head_size; d += nthr) {
            acc[d] = acc[d] * rescale + p_exp * v_p[d];
        }
        __syncthreads();
        if (tid == 0) { s_l = s_l * rescale + p_exp; s_m = m_new; }
        __syncthreads();
    }

    const float inv_l = 1.0f / s_l;
    for (int d = tid; d < head_size; d += nthr) {
        o_row[d] = acc[d] * inv_l;
    }
}
