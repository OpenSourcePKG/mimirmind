// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// paged_attention_v1 — SKELETON (M-Cuda.Batch Sub-Step B2)
//
// This is a compile-only launch-surface skeleton. The body is NOT
// implemented — the FA2-Ampere+PTX compute layer will land after
// (a) Blackwell hardware is in-house and (b) FlashInfer FMHA_V2 SM120
// has been read as reference-vault. Any invocation of this kernel
// before the body ships will `__trap()` — that is intentional, so
// nobody silently runs against zeroed output.
//
// The **signature** IS load-bearing and MUST be considered final for
// Bragi-v1:
//   - Callers (M-Cuda.Batch Phase C scheduler + Phase D
//     InferenceEngine::generateServing) will bind against the argument
//     order committed here.
//   - The C++ launch wrapper in src/core/gpu/cuda/PagedAttentionV1.hpp
//     mirrors this signature slot-for-slot and calls it via
//     CudaModule::getFunction("paged_attention_v1"). Changing the
//     signature after Phase C wire-up requires a coordinated update
//     across the scheduler + wrapper + this file.
//
// Design references (do NOT link as runtime deps):
//   - Kwon et al 2023 "PagedAttention" — original paper, blocked KV
//     with per-sequence block-table indirection.
//   - vLLM V1 csrc/attention/paged_attention_v1_kernel.cuh — signature
//     shape, layout conventions. Softcap parameter is our Gemma-4
//     extension (final-logit soft-cap fix c69f012).
//   - gau-nernst "Speed-of-Light Flash Attention for 5090 in CUDA C++"
//     — Ampere-idiom + inline PTX blueprint for the compute layer,
//     picked because Dao-AILab flash-attention does not whitelist
//     sm_120 and FA3/FA4 require Datacenter-Blackwell (tcgen05).
//   - FlashInfer FMHA_V2 SM120 kernels — reference-vault for how a
//     paged-KV FA2 kernel is structured on sm_120 specifically.
//
// Design constants (target values for the eventual body):
//   BLOCK_KV        = 64   // K-tile per iteration
//   WARPS_PER_BLOCK = 4    // producer/consumer split
//   SMEM_BUDGET     = 99 KiB on sm_120
//   Warp-specialisation with mbarrier for cross-warp sync
//
// Kernel layout conventions (match vLLM):
//   query        [num_seqs, num_heads, head_size]         row-major fp32
//   out          [num_seqs, num_heads, head_size]         row-major fp32
//   key_cache    Paged pool. Layout depends on kv_cache_dtype:
//                fp32: [num_blocks, num_kv_heads, head_size / X, block_size, X]
//                where X = 16/sizeof(dtype) (vLLM convention for
//                coalesced access; may simplify to
//                [num_blocks, block_size, num_kv_heads, head_size] on
//                sm_120 if the empirical benchmark favours it — decide
//                during the body implementation).
//   value_cache  Same paged pool convention.
//   block_tables [num_seqs, max_num_blocks_per_seq]  int32, -1 sentinel
//   seq_lens     [num_seqs]                          int32
//
// GQA: hq -> hkv = (hq * num_kv_heads) / num_heads. Same rule as the
// existing kernels_cuda/attention_prefill_flash*.cu files.
//
// Causal masking: implied by seq_lens[i] being the *decoded* length —
// this kernel serves the decode pass (one query token per sequence).
// A separate paged_attention_v2 will handle chunked-KV / prefill.

#include <cuda_runtime.h>

// Placeholder launch bounds — the eventual body will retune these
// against sm_120 occupancy. Kept as a compile-time constant so the
// wrapper can query `_LAUNCH_BOUNDS` and pass matching blockDim.
#ifndef PAGED_ATTN_V1_LOCAL
#define PAGED_ATTN_V1_LOCAL 128
#endif

// kv_cache_dtype values — mirrors the wrapper enum.
#define PAGED_ATTN_KV_DTYPE_FP32  0
#define PAGED_ATTN_KV_DTYPE_FP16  1
#define PAGED_ATTN_KV_DTYPE_Q8_0  2

extern "C" __global__ __launch_bounds__(PAGED_ATTN_V1_LOCAL)
void paged_attention_v1(
          float* __restrict__ out,                 // [num_seqs, num_heads, head_size]
    const float* __restrict__ query,               // [num_seqs, num_heads, head_size]
    const void*  __restrict__ key_cache,           // paged pool, raw bytes; layout per kv_cache_dtype
    const void*  __restrict__ value_cache,         // paged pool, raw bytes
    const int*   __restrict__ block_tables,        // [num_seqs, max_num_blocks_per_seq]
    const int*   __restrict__ seq_lens,            // [num_seqs]
    const int                 num_seqs,
    const int                 num_heads,
    const int                 num_kv_heads,
    const int                 head_size,
    const int                 block_size,
    const int                 max_num_blocks_per_seq,
    const float               scale,               // 1/sqrt(head_size), pre-baked
    const float               softcap,             // 0.0f = disabled; > 0 = Gemma-4 final-logit-softcap
    const int                 kv_cache_dtype)      // PAGED_ATTN_KV_DTYPE_*
{
    // ---- Baseline correctness body (M-Cuda.Batch B2) --------------------
    // fp32 paged-KV decode attention: one query token per sequence, online
    // (streaming) softmax over the sequence's KV via block-table
    // indirection, GQA head mapping, optional Gemma-4 logit soft-cap.
    //
    // This is the CORRECT-but-not-fast body. The FA2-Ampere speed-of-light
    // variant (BLOCK_KV tiling, warp-specialisation, mbarrier producer/
    // consumer, inline PTX) is a deferred perf task — see the file header.
    // One workgroup owns one (head, sequence); threadIdx.x strides the
    // head_size dimension. Physical KV layout is the simple vLLM-alt form
    //   key/value_cache[num_blocks, block_size, num_kv_heads, head_size]
    // chosen here (see header — the coalesced [.., head_size/X, .., X] form
    // is left to the FA2 body). Both caches share this layout.
    if (kv_cache_dtype != PAGED_ATTN_KV_DTYPE_FP32) {
        __trap();   // fp16 / Q8_0 paged KV land with the FA2 body
    }

    const int hq  = static_cast<int>(blockIdx.x);   // query head
    const int seq = static_cast<int>(blockIdx.y);   // sequence
    if (hq >= num_heads || seq >= num_seqs) {
        return;
    }

    const int tid  = static_cast<int>(threadIdx.x);
    const int nthr = static_cast<int>(blockDim.x);  // == PAGED_ATTN_V1_LOCAL

    // GQA: many query heads share one KV head (same rule as the contiguous
    // attention kernels). num_kv_heads == num_heads collapses to identity.
    const int hkv     = (hq * num_kv_heads) / num_heads;
    const int seq_len = seq_lens[seq];

    const float* __restrict__ q_row =
        query + (static_cast<long>(seq) * num_heads + hq) * head_size;
    float* __restrict__ o_row =
        out + (static_cast<long>(seq) * num_heads + hq) * head_size;

    // Dynamic shared memory laid out as
    //   [ head_size (query) | head_size (accumulator) | nthr (reduction) ].
    // The launch must reserve (2*head_size + nthr) * sizeof(float) bytes;
    // the C++ wrapper computes this. head_size <= 256 for qwen35moe, so the
    // budget stays a few KiB — far under the sm_120 SMEM cap.
    extern __shared__ float smem[];
    float* __restrict__ sq  = smem;                     // [head_size]
    float* __restrict__ acc = smem + head_size;         // [head_size]
    float* __restrict__ red = smem + 2 * head_size;     // [nthr]

    for (int d = tid; d < head_size; d += nthr) {
        sq[d]  = q_row[d];
        acc[d] = 0.0f;
    }
    __syncthreads();

    if (seq_len <= 0) {   // defensive: scheduler should never admit len 0
        for (int d = tid; d < head_size; d += nthr) {
            o_row[d] = 0.0f;
        }
        return;
    }

    __shared__ float s_m;      // running row-max of the logits
    __shared__ float s_l;      // running softmax denominator
    __shared__ float s_score;  // broadcast of the current position's logit
    if (tid == 0) {
        s_m = -1.0e30f;        // effectively -inf; first logit always wins
        s_l = 0.0f;
    }
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

        // logit = scale * (q . k_p), reduced across the block.
        float partial = 0.0f;
        for (int d = tid; d < head_size; d += nthr) {
            partial += sq[d] * k_p[d];
        }
        red[tid] = partial;
        __syncthreads();
        for (int stride = nthr >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                red[tid] += red[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            float logit = red[0] * scale;
            if (softcap > 0.0f) {
                logit = softcap * tanhf(logit / softcap);
            }
            s_score = logit;
        }
        __syncthreads();

        // Streaming-softmax update: rescale the running accumulator by the
        // shift in the row-max, then fold in this position's contribution.
        const float score   = s_score;
        const float m_new   = fmaxf(s_m, score);
        const float rescale = __expf(s_m - m_new);   // 0 on the first step
        const float p_exp   = __expf(score - m_new);
        const float* __restrict__ v_p = vbase + kv_off;
        for (int d = tid; d < head_size; d += nthr) {
            acc[d] = acc[d] * rescale + p_exp * v_p[d];
        }
        __syncthreads();
        if (tid == 0) {
            s_l = s_l * rescale + p_exp;
            s_m = m_new;
        }
        __syncthreads();
    }

    const float inv_l = 1.0f / s_l;
    for (int d = tid; d < head_size; d += nthr) {
        o_row[d] = acc[d] * inv_l;
    }
}