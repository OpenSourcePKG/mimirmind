// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// paged_attention_v2 + paged_attention_v2_reduce — SKELETON
// (M-Cuda.Batch Sub-Step B3)
//
// vLLM's paged_attention_v2 pattern: **two-kernel chunked-KV** for
// large context decode. Where v1 processes each (sequence, head) with
// a single workgroup that streams the full KV in one pass, v2 splits
// the KV into `PARTITION_SIZE`-token chunks so multiple workgroups can
// work on the same (sequence, head) in parallel. A second reduce-
// kernel then combines the per-partition partials into the final
// output using the online-softmax rescale trick.
//
// **When to use v2 vs v1** (Phase C dispatch decision):
//   - seq_len <= PARTITION_SIZE  → v1 (single-pass, no reduce overhead)
//   - seq_len >  PARTITION_SIZE  → v2 (partition-parallel, worth reduce)
//
// This is a compile-only launch-surface skeleton. Bodies are NOT
// implemented — both kernels `__trap()`. The FA2-Ampere+PTX compute
// layer for BOTH v1 and v2 lands together after Blackwell arrives and
// FlashInfer FMHA_V2 SM120 has been read as reference-vault.
//
// The **signatures ARE load-bearing** for Bragi-v1: Phase C scheduler
// + Phase D InferenceEngine::generateServing will bind against the
// argument order committed here. The v2 signature adds three
// workspace pointers (exp_sums, max_logits, tmp_out) + a
// max_num_partitions parameter to v1's signature; everything else
// is slot-for-slot identical.
//
// Design references (mirrored from v1 file, no runtime deps):
//   - Kwon et al 2023 "PagedAttention" — original paper
//   - vLLM V1 csrc/attention/paged_attention_v2_kernel.cuh —
//     signature shape + two-kernel split
//   - gau-nernst 5090 FA2 blueprint — Ampere-idiom + inline PTX
//   - FlashInfer FMHA_V2 SM120 — sm_120-specific paged-KV FA2 layout
//
// Layout conventions (match v1 file for the shared args):
//   query        [num_seqs, num_heads, head_size]         row-major fp32
//   key_cache    Paged pool, layout per kv_cache_dtype (see v1 header)
//   value_cache  Paged pool, same convention
//   block_tables [num_seqs, max_num_blocks_per_seq]  int32, -1 sentinel
//   seq_lens     [num_seqs]                          int32
//
// New in v2 (per-partition workspace):
//   tmp_out      [num_seqs, num_heads, max_num_partitions, head_size]
//                fp32, per-partition partial outputs
//   exp_sums     [num_seqs, num_heads, max_num_partitions]  fp32
//                per-partition softmax denominator (sum of exp scores)
//   max_logits   [num_seqs, num_heads, max_num_partitions]  fp32
//                per-partition max logit (for online-softmax rescale
//                in the reduce kernel — same trick as our existing
//                attention_flash_partial.cu + attention_flash_merge.cu
//                pair on the L0 side, adapted to paged-KV)
//
// GQA + causal masking rules mirror v1 exactly.

#include <cuda_runtime.h>

// Placeholder launch bounds — the eventual body will retune these
// against sm_120 occupancy. Kept as compile-time constants so the
// wrappers can query them and pass matching blockDim.
#ifndef PAGED_ATTN_V2_LOCAL
#define PAGED_ATTN_V2_LOCAL 128
#endif

#ifndef PAGED_ATTN_V2_REDUCE_LOCAL
#define PAGED_ATTN_V2_REDUCE_LOCAL 128
#endif

// PARTITION_SIZE — the K-chunk granularity across which v2 splits the
// KV traversal. 512 is the vLLM default. Must be a multiple of the
// block-size (16 in our Allocator) so partitions align with paged
// blocks (32 blocks per partition at block-size 16).
#ifndef PAGED_ATTN_V2_PARTITION_SIZE
#define PAGED_ATTN_V2_PARTITION_SIZE 512
#endif

// kv_cache_dtype values — mirrors the wrapper enum + v1 file.
#define PAGED_ATTN_KV_DTYPE_FP32  0
#define PAGED_ATTN_KV_DTYPE_FP16  1
#define PAGED_ATTN_KV_DTYPE_Q8_0  2

// ---------------------------------------------------------------------
// Kernel 1: per-partition attention. One workgroup per
// (head, sequence, partition_idx). Writes partial (tmp_out, exp_sums,
// max_logits) for its partition. Does NOT produce the final output.
// ---------------------------------------------------------------------

extern "C" __global__ __launch_bounds__(PAGED_ATTN_V2_LOCAL)
void paged_attention_v2(
          float* __restrict__ tmp_out,                  // [num_seqs, num_heads, max_num_partitions, head_size]
          float* __restrict__ exp_sums,                 // [num_seqs, num_heads, max_num_partitions]
          float* __restrict__ max_logits,               // [num_seqs, num_heads, max_num_partitions]
    const float* __restrict__ query,                    // [num_seqs, num_heads, head_size]
    const void*  __restrict__ key_cache,                // paged pool, raw bytes
    const void*  __restrict__ value_cache,              // paged pool, raw bytes
    const int*   __restrict__ block_tables,             // [num_seqs, max_num_blocks_per_seq]
    const int*   __restrict__ seq_lens,                 // [num_seqs]
    const int                 num_seqs,
    const int                 num_heads,
    const int                 num_kv_heads,
    const int                 head_size,
    const int                 block_size,
    const int                 max_num_blocks_per_seq,
    const int                 max_num_partitions,       // for tmp_out stride
    const float               scale)
{
    // ---- Partition-parallel paged-KV decode (M-Cuda.Batch B3 body) ------
    // Same streaming-softmax compute as paged_attention_v1, but each
    // workgroup walks only its [p_start, p_end) K-range (one partition) and
    // emits the UNNORMALISED partial (acc, m, l) for the reduce kernel to
    // merge — the FlashDecoding / vLLM-v2 split-K pattern. fp32 KV only + no
    // soft-cap (qwen35moe full-attn); the CudaKernel 16-arg cap forced
    // partition_size to the compile-time constant and dropped the softcap /
    // dtype args — the C++ wrapper routes softcap>0 to V1.
    constexpr int partition_size = PAGED_ATTN_V2_PARTITION_SIZE;

    const int hq   = static_cast<int>(blockIdx.x);   // query head
    const int seq  = static_cast<int>(blockIdx.y);   // sequence
    const int part = static_cast<int>(blockIdx.z);   // partition index
    if (hq >= num_heads || seq >= num_seqs || part >= max_num_partitions) {
        return;
    }

    const int tid  = static_cast<int>(threadIdx.x);
    const int nthr = static_cast<int>(blockDim.x);

    const int hkv     = (hq * num_kv_heads) / num_heads;
    const int seq_len = seq_lens[seq];
    const int p_start = part * partition_size;
    int       p_end   = p_start + partition_size;
    if (p_end > seq_len) {
        p_end = seq_len;
    }

    // Partial-output slot for this (seq, head, partition).
    const long part_idx =
        (static_cast<long>(seq) * num_heads + hq) * max_num_partitions + part;
    float* __restrict__ o_row = tmp_out + part_idx * head_size;

    const float* __restrict__ q_row =
        query + (static_cast<long>(seq) * num_heads + hq) * head_size;

    // Warp-cooperative decode attention: block = nWarps warps of 32 lanes.
    // Each warp streams a strided subset of the partition's KV positions and
    // keeps its OWN online-softmax (m, l, acc) with acc distributed across the
    // 32 lanes (lane owns dims { lane + i*32 }). The per-position q.k reduction
    // is a sync-free warp shuffle; the warps are merged once at the end. This
    // removes V1's per-position __syncthreads tree-reduction (the bottleneck).
    // Requires head_size % 32 == 0 and <= 256 (128 for qwen35moe); fp32 KV.
    constexpr int WARP = 32;
    if ((head_size % WARP) != 0 || head_size > 256) {
        __trap();
    }
    const int lane     = tid % WARP;
    const int warpId   = tid / WARP;
    const int nWarps   = nthr / WARP;
    const int dPerLane = head_size / WARP;   // <= 8 for head_size <= 256

    // Empty partition (past this sequence's length): neutral partial so the
    // reduce kernel skips it cleanly.
    if (p_start >= p_end) {
        if (tid == 0) {
            max_logits[part_idx] = -1.0e30f;
            exp_sums[part_idx]   = 0.0f;
        }
        for (int d = tid; d < head_size; d += nthr) {
            o_row[d] = 0.0f;
        }
        return;
    }

    // Per-lane registers: this lane owns dims { lane + i*WARP : i < dPerLane }.
    float qreg[8];
    float acc[8];
    #pragma unroll
    for (int i = 0; i < dPerLane; ++i) {
        qreg[i] = q_row[lane + i * WARP];
        acc[i]  = 0.0f;
    }
    float m = -1.0e30f;
    float l = 0.0f;

    const float* __restrict__ kbase = static_cast<const float*>(key_cache);
    const float* __restrict__ vbase = static_cast<const float*>(value_cache);
    const int*   __restrict__ bt =
        block_tables + static_cast<long>(seq) * max_num_blocks_per_seq;

    for (int p = p_start + warpId; p < p_end; p += nWarps) {
        const int  blk    = bt[p / block_size];
        const int  slot   = p % block_size;
        const long kv_off =
            ((static_cast<long>(blk) * block_size + slot) * num_kv_heads + hkv)
            * head_size;
        const float* __restrict__ k_p = kbase + kv_off;

        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < dPerLane; ++i) {
            dot += qreg[i] * k_p[lane + i * WARP];
        }
        #pragma unroll
        for (int off = WARP / 2; off > 0; off >>= 1) {
            dot += __shfl_xor_sync(0xffffffffu, dot, off);
        }
        const float logit   = dot * scale;   // identical across the warp's lanes
        const float m_new    = fmaxf(m, logit);
        const float rescale  = __expf(m - m_new);
        const float p_exp    = __expf(logit - m_new);
        const float* __restrict__ v_p = vbase + kv_off;
        #pragma unroll
        for (int i = 0; i < dPerLane; ++i) {
            acc[i] = acc[i] * rescale + p_exp * v_p[lane + i * WARP];
        }
        l = l * rescale + p_exp;
        m = m_new;
    }

    // Cross-warp merge in shared memory (once). Layout:
    //   [ nWarps * head_size (acc) | nWarps (m) | nWarps (l) ].
    extern __shared__ float smem[];
    float* __restrict__ sAcc = smem;                       // [nWarps][head_size]
    float* __restrict__ sM   = smem + nWarps * head_size;  // [nWarps]
    float* __restrict__ sL   = sM + nWarps;                // [nWarps]
    #pragma unroll
    for (int i = 0; i < dPerLane; ++i) {
        sAcc[warpId * head_size + lane + i * WARP] = acc[i];
    }
    if (lane == 0) {
        sM[warpId] = m;
        sL[warpId] = l;
    }
    __syncthreads();

    // Global max + denominator (nWarps <= 8, so every thread recomputes them
    // cheaply), then each thread merges the head dims it owns. Emit the
    // UNNORMALISED acc + (m, l) for the reduce kernel.
    float mg = -1.0e30f;
    for (int w = 0; w < nWarps; ++w) {
        mg = fmaxf(mg, sM[w]);
    }
    float lg = 0.0f;
    for (int w = 0; w < nWarps; ++w) {
        lg += sL[w] * __expf(sM[w] - mg);
    }
    for (int d = tid; d < head_size; d += nthr) {
        float a = 0.0f;
        for (int w = 0; w < nWarps; ++w) {
            a += sAcc[w * head_size + d] * __expf(sM[w] - mg);
        }
        o_row[d] = a;
    }
    if (tid == 0) {
        max_logits[part_idx] = mg;
        exp_sums[part_idx]   = lg;
    }
}

// ---------------------------------------------------------------------
// Kernel 2: reduce partial outputs into the final `out`. One workgroup
// per (head, sequence). Reads the per-partition partials from
// tmp_out/exp_sums/max_logits, applies the online-softmax rescale
// (max-shift + exp-sum), and writes the final head_size vector.
// ---------------------------------------------------------------------

extern "C" __global__ __launch_bounds__(PAGED_ATTN_V2_REDUCE_LOCAL)
void paged_attention_v2_reduce(
          float* __restrict__ out,                      // [num_seqs, num_heads, head_size]
    const float* __restrict__ exp_sums,                 // [num_seqs, num_heads, max_num_partitions]
    const float* __restrict__ max_logits,               // [num_seqs, num_heads, max_num_partitions]
    const float* __restrict__ tmp_out,                  // [num_seqs, num_heads, max_num_partitions, head_size]
    const int*   __restrict__ seq_lens,                 // [num_seqs]
    const int                 num_seqs,
    const int                 num_heads,
    const int                 head_size,
    const int                 max_num_partitions)
{
    // ---- Online-softmax merge across partitions (M-Cuda.Batch B3) -------
    constexpr int partition_size = PAGED_ATTN_V2_PARTITION_SIZE;
    const int hq  = static_cast<int>(blockIdx.x);   // query head
    const int seq = static_cast<int>(blockIdx.y);   // sequence
    if (hq >= num_heads || seq >= num_seqs) {
        return;
    }
    const int tid  = static_cast<int>(threadIdx.x);
    const int nthr = static_cast<int>(blockDim.x);

    const int seq_len   = seq_lens[seq];
    int       num_parts = (seq_len + partition_size - 1) / partition_size;
    if (num_parts > max_num_partitions) {
        num_parts = max_num_partitions;
    }

    const long base = static_cast<long>(seq) * num_heads + hq;
    float* __restrict__ o_row = out + base * head_size;

    if (num_parts <= 0) {
        for (int d = tid; d < head_size; d += nthr) {
            o_row[d] = 0.0f;
        }
        return;
    }

    const float* __restrict__ ml = max_logits + base * max_num_partitions;
    const float* __restrict__ es = exp_sums   + base * max_num_partitions;
    const float* __restrict__ to =
        tmp_out + base * max_num_partitions * head_size;   // [max_num_partitions, head_size]

    extern __shared__ float rsmem[];   // [nthr]

    // Phase 1: global max of the per-partition max-logits.
    float lm = -1.0e30f;
    for (int i = tid; i < num_parts; i += nthr) {
        lm = fmaxf(lm, ml[i]);
    }
    rsmem[tid] = lm;
    __syncthreads();
    for (int stride = nthr >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            rsmem[tid] = fmaxf(rsmem[tid], rsmem[tid + stride]);
        }
        __syncthreads();
    }
    __shared__ float s_mg;
    if (tid == 0) {
        s_mg = rsmem[0];
    }
    __syncthreads();
    const float mg = s_mg;

    // Phase 2: global softmax denominator l = sum_i es[i] * exp(ml[i] - mg).
    float ll = 0.0f;
    for (int i = tid; i < num_parts; i += nthr) {
        ll += es[i] * __expf(ml[i] - mg);
    }
    rsmem[tid] = ll;
    __syncthreads();
    for (int stride = nthr >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            rsmem[tid] += rsmem[tid + stride];
        }
        __syncthreads();
    }
    __shared__ float s_lg;
    if (tid == 0) {
        s_lg = rsmem[0];
    }
    __syncthreads();
    const float inv_l = (s_lg > 0.0f) ? (1.0f / s_lg) : 0.0f;

    // Phase 3: o[d] = (sum_i to[i][d] * exp(ml[i] - mg)) / l. Each thread owns
    // a subset of the head_size dims and loops over the (few) partitions.
    for (int d = tid; d < head_size; d += nthr) {
        float o = 0.0f;
        for (int i = 0; i < num_parts; ++i) {
            o += to[static_cast<long>(i) * head_size + d] * __expf(ml[i] - mg);
        }
        o_row[d] = o * inv_l;
    }
}