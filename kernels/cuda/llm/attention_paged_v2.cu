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
    const int                 partition_size,           // = PAGED_ATTN_V2_PARTITION_SIZE, passed for consistency
    const float               scale,
    const float               softcap,                  // 0.0f = disabled; > 0 = Gemma-4
    const int                 kv_cache_dtype)
{
    // Compile-time sanity: keep the signature stable. Any change here
    // must be mirrored in src/core/gpu/cuda/PagedAttentionV2.hpp.
    (void)tmp_out;
    (void)exp_sums;
    (void)max_logits;
    (void)query;
    (void)key_cache;
    (void)value_cache;
    (void)block_tables;
    (void)seq_lens;
    (void)num_seqs;
    (void)num_heads;
    (void)num_kv_heads;
    (void)head_size;
    (void)block_size;
    (void)max_num_blocks_per_seq;
    (void)max_num_partitions;
    (void)partition_size;
    (void)scale;
    (void)softcap;
    (void)kv_cache_dtype;

    // TODO(M-Cuda.Batch B3 body): partition-parallel FA2-Ampere with
    // paged-KV indirection. Same compute pattern as v1's eventual
    // body, but only walks a [partition_start, partition_end) K-range
    // instead of the full KV. Writes partial (o, m, l) instead of the
    // final output. Blueprint refs in the file header.
    __trap();
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
    const int                 max_num_partitions,
    const int                 partition_size)
{
    // Compile-time sanity — mirror in PagedAttentionV2.hpp.
    (void)out;
    (void)exp_sums;
    (void)max_logits;
    (void)tmp_out;
    (void)seq_lens;
    (void)num_seqs;
    (void)num_heads;
    (void)head_size;
    (void)max_num_partitions;
    (void)partition_size;

    // TODO(M-Cuda.Batch B3 body): online-softmax merge across
    // partitions. Same math as our existing L0 kernel
    // attention_flash_merge.cl adapted to paged-partition layout:
    //   m_global = max_i(max_logits[i])
    //   l_global = sum_i(exp_sums[i] * exp(max_logits[i] - m_global))
    //   o_global = sum_i(tmp_out[i] * exp(max_logits[i] - m_global))
    //              / l_global
    // Only partitions with i < ceil(seq_len / partition_size) contribute.
    __trap();
}