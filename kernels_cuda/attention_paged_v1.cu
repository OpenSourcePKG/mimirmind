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
    // Compile-time sanity: keep the signature stable. Any change here
    // must be mirrored in src/core/gpu/cuda/PagedAttentionV1.hpp.
    (void)out;
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
    (void)scale;
    (void)softcap;
    (void)kv_cache_dtype;

    // TODO(M-Cuda.Batch B2 body): FA2-Ampere compute with paged-KV
    // indirection. See file header for blueprint references. Until the
    // body lands, invocation is a hard runtime error so the scheduler
    // never operates against garbage output.
    __trap();
}