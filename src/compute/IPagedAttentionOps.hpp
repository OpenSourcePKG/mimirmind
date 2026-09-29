// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "runtime/KvCache.hpp"   // runtime::KvDtype

#include <cstddef>
#include <cstdint>

namespace mimirmind::compute {

/// Paged (block-table) attention device ops for serving-class continuous
/// batching. Segregated out of the backend-neutral `ComputeOps` (8.30.6
/// ISP-split) because only the CUDA/Bragi backend implements them — the L0
/// serving substrate uses the non-paged slab path instead. A backend that
/// supports them returns a non-null `IPagedAttentionOps*` from
/// `ComputeOps::pagedAttentionOps()`; others return nullptr, so the neutral
/// base no longer carries throw-default stubs for this concern.
///
/// The cuDNN-SDPA query/graceful-fallback pair (pagedPrefillCudnnAvailable /
/// pagedPrefillAttentionCudnnAsync) stays on `ComputeOps` — both return safe
/// false defaults (graceful "not available"), consulted to pick the path
/// before this interface is reached.
class IPagedAttentionOps {
public:
    virtual ~IPagedAttentionOps() = default;

    /// Paged GQA decode attention (kernel `paged_attention_v1`, fp32 baseline).
    /// keyCache/valueCache are the per-layer pool bases; blockTables [numSeqs,
    /// maxNumBlocksPerSeq] int32 (-1 sentinel); seqLens [numSeqs] int32.
    /// query/out [numSeqs, numHeads, headSize].
    virtual void pagedAttentionDecodeV1Async(
        float* out, const float* query, const float* keyCache,
        const float* valueCache, const std::int32_t* blockTables,
        const std::int32_t* seqLens, std::size_t numSeqs,
        std::size_t numHeads, std::size_t numKvHeads, std::size_t headSize,
        std::size_t blockSize, std::size_t maxNumBlocksPerSeq, float scale,
        float softcap) = 0;

    /// Split-K (partition-parallel) paged decode attention (`paged_attention_v2`
    /// + reduce) — same result as V1 but parallelises the KV traversal across
    /// ceil(maxSeqLen/512) partitions so long-context decode fills the GPU.
    virtual void pagedAttentionDecodeV2Async(
        float* out, const float* query, const float* keyCache,
        const float* valueCache, const std::int32_t* blockTables,
        const std::int32_t* seqLens, std::size_t numSeqs,
        std::size_t numHeads, std::size_t numKvHeads, std::size_t headSize,
        std::size_t blockSize, std::size_t maxNumBlocksPerSeq,
        std::size_t maxSeqLen, float scale, float softcap,
        runtime::KvDtype kvDtype = runtime::KvDtype::F32) = 0;

    /// 5.21 Increment II — paged, batched, CAUSAL, T>1 prefill attention. Slot
    /// seq carries seqT[seq] ragged query tokens at token offset queryOff[seq];
    /// query pq attends KV [0, startPos[seq]+pq] via the block table. maxT =
    /// max(seqT) sizes the grid's query dim.
    virtual void pagedAttentionPrefillCausalAsync(
        float* out, const float* query, const float* keyCache,
        const float* valueCache, const std::int32_t* blockTables,
        const std::int32_t* seqT, const std::int32_t* queryOff,
        const std::int32_t* startPos, std::size_t numSeqs,
        std::size_t numHeads, std::size_t numKvHeads, std::size_t headSize,
        std::size_t blockSize, std::size_t maxNumBlocksPerSeq,
        std::size_t maxT, float scale, float softcap,
        runtime::KvDtype kvDtype = runtime::KvDtype::F32) = 0;
};

} // namespace mimirmind::compute
