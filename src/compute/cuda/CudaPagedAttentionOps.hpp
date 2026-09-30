// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "compute/ComputeBuffer.hpp"
#include "compute/IPagedAttentionOps.hpp"
#include "core/gpu/cuda/CudaKernel.hpp"
#include "core/gpu/cuda/CudaModule.hpp"

#include <cstddef>
#include <cstdint>

namespace mimirmind::core::cuda {
class CudaComputeContext;
}

namespace mimirmind::compute::cuda {

/**
 * CUDA implementation of the segregated IPagedAttentionOps interface
 * (8.30.11.4): the 3 paged (block-table) attention device ops — fp32 decode V1,
 * split-K decode V2 (+reduce), and batched causal T>1 prefill — carved out of
 * the GpuOps god-class into their own collaborator. Owns the three kernel
 * modules the ops need (attention_paged_v1 / _prefill_causal / _v2, 8 kernels
 * total) plus the split-K V2 per-partition workspace (grown on demand via a
 * private allocate() that mirrors GpuOps::allocate). GpuOps holds one instance
 * and hands it back from pagedAttentionOps(); the cuDNN-SDPA query/fallback
 * pair (pagedPrefillCudnnAvailable / pagedPrefillAttentionCudnnAsync) stays on
 * GpuOps (consulted to pick the path before this interface is reached).
 *
 * CUDA-only. Bodies are the former GpuOps::pagedAttention*Async methods
 * verbatim.
 */
class CudaPagedAttentionOps final : public ::mimirmind::compute::IPagedAttentionOps {
public:
    explicit CudaPagedAttentionOps(core::cuda::CudaComputeContext& ctx);

    void pagedAttentionDecodeV1Async(
        float* out, const float* query, const float* keyCache,
        const float* valueCache, const std::int32_t* blockTables,
        const std::int32_t* seqLens, std::size_t numSeqs,
        std::size_t numHeads, std::size_t numKvHeads, std::size_t headSize,
        std::size_t blockSize, std::size_t maxNumBlocksPerSeq, float scale,
        float softcap) override;

    void pagedAttentionDecodeV2Async(
        float* out, const float* query, const float* keyCache,
        const float* valueCache, const std::int32_t* blockTables,
        const std::int32_t* seqLens, std::size_t numSeqs,
        std::size_t numHeads, std::size_t numKvHeads, std::size_t headSize,
        std::size_t blockSize, std::size_t maxNumBlocksPerSeq,
        std::size_t maxSeqLen, float scale, float softcap,
        runtime::KvDtype kvDtype = runtime::KvDtype::F32) override;

    void pagedAttentionPrefillCausalAsync(
        float* out, const float* query, const float* keyCache,
        const float* valueCache, const std::int32_t* blockTables,
        const std::int32_t* seqT, const std::int32_t* queryOff,
        const std::int32_t* startPos, std::size_t numSeqs,
        std::size_t numHeads, std::size_t numKvHeads, std::size_t headSize,
        std::size_t blockSize, std::size_t maxNumBlocksPerSeq,
        std::size_t maxT, float scale, float softcap,
        runtime::KvDtype kvDtype = runtime::KvDtype::F32) override;

private:
    // Split-K V2 per-partition workspace grower — mirrors GpuOps::allocate
    // (Managed on unified GB10, Device on discrete; captureless allocator
    // deleter). ComputeBuffer RAII-frees on cap-grow / destruction.
    compute::ComputeBuffer allocate(std::size_t bytes);

    core::cuda::CudaComputeContext& _ctx;

    core::cuda::CudaModule _pagedAttentionV1Module;
    core::cuda::CudaKernel _pagedAttentionV1Kernel;
    core::cuda::CudaModule _pagedAttentionPrefillCausalModule;
    core::cuda::CudaKernel _pagedAttentionPrefillCausalKernel;
    core::cuda::CudaKernel _pagedAttentionPrefillCausalFp16Kernel;  // fp16 KV (5.16)
    core::cuda::CudaKernel _pagedAttentionPrefillCausalFp8Kernel;   // fp8 KV (5.16)
    core::cuda::CudaModule _pagedAttentionV2Module;
    core::cuda::CudaKernel _pagedAttentionV2Kernel;
    core::cuda::CudaKernel _pagedAttentionV2Fp16Kernel;   // fp16 KV (5.14 I1)
    core::cuda::CudaKernel _pagedAttentionV2Fp8Kernel;    // fp8 KV (5.16)
    core::cuda::CudaKernel _pagedAttentionV2ReduceKernel;

    // Split-K V2 per-partition workspace (grown on demand; RAII-freed).
    compute::ComputeBuffer _pagedV2TmpOut;      // [slots, headSize] fp32
    compute::ComputeBuffer _pagedV2ExpSums;     // [slots] fp32
    compute::ComputeBuffer _pagedV2MaxLogits;   // [slots] fp32
    std::size_t            _pagedV2SlotCap{0};   // slots = nSeq*nHeads*maxNumPartitions
    std::size_t            _pagedV2HeadDimCap{0};
};

} // namespace mimirmind::compute::cuda
