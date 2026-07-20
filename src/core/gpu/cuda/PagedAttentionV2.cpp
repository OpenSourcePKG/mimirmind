// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/gpu/cuda/PagedAttentionV2.hpp"

namespace mimirmind::core::cuda {

PagedAttentionV2::PagedAttentionV2(CudaModule& module)
    : _partial(module.getFunction(kPartialKernelSymbol))
    , _reduce (module.getFunction(kReduceKernelSymbol))
{}

void PagedAttentionV2::launch(CudaStream&  stream,
                              const void*  out,
                              const void*  tmpOut,
                              const void*  expSums,
                              const void*  maxLogits,
                              const void*  query,
                              const void*  keyCache,
                              const void*  valueCache,
                              const void*  blockTables,
                              const void*  seqLens,
                              std::int32_t numSeqs,
                              std::int32_t numHeads,
                              std::int32_t numKvHeads,
                              std::int32_t headSize,
                              std::int32_t blockSize,
                              std::int32_t maxNumBlocksPerSeq,
                              std::int32_t maxNumPartitions,
                              float        scale,
                              float        softcap,
                              PagedKvDtype kvDtype,
                              std::size_t  partialSharedMemBytes,
                              std::size_t  reduceSharedMemBytes)
{
    // ---- Kernel 1: per-partition partial attention ---------------
    //
    // Argument order MUST match kernels_cuda/attention_paged_v2.cu
    // paged_attention_v2 slot-for-slot. 19 args total.
    _partial.clearArgs();
    _partial.setPtr(0,  tmpOut);
    _partial.setPtr(1,  expSums);
    _partial.setPtr(2,  maxLogits);
    _partial.setPtr(3,  query);
    _partial.setPtr(4,  keyCache);
    _partial.setPtr(5,  valueCache);
    _partial.setPtr(6,  blockTables);
    _partial.setPtr(7,  seqLens);
    _partial.setValue(8,  numSeqs);
    _partial.setValue(9,  numHeads);
    _partial.setValue(10, numKvHeads);
    _partial.setValue(11, headSize);
    _partial.setValue(12, blockSize);
    _partial.setValue(13, maxNumBlocksPerSeq);
    _partial.setValue(14, maxNumPartitions);
    _partial.setValue(15, kPartitionSize);
    _partial.setValue(16, scale);
    _partial.setValue(17, softcap);
    _partial.setValue(18, static_cast<int>(kvDtype));

    // Grid: one workgroup per (head, sequence, partition). Matches
    // vLLM V2 convention and lets a single-block reduction over
    // K-tiles inside each partition stay warp-local.
    const std::uint32_t partialGridX = static_cast<std::uint32_t>(numHeads);
    const std::uint32_t partialGridY = static_cast<std::uint32_t>(numSeqs);
    const std::uint32_t partialGridZ = static_cast<std::uint32_t>(maxNumPartitions);

    _partial.launch(stream,
                    partialGridX, partialGridY, partialGridZ,
                    kPartialBlockThreads, /*blockY=*/1, /*blockZ=*/1,
                    partialSharedMemBytes);

    // ---- Kernel 2: reduce partials into final out ----------------
    //
    // Argument order MUST match paged_attention_v2_reduce slot-for-
    // slot. 10 args. Stream-ordered after the partial kernel — CUDA
    // guarantees serial execution within the same stream, no explicit
    // sync required.
    _reduce.clearArgs();
    _reduce.setPtr(0, out);
    _reduce.setPtr(1, expSums);
    _reduce.setPtr(2, maxLogits);
    _reduce.setPtr(3, tmpOut);
    _reduce.setPtr(4, seqLens);
    _reduce.setValue(5, numSeqs);
    _reduce.setValue(6, numHeads);
    _reduce.setValue(7, headSize);
    _reduce.setValue(8, maxNumPartitions);
    _reduce.setValue(9, kPartitionSize);

    // Grid: one workgroup per (head, sequence). Reduce is embarrass-
    // ingly parallel across (head, sequence) — no cross-block sync.
    const std::uint32_t reduceGridX = static_cast<std::uint32_t>(numHeads);
    const std::uint32_t reduceGridY = static_cast<std::uint32_t>(numSeqs);

    _reduce.launch(stream,
                   reduceGridX, reduceGridY, /*gridZ=*/1,
                   kReduceBlockThreads, /*blockY=*/1, /*blockZ=*/1,
                   reduceSharedMemBytes);
}

} // namespace mimirmind::core::cuda