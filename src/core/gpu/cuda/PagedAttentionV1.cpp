// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/gpu/cuda/PagedAttentionV1.hpp"

namespace mimirmind::core::cuda {

PagedAttentionV1::PagedAttentionV1(CudaModule& module)
    : _kernel(module.getFunction(kKernelSymbol))
{}

void PagedAttentionV1::launch(CudaStream&  stream,
                              const void*  out,
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
                              float        scale,
                              float        softcap,
                              PagedKvDtype kvDtype,
                              std::size_t  sharedMemBytes)
{
    // Argument order MUST match kernels_cuda/attention_paged_v1.cu
    // signature slot-for-slot — the kernel reads a raw void** so any
    // divergence here is a silent memory corruption.
    _kernel.clearArgs();
    _kernel.setPtr(0,  out);
    _kernel.setPtr(1,  query);
    _kernel.setPtr(2,  keyCache);
    _kernel.setPtr(3,  valueCache);
    _kernel.setPtr(4,  blockTables);
    _kernel.setPtr(5,  seqLens);
    _kernel.setValue(6,  numSeqs);
    _kernel.setValue(7,  numHeads);
    _kernel.setValue(8,  numKvHeads);
    _kernel.setValue(9,  headSize);
    _kernel.setValue(10, blockSize);
    _kernel.setValue(11, maxNumBlocksPerSeq);
    _kernel.setValue(12, scale);
    _kernel.setValue(13, softcap);
    _kernel.setValue(14, static_cast<int>(kvDtype));

    // Grid: one workgroup per (head, sequence). Matches the pattern
    // used by the contiguous attention_prefill_flash.cu — Phase C can
    // A/B this against a (sequence, head) grid once the body ships.
    const std::uint32_t gridX = static_cast<std::uint32_t>(numHeads);
    const std::uint32_t gridY = static_cast<std::uint32_t>(numSeqs);

    _kernel.launch(stream,
                   gridX, gridY, /*gridZ=*/1,
                   kBlockThreads, /*blockY=*/1, /*blockZ=*/1,
                   sharedMemBytes);
}

} // namespace mimirmind::core::cuda