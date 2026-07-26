// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/gpu/cuda/CudaKernel.hpp"
#include "core/gpu/cuda/CudaModule.hpp"
#include "runtime/nvfp4/NvFp4Materializer.hpp"

#include <cstddef>
#include <cstdint>

namespace mimirmind::compute {
class ComputeOps;
}
namespace mimirmind::core::cuda {
class CudaComputeContext;
}

namespace mimirmind::compute::cuda {

/**
 * CUDA implementation of the NVFP4 materializer's device seam: allocation
 * and D2D copy go through the compute backend (`ComputeOps`), the two dequant
 * kernels launch from the `dequant_nvfp4` / `dequant_fp8` PTX modules, and
 * scale reads are a small D2H `cudaMemcpy`. All kernel launches and the D2D
 * copies are enqueued on the context's stream — the caller
 * (`InferenceEngine::loadModelNvfp4`) must sync the stream once after the
 * materialization run before the BF16 weights are used.
 */
class CudaMaterializerOps final : public runtime::nvfp4::MaterializerDeviceOps {
public:
    CudaMaterializerOps(core::cuda::CudaComputeContext& ctx, ComputeOps& ops);

    [[nodiscard]] ComputeBuffer allocate(std::size_t bytes) override;

    void dequantNvfp4(const void* packed, const void* blockScale, float global,
                      std::uint64_t rows, std::uint64_t in, void* dstBf16) override;

    void dequantFp8(const void* weight, float scale,
                    std::uint64_t n, void* dstBf16) override;

    void copyBytes(void* dst, const void* src, std::size_t bytes) override;

    void widenToF32(void* dstF32, const void* src,
                    core::safetensors::SafetensorsDtype srcDtype,
                    std::uint64_t n) override;

    [[nodiscard]] float readF32(const void* devPtr) override;

    void negExpInPlaceF32(void* f32, std::uint64_t n) override;

    void addOneInPlaceF32(void* f32, std::uint64_t n) override;

    /**
     * BF16 -> Q8_0 weight re-quantisation (not part of the materializer
     * interface — the NVFP4 loader calls it directly on the concrete ops to
     * shrink the materialised attention projections). `srcBf16` holds `rows`
     * rows of `K` BF16 elements (K % 32 == 0); `dstQ8_0` receives the Q8_0
     * blocks in the ggml layout the matmul_q8_0 kernels expect. Enqueued on
     * the context stream; caller syncs.
     */
    void quantizeBf16ToQ8_0(void* dstQ8_0, const void* srcBf16,
                            std::uint64_t rows, std::uint64_t K);

    /**
     * BF16 -> Q4_K weight re-quantisation for the MoE experts. `srcBf16` holds
     * `totalElems` BF16 values (totalElems % 256 == 0); `dstQ4K` receives the
     * Q4_K super-blocks (144 B / 256 elems) in the ggml layout matmul_q4k
     * expects. One thread block per 256-element super-block. Enqueued on the
     * context stream; caller syncs.
     */
    void quantizeBf16ToQ4K(void* dstQ4K, const void* srcBf16,
                           std::uint64_t totalElems);

    /** BF16 -> Q6_K (210 B / 256 elems), for the MoE down-expert banks. */
    void quantizeBf16ToQ6K(void* dstQ6K, const void* srcBf16,
                           std::uint64_t totalElems);

    /**
     * BF16 -> blocked-FP8 E4M3 (34 B / 32 elems) for the attention projections.
     * `rows` rows of `K` BF16 elements (K % 32 == 0); dst gets the blocked-FP8
     * in the layout matmul_fp8_vec/_gemm expect. Log-format (E4M3) preserves the
     * small weights a linear Q8_0 would crush. Enqueued on the stream.
     */
    void quantizeBf16ToFp8(void* dstFp8, const void* srcBf16,
                           std::uint64_t rows, std::uint64_t K);

    /**
     * NVFP4 -> blocked-NVFP4 repackage (LOSSLESS — keeps the E2M1 nibbles).
     * `packed` [rows, in/2] E2M1, `blockScale` [rows, in/16] E4M3, `global`
     * scalar; `dst` receives 20-byte / 32-element super-blocks matmul_nvfp4blk
     * expects. `in` % 32 == 0. Enqueued on the stream.
     */
    void repackageNvfp4ToBlk(void* dst, const void* packed,
                             const void* blockScale, float global,
                             std::uint64_t rows, std::uint64_t in);

private:
    core::cuda::CudaComputeContext& _ctx;
    ComputeOps&                     _ops;
    core::cuda::CudaModule          _nvfp4Module;
    core::cuda::CudaModule          _fp8Module;
    core::cuda::CudaModule          _castModule;
    core::cuda::CudaModule          _negExpModule;
    core::cuda::CudaModule          _addOneModule;
    core::cuda::CudaModule          _quantQ8Module;
    core::cuda::CudaModule          _quantQ4KModule;
    core::cuda::CudaModule          _quantQ6KModule;
    core::cuda::CudaModule          _quantFp8Module;
    core::cuda::CudaModule          _repackNvblkModule;
    core::cuda::CudaKernel          _dqNvfp4;
    core::cuda::CudaKernel          _dqFp8;
    core::cuda::CudaKernel          _castBf16;
    core::cuda::CudaKernel          _castF16;
    core::cuda::CudaKernel          _negExp;
    core::cuda::CudaKernel          _addOne;
    core::cuda::CudaKernel          _quantQ8;
    core::cuda::CudaKernel          _quantQ4K;
    core::cuda::CudaKernel          _quantQ6K;
    core::cuda::CudaKernel          _quantFp8;
    core::cuda::CudaKernel          _repackNvblk;
};

} // namespace mimirmind::compute::cuda