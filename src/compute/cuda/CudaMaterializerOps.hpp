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

private:
    core::cuda::CudaComputeContext& _ctx;
    ComputeOps&                     _ops;
    core::cuda::CudaModule          _nvfp4Module;
    core::cuda::CudaModule          _fp8Module;
    core::cuda::CudaModule          _castModule;
    core::cuda::CudaKernel          _dqNvfp4;
    core::cuda::CudaKernel          _dqFp8;
    core::cuda::CudaKernel          _castBf16;
    core::cuda::CudaKernel          _castF16;
};

} // namespace mimirmind::compute::cuda