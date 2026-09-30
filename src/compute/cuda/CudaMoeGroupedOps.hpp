// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "compute/IMoeGroupedOps.hpp"
#include "core/gpu/cuda/CudaKernel.hpp"
#include "core/gpu/cuda/CudaModule.hpp"

#include <cstddef>
#include <cstdint>

namespace mimirmind::core::cuda {
class CudaComputeContext;
}

namespace mimirmind::compute::cuda {

/**
 * CUDA implementation of the segregated IMoeGroupedOps interface (8.30.11.4):
 * the 11 FP4-tensor-core grouped-MoE device ops, carved out of the GpuOps
 * god-class into their own collaborator. Owns the two kernel modules the ops
 * need (moe_pad + moe_act_quant_nvfp4, 8 kernels total); the two TC-banks
 * grouped GEMMs call the CUTLASS free functions. GpuOps holds one instance and
 * hands it back from moeGroupedOps(); the availability + scratch-size queries
 * stay on GpuOps (they are consulted before this interface is reached).
 *
 * CUDA-only. Bodies are the former GpuOps::moe*Async methods verbatim.
 */
class CudaMoeGroupedOps final : public ::mimirmind::compute::IMoeGroupedOps {
public:
    explicit CudaMoeGroupedOps(core::cuda::CudaComputeContext& ctx);

    void moeZeroBytesAsync(void* dst, std::size_t bytes) override;
    void moePadOffsetsAsync(const std::int32_t* expOffset,
                            std::int32_t* padOffset, std::size_t nExperts) override;
    void moeContigToPadAsync(const std::int32_t* expOffset,
                             const std::int32_t* padOffset,
                             std::int32_t* contigToPad,
                             std::size_t nExperts, std::size_t R) override;
    void moeRowsScatterF32Async(const float* src, const std::int32_t* idxMap,
                                float* dst, std::size_t nRows,
                                std::size_t dim) override;
    void moeIndexGatherI32Async(const std::int32_t* src, const std::int32_t* idxMap,
                                std::int32_t* dst, std::size_t n) override;
    void moeActQuantNvfp4Async(const float* in, unsigned char* outNib,
                               unsigned char* outSf, float gscale,
                               std::size_t M, std::size_t K) override;
    void moeActQuantNvfp4RowsAsync(const float* in, unsigned char* outNib,
                                   unsigned char* outSf, float gscale,
                                   const std::int32_t* rowMap,
                                   std::size_t nRows, std::size_t K) override;
    void moeActQuantNvfp4GatherRowsAsync(const float* in, unsigned char* outNib,
                                         unsigned char* outSf, float gscale,
                                         const std::int32_t* rowMap,
                                         std::size_t nRows, std::size_t K,
                                         const std::int32_t* srcMap = nullptr) override;
    void moeSiluMulQuantNvfp4RowsAsync(const float* gate, const float* up,
                                       unsigned char* outNib, unsigned char* outSf,
                                       float gscale, const std::int32_t* rowMap,
                                       std::size_t nRows, std::size_t K) override;
    void moeGroupedGemmNvfp4TcBanksAsync(
        std::size_t nExperts, std::size_t N, std::size_t K,
        const std::int32_t* expOffset, const std::int32_t* padOffset,
        const void* aBank, const void* sfaBank,
        const void* bBank, const void* sfbBank,
        const float* globalsBank, void* dBank,
        void* scratch, std::size_t scratchBytes) override;
    void moeGroupedGemmNvfp4TcBanksGateUpAsync(
        std::size_t nExperts, std::size_t N, std::size_t K,
        const std::int32_t* expOffset, const std::int32_t* padOffset,
        const void* aBank, const void* sfaBank,
        const void* gateBBank, const void* gateSfbBank,
        const float* gateGlobalsBank, void* gateDBank,
        const void* upBBank, const void* upSfbBank,
        const float* upGlobalsBank, void* upDBank,
        void* scratch, std::size_t scratchBytes) override;

private:
    core::cuda::CudaComputeContext& _ctx;
    core::cuda::CudaModule _moePadModule;
    core::cuda::CudaModule _moeActQuantModule;
    core::cuda::CudaKernel _moePadOffsetsKernel;
    core::cuda::CudaKernel _moeContigToPadKernel;
    core::cuda::CudaKernel _moeRowsScatterKernel;
    core::cuda::CudaKernel _moeIndexGatherKernel;
    core::cuda::CudaKernel _moeActQuantKernel;
    core::cuda::CudaKernel _moeActQuantRowsKernel;
    core::cuda::CudaKernel _moeActQuantGatherRowsKernel;
    core::cuda::CudaKernel _moeSiluMulQuantRowsKernel;
};

} // namespace mimirmind::compute::cuda
