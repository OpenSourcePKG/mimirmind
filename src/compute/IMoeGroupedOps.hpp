// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstddef>
#include <cstdint>

namespace mimirmind::compute {

/// FP4-tensor-core grouped-MoE device ops (M-Cuda.MoeGroup E-d.4b): the
/// padded-per-expert building blocks + the CUTLASS block-scaled NVFP4 grouped
/// GEMM. Segregated out of the backend-neutral `ComputeOps` (8.30.6 ISP-split)
/// because only the CUDA/Bragi backend (CUTLASS-linked) implements them — every
/// other backend used to satisfy the contract by throwing at runtime. A backend
/// that supports them returns a non-null `IMoeGroupedOps*` from
/// `ComputeOps::moeGroupedOps()`; others return nullptr, so the neutral base no
/// longer carries throw-default stubs for this concern.
///
/// Availability + scratch-size queries stay on `ComputeOps`
/// (moeGroupedGemmNvfp4TcAvailable / *ScratchBytes) — they return safe
/// false/0 defaults and are consulted (some ungated) before this interface is
/// reached.
class IMoeGroupedOps {
public:
    virtual ~IMoeGroupedOps() = default;

    /// Async device memset to zero (pre-zero the swizzled SF banks' padding).
    virtual void moeZeroBytesAsync(void* dst, std::size_t bytes) = 0;

    /// padOffset = prefix of round_up(count_e, 128); padOffset[nExperts]=totalPad.
    virtual void moePadOffsetsAsync(const std::int32_t* expOffset,
                                    std::int32_t* padOffset, std::size_t nExperts) = 0;

    /// contigToPad[r] = padded row of contiguous gathered row r.
    virtual void moeContigToPadAsync(const std::int32_t* expOffset,
                                     const std::int32_t* padOffset,
                                     std::int32_t* contigToPad,
                                     std::size_t nExperts, std::size_t R) = 0;

    /// dst[idxMap[r]] = src[r] over `dim`-wide rows (spread to padded slots).
    virtual void moeRowsScatterF32Async(const float* src, const std::int32_t* idxMap,
                                        float* dst, std::size_t nRows,
                                        std::size_t dim) = 0;

    /// dst[i] = (src[i] < 0) ? -1 : idxMap[src[i]]  (remap an index array).
    virtual void moeIndexGatherI32Async(const std::int32_t* src,
                                        const std::int32_t* idxMap,
                                        std::int32_t* dst, std::size_t n) = 0;

    /// F32 [M,K] activations -> NVFP4 nibbles [M,K/2] + swizzled UE4M3 SFA. The
    /// caller pre-zeroes `outNib`/`outSf` (padding). K % 16 == 0.
    virtual void moeActQuantNvfp4Async(const float* in, unsigned char* outNib,
                                       unsigned char* outSf, float gscale,
                                       std::size_t M, std::size_t K) = 0;

    /// Row-mapped activation quantiser: quantise only `nRows` rows, each read
    /// from / written to padded row `rowMap[i]` (skips the 128-row padding).
    virtual void moeActQuantNvfp4RowsAsync(const float* in, unsigned char* outNib,
                                           unsigned char* outSf, float gscale,
                                           const std::int32_t* rowMap,
                                           std::size_t nRows, std::size_t K) = 0;

    /// 5.21.10: fused gather + row-mapped NVFP4 act-quant — reads the COMPACT
    /// gathered rows and writes nibbles/SF at padded row `rowMap[logical]`.
    virtual void moeActQuantNvfp4GatherRowsAsync(const float* in,
                                                 unsigned char* outNib,
                                                 unsigned char* outSf, float gscale,
                                                 const std::int32_t* rowMap,
                                                 std::size_t nRows, std::size_t K,
                                                 const std::int32_t* srcMap = nullptr) = 0;

    /// 5.21.8: fused silu(gate)*up + row-mapped NVFP4 act-quant in one pass.
    virtual void moeSiluMulQuantNvfp4RowsAsync(const float* gate, const float* up,
                                               unsigned char* outNib,
                                               unsigned char* outSf, float gscale,
                                               const std::int32_t* rowMap,
                                               std::size_t nRows, std::size_t K) = 0;

    /// CUTLASS block-scaled NVFP4 grouped GEMM, one expert per group, F32 out.
    /// `scratch` is caller-owned, sized >=
    /// ComputeOps::moeGroupedGemmNvfp4TcBanksScratchBytes(nExperts).
    virtual void moeGroupedGemmNvfp4TcBanksAsync(
        std::size_t nExperts, std::size_t N, std::size_t K,
        const std::int32_t* expOffset, const std::int32_t* padOffset,
        const void* aBank, const void* sfaBank,
        const void* bBank, const void* sfbBank,
        const float* globalsBank, void* dBank,
        void* scratch, std::size_t scratchBytes) = 0;

    /// 5.18.21: gate+up FUSED CUTLASS grouped GEMM — both projections in ONE
    /// launch (2*nExperts groups) sharing the activation banks. `scratch`
    /// caller-owned, sized >=
    /// ComputeOps::moeGroupedGemmNvfp4TcBanksGateUpScratchBytes(nExperts).
    virtual void moeGroupedGemmNvfp4TcBanksGateUpAsync(
        std::size_t nExperts, std::size_t N, std::size_t K,
        const std::int32_t* expOffset, const std::int32_t* padOffset,
        const void* aBank, const void* sfaBank,
        const void* gateBBank, const void* gateSfbBank,
        const float* gateGlobalsBank, void* gateDBank,
        const void* upBBank, const void* upSfbBank,
        const float* upGlobalsBank, void* upDBank,
        void* scratch, std::size_t scratchBytes) = 0;
};

} // namespace mimirmind::compute
