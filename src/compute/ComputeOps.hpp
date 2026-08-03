// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "compute/ComputeBuffer.hpp"
#include "core/config/Config.hpp"
#include "runtime/KvCache.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace mimirmind::compute {

/**
 * Backend-neutral kernel-launch interface. Every element-wise +
 * normalisation + attention kernel that the transformer block hits
 * shows up here as a pure virtual — one method per fused-shape.
 *
 * Concrete backends (`GpuOps` on Level-Zero, `HipGpuOps` on ROCm/HIP)
 * implement the launches against their native APIs and own the module
 * loading / kernel handles / persistent scratch buffers. Backend-
 * specific escape hatches (raw command queue, USM/HIP allocator,
 * `curLenSlot()` USM int) stay on the concrete class — consumers that
 * need them still downcast to the concrete backend, exactly the same
 * way they downcast `ComputeContext&` to `L0ComputeContext&`.
 *
 * Schicht 3 of the HW-abstraction strategy: introducing the abstract
 * base establishes the interface contract without yet forcing every
 * caller to migrate to it. `GpuOps` inherits and marks its overrides;
 * backends can keep holding `GpuOps&` (or `HipGpuOps&` when it lands)
 * until Schicht 4 unifies the launch API. This split keeps the diff
 * bounded — the alternative was a mass-rename of every backend at the
 * same time as the interface split, which was rejected as too risky
 * for one commit.
 *
 * All async methods append a kernel to the backend's queue WITHOUT
 * syncing — the caller flushes/syncs on the concrete backend before
 * reading results. Not thread-safe; construct once at startup.
 */
class ComputeOps {
public:
    virtual ~ComputeOps() = default;

    ComputeOps(const ComputeOps&)            = delete;
    ComputeOps& operator=(const ComputeOps&) = delete;
    ComputeOps(ComputeOps&&)                 = delete;
    ComputeOps& operator=(ComputeOps&&)      = delete;

    // ---- Element-wise + normalisation ----------------------------------

    virtual void rmsNormAsync(const float* x,
                              std::size_t  M,
                              std::size_t  K,
                              const float* weight,
                              float        eps,
                              float*       y) = 0;

    virtual void rmsNormGemmaAsync(const float* x,
                                   std::size_t  M,
                                   std::size_t  K,
                                   const float* weight,
                                   float        eps,
                                   float*       y) = 0;

    virtual void rmsNormNoWeightAsync(const float* x,
                                      std::size_t  M,
                                      std::size_t  K,
                                      float        eps,
                                      float*       y) = 0;

    virtual void rmsNormQkvAsync(float*           qBuf,   const float* qWeight,
                                 void*            kBase,  const float* kWeight,
                                 void*            vBase,
                                 std::size_t      qRows,
                                 std::size_t      kvRows,
                                 std::size_t      headDim,
                                 float            eps,
                                 std::size_t      writeOffset,
                                 std::size_t      kvDim,
                                 runtime::KvDtype kvDtype        = runtime::KvDtype::F32,
                                 bool             useStagingSlot = false) = 0;

    virtual void addRmsNormAsync(float*       x,
                                 const float* delta,
                                 std::size_t  M,
                                 std::size_t  K,
                                 const float* weight,
                                 float        eps,
                                 float*       y) = 0;

    virtual void addBiasAsync(float*       y,
                              std::size_t  M,
                              std::size_t  K,
                              const float* bias) = 0;

    virtual void addResidualAsync(float*       y,
                                  const float* x,
                                  std::size_t  n) = 0;

    virtual void siluMulAsync(float*       gate,
                              const float* up,
                              std::size_t  n) = 0;

    virtual void geluMulAsync(float*       gate,
                              const float* up,
                              std::size_t  n) = 0;

    virtual void mulScalarAsync(float*       y,
                                float        s,
                                std::size_t  n) = 0;

    virtual void scaledAddResidualAsync(float*       dst,
                                        const float* src,
                                        float        scale,
                                        std::size_t  n) = 0;

    // De-interleave a fused per-head [block_a | block_b] projection into
    // two contiguous [seqLen, numHeads, headDim] buffers. Qwen3-Next fuses
    // the query and a per-head output gate into one `attn_q` weight whose
    // output is `[Q_h | gate_h]` per head (head stride 2*headDim); this
    // splits it so RoPE/attention see a contiguous Q and the gate is
    // applied post-attention. `a` = first block per head, `b` = second.
    virtual void splitHeadPairAsync(const float* src,
                                    float*       a,
                                    float*       b,
                                    std::size_t  seqLen,
                                    std::size_t  numHeads,
                                    std::size_t  headDim) = 0;

    // In-place sigmoid gating: y[r,c] *= sigmoid(g[r, gateDim==1?0:c]).
    // `gateDim == dim` = per-element gate (Qwen3-Next attention output
    // gate); `gateDim == 1` = per-row scalar broadcast (shared-expert
    // gate). y is [rows, dim] f32 row-major; g is [rows, gateDim].
    virtual void sigmoidGateMulAsync(float*       y,
                                     const float* g,
                                     std::size_t  rows,
                                     std::size_t  dim,
                                     std::size_t  gateDim) = 0;

    // ---- GatedDeltaNet (Qwen3-Next linear attention, M-Q3N.3) ---------
    //
    // GPU counterparts of the compute/GatedDeltaNet.* CPU reference. The
    // CPU backend delegates straight to that reference; the GPU backends
    // dispatch dedicated kernels validated against it. All F32.

    /// In-place L2 normalisation over the innermost `dim` (head_dim):
    /// `x[r,:] /= max(sqrt(sum_j x[r,j]^2), eps)`. `rows` = number of
    /// length-`dim` vectors (== T*H). Applied to q/k in the linear layer.
    virtual void l2NormInPlaceAsync(float*      x,
                                    std::size_t rows,
                                    std::size_t dim,
                                    float       eps) = 0;

    /// Causal depthwise 1-D convolution + SiLU (Qwen3-Next `ssm_conv1d`):
    /// `out[t,c] = silu( sum_{k} convInput[t+k, c] * kernel[k, c] )`.
    /// `convInput` is [(kernelSize-1)+T, channels]; `kernel` is
    /// [kernelSize, channels] (tap-major); `out` is [T, channels].
    virtual void causalConv1dSiluAsync(const float* convInput,
                                       const float* kernel,
                                       float*       out,
                                       std::size_t  T,
                                       std::size_t  channels,
                                       std::size_t  kernelSize) = 0;

    /// Autoregressive gated delta-rule recurrence over `T` tokens, per
    /// value-head. Updates `state` [H, S, S] in place and writes `out`
    /// [T, H, S]. q/k/v are [T, H, S]; gLog/beta are [T, H] (gLog is the
    /// raw log-decay — exp applied internally). Reference:
    /// compute::gatedDeltaNetRecurrent. Single sequence (n_seqs == 1).
    virtual void gatedDeltaNetRecurrentAsync(const float* q,
                                             const float* k,
                                             const float* v,
                                             const float* gLog,
                                             const float* beta,
                                             float*       state,
                                             float*       out,
                                             std::size_t  T,
                                             std::size_t  H,
                                             std::size_t  S) = 0;

    /// GatedDeltaNet decay gate: gLog[t,h] = -exp(ssmA[h]) *
    /// softplus(alpha[t,h] + ssmDt[h]). alpha [T,H]; ssmA/ssmDt [H];
    /// gLog (out) [T,H]. Reference: compute::deltanetGate.
    virtual void deltanetGateAsync(const float* alpha,
                                   const float* ssmA,
                                   const float* ssmDt,
                                   float*       gLog,
                                   std::size_t  T,
                                   std::size_t  H) = 0;

    /// Chunked-prefill stage K0: inclusive prefix-sum of gLog within each
    /// chunk, per head. gLog/gCum [T,H]; chunkSize C (0 -> 64). Reference:
    /// compute::deltanetChunkCumGate. Default: unsupported — the chunked-
    /// prefill GPU path is CUDA-first; CUDA overrides.
    virtual void deltanetChunkCumGateAsync(const float* gLog,
                                           float*       gCum,
                                           std::size_t  T,
                                           std::size_t  H,
                                           std::size_t  chunkSize) {
        (void)gLog; (void)gCum; (void)T; (void)H; (void)chunkSize;
        throw std::runtime_error(
            "deltanetChunkCumGateAsync: not supported on this backend");
    }

    /// Chunked-prefill stage K2: chunk forward (readout + state carry).
    /// Consumes gCum (K0) and a0 (K1); writes out [T,H,S] and carries state
    /// [H,S,S] in place. Reference: compute::deltanetChunkForward. Default:
    /// unsupported; CUDA overrides.
    virtual void deltanetChunkForwardAsync(const float* q, const float* k,
                                           const float* v, const float* gCum,
                                           const float* beta, const float* a0,
                                           float* state, float* out,
                                           std::size_t T, std::size_t H,
                                           std::size_t S, std::size_t chunkSize) {
        (void)q; (void)k; (void)v; (void)gCum; (void)beta; (void)a0;
        (void)state; (void)out; (void)T; (void)H; (void)S; (void)chunkSize;
        throw std::runtime_error(
            "deltanetChunkForwardAsync: not supported on this backend");
    }

    // ---- M-Cuda.Batch batched (nSeq) variants -----------------------------
    // Serving-class batched decode/prefill ops. CUDA-first; every default
    // throws so non-CUDA backends (single-session) stay unaffected.

    virtual void gatedDeltaNetRecurrentBatchedAsync(
            const float* q, const float* k, const float* v, const float* gLog,
            const float* beta, float* state, float* out, std::size_t nSeq,
            std::size_t T, std::size_t H, std::size_t S) {
        (void)q; (void)k; (void)v; (void)gLog; (void)beta; (void)state;
        (void)out; (void)nSeq; (void)T; (void)H; (void)S;
        throw std::runtime_error(
            "gatedDeltaNetRecurrentBatchedAsync: not supported on this backend");
    }

    // MV-a (MTP verify): batched GDN over the T=K+1 window with per-position
    // state export (stateOut [T,nSeq,H,S,S]) instead of a per-step snapshot.
    virtual void gatedDeltaNetVerifyBatchedAsync(
            const float* q, const float* k, const float* v, const float* gLog,
            const float* beta, const float* stateIn, float* stateOut,
            float* out, std::size_t nSeq, std::size_t T, std::size_t H,
            std::size_t S) {
        (void)q; (void)k; (void)v; (void)gLog; (void)beta; (void)stateIn;
        (void)stateOut; (void)out; (void)nSeq; (void)T; (void)H; (void)S;
        throw std::runtime_error(
            "gatedDeltaNetVerifyBatchedAsync: not supported on this backend");
    }

    virtual void causalConv1dSiluBatchedAsync(
            const float* convInput, const float* kernel, float* out,
            std::size_t nSeq, std::size_t T, std::size_t channels,
            std::size_t kernelSize) {
        (void)convInput; (void)kernel; (void)out; (void)nSeq; (void)T;
        (void)channels; (void)kernelSize;
        throw std::runtime_error(
            "causalConv1dSiluBatchedAsync: not supported on this backend");
    }

    virtual void mropeInPlaceBatchedAsync(
            void* xBase, std::size_t nSeq, std::size_t xSeqStride,
            std::size_t seqLen, std::size_t numHeads, std::size_t headDim,
            const std::int32_t* startPosDev, float base,
            const std::int32_t* sections, std::size_t writeOffsetStride,
            runtime::KvDtype kvDtype) {
        (void)xBase; (void)nSeq; (void)xSeqStride; (void)seqLen; (void)numHeads;
        (void)headDim; (void)startPosDev; (void)base; (void)sections;
        (void)writeOffsetStride; (void)kvDtype;
        throw std::runtime_error(
            "mropeInPlaceBatchedAsync: not supported on this backend");
    }

    virtual void attentionDecodeFlashBatchedAsync(
            const float* q, const float* k, const float* v,
            float* partialScratch, float* out, std::size_t nSeq,
            std::size_t maxKTiles, std::size_t qSeqStride,
            std::size_t kvSeqStride, std::size_t partialSeqStride,
            std::size_t outSeqStride, std::size_t nHeads, std::size_t nKvHeads,
            std::size_t headDim, const std::int32_t* curLenDev, float scale,
            std::size_t slidingWindow, runtime::KvDtype kvDtype) {
        (void)q; (void)k; (void)v; (void)partialScratch; (void)out; (void)nSeq;
        (void)maxKTiles; (void)qSeqStride; (void)kvSeqStride;
        (void)partialSeqStride; (void)outSeqStride; (void)nHeads;
        (void)nKvHeads; (void)headDim; (void)curLenDev; (void)scale;
        (void)slidingWindow; (void)kvDtype;
        throw std::runtime_error(
            "attentionDecodeFlashBatchedAsync: not supported on this backend");
    }

    virtual void deltanetChunkCumGateBatchedAsync(
            const float* gLog, float* gCum, std::size_t nSeq, std::size_t T,
            std::size_t H, std::size_t chunkSize) {
        (void)gLog; (void)gCum; (void)nSeq; (void)T; (void)H; (void)chunkSize;
        throw std::runtime_error(
            "deltanetChunkCumGateBatchedAsync: not supported on this backend");
    }

    virtual void deltanetChunkForwardBatchedAsync(
            const float* q, const float* k, const float* v, const float* gCum,
            const float* beta, const float* a0, float* state, float* out,
            std::size_t nSeq, std::size_t T, std::size_t H, std::size_t S,
            std::size_t chunkSize) {
        (void)q; (void)k; (void)v; (void)gCum; (void)beta; (void)a0;
        (void)state; (void)out; (void)nSeq; (void)T; (void)H; (void)S;
        (void)chunkSize;
        throw std::runtime_error(
            "deltanetChunkForwardBatchedAsync: not supported on this backend");
    }

    /// M-Cuda.Batch B2/D2a — paged-KV decode attention. One query token per
    /// sequence; each sequence reads its KV from the shared physical pool
    /// (PagedKvPool) via its block table. `keyCache`/`valueCache` are the
    /// per-layer pool bases [numBlocks, blockSize, numKvHeads, headSize];
    /// `blockTables` [numSeqs, maxNumBlocksPerSeq] int32 (-1 sentinel);
    /// `seqLens` [numSeqs] int32. `query`/`out` [numSeqs, numHeads, headSize].
    /// fp32 baseline (kernel `paged_attention_v1`). Default: unsupported;
    /// CUDA overrides.
    virtual void pagedAttentionDecodeV1Async(
            float* out, const float* query, const float* keyCache,
            const float* valueCache, const std::int32_t* blockTables,
            const std::int32_t* seqLens, std::size_t numSeqs,
            std::size_t numHeads, std::size_t numKvHeads, std::size_t headSize,
            std::size_t blockSize, std::size_t maxNumBlocksPerSeq, float scale,
            float softcap) {
        (void)out; (void)query; (void)keyCache; (void)valueCache;
        (void)blockTables; (void)seqLens; (void)numSeqs; (void)numHeads;
        (void)numKvHeads; (void)headSize; (void)blockSize;
        (void)maxNumBlocksPerSeq; (void)scale; (void)softcap;
        throw std::runtime_error(
            "pagedAttentionDecodeV1Async: not supported on this backend");
    }

    /// Chunked-prefill stage K1: per-chunk ungated triangular inverse A0
    /// (a0 [nChunks,H,C,C]). Reference: compute::deltanetKktSolveInverse.
    /// Default: unsupported; CUDA overrides.
    virtual void deltanetKktSolveInverseAsync(const float* k, const float* beta,
                                              float* a0, std::size_t T,
                                              std::size_t H, std::size_t S,
                                              std::size_t chunkSize) {
        (void)k; (void)beta; (void)a0; (void)T; (void)H; (void)S; (void)chunkSize;
        throw std::runtime_error(
            "deltanetKktSolveInverseAsync: not supported on this backend");
    }

    /// M-Q3N.5: device-side MoE top-K router. Replaces the host
    /// compute::moeTopKRoute + host->USM copy so the MoE block has no host
    /// sync (precondition for decode graph capture). logits [T,nExperts] F32;
    /// outIdx [T,K] int32; outWeight [T,K] F32 (renormalised, *wScale).
    /// Reference: compute::moeTopKRoute. Default: unsupported; CUDA overrides.
    virtual void moeTopKRouteDeviceAsync(const float*  logits,
                                         std::int32_t* outIdx,
                                         float*        outWeight,
                                         std::size_t   T,
                                         std::size_t   nExperts,
                                         std::size_t   K,
                                         float         wScale) {
        (void)logits; (void)outIdx; (void)outWeight;
        (void)T; (void)nExperts; (void)K; (void)wScale;
        throw std::runtime_error(
            "moeTopKRouteDeviceAsync: not supported on this backend");
    }

    /// M-Cuda.MoeGroup: grouped-by-expert MoE prefill building blocks. Turns
    /// flat per-assignment routing (expIdx/kw, R=T*K) into an offset table +
    /// permutation so each expert weight is read once (moeGroupBuildAsync),
    /// gathers the per-expert-grouped activations (moeGatherRowsAsync), and
    /// folds the grouped GEMM output back to token order with router weights
    /// (moeScatterExpertOutAsync). Default: unsupported; CUDA overrides.
    /// See kernels/cuda/llm/moe_group_build.cu et al. for the exact contracts.
    virtual void moeGroupBuildAsync(const std::int32_t* expIdx, const float* kw,
                                    std::int32_t* expOffset,
                                    std::int32_t* rowSrcTok, float* rowKw,
                                    std::int32_t* asnToRow, std::size_t R,
                                    std::size_t nExperts, std::size_t K) {
        (void)expIdx; (void)kw; (void)expOffset; (void)rowSrcTok; (void)rowKw;
        (void)asnToRow; (void)R; (void)nExperts; (void)K;
        throw std::runtime_error(
            "moeGroupBuildAsync: not supported on this backend");
    }
    virtual void moeGatherRowsAsync(const float* x, const std::int32_t* rowSrcTok,
                                    float* xCompact, std::size_t dModel,
                                    std::size_t R) {
        (void)x; (void)rowSrcTok; (void)xCompact; (void)dModel; (void)R;
        throw std::runtime_error(
            "moeGatherRowsAsync: not supported on this backend");
    }
    virtual void moeScatterExpertOutAsync(const float* y,
                                          const std::int32_t* asnToRow,
                                          const float* kw, float* accum,
                                          std::size_t dModel, std::size_t T,
                                          std::size_t K) {
        (void)y; (void)asnToRow; (void)kw; (void)accum;
        (void)dModel; (void)T; (void)K;
        throw std::runtime_error(
            "moeScatterExpertOutAsync: not supported on this backend");
    }

    /// M-Cuda.MoeGroup Sub-Step E-a: turn the per-expert row ranges
    /// (expOffset, exclusive prefix sum) into a compact per-tile schedule a
    /// single device-driven grouped GEMM consumes in ONE launch — no expOffset
    /// D2H, no per-expert host loop. `maxTiles` is the host-computed static
    /// upper bound ceil(R/tileM)+nExperts; unused tail tiles carry the -1
    /// sentinel. Default: unsupported; CUDA overrides. See
    /// kernels/cuda/llm/moe_group_tiles.cu.
    virtual void moeGroupTilesAsync(const std::int32_t* expOffset,
                                    std::int32_t* tileExpert,
                                    std::int32_t* tileRow0,
                                    std::int32_t* tileRows,
                                    std::int32_t* nTiles,
                                    std::size_t nExperts, std::size_t maxTiles,
                                    std::size_t tileM) {
        (void)expOffset; (void)tileExpert; (void)tileRow0; (void)tileRows;
        (void)nTiles; (void)nExperts; (void)maxTiles; (void)tileM;
        throw std::runtime_error(
            "moeGroupTilesAsync: not supported on this backend");
    }

    /// M-Cuda.MoeGroup Sub-Step E-b: device-driven grouped GEMM over the
    /// contiguous [nExperts][N][K] blocked-NVFP4 expert bank. Each grid.y block
    /// reads its (expert, row-range) from the tile schedule ON THE DEVICE and
    /// GEMMs its <= tileM rows into Y[row0.., :]. `maxTiles` sizes grid.y
    /// (over-provisioned to the static bound; -1 sentinel tiles early-exit).
    /// Default: unsupported; CUDA overrides. See
    /// kernels/cuda/llm/moe_grouped_gemm_nvfp4blk.cu.
    virtual void moeGroupedGemmNvfp4Async(const float* x,
                                          const unsigned char* w, float* y,
                                          const std::int32_t* tileExpert,
                                          const std::int32_t* tileRow0,
                                          const std::int32_t* tileRows,
                                          std::size_t K, std::size_t N,
                                          std::size_t maxTiles,
                                          bool decodeSmallM = false) {
        (void)x; (void)w; (void)y; (void)tileExpert; (void)tileRow0;
        (void)tileRows; (void)K; (void)N; (void)maxTiles; (void)decodeSmallM;
        throw std::runtime_error(
            "moeGroupedGemmNvfp4Async: not supported on this backend");
    }

    // === M-Cuda.MoeGroup Sub-Step E-d.4b: FP4-tensor-core grouped MoE ========
    // The padded-per-expert building blocks + the CUTLASS block-scaled NVFP4
    // grouped GEMM. CUDA-only; every default is unsupported / unavailable.

    /// True if this build can run the FP4-TC grouped GEMM (CUTLASS linked).
    [[nodiscard]] virtual bool moeGroupedGemmNvfp4TcAvailable() const noexcept {
        return false;
    }

    /// Async device memset to zero (pre-zero the swizzled SF banks' padding).
    virtual void moeZeroBytesAsync(void* dst, std::size_t bytes) {
        (void)dst; (void)bytes;
        throw std::runtime_error("moeZeroBytesAsync: not supported on this backend");
    }

    /// padOffset = prefix of round_up(count_e, 128); padOffset[nExperts]=totalPad.
    virtual void moePadOffsetsAsync(const std::int32_t* expOffset,
                                    std::int32_t* padOffset, std::size_t nExperts) {
        (void)expOffset; (void)padOffset; (void)nExperts;
        throw std::runtime_error("moePadOffsetsAsync: not supported on this backend");
    }

    /// contigToPad[r] = padded row of contiguous gathered row r.
    virtual void moeContigToPadAsync(const std::int32_t* expOffset,
                                     const std::int32_t* padOffset,
                                     std::int32_t* contigToPad,
                                     std::size_t nExperts, std::size_t R) {
        (void)expOffset; (void)padOffset; (void)contigToPad; (void)nExperts; (void)R;
        throw std::runtime_error("moeContigToPadAsync: not supported on this backend");
    }

    /// dst[idxMap[r]] = src[r] over `dim`-wide rows (spread to padded slots).
    virtual void moeRowsScatterF32Async(const float* src, const std::int32_t* idxMap,
                                        float* dst, std::size_t nRows, std::size_t dim) {
        (void)src; (void)idxMap; (void)dst; (void)nRows; (void)dim;
        throw std::runtime_error("moeRowsScatterF32Async: not supported on this backend");
    }

    /// dst[i] = (src[i] < 0) ? -1 : idxMap[src[i]]  (remap an index array).
    virtual void moeIndexGatherI32Async(const std::int32_t* src,
                                        const std::int32_t* idxMap,
                                        std::int32_t* dst, std::size_t n) {
        (void)src; (void)idxMap; (void)dst; (void)n;
        throw std::runtime_error("moeIndexGatherI32Async: not supported on this backend");
    }

    /// F32 [M,K] activations -> NVFP4 nibbles [M,K/2] + swizzled UE4M3 SFA. The
    /// caller pre-zeroes `outNib`/`outSf` (padding). K % 16 == 0.
    virtual void moeActQuantNvfp4Async(const float* in, unsigned char* outNib,
                                       unsigned char* outSf, float gscale,
                                       std::size_t M, std::size_t K) {
        (void)in; (void)outNib; (void)outSf; (void)gscale; (void)M; (void)K;
        throw std::runtime_error("moeActQuantNvfp4Async: not supported on this backend");
    }

    /// Device scratch bytes moeGroupedGemmNvfp4TcBanksAsync needs for
    /// `nExperts` groups (per-group CUTLASS arrays + workspace). Zero if the
    /// FP4-TC path is unavailable. The caller owns the buffer (per-slot).
    [[nodiscard]] virtual std::size_t
    moeGroupedGemmNvfp4TcBanksScratchBytes(std::size_t nExperts) const noexcept {
        (void)nExperts;
        return 0;
    }

    /// CUTLASS block-scaled NVFP4 grouped GEMM, one expert per group, F32 out.
    /// Banks + device expOffset/padOffset; all per-group pointers built on
    /// device (no D2H). N,K shared. `scratch` is caller-owned (per-slot), sized
    /// >= moeGroupedGemmNvfp4TcBanksScratchBytes(nExperts). See
    /// runGroupedNvfp4TcF32Banks.
    virtual void moeGroupedGemmNvfp4TcBanksAsync(
        std::size_t nExperts, std::size_t N, std::size_t K,
        const std::int32_t* expOffset, const std::int32_t* padOffset,
        const void* aBank, const void* sfaBank,
        const void* bBank, const void* sfbBank,
        const float* globalsBank, void* dBank,
        void* scratch, std::size_t scratchBytes) {
        (void)nExperts; (void)N; (void)K; (void)expOffset; (void)padOffset;
        (void)aBank; (void)sfaBank; (void)bBank; (void)sfbBank;
        (void)globalsBank; (void)dBank; (void)scratch; (void)scratchBytes;
        throw std::runtime_error(
            "moeGroupedGemmNvfp4TcBanksAsync: not supported on this backend");
    }

    /// M-CLR.MoE Increment 2: device-indexed fused gate+up projection for
    /// Gemma 4 MoE T=1 decode. Reads the router pick `expIdx[k]` on the
    /// device (no host round-trip on the routing) and folds the per-expert
    /// down-scale into the activation, so the companion fused-K down kernel
    /// can take the raw router weight as `kw`. Together they remove every
    /// host read of the routing from the decode MoE block — the
    /// precondition for Command-List-Replay capture (Xe-LPG / NUC).
    ///   gateActOut[k, f] = gelu_tanh(gate_f . x) * (up_f . x) * downScale[e]
    /// with gate_f = row f, up_f = row (nFf + f) of expert e's fused
    /// gate_up bank. Reference: the sequential (gate GEMV, up GEMV,
    /// geluMul, downScale) host path. Default: unsupported; L0 overrides
    /// for Q6_K expert banks.
    virtual void moeGateUpFusedKGeluAsync(const float*        x,
                                          const void*         W,
                                          const std::int32_t* expIdx,
                                          const float*        downScale,
                                          float*              gateActOut,
                                          std::size_t         dModel,
                                          std::size_t         nFf,
                                          std::size_t         kActive,
                                          std::size_t         expertBytes) {
        (void)x; (void)W; (void)expIdx; (void)downScale; (void)gateActOut;
        (void)dModel; (void)nFf; (void)kActive; (void)expertBytes;
        throw std::runtime_error(
            "moeGateUpFusedKGeluAsync: not supported on this backend");
    }

    /// True when the fused device gate+up kernel is loaded and this dModel
    /// fits its SLM-resident input ceiling. The caller must additionally
    /// confirm the expert gate_up bank is Q6_K (the only type wired today).
    [[nodiscard]] virtual bool moeGateUpFusedKGeluAvailable(
        std::size_t /*dModel*/) const noexcept { return false; }

    /// In-place logistic sigmoid: y[i] = 1/(1+exp(-y[i])). GatedDeltaNet
    /// `beta` gate. Reference: compute::sigmoidInPlace.
    virtual void sigmoidInPlaceAsync(float* y, std::size_t n) = 0;

    /// Per-head channel slice + optional GQA head repeat: turns the fused
    /// conv output into contiguous q / k / v. dst[t,hd,s] =
    /// src[t*convTotalWidth + offset + (hd % srcHeads)*S + s]. dst is
    /// [T, dstHeads, S]. Reference: compute::gatherHeadsFromChannels.
    virtual void gatherHeadsFromChannelsAsync(const float* src,
                                              float*       dst,
                                              std::size_t  T,
                                              std::size_t  offset,
                                              std::size_t  srcHeads,
                                              std::size_t  dstHeads,
                                              std::size_t  S,
                                              std::size_t  convTotalWidth) = 0;

    // ---- RoPE ---------------------------------------------------------

    virtual void ropeInPlaceAsync(void*            xBase,
                                  std::size_t      seqLen,
                                  std::size_t      numHeads,
                                  std::size_t      headDim,
                                  std::size_t      startPos,
                                  float            base,
                                  std::size_t      writeOffsetStride = 0,
                                  runtime::KvDtype kvDtype           = runtime::KvDtype::F32) = 0;

    virtual void ropeInPlaceWithFactorsAsync(void*            xBase,
                                             const float*     freqFactors,
                                             std::size_t      seqLen,
                                             std::size_t      numHeads,
                                             std::size_t      headDim,
                                             std::size_t      startPos,
                                             float            base,
                                             std::size_t      writeOffsetStride = 0,
                                             runtime::KvDtype kvDtype           = runtime::KvDtype::F32) = 0;

    // Interleaved multi-axis RoPE (IMRoPE) — Qwen3-Next / Qwen3.5-VL
    // full-attention layers (`LLM_ROPE_TYPE_IMROPE`). Same split-pair
    // rotation as ropeInPlaceAsync but the per-pair angle base is chosen
    // across four position axes via the IMRoPE sector rule. `sections`
    // points at 4 int32 dimension-section widths (GGUF
    // `<arch>.rope.dimension_sections`). For text-only positions (all
    // axes equal) this is bit-identical to ropeInPlaceAsync. Only the
    // F32 storage path is implemented in M-Q3N.2; FP16/Q8_0 KV throws.
    virtual void mropeInPlaceAsync(void*              xBase,
                                   std::size_t        seqLen,
                                   std::size_t        numHeads,
                                   std::size_t        headDim,
                                   std::size_t        startPos,
                                   float              base,
                                   const std::int32_t* sections,
                                   std::size_t        writeOffsetStride = 0,
                                   runtime::KvDtype   kvDtype           = runtime::KvDtype::F32) = 0;

    // ---- Quantisation + KV commit -------------------------------------

    virtual void xQuantI8Async(const float* x,
                               std::int8_t* y,
                               float*       scale,
                               std::size_t  M,
                               std::size_t  K) = 0;

    virtual void kvQuantCommitQ8Async(const float* xSrc,
                                      void*        kvDst,
                                      std::size_t  T,
                                      std::size_t  kvDim,
                                      std::size_t  writeOffset) = 0;

    virtual void qkvSplitAsync(const float*     fused,
                               float*           Yq,
                               void*            YkBase,
                               void*            YvBase,
                               std::size_t      M,
                               std::size_t      Nq,
                               std::size_t      Nkv,
                               bool             hasV,
                               std::size_t      writeOffset    = 0,
                               runtime::KvDtype kvDtype        = runtime::KvDtype::F32,
                               bool             useStagingSlot = false) = 0;

    // ---- Attention ----------------------------------------------------

    virtual void attentionAsync(const float*     q,
                                const void*      k,
                                const void*      v,
                                std::size_t      T_q,
                                std::size_t      T_k,
                                std::size_t      nHeads,
                                std::size_t      nKvHeads,
                                std::size_t      headDim,
                                std::size_t      positionOffset,
                                float            scale,
                                float*           out,
                                std::size_t      slidingWindow = 0,
                                runtime::KvDtype kvDtype       = runtime::KvDtype::F32) = 0;

    // ---- Reordered-Q8_0 matvec (test-facing) --------------------------

    virtual void matmulQ8_0VecReorderAsync(const void*  wReordered,
                                           std::size_t  N,
                                           std::size_t  K,
                                           const float* x,
                                           float*       y) = 0;

    // ---- Recording-side knobs -----------------------------------------

    /// Right-size the FlashAttention partial launch geometry during
    /// recording. When 0 (default), the launch uses the actual `nKTiles`
    /// derived from `positionOffset+1`. See `GpuOps::setReplayMaxKTiles`
    /// for the CLR context; HIP's hipGraph equivalent will consume the
    /// same knob when it lands.
    virtual void setReplayMaxKTiles(std::size_t n) noexcept = 0;

    // ---- Feature-flag + status accessors ------------------------------
    //
    // These reflect the ctor-time resolution of `features.*` in
    // config.json plus any post-load autotune result. Numerics-neutral
    // bookkeeping — every backend surfaces the same values through
    // `/v1/system/status`, so the interface pins them here.

    [[nodiscard]] virtual std::string_view selfTestStatus() const noexcept = 0;

    [[nodiscard]] virtual bool prefillFlashEnabled() const noexcept = 0;
    [[nodiscard]] virtual bool prefillFlashGqaQ8Enabled() const noexcept = 0;
    [[nodiscard]] virtual std::size_t prefillFlashKTileQ8() const noexcept = 0;
    [[nodiscard]] virtual std::string_view prefillFlashKTileQ8Source() const noexcept = 0;

    [[nodiscard]] virtual core::config::TriState q8_0ReorderMode() const noexcept = 0;
    [[nodiscard]] virtual std::string_view q8_0ReorderModeName() const noexcept = 0;
    [[nodiscard]] virtual std::size_t q8_0ReorderTensorCount() const noexcept = 0;
    [[nodiscard]] virtual std::size_t q8_0ReorderTotalBytes() const noexcept = 0;

    /// Load-time hook: backends invoke this once per Q8_0 tensor they
    /// actually reordered so the counters + info log stay accurate. See
    /// `GpuOps::noteQ8_0ReorderApplied` for the full contract.
    virtual void noteQ8_0ReorderApplied(std::size_t bytes,
                                        std::string_view label) noexcept = 0;

    // ---- Stream / recording ops (Schritt 3c.1) ------------------------
    //
    // Neutral wrappers around the two dominant L0-specific operations
    // that backends previously reached through `_ops.queue()`:
    // `UnorderedScope` (concurrent-dispatch window on the command
    // queue) and `appendMemoryCopy` (device↔device or device↔host
    // copy queued into the same stream as kernel launches).
    //
    // L0 impl forwards to `runtime::CommandQueue::{push,pop}Unordered`
    // and `appendMemoryCopy`. HIP impl leaves push/pop as no-ops
    // (streams already reorder freely at driver level) and routes
    // the copy through `hipMemcpyAsync` on the shared HIP stream.

    /// Enter concurrent-dispatch scope. Launches appended inside the
    /// scope may reorder / overlap; the runtime restores strict order
    /// on `popUnorderedScope()`. Use the RAII `compute::UnorderedScope`
    /// helper below rather than calling these two directly.
    virtual void pushUnorderedScope() = 0;
    virtual void popUnorderedScope()  = 0;

    /// Queue a memory copy into the same stream as kernel launches.
    /// Both pointers must be reachable from the device (backend-
    /// dependent — for L0 USM they can be host/shared/device; for HIP
    /// they can be host-pinned or device). Not synchronous — call
    /// `flush()` before the CPU reads the result.
    virtual void appendMemoryCopy(void*       dst,
                                  const void* src,
                                  std::size_t bytes) = 0;

    /// Flush pending work + wait for it. On L0 this closes the current
    /// command list, executes it, and syncs; on HIP it calls
    /// `hipStreamSynchronize`. Same semantic as `ComputeMatmul::sync()`
    /// but reachable through the ops interface for callers that only
    /// hold `ComputeOps&`.
    virtual void flush() = 0;

    /// Copy `bytes` from a device buffer to a plain host buffer
    /// synchronously — the CPU can dereference `dst` immediately after
    /// return.
    ///
    /// **Why this exists (Session 2026-07-18 finding):** L0 on Meteor
    /// Lake uses UMA/USM so a device pointer is directly host-readable
    /// with cache-line-speed access. HIP on discrete gfx1101 does NOT
    /// share memory: every host-side `logits[i]` on a `hipMalloc`'d
    /// buffer traverses PCIe, ~700 ns/access × 152 k vocab = ~110 ms
    /// per token. That was the dominant unaccounted decode cost.
    /// Callers that need to CPU-scan a device buffer (argmax, top-k,
    /// etc.) must go through this method — L0 falls back to std::memcpy
    /// (which the compiler often elides when src == dst), HIP issues a
    /// single bulk `hipMemcpy(D→H)`.
    virtual void readbackToHost(void*       hostDst,
                                const void* deviceSrc,
                                std::size_t bytes) = 0;

    // ---- Allocation (Schritt 3c.2) ------------------------------------
    //
    // Neutral buffer factory. Consumers that used to construct
    // `core::l0::UsmHandle{allocator(), bytes}` now call
    // `ops.allocate(bytes)` and hold the resulting `ComputeBuffer` as
    // a value member (see `runtime::BlockBuffers`). The concrete
    // backend installs a deleter closure at allocate() time; the
    // buffer releases itself on destruction without depending on any
    // backend type.
    //
    // A zero-byte request returns an empty `ComputeBuffer` (deleter
    // unset, dtor is a no-op) — mirrors the current UsmHandle default-
    // ctor semantics.

    /// Allocate `bytes` of device-visible memory. Throws (backend-
    /// dependent exception type) on driver failure. Returned buffer
    /// owns the allocation and frees it via the deleter installed by
    /// this ops instance's backing allocator.
    [[nodiscard]] virtual ComputeBuffer allocate(std::size_t bytes) = 0;

    /// Synchronous host-to-device copy. Blocks until the transfer is
    /// visible to subsequent GPU dispatches. Used by loaders that need
    /// "the mmap bytes must be on the device before we return".
    ///
    /// L0 impl is a plain `std::memcpy` — USM is host-visible on the
    /// target GPU. HIP impl is a blocking `hipMemcpy(hipMemcpyHostToDevice)`.
    /// Callers should batch multi-tensor loads with a single trailing
    /// `flush()` if they need cross-tensor ordering; per-tensor this
    /// call is already ordered against subsequent kernel launches.
    virtual void uploadHostBytes(void*       deviceDst,
                                 const void* hostSrc,
                                 std::size_t bytes) = 0;

protected:
    ComputeOps() = default;
};

/**
 * RAII helper: enters an unordered-dispatch scope on construction,
 * exits on destruction. Non-copyable + non-movable to avoid dangling
 * push/pop pairings. Same shape as the L0-native
 * `runtime::UnorderedScope` — direct drop-in for callers that migrate
 * from `_ops.queue()` to `_ops`.
 */
class UnorderedScope {
public:
    explicit UnorderedScope(ComputeOps& ops) : _ops{ops} {
        _ops.pushUnorderedScope();
    }
    ~UnorderedScope() { _ops.popUnorderedScope(); }

    UnorderedScope(const UnorderedScope&)            = delete;
    UnorderedScope& operator=(const UnorderedScope&) = delete;
    UnorderedScope(UnorderedScope&&)                 = delete;
    UnorderedScope& operator=(UnorderedScope&&)      = delete;

private:
    ComputeOps& _ops;
};

} // namespace mimirmind::compute