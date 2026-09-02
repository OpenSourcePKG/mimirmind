// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "runtime/arch/Qwen3_5Backend.hpp"

namespace mimirmind::runtime::arch {

/**
 * Qwen3.5-MoE (`qwen3_5_moe`) — the routed-expert FFN variant of the shared
 * Qwen3_5Backend hybrid architecture (GatedDeltaNet linear attention +
 * periodic full attention). First consumer: Qwen3.6-35B-A3B (Bragi target).
 *
 * Over the shared base it adds:
 *   - runFfn = routed top-K softmax experts + gated shared expert (the FFN
 *     seam the base calls after the post-attention norm),
 *   - the serving-class batched decode (runBlockBatched + the batched
 *     full-attention / GatedDeltaNet layers),
 *   - the MTP draft (native nextn module) + the DFlash verify layer.
 *
 * The dense variant of the same arch is Qwen3_5DenseBackend (SwiGLU FFN, no
 * experts) — it shares this base but not these routed-expert / serving paths.
 *
 * NOTE (roadmap 5.27): no longer `final` — Qwen4ExpBackend (Qwen3.8-Flash-Next,
 * `qwen4_exp`) derives from this to reuse the whole routed-expert + serving +
 * MTP stack, adding only Hyper-Connections + PLE n-gram embeddings on top.
 */
class Qwen3_5MoeBackend : public Qwen3_5Backend {
public:
    Qwen3_5MoeBackend(const model::LlmConfig&       config,
                      const core::gguf::WeightsMap& weights,
                      const model::FusedQkvWeights* fusedQkv,
                      compute::ComputeOps&          ops,
                      compute::ComputeMatmul&       gmm,
                      runtime::OpProfiler&          opProfiler,
                      bool                          moeGroupEnabled     = true,
                      bool                          moeFusedDownEnabled = false);

    /// This concrete CUDA serving overload shares a name with the base
    /// ArchBackend::runBlockBatched (the L0 synchronized-decode interface,
    /// which this backend does not implement). Pull the base overload into
    /// scope so the concrete one below overloads rather than name-hides it.
    using ArchBackend::runBlockBatched;

    /// M-Cuda.Batch D2 — batched serving decode of one layer over nSeq
    /// sequences (one query token each). Dispatches full-attention vs
    /// GatedDeltaNet by blockIdx. `x` is [nSeq, d_model].
    void runBlockBatched(std::size_t              blockIdx,
                         float*                   x,
                         const BatchedDecodeCtx&  ctx,
                         BlockBuffers&            s);

    /// M-Cuda.MTP-VerifyChunked — batched GatedDeltaNet VERIFY layer for the
    /// spec-decode weight-read amortisation (see the .cpp for the state-export
    /// contract). `ssmExport` writes the [blockIdx] slab.
    void runLinearBlockVerify(std::size_t         blockIdx,
                              float*              x,
                              std::size_t         N,
                              std::size_t         Kp1,
                              std::int32_t*       expIdxSlot,
                              float*              kwSlot,
                              const std::uint8_t* gdnSeqStart,
                              std::size_t         maxBatch,
                              float*              ssmExport,
                              float* const*       convSnap,
                              BlockBuffers&       s);

    /// Prefill-only routing hook: when set, the routed-MoE of subsequent
    /// single-session runBlock(T>1) calls goes through the amortised batched
    /// fused-K path instead of the per-token path. Cleared (nullptr) after a
    /// prefillSlot block loop.
    void setPrefillMoeScratch(std::int32_t* expIdx, float* kw) noexcept {
        _prefillMoeExpIdx = expIdx;
        _prefillMoeKw     = kw;
    }

    /// M-Cuda.MTP — one Multi-Token-Prediction draft step (native nextn module
    /// blk.<blockCount>). Writes vocab logits into `logitsOut`, leaves the
    /// next-step hidden in `ehScratch`. Does NOT commit mtpCache.
    void runMtpDraftStep(const float*   hidden,
                         std::int32_t   prevTok,
                         KvCache&       mtpCache,
                         BlockBuffers&  s,
                         float*         embScratch,
                         float*         catScratch,
                         float*         ehScratch,
                         float*         logitsOut,
                         float*         logitsScratch);

    /// M-Cuda.MTP E5b — BATCHED MTP draft step over `nSeq` slots (one nextn
    /// forward for all slots). `skipHead` skips the shared head (prefill seed).
    void runMtpDraftStepBatched(const float*            hidden,
                                const std::int32_t*     prevTok,
                                std::size_t             nSeq,
                                const BatchedDecodeCtx& ctx,
                                std::size_t             kvPoolLayer,
                                BlockBuffers&           s,
                                float*                  embScratch,
                                float*                  catScratch,
                                float*                  ehScratch,
                                float*                  tmpE,
                                float*                  tmpH,
                                float*                  logitsOut,
                                float*                  logitsScratch,
                                bool                    skipHead);

protected:
    /// FFN seam: routed top-K experts + gated shared expert.
    void runFfn(std::size_t   blockIdx,
                const float*  moeInput,
                std::size_t   T,
                BlockBuffers& s) override;

    /// MoE FFN — routed top-K softmax experts + gated shared expert (per-token).
    void runMoeFfn(std::size_t         blockIdx,
                   const float*        moeInput,
                   std::size_t         T,
                   BlockBuffers&       s);

    /// Batched (nSeq) MoE FFN for serving-class decode (fused-K *_Batched).
    void runMoeFfnBatched(std::size_t    blockIdx,
                          const float*   moeInput,
                          std::size_t    nSeq,
                          std::int32_t*  expIdxSlot,
                          float*         kwSlot,
                          BlockBuffers&  s);

    /// True grouped-by-expert MoE (device-driven / FP4-TC). `preferBlocked`
    /// (GD-a decode) forces the blocked grouped GEMM.
    void runMoeFfnGrouped(std::size_t    blockIdx,
                          const float*   moeInput,
                          std::size_t    nSeq,
                          std::int32_t*  expIdxSlot,
                          float*         kwSlot,
                          BlockBuffers&  s,
                          bool           preferBlocked = false);

    /// Track B — one shared-expert projection through the CUTLASS block-scaled
    /// NVFP4 tensor-core GEMM as a single group (nExp=1).
    void sharedExpertTcGemm(std::size_t   N,
                            std::size_t   K,
                            const float*  X,
                            std::size_t   M,
                            const void*   wNib,
                            const void*   wSfb,
                            const float*  wGlob,
                            float*        Y,
                            BlockBuffers& s);

    /// M-Cuda.Batch D2a — batched full-attention layer (paged KV).
    void runFullAttentionBlockBatched(
        std::size_t             blockIdx,
        float*                  x,
        const BatchedDecodeCtx& ctx,
        BlockBuffers&           s,
        std::size_t             kvPoolLayer =
            std::numeric_limits<std::size_t>::max());

    /// M-Cuda.Batch D2b — batched GatedDeltaNet layer.
    void runLinearBlockBatched(std::size_t             blockIdx,
                               float*                  x,
                               const BatchedDecodeCtx& ctx,
                               BlockBuffers&           s);
};

} // namespace mimirmind::runtime::arch
