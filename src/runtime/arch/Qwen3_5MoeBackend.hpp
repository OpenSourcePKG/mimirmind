// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "compute/IMoeGroupedOps.hpp"
#include "compute/IPagedAttentionOps.hpp"
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
    // The backend's FP4-tensor-core grouped-MoE ops (8.30.6 ISP-split). Only
    // reached inside moeGroupedGemmNvfp4TcAvailable()-gated paths (CUDA/Bragi);
    // mgOps() hands the interface back and throws a clear error if the backend
    // does not provide it — the same unsupported-backend failure the former
    // base-class throw-defaults gave, at one named seam.
    [[nodiscard]] compute::IMoeGroupedOps& mgOps() const;

    // The backend's paged-attention ops (8.30.6 ISP-split). Only reached on the
    // CUDA paged-KV serving path (the L0 substrate is non-paged slab); paOps()
    // hands the interface back and throws a clear error if the backend does not
    // provide it — the same unsupported-backend failure the former base-class
    // throw-defaults gave.
    [[nodiscard]] compute::IPagedAttentionOps& paOps() const;

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

    // runMoeFfnGrouped helpers (extracted verbatim; 8.30.11.4). runMoeGroupedTc
    // is the FP4-tensor-core grouped-GEMM branch (act-quant -> gate+up -> silu ->
    // down -> scatter). runMoeOeaUnionProfile is the env-gated OEA union
    // diagnostic (MIMIRMIND_MOE_UNION_PROFILE); no-op unless enabled + nSeq>1.
    void runMoeGroupedTc(BlockBuffers& s, const float* moeInput,
                         const core::gguf::GgufTensor& gateExps,
                         const core::gguf::GgufTensor& upExps,
                         const core::gguf::GgufTensor& downExps,
                         const std::int32_t* expOffset,
                         const std::int32_t* asnToRow,
                         const std::int32_t* rowSrcTok, const float* kwSlot,
                         float* moeAccumBuf, std::size_t R, std::size_t nExperts,
                         std::size_t K, std::size_t d_model, std::size_t n_ff_exp,
                         std::size_t nSeq);
    void runMoeOeaUnionProfile(const std::int32_t* expOffset, std::size_t nExperts,
                               std::size_t nSeq, std::size_t R,
                               std::size_t blockIdx);

    // runMoeFfnGrouped scalar (non-TC) grouped-GEMM branches (8.30.11.4),
    // extracted verbatim. deviceDriven = the fully device-scheduled blocked-NVFP4
    // path (moe_group_tiles -> one grouped GEMM/projection, nothing crosses to
    // the host); hostDriven = the correct-but-slower per-expert loop (one D2H of
    // the group offsets, one dense GEMM per expert). Shared compact buffers + dims
    // are one named MoeGroupedArgs struct (same swap-proofing as GdnConvArgs);
    // each helper destructures the fields it uses back into identical locals.
    struct MoeGroupedArgs {
        float*        xComp;
        float*        gateComp;
        float*        upComp;
        float*        downComp;
        std::int32_t* expOffset;
        std::size_t   R;
        std::size_t   nExperts;
        std::size_t   d_model;
        std::size_t   n_ff_exp;
        std::size_t   nSeq;
    };
    void runMoeGroupedDeviceDriven(std::size_t blockIdx, bool preferBlocked,
                                   BlockBuffers& s,
                                   const core::gguf::GgufTensor& gateExps,
                                   const core::gguf::GgufTensor& upExps,
                                   const core::gguf::GgufTensor& downExps,
                                   const MoeGroupedArgs& a);
    void runMoeGroupedHostDriven(const core::gguf::GgufTensor& gateExps,
                                 const core::gguf::GgufTensor& upExps,
                                 const core::gguf::GgufTensor& downExps,
                                 float* matmulScratch, const MoeGroupedArgs& a);

    // runLinearBlockBatched GDN gated-delta-rule recurrence stage (8.30.11.4),
    // extracted verbatim. Its ~14 inputs are passed as one named struct (not
    // positional) so the many same-type float* buffers cannot be swapped; the
    // helper destructures them back into identically-named locals, keeping the
    // moved body byte-identical.
    struct GdnRecurArgs {
        float*      stateBase;
        std::size_t stateElems;
        float*      qBuf;
        float*      kBuf;
        float*      vBuf;
        float*      alphaBuf;
        float*      betaBuf;
        float*      gateBuf;
        float*      deltaOut;
        std::size_t nRow;
        std::size_t hV;
        std::size_t S;
        std::size_t nSeq;
        bool        ragged;
    };
    void runGdnRecurrence(const BatchedDecodeCtx& ctx, BlockBuffers& s,
                          const core::gguf::GgufTensor& ssmA,
                          const core::gguf::GgufTensor& ssmDt,
                          const GdnRecurArgs& r);

    // runLinearBlockBatched causal-conv1d+silu -> q/k/v split stage (8.30.11.4),
    // extracted verbatim. Same named-struct discipline as GdnRecurArgs so the
    // many same-type float* buffers cannot be swapped; the helper destructures
    // them back into identically-named locals, keeping the moved body
    // byte-identical. `convSplitFused` stays internal to the stage.
    struct GdnConvArgs {
        float*                        convBase;
        float*                        qkvMixed;
        float*                        convInput;
        float*                        qBuf;
        float*                        kBuf;
        float*                        vBuf;
        const core::gguf::GgufTensor* convW;
        std::size_t                   nSeq;
        std::size_t                   nRow;
        std::size_t                   convStateElems;
        std::size_t                   convDim;
        std::size_t                   dConv;
        std::size_t                   stateRows;
        std::size_t                   S;
        std::size_t                   hK;
        std::size_t                   hV;
        std::size_t                   keyDim;
        float                         eps;
        bool                          ragged;
    };
    void runGdnConvAndSplit(const BatchedDecodeCtx& ctx, const GdnConvArgs& r);

    // runLinearBlockBatched GDN input-projection stage (8.30.11.4), extracted
    // verbatim: the 3-way qkv/gate/beta/alpha projection (fused-nSeq1 / fused-
    // batch / 4-matmul). Inputs go through a named GdnProjArgs struct; the four
    // OUTPUT buffers are float*& (the fused-nSeq1 branch REASSIGNS them to the
    // fused-output slices), so the helper writes them back to the caller.
    struct GdnProjArgs {
        const core::gguf::GgufTensor* qkvW;
        const core::gguf::GgufTensor* gateW;
        const core::gguf::GgufTensor* betaW;
        const core::gguf::GgufTensor* alphaW;
        float*                        normBuf;
        float*                        mmScratch;
        std::size_t                   nSeq;
        std::size_t                   nRow;
        std::size_t                   d_model;
        std::size_t                   convDim;
        std::size_t                   valueDim;
        std::size_t                   hV;
        bool                          ragged;
    };
    void runGdnProjections(std::size_t blockIdx, const GdnProjArgs& r,
                           float*& qkvMixed, float*& zBuf,
                           float*& betaBuf, float*& alphaBuf);

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
