// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "runtime/arch/ArchBackend.hpp"
#include "compute/ComputeBuffer.hpp"  // GDN-Inc 1 fused-proj weight/scratch buffers

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace mimirmind::compute {
class ComputeMatmul;
class ComputeOps;
} // namespace mimirmind::compute

namespace mimirmind::core::gguf {
class WeightsMap;
struct GgufTensor;
} // namespace mimirmind::core::gguf

namespace mimirmind::model {
class FusedQkvWeights;
struct LlmConfig;
} // namespace mimirmind::model

namespace mimirmind::runtime::serving { class PagedKvPool; }

namespace mimirmind::runtime::arch {

/**
 * M-Cuda.Batch D2 — per-decode-step batched context (nSeq active
 * sequences, exactly one query token each). Assembled once per decode
 * step by the serving engine and passed to every batched block. All
 * pointers are caller-owned and outlive the block call.
 */
struct BatchedDecodeCtx {
    std::size_t nSeq{0};

    // ---- Paged KV (full-attention layers) ----
    serving::PagedKvPool* pool{nullptr};
    const std::uint32_t*  writeBlockId{nullptr};   // [nSeq] host: block for current pos
    const std::int32_t*   writeSlot{nullptr};      // [nSeq] host: slot for current pos
    // Device copies of writeBlockId/writeSlot (updated outside a decode graph);
    // when set, the KV write scatters on-device so it is graph-capturable.
    const std::uint32_t*  writeBlockIdDev{nullptr}; // [nSeq] device
    const std::int32_t*   writeSlotDev{nullptr};    // [nSeq] device
    const std::int32_t*   blockTablesDev{nullptr}; // [nSeq, maxBlocksPerSeq] device
    const std::int32_t*   seqLensDev{nullptr};     // [nSeq] device: length incl. current token
    std::size_t           maxBlocksPerSeq{0};
    std::int32_t          maxSeqLen{0};            // host: max seq_lens over the batch
                                                   // (0 => unknown; full-attn decode
                                                   //  falls back to paged V1). Drives the
                                                   //  split-K partition count for V2.
    const std::int32_t*   startPosDev{nullptr};    // [nSeq] device: per-seq current position

    // ---- MoE routing scratch (USM) ----
    std::int32_t*         expIdxSlot{nullptr};     // [nSeq * expertUsedCount]
    float*                kwSlot{nullptr};         // [nSeq * expertUsedCount]

    // ---- GatedDeltaNet (linear layers) ----
    const std::uint8_t*   isSeqStart{nullptr};     // [nSeq]: 1 => zero SsmState this step
    // 5.21 Increment I — per-slot active mask. When set, row seq with
    // activeMask[seq]==0 is FROZEN: all its persistent-state writes (KV scatter,
    // GDN recurrence, conv tail, seqStart reset) are skipped so its state stays
    // byte-identical. Enables a prefilling/parked slot to coexist in the batch
    // without corruption. nullptr => all rows active (bit-identical legacy path).
    // Mask DOMINATES isSeqStart (freeze wins over zero). `activeMask` = device
    // pointer (kernels); `activeMaskHost` = host copy (host-side guard loops).
    const std::uint8_t*   activeMask{nullptr};     // [nSeq] device
    const std::uint8_t*   activeMaskHost{nullptr}; // [nSeq] host

    // ---- 5.21 Increment III — ragged (varlen) mixed prefill+decode ----
    // When seqTDev != nullptr the batched forward runs the RAGGED path: slot seq
    // carries seqT[seq] tokens (T=1 decode rows OR T=chunk prefill rows), all
    // packed into nRow = sum(seqT) activation rows. Every M-op runs over nRow;
    // per-token rope positions come from ropePosDev; the GDN conv/recurrence use
    // seqOffDev (token offset = prefix-sum seqT) + convInOffDev (prefix-sum of
    // seqT+dConv-1); attention branches per row (T=1 -> paged decode-V2, T>1 ->
    // paged causal prefill with queryOff=seqOffDev, startPos=startPosDev). The
    // per-token KV targets reuse writeBlockIdDev/writeSlotDev sized [nRow].
    // nullptr/0 => the T=1 decode path (bit-identical legacy).
    std::size_t           nRow{0};                 // total tokens (0 => == nSeq)
    const std::int32_t*   seqTDev{nullptr};        // [nSeq] device: tokens per slot
    const std::int32_t*   seqTHost{nullptr};       // [nSeq] host copy (host loops)
    const std::int32_t*   seqOffDev{nullptr};      // [nSeq] device: token offset
    const std::int32_t*   convInOffDev{nullptr};   // [nSeq] device: conv-input offset
    const std::int32_t*   ropePosDev{nullptr};     // [nRow] device: per-token rope pos
    std::size_t           maxSeqT{0};              // host: max(seqT) (attn/conv grid)

    // 5.21-III HYBRID ATTENTION — in a TRUE mixed prefill+decode forward, route
    // seqT==1 (decode) rows through the split-K paged decode-V2 kernel instead of
    // the O(seq_len) prefill-causal path (which is kept only for the seqT>1
    // prefill rows). Built once per forward by ServingSession::runVarlenPrefill
    // when the batch holds BOTH row classes. hybDecodeCount==0 => plain ragged
    // (all rows via prefill-causal), bit-identical to the non-hybrid mixed step.
    // Prefill-causal skips the decode rows automatically: hybSeqTPrefillDev is
    // seqTDev with the decode slots' entries set to 0 (its `pq >= seqT` guard).
    std::size_t           hybDecodeCount{0};              // D = #seqT==1 rows
    const std::int32_t*   hybDecodeRowMapDev{nullptr};    // [D] query rows (nRow layout)
    const std::int32_t*   hybDecodeSeqLensDev{nullptr};   // [D] KV length (startPos+1)
    const std::int32_t*   hybDecodeBlockTablesDev{nullptr}; // [D*maxBlocksPerSeq]
    const std::int32_t*   hybSeqTPrefillDev{nullptr};    // [nSeq] seqT, decode slots -> 0
    float*                hybQDecodeScratch{nullptr};     // [D*q_dim] gathered decode Q
    float*                hybAttnDecodeScratch{nullptr};  // [D*q_dim] decode-V2 out
    std::size_t           hybMaxDecodeSeqLen{0};          // max(startPos+1) over decode
};

/**
 * Qwen3-Next / Qwen3.5-MoE (`qwen35moe`) decoder block — the hybrid
 * linear-attention + full-attention MoE architecture (Bragi target, see
 * research/qwen3next-gated-deltanet-recon-2026-07-21).
 *
 * Layer topology is interleaved: every `full_attention_interval`-th layer
 * is a full (softmax) attention layer, the rest are GatedDeltaNet linear-
 * attention layers (`config.isRecurrentLayer(blockIdx)`). Every layer runs
 * the same MoE FFN (routed experts + gated shared expert).
 *
 * M-Q3N.2 scope: the FULL-attention layers only.
 *
 *   x -> rmsNorm(attn_norm)
 *     -> attn_q proj  = [Q | gate] per head  (splitHeadPair)
 *     -> QK-norm(Q,K) -> IMRoPE(Q,K) -> GQA attention
 *     -> attn * sigmoid(gate)  (output gate)  -> attn_output proj
 *     -> attn residual
 *     -> rmsNorm(attn_post_norm)
 *     -> MoE FFN (routed top-K softmax + gated shared expert)
 *     -> FFN residual
 *
 * The recurrent (GatedDeltaNet) layers throw until M-Q3N.3 lands the
 * conv1d + delta-rule kernels. Reaching a full-attention block end-to-end
 * therefore needs input injection (parity harness) until then.
 */
class Qwen3_5Backend : public ArchBackend {
public:
    Qwen3_5Backend(const model::LlmConfig&       config,
                     const core::gguf::WeightsMap& weights,
                     const model::FusedQkvWeights* fusedQkv,
                     compute::ComputeOps&          ops,
                     compute::ComputeMatmul&       gmm,
                     runtime::OpProfiler&          opProfiler);

    ~Qwen3_5Backend() override = default;

    void runBlock(std::size_t   blockIdx,
                  float*        x,
                  std::size_t   T,
                  KvCache&      cache,
                  BlockBuffers& buffers,
                  bool          traceBlock0) override;

    /// Dense PagedKvPool layer index for full-attention block `blockIdx`, or
    /// `SIZE_MAX` for recurrent (GatedDeltaNet) blocks (they hold no KV).
    /// Exposes the private `_fullAttnDense` map so the serving prefill path
    /// (ServingSession::prefillSlot) can build a per-block KvCache view over a
    /// slot's contiguous pool region and reuse the single-session
    /// `runBlock`(T>1) forward on it.
    [[nodiscard]] std::size_t fullAttnDenseLayer(std::size_t blockIdx) const {
        return _fullAttnDense[blockIdx];
    }

    /// M-Cuda.DFlash Phase 2 — hidden-state tap. When configured, `runBlock`
    /// copies the residual stream `x` after each tapped block into the caller's
    /// per-tap device sink, position-major [maxPos, d_model] (row = absolute
    /// sequence position `cache.length() + r`). `tapLayers[k]` is the block
    /// index whose output goes to sink `tapDst[k]`; the DFlash drafter later
    /// concatenates the 8 sinks per position. The copy is CLR-safe
    /// (`appendMemoryCopy` on the compute stream, no host op — see
    /// lesson host-ops-in-runBlock-are-CLR-landmines). Passing an empty span
    /// disables the tap (the default) — the hot path then costs one
    /// empty-vector check per block. The caller owns the sink buffers and their
    /// lifetime; each must hold at least `maxPos * d_model` floats.
    void configureHiddenTap(std::span<const std::size_t> tapLayers,
                            std::span<float* const>      tapDst);

    /// Disable the hidden tap (restore the prod-inert path).
    void clearHiddenTap() noexcept {
        _hiddenTapSlot.clear();
        _hiddenTapDst.clear();
    }

    [[nodiscard]] bool hiddenTapActive() const noexcept {
        return !_hiddenTapDst.empty();
    }

    /// GDN ReplaySSM verify capture (DFlash partial-accept fold). When
    /// configured, runLinearBlock copies each recurrent block's gated delta-rule
    /// recurrence inputs (post-L2norm k, v, gLog=gateBuf, beta) and its conv
    /// input ([conv_state | qkv_mixed], pre-conv) into the caller's per-recurrent
    /// -block device sinks, so a DFlash partial accept can fold the accepted
    /// prefix instead of re-forwarding the trunk. `recurBlocks[slot]` is the
    /// block index for sink slot `slot`; each k/v sink holds `maxT*H_v*S`, each
    /// g/b sink `maxT*H_v`, each conv sink `((d_conv-1)+maxT)*conv_dim` floats.
    /// Capture is skipped for T > maxT (prefill). Empty = prod-inert.
    void configureGdnCapture(std::span<const std::size_t> recurBlocks,
                             std::span<float* const>      kSinks,
                             std::span<float* const>      vSinks,
                             std::span<float* const>      gSinks,
                             std::span<float* const>      bSinks,
                             std::span<float* const>      convSinks,
                             std::size_t                  maxT);

    void clearGdnCapture() noexcept {
        _gdnCapSlot.clear();
        _gdnCapK.clear(); _gdnCapV.clear(); _gdnCapG.clear();
        _gdnCapB.clear(); _gdnCapConv.clear();
    }

    // GDN dims for the DFlash ReplaySSM ring allocation (defined out-of-line;
    // LlmConfig is only forward-declared in this header).
    [[nodiscard]] std::size_t gdnVHeads()     const noexcept;
    [[nodiscard]] std::size_t gdnStateSize()  const noexcept;
    [[nodiscard]] std::size_t gdnConvDim()    const noexcept;
    [[nodiscard]] std::size_t gdnConvKernel() const noexcept;
    [[nodiscard]] std::size_t layerCount()    const noexcept;
    [[nodiscard]] bool        isRecurrent(std::size_t b) const noexcept;

    [[nodiscard]] bool        scalesEmbedding()   const noexcept override { return false; }
    // FP16-KV prefill routes K/V through an fp32 staging redirect + kv_commit_fp16
    // (runFullAttentionBlock), so a raw fp32 matmul never hits the fp16 slot.
    [[nodiscard]] bool supportsFp16KvStaging() const noexcept override { return true; }
    [[nodiscard]] const char* name()              const noexcept override { return "qwen35moe"; }
    [[nodiscard]] bool        needsQGateScratch() const noexcept override { return true; }
    [[nodiscard]] bool        needsSsmScratch()   const noexcept override;

    [[nodiscard]] std::vector<std::size_t>
        kvDimPerLayer() const override;
    [[nodiscard]] std::pair<std::size_t, std::size_t>
        maxQKVDims() const override;

protected:
    /// Polymorphic FFN seam (5.20). Called by runFullAttentionBlock /
    /// runLinearBlock after the post-attention norm; the concrete subclass
    /// runs its FFN over `ffnInput` [T, d_model] and writes the result into
    /// `s.moeAccumBuf`, which the caller adds to the residual.
    /// Qwen3_5MoeBackend = routed experts + shared expert; Qwen3_5DenseBackend
    /// = plain SwiGLU.
    virtual void runFfn(std::size_t   blockIdx,
                        const float*  ffnInput,
                        std::size_t   T,
                        BlockBuffers& s) = 0;

    /// Full-attention layer forward (the "easy" 1-in-interval layers).
    /// `kvLayerIdx` selects the KvCache layer for the self-attn K/V (defaults
    /// to `blockIdx`). The MTP module reuses this block with its OWN 1-layer
    /// KvCache addressed at layer 0 while reading blk.<mtp> weights — hence
    /// the decoupling of weight-block-index from KV-layer-index.
    void runFullAttentionBlock(std::size_t   blockIdx,
                               float*        x,
                               std::size_t   T,
                               KvCache&      cache,
                               BlockBuffers& s,
                               bool          diag,
                               std::size_t   kvLayerIdx =
                                   std::numeric_limits<std::size_t>::max());

    /// GatedDeltaNet linear-attention layer forward (M-Q3N.3.2). Runs the
    /// conv1d → delta-rule recurrence → gated norm → out projection, then
    /// the shared MoE FFN. The recurrent state (s.ssmStatePtr /
    /// ssmConvStatePtr) is per-sequence (owned by SsmState) and persists
    /// across decode steps; it is zeroed at sequence start (cache.length()==0).
    void runLinearBlock(std::size_t   blockIdx,
                        float*        x,
                        std::size_t   T,
                        KvCache&      cache,
                        BlockBuffers& s,
                        bool          diag);

    const model::LlmConfig&       _config;
    const core::gguf::WeightsMap& _weights;
    const model::FusedQkvWeights* _fusedQkv{nullptr};
    compute::ComputeOps&          _ops;
    compute::ComputeMatmul&       _gmm;
    runtime::OpProfiler&          _op;

    // MoE-only config (set by the Qwen3_5MoeBackend subclass ctor; unused by
    // the dense subclass). Kept here as protected storage so the base's member
    // layout stays one place; only the routed-expert FFN path reads them.
    bool _moeGroupEnabled{true};
    bool _moeFusedDownEnabled{false};
    // M-dependent dense FP8 (env MIMIRMIND_DENSE_FP8_LOWM). Max batch (nSeq/T)
    // at which the FP8 ".fp8" dense variants are preferred over BF16; 0 = off.
    // The loader adds the variants only when this is set, so pickDense() falls
    // back to BF16 whenever they are absent.
    std::size_t _denseFp8MaxT{0};
    // GDN-Inc 1 (2026-08-09): vLLM-style fused GDN input projections. Concats
    // attn_qkv+attn_gate -> qkvz and ssm_beta+ssm_alpha -> ba into one BF16
    // weight each (runtime-cached, per block), so each dispatches as ONE fp8
    // GEMV (quantised with a single per-tensor scale, matching vLLM's
    // in_proj_qkvz / in_proj_ba) instead of 2+2 matmuls. nSeq==1 decode only
    // (pointer-split output). MIMIRMIND_GDN_PROJ_FUSE=1; off by default.
    bool                                _gdnProjFuse{false};
    std::vector<compute::ComputeBuffer> _gdnQkvzW;    // [convDim+valueDim, d_model] BF16
    std::vector<compute::ComputeBuffer> _gdnBaW;      // [2*hV, d_model] BF16
    compute::ComputeBuffer              _gdnQkvzOut;  // scratch [(convDim+valueDim)]
    compute::ComputeBuffer              _gdnBaOut;    // scratch [2*hV]
    // GDN-Inc 2 (2026-08-09): fold deltanet_gate + sigmoid(beta) into the v3
    // recurrence kernel (vLLM fused_sigmoid_gating_delta_rule_update), removing
    // two elementwise launches per linear layer. Bit-identical, no memory cost →
    // DEFAULT ON; MIMIRMIND_GDN_GATE_FUSE=0 disables.
    bool                                _gdnGateFuse{true};
    // GDN-Inc 2b (2026-08-09): fuse the 3 head-gathers + 2 q/k L2-norms after
    // conv into one launch (vLLM fused_post_conv_prep). Bit-identical, no memory
    // cost → DEFAULT ON; MIMIRMIND_GDN_PREP_FUSE=0 disables.
    bool                                _gdnPrepFuse{true};
    // M-Q3N.5: device-side MoE top-K (env MIMIRMIND_MOE_DEVICE_TOPK). When on
    // AND the fully-fused decode path applies, top-K runs on the device and
    // the host moeTopKRoute + host->USM copy are skipped (no per-layer host
    // sync). Default off until parity-gated in prod.
    bool _moeDeviceTopKEnabled{false};

    // M-Cuda.MoeGroup: route the T>1 prefill routed-MoE through the true
    // grouped-by-expert path (runMoeFfnGrouped) instead of the fused-K
    // batched path. Env MIMIRMIND_GROUPED_MOE (opt-in `=1`).
    //
    // DEFAULT OFF — the host-driven grouped path (Option 1) is correct but
    // MEASURED ~25-35% SLOWER than the fused-K batched path on GB10 (A/B on
    // xd-ki-pkg1, qwen3.6-35B-NVFP4: batched 11.3-13.8 vs grouped 15.3-17.1
    // ms/prompt-tok across 185..4337 tokens). Reading each expert weight once
    // is outweighed by the per-MoE-layer host sync (one expOffset D2H drains
    // the stream ~48x/chunk) + the ~nExperts*3 per-expert launches. The
    // fused-K path is already host-sync-free. The real win needs a
    // DEVICE-DRIVEN grouped GEMM (single kernel reading expOffset on device,
    // no D2H, collapsed launches, FP4-TC) — the M-Cuda.MoeGroup follow-up.
    bool _moeGroupedPrefill{false};
    // M-Cuda.MoeGroup Sub-Step E (Option 2): when set (env
    // MIMIRMIND_GROUPED_MOE=2), runMoeFfnGrouped drives the grouped GEMM
    // entirely on the device — moe_group_tiles builds a compact tile schedule
    // and a single moe_grouped_gemm_nvfp4blk launch per projection consumes it,
    // reading the per-tile (expert, row-range) on the device. No expOffset D2H,
    // no per-expert host loop — the shape that can beat fused-K. NVFP4_BLK
    // expert weights only (the serving format); other types fall back to the
    // host-driven loop. Opt-in, off by default.
    bool _moeGroupedDeviceDriven{false};
    // E-d.4b: FP4-tensor-core grouped prefill (MIMIRMIND_GROUPED_MOE=3).
    bool _moeGroupedTc{false};
    // M-Cuda.MoeGroup-Decode GD-a: route the batched *decode* MoE through the
    // device-driven blocked grouped GEMM instead of the per-token fused-K path
    // (env MIMIRMIND_GROUPED_MOE_DECODE=1). The decode ceiling is expert-weight
    // bandwidth — fused-K re-reads each expert per token; grouped reads it once
    // per tile. Forces the blocked branch (never TC — decode M is tiny, TC's
    // 128-row pad is pure waste); needs NVFP4_BLK banks (the additive-default
    // decode format; tc-only/other layouts fall back to fused-K via the guard).
    // DEFAULT-ON 2026-08-03: +5% decode, greedy-bit-identical to fused-K (3/3
    // prompts), 0 memory cost. MIMIRMIND_GROUPED_MOE_DECODE=0 disables.
    bool _moeGroupedDecode{true};
    // GD-c: route the *decode* routed-MoE through the FP4-tensor-core grouped
    // GEMM instead of the blocked GD-b kernel. Requires the experts to carry TC
    // sidecar banks (additive-default / =3 load); with blocked-only (=0) banks
    // absent, decode falls through to the device-driven blocked branch. W4A4
    // numerics — coherence-gated. OPT-IN, DEFAULT-OFF: the isolated D2e batched
    // -decode bench shows +28% @nSeq64 (short context), BUT the real HTTP
    // serving aggregate (conc32/64, growing KV context, continuous batching) is
    // ~3.5% SLOWER than device-driven blocked GD-b (99.3 vs 103; measured
    // 2026-08-04) — the isolated win inverts under real serving, so blocked
    // stays the serving default. Enable with MIMIRMIND_GROUPED_MOE_DECODE_TC=1.
    bool _moeGroupedDecodeTc{false};
    // Track B: run the dense shared-expert projections through the FP4-TC
    // grouped GEMM (nExp=1) at prefill (M>=kShexpTcMinM) instead of the blocked
    // kernel whose kGemmMaxM=16 loop re-streams the weight ~M/16 times (the
    // measured stage-R prefill bottleneck). Default-on; only fires when the
    // loader built the shexp TC sidecars. MIMIRMIND_SHEXP_TC=0 rolls back.
    bool _shexpTc{true};
    // Force the single-pass paged decode (V1) instead of the split-K V2 for
    // full-attention layers. A/B + rollback lever (MIMIRMIND_PAGED_V1=1).
    bool _forcePagedV1{false};
    // Route the routed-MoE decode grouped GEMM through the de-interleaved
    // uint4-coalesced kernel (nSeq==1 only). A/B (MIMIRMIND_NVFP4_DEINT=1).
    bool _useDeintMoe{false};
    // Default-on: single-user (nSeq==1) decode uses the register-staged MoE GEMV
    // (m1reg, +2-4% vs m4). MIMIRMIND_MOE_M1NB=0 disables it.
    bool _useM1nb{true};
    // 5.18.8: fuse the routed-expert gate+up projections into ONE stacked-w13
    // grouped GEMM (N=2*n_ff) on the device-driven deint decode path, replacing
    // the two separate gate/up launches. Halves gate/up launches AND doubles N
    // per launch (better SM fill on the tileM=4 M=1 kernel — the 5.18.5 dispatch/
    // occupancy residual). Mirrors the GDN in_proj concat-cache + pointer-split
    // (_gdnQkvzW). Bit-identical (same weights/math, concatenated layout + fused
    // silu split). nSeq==1 deint only. OPT-IN: MIMIRMIND_MOE_W13_FUSE=1.
    bool                                _moeW13Fuse{false};
    std::vector<compute::ComputeBuffer> _moeW13W;  // per-block [nExp][2*n_ff][d_model] blocked
    // Host mirror of the device expert-offset table (moe_group_build output),
    // read back once per grouped MoE layer to drive the per-expert launches
    // (host-driven Option 1 only; the device-driven path never reads it back).
    std::vector<std::int32_t> _groupOffsetHost;

    // Diagnostic: when MIMIRMIND_SSM_TRACE is set, log per-linear-layer
    // recurrent-state / output norms and per-block residual-stream norms
    // each forward. Localises the M-Q3N.3 length-degeneration bug (state
    // saturation) without an external reference. No-op / zero cost when off.
    bool _ssmTrace{false};

    // MIMIRMIND_SSM_DUMP=<dir>: directional per-block residual-stream dump.
    // Unlike _ssmTrace (l2+max, norm-blind), this writes the RAW last-token
    // hidden-state vector after every block to
    //   <dir>/pos<pos>-blk<blockIdx>-xout.bin   (d_model f32, little-endian)
    // so it can be diffed against an external oracle (vLLM/HF hidden_states)
    // to localise the Qwen3.6 long-gen directional drift, which the norm
    // trace is blind to. MIMIRMIND_SSM_DUMP_POS=<N> restricts dumping to the
    // single decode position N (-1 / unset = every position). No-op when off.
    bool        _ssmDump{false};
    std::string _ssmDumpDir{};
    long        _ssmDumpPos{-1};

    // M-Cuda.DFlash Phase 2 — hidden-state tap. `_hiddenTapSlot[blockIdx]` is
    // the sink index for a tapped block, or -1. `_hiddenTapDst[k]` is the
    // caller's device buffer for tap k (position-major [maxPos, d_model]). Both
    // empty => tap disabled (prod-inert default; one empty-check per block).
    std::vector<int>    _hiddenTapSlot;
    std::vector<float*> _hiddenTapDst;

    // GDN ReplaySSM verify capture (DFlash partial-accept fold). _gdnCapSlot
    // [blockIdx] = per-recurrent-block sink slot, or -1. Per-slot device sinks
    // for the recurrence inputs + conv input. Empty => prod-inert. Skipped for
    // T > _gdnCapMaxT (prefill).
    std::vector<int>    _gdnCapSlot;
    std::vector<float*> _gdnCapK, _gdnCapV, _gdnCapG, _gdnCapB, _gdnCapConv;
    std::size_t         _gdnCapMaxT{0};

    // MIMIRMIND_GDN_DUMP=<dir>: dump the GatedDeltaNet recurrence in/out tensors
    // for block MIMIRMIND_GDN_DUMP_BLK (default 0) as one file per tensor:
    //   <dir>/blk<b>-{q,k,v,glog,beta,dnet}.bin  ([T,...] f32, one prefill call)
    // to isolate the recurrence math (feed q/k/v/glog/beta to the fp64 HF
    // reference, compare dnet) from upstream conv/projections. No-op when off.
    bool        _gdnDump{false};
    std::string _gdnDumpDir{};
    std::size_t _gdnDumpBlk{0};

    // MIMIRMIND_Q8_DP4A: route the Q8_0 shared-expert GEMVs through the
    // dp4a (int8) path at T=1 decode (M-Q3N.4e). Experimental A/B toggle
    // for the perf measurement; default off.
    bool _q8Dp4a{false};

    // 5.22 OEA — Opportunistic Expert Activation (batch-aware routing). Default
    // OFF, lossy. MinShare = expert kept if shared by >= this many batch tokens;
    // MaxBatch caps OEA to decode-sized batches (prefill must not be rerouted).
    bool        _moeOeaEnabled{false};
    int         _moeOeaMinShare{2};
    std::size_t _moeOeaMaxBatch{128};

    // Chunked GatedDeltaNet prefill (K0->K1->K2, M-Q3N.4) auto-gate.
    //
    // For a prefill of length T the delta-rule step runs either the
    // T-sequential AR recurrence or the parallel chunked pipeline
    // (parity-equivalent, cuda_parity 10/10). The chunked path only wins once
    // T is large enough that the sequential dependency dominates -- at short
    // prefills it is correct but no faster (M-Q3N.4l). `_gdnChunkMinT` is the
    // smallest prefill T that switches to the chunked path; decode (T==1)
    // always uses AR.
    //
    // Resolution (see ctor), highest precedence first:
    //   MIMIRMIND_GDN_CHUNK_MIN_T=<N>  -> chunk when T >= N  (A/B sweep knob)
    //   MIMIRMIND_GDN_CHUNK[=1]        -> chunk for every prefill (T >= 2)
    //   (neither set)                  -> kGdnChunkMinTDefault
    //
    // The default is "disabled" (SIZE_MAX) until the GB10 crossover A/B fixes
    // the break-even T; prod stays on the AR path meanwhile.
    static constexpr std::size_t kGdnChunkMinTDefault =
        std::numeric_limits<std::size_t>::max();
    std::size_t _gdnChunkMinT{kGdnChunkMinTDefault};

    /// Host-side L2 norm + max|.| of a compute buffer, after a sync. Only
    /// called on the diagnostic trace path.
    void traceNorm(const char* tag, std::size_t blockIdx,
                   std::size_t pos, const float* p, std::size_t n) const;

    /// Directional dump: writes `n` raw f32 from unified-memory `p` (after a
    /// sync) to `<_ssmDumpDir>/pos<pos>-blk<blockIdx>-<tag>.bin`. Gated by
    /// MIMIRMIND_SSM_DUMP / MIMIRMIND_SSM_DUMP_POS (see `_ssmDump`). Only
    /// called on the diagnostic path.
    void traceDump(const char* tag, std::size_t blockIdx,
                   std::size_t pos, const float* p, std::size_t n) const;

    // IMRoPE dimension sections (config.ropeSections), padded to 4 int32
    // so the op always receives a valid 4-element pointer. Zeroed when the
    // model ships no sections (degenerates to plain RoPE).
    std::int32_t _ropeSections[4]{0, 0, 0, 0};

    // M-Cuda.Batch D2a — maps a global blockIdx to its dense index among
    // full-attention layers (the PagedKvPool layer index). SIZE_MAX for
    // recurrent (GatedDeltaNet) layers, which hold an SsmState instead of
    // paged KV. Built once in the ctor.
    std::vector<std::size_t> _fullAttnDense;

    // Router scratch, hoisted so steady-state runBlock does no allocation.
    std::vector<std::int32_t> _topKIdx;      // [T*K]
    std::vector<float>        _topKWeight;   // [T*K]

    // Prefill-only MoE routing scratch (device), set via setPrefillMoeScratch.
    // nullptr => per-token runMoeFfn (the historical single-session path).
    std::int32_t* _prefillMoeExpIdx{nullptr};
    float*        _prefillMoeKw{nullptr};
};

} // namespace mimirmind::runtime::arch