// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "runtime/arch/ArchBackend.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
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
    const std::int32_t*   blockTablesDev{nullptr}; // [nSeq, maxBlocksPerSeq] device
    const std::int32_t*   seqLensDev{nullptr};     // [nSeq] device: length incl. current token
    std::size_t           maxBlocksPerSeq{0};
    const std::int32_t*   startPosDev{nullptr};    // [nSeq] device: per-seq current position

    // ---- MoE routing scratch (USM) ----
    std::int32_t*         expIdxSlot{nullptr};     // [nSeq * expertUsedCount]
    float*                kwSlot{nullptr};         // [nSeq * expertUsedCount]

    // ---- GatedDeltaNet (linear layers) ----
    const std::uint8_t*   isSeqStart{nullptr};     // [nSeq]: 1 => zero SsmState this step
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
class Qwen35MoeBackend final : public ArchBackend {
public:
    Qwen35MoeBackend(const model::LlmConfig&       config,
                     const core::gguf::WeightsMap& weights,
                     const model::FusedQkvWeights* fusedQkv,
                     compute::ComputeOps&          ops,
                     compute::ComputeMatmul&       gmm,
                     runtime::OpProfiler&          opProfiler,
                     bool                          moeGroupEnabled     = true,
                     bool                          moeFusedDownEnabled = false);

    void runBlock(std::size_t   blockIdx,
                  float*        x,
                  std::size_t   T,
                  KvCache&      cache,
                  BlockBuffers& buffers,
                  bool          traceBlock0) override;

    /// This concrete CUDA serving overload shares a name with the new
    /// ArchBackend::runBlockBatched (the L0 synchronized-decode interface,
    /// which this backend does not implement — supportsBatchedDecode()
    /// stays false). Pull the base overload into scope so the concrete one
    /// below overloads rather than name-hides it (-Woverloaded-virtual).
    using ArchBackend::runBlockBatched;

    /// M-Cuda.Batch D2 — batched serving decode of one layer over nSeq
    /// sequences (one query token each). CUDA-only concrete entry point
    /// (not on the ArchBackend interface); the serving engine calls it
    /// directly. Dispatches full-attention vs GatedDeltaNet by blockIdx.
    /// `x` is [nSeq, d_model]; `s` scratch must be sized for maxT >= nSeq.
    void runBlockBatched(std::size_t              blockIdx,
                         float*                   x,
                         const BatchedDecodeCtx&  ctx,
                         BlockBuffers&            s);

    /// M-Cuda.MTP-VerifyChunked MV-c/d — batched GatedDeltaNet VERIFY layer.
    /// N slots' K+1-token verify window laid time-major in `x` (row j*N+s;
    /// position j's N slots contiguous). Batches proj/out-proj/MoE over
    /// M=N*(K+1) (each weight read once vs K+1x for the sequential path — the
    /// spec-decode weight-read amortisation). The gated-delta recurrence runs
    /// as ONE fused verify kernel (`gatedDeltaNetVerifyBatchedAsync`): the
    /// [S,S] state stays resident across the K+1 positions (no per-token global
    /// state round-trip) and the post-step state of every position is exported
    /// to `ssmExport` (per-layer slab, time-major [Kp1, maxBatch, stateElems],
    /// packed with stride N) so a partial accept picks the accepted-prefix
    /// state without re-forwarding and without the K+1 full-slab snapshots the
    /// old path used. The kernel does NOT advance the live `stateBase`, so the
    /// caller commits from `ssmExport[a]` for EVERY accept a (0..K). conv1d
    /// stays per-position (cheap) and is snapshotted into convSnap[j].
    /// gdnSeqStart is [(K+1)*maxBatch]. `ssmExport` is the whole buffer base;
    /// this call writes the [blockIdx] slab.
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

    /// Dense PagedKvPool layer index for full-attention block `blockIdx`, or
    /// `SIZE_MAX` for recurrent (GatedDeltaNet) blocks (they hold no KV).
    /// Exposes the private `_fullAttnDense` map so the serving prefill path
    /// (ServingSession::prefillSlot) can build a per-block KvCache view over a
    /// slot's contiguous pool region and reuse the single-session
    /// `runBlock`(T>1) forward on it.
    [[nodiscard]] std::size_t fullAttnDenseLayer(std::size_t blockIdx) const {
        return _fullAttnDense[blockIdx];
    }

    /// Prefill-only routing hook: when set, the routed-MoE of subsequent
    /// single-session runBlock(T>1) calls goes through the amortised batched
    /// fused-K path (runMoeFfnBatched) instead of the per-token runMoeFfn,
    /// using the caller-provided device routing scratch (expIdx/kw, each sized
    /// >= T*expertUsedCount). This turns a T-token prefill's MoE from
    /// O(T) weight re-reads into one batched pass (the TTFT lever). Set before
    /// a prefillSlot block loop, cleared (nullptr) after — nullptr keeps the
    /// historical per-token path (single-session generate is unaffected).
    void setPrefillMoeScratch(std::int32_t* expIdx, float* kw) noexcept {
        _prefillMoeExpIdx = expIdx;
        _prefillMoeKw     = kw;
    }

    /// M-Cuda.MTP — one Multi-Token-Prediction draft step. Runs the model's
    /// native nextn module (blk.<blockCount>): eh_proj(concat(RMSNorm(hidden,
    /// hnorm), RMSNorm(embed(prevTok), enorm))) -> block-<mtp> attn+MoE (own
    /// KV via `mtpCache` layer 0) -> RMSNorm(shared_head_norm) -> shared
    /// lm_head. Writes vocab logits into `logitsOut` and leaves the block
    /// output (the next-step hidden) in `ehScratch`. Does NOT commit mtpCache.
    /// Scratch (caller-owned): embScratch[d], catScratch[2*d], ehScratch[d],
    /// logitsScratch (lm-head). CUDA-only concrete entry point for the engine.
    void runMtpDraftStep(const float*   hidden,
                         std::int32_t   prevTok,
                         KvCache&       mtpCache,
                         BlockBuffers&  s,
                         float*         embScratch,
                         float*         catScratch,
                         float*         ehScratch,
                         float*         logitsOut,
                         float*         logitsScratch);

    /// M-Cuda.MTP Increment E5b — BATCHED MTP draft step over `nSeq` slots.
    /// The throughput fix: one nextn forward (eh_proj → blk.<blockCount>
    /// attn+MoE via the paged pool → shared head) over all slots instead of
    /// N sequential runMtpDraftStep calls with a flush per step. `hidden`
    /// [nSeq,d] and `prevTok` (host [nSeq]) are per-slot; `ctx` carries the
    /// nextn pool + per-slot positions (via `kvPoolLayer`). Writes
    /// `logitsOut` [nSeq,vocab] and leaves the next-step hiddens in
    /// `ehScratch` [nSeq,d]. When `skipHead` (prefill seed) the shared head
    /// is skipped (no logits). Scratch is caller-owned, sized for nSeq rows:
    /// embScratch[nSeq,d], catScratch[nSeq,2d], ehScratch[nSeq,d],
    /// tmpE/tmpH[nSeq,d]. Does NOT advance the pool position (caller commits).
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

    [[nodiscard]] bool        scalesEmbedding()   const noexcept override { return false; }
    [[nodiscard]] const char* name()              const noexcept override { return "qwen35moe"; }
    [[nodiscard]] bool        needsQGateScratch() const noexcept override { return true; }
    [[nodiscard]] bool        needsSsmScratch()   const noexcept override;

    [[nodiscard]] std::vector<std::size_t>
        kvDimPerLayer() const override;
    [[nodiscard]] std::pair<std::size_t, std::size_t>
        maxQKVDims() const override;

private:
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

    /// MoE FFN shared by full-attention (and, later, linear) layers:
    /// routed top-K softmax experts + gated shared expert. Reads its input
    /// from `moeInput` [T, d_model], writes the block-summed result into
    /// `moeAccumBuf`.
    void runMoeFfn(std::size_t         blockIdx,
                   const float*        moeInput,
                   std::size_t         T,
                   BlockBuffers&       s);

    /// M-Cuda.Batch D2c — batched (nSeq) MoE FFN for serving-class decode.
    /// Every sequence contributes exactly one token (T==1 per seq); the nSeq
    /// rows of `moeInput` [nSeq, d_model] take the place of the single-session
    /// `T` batch dimension for the row-parallel ops (router matmul, shared
    /// expert), while the routed experts go through the fused-K *_Batched
    /// kernels (grid.y = nSeq). Per-sequence top-K is computed on the host and
    /// scattered into caller-owned USM slots `expIdxSlot` / `kwSlot`, each
    /// [nSeq * expertUsedCount]. Result is summed into `s.moeAccumBuf`
    /// [nSeq, d_model], byte-identical to nSeq separate runMoeFfn(T==1) calls.
    ///
    /// Preconditions (serving path only — throws otherwise): routed experts
    /// use SEPARATE Q4_K gate/up banks + a Q5_K down bank, the fused-K kernels
    /// are available, d_model % 256 == 0 and n_ff_exp % 256 == 0. Scratch
    /// `s.moeGateCompact` must hold [nSeq * K, n_ff_exp]; `s.scoreScratch`
    /// must hold >= nSeq scalars for the shared-expert gate.
    void runMoeFfnBatched(std::size_t    blockIdx,
                          const float*   moeInput,
                          std::size_t    nSeq,
                          std::int32_t*  expIdxSlot,
                          float*         kwSlot,
                          BlockBuffers&  s);

    /// M-Cuda.MoeGroup — true grouped-by-expert MoE for the T>1 prefill.
    /// Same preconditions and result as runMoeFfnBatched (separate
    /// NVFP4_BLK/Q4_K gate/up + NVFP4_BLK/Q5_K/Q6_K down banks), but instead
    /// of the fused-K kernels (which re-read each expert weight once per token
    /// routing to it) it groups the T*K assignments by expert on the device
    /// (moe_group_build), gathers the per-expert-grouped activations
    /// (moe_gather_rows), runs ONE dense GEMM per expert over its M=count[e]
    /// rows (each weight read once per 16-row chunk), and folds the result
    /// back to token order with the router weights (moe_scatter_expert_out).
    /// Only expOffset crosses to the host (one small D2H per layer) to drive
    /// the per-expert launch bounds; routing/gather/scatter stay device-side.
    /// Numerically a different fold order than the fused-K path — validated by
    /// relative logits parity + coherence, not bit-exactness.
    /// `preferBlocked` (GD-a, decode): force the device-driven blocked grouped
    /// GEMM and never the FP4-TC branch, even when the experts carry TC sidecar
    /// pointers (additive-default decode). Decode M is ~1 row/expert, where TC's
    /// 128-row padding is pure waste; the win is weight-read amortisation on the
    /// blocked kernel.
    void runMoeFfnGrouped(std::size_t    blockIdx,
                          const float*   moeInput,
                          std::size_t    nSeq,
                          std::int32_t*  expIdxSlot,
                          float*         kwSlot,
                          BlockBuffers&  s,
                          bool           preferBlocked = false);

    /// M-Cuda.Batch D2a — batched full-attention layer (paged KV). Mirrors
    /// runFullAttentionBlock with the T dimension replaced by nSeq and the
    /// KV write/read going through the PagedKvPool + paged_attention_v1.
    /// `kvPoolLayer` decouples the pool layer from the weight-block index
    /// (default = `_fullAttnDense[blockIdx]`); the batched MTP draft passes
    /// the dedicated nextn pool layer while reading blk.<blockCount> weights.
    void runFullAttentionBlockBatched(
        std::size_t             blockIdx,
        float*                  x,
        const BatchedDecodeCtx& ctx,
        BlockBuffers&           s,
        std::size_t             kvPoolLayer =
            std::numeric_limits<std::size_t>::max());

    /// M-Cuda.Batch D2b — batched GatedDeltaNet layer. Mirrors
    /// runLinearBlock with per-sequence SsmState[nSeq] recurrent + conv
    /// state and the *_Batched conv/recurrence kernels.
    void runLinearBlockBatched(std::size_t             blockIdx,
                               float*                  x,
                               const BatchedDecodeCtx& ctx,
                               BlockBuffers&           s);

    const model::LlmConfig&       _config;
    const core::gguf::WeightsMap& _weights;
    const model::FusedQkvWeights* _fusedQkv{nullptr};
    compute::ComputeOps&          _ops;
    compute::ComputeMatmul&       _gmm;
    runtime::OpProfiler&          _op;

    bool _moeGroupEnabled;
    bool _moeFusedDownEnabled;
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
    // Opt-in (MIMIRMIND_GROUPED_MOE_DECODE_TC=1): route the *decode* routed-MoE
    // through the FP4-tensor-core grouped GEMM instead of the blocked GD-b
    // kernel. Requires the experts to carry TC sidecar banks (additive / =3
    // load). W4A4 numerics — coherence-gated, off by default.
    bool _moeGroupedDecodeTc{false};
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