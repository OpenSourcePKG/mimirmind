// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "compute/ComputeBuffer.hpp"
#include "model/LlmConfig.hpp"

#include <cstddef>
#include <cstdint>

namespace mimirmind::compute {
class ComputeOps;
}

namespace mimirmind::runtime {

using ::mimirmind::compute::ComputeBuffer;

/**
 * Per-call transformer-block scratch. Shared by every architecture
 * backend — the dense fields cover Qwen2 / Llama-family inference, the
 * trailing MoE fields are only allocated when the model is MoE
 * (currently Gemma 4 26B-A4B).
 *
 * `q_dim` / `kv_dim` are sized for the LARGEST layer the backend will
 * dispatch — Gemma 4's full-attention layers can need double the SWA
 * Q/K stride. Block buffers therefore use the backend-reported
 * `maxQKVDims()` instead of derived `headCount * headDim`.
 *
 * Schritt 3c.2 of the HW-abstraction consumer migration: each field is
 * a `compute::ComputeBuffer` value member (backend-neutral RAII buffer),
 * allocated via `compute::ComputeOps::allocate()`. Layout and lifetime
 * stay identical to the previous `core::l0::UsmHandle` version —
 * only the type changes, so downstream code sees the same
 * `.get<T>()` / `.bytes()` accessors.
 */
struct BlockBuffers {
    std::size_t maxT{0};
    std::size_t maxSeq{0};
    std::size_t d_model{0};
    std::size_t q_dim{0};
    std::size_t kv_dim{0};
    std::size_t ff_dim{0};

    ComputeBuffer qBuf;          // [maxT, q_dim]
    ComputeBuffer normBuf;       // [maxT, d_model]
    ComputeBuffer attnOut;       // [maxT, q_dim]
    ComputeBuffer projOut;       // [maxT, d_model]
    ComputeBuffer gateOut;       // [maxT, max(ff_dim, gate_up_per_expert)]
    ComputeBuffer upOut;         // [maxT, ff_dim]
    ComputeBuffer matmulScratch; // max(d_model, q_dim, ff_dim)
    ComputeBuffer scoreScratch;  // [maxSeq]

    // Fused QKV: staging output for the concatenated matmul, split by a
    // downstream kernel into qBuf / kSlot / vSlot. Sized for the widest
    // layer (Q + K + V heads). Only allocated when fused QKV is enabled.
    ComputeBuffer qkvFusedScratch; // [maxT, q_dim + 2*kv_dim]

    // Qwen3-Next full-attention: the `attn_q` weight fuses the query and a
    // per-head output gate, so its matmul output is [maxT, 2*q_dim]
    // ([Q_h|gate_h] per head). `qGateFused` receives that matmul;
    // `splitHeadPairAsync` de-interleaves it into `qBuf` (Q) and
    // `gateScratch` (the gate, applied post-attention). Both zero-sized
    // unless the arch reports `needsQGateScratch()`.
    ComputeBuffer qGateFused;   // [maxT, 2*q_dim]
    ComputeBuffer gateScratch;  // [maxT, q_dim]

    // M10.2 Phase 1a Commit 5 — persistent fp32 K/V staging for the Q8_0
    // KV cache path. rmsnorm_qkv + RoPE stay fp32 in registers/USM; then
    // `kv_quant_commit_q8_0` folds each row into a 32-elem Q8_0 block
    // (fp16 scale + 32 int8) inside the cache slot. Only allocated when
    // the engine is configured with KvDtype::Q8_0; zero-sized otherwise.
    // Layout: [maxT, kv_dim] fp32 each. Layer-lokal reused across the
    // forward pass (sequential runBlock), so 2 buffers cover every layer.
    ComputeBuffer kvKFp32Scratch;  // [maxT, kv_dim]
    ComputeBuffer kvVFp32Scratch;  // [maxT, kv_dim]

    // Gemma 4 Path B (MoE) scratch. Zero-sized for non-MoE blocks.
    ComputeBuffer moeAccumBuf;   // [maxT, d_model]   weighted sum of experts
    ComputeBuffer expertOutBuf;  // [maxT, d_model]   per-expert down output

    // Expert-grouping scratch (M5i.F). Compact buffers ordered by
    // expert: row i belongs to expert experts[i]. Sized for the
    // worst-case gather where every (token, top-k slot) fires:
    //   nRowsMax = maxT * expertUsedCount
    // Only allocated when the model has experts AND grouping is on.
    ComputeBuffer moeXCompact;    // [nRowsMax, d_model]
    ComputeBuffer moeGateCompact; // [nRowsMax, ffPerExpert]
    ComputeBuffer moeUpCompact;   // [nRowsMax, ffPerExpert]
    ComputeBuffer moeW13Compact;  // [nRowsMax, 2*ffPerExpert] (5.18.8 fused gate+up out)
    ComputeBuffer moeDownCompact; // [nRowsMax, d_model]

    // M-Cuda.MoeGroup — device token-grouping build outputs (moe_group_build).
    ComputeBuffer moeGroupOffset; // [nExperts+1] int32 exclusive prefix sum
    ComputeBuffer moeGroupRowTok; // [nRowsMax]   int32 row -> source token
    ComputeBuffer moeGroupRowKw;  // [nRowsMax]   f32   row -> router weight
    ComputeBuffer moeGroupAsnRow; // [nRowsMax]   int32 assignment -> row (inverse)

    // M-Cuda.MoeGroup Sub-Step E — device-driven grouped GEMM tile schedule
    // (moe_group_tiles). Sized to the static upper bound maxTiles =
    // ceil(nRowsMax/16) + nExperts so the grouped-GEMM grid.y never reads a
    // device value. Only allocated on the device-driven grouped path.
    ComputeBuffer moeGroupTileExpert; // [maxTiles] int32 expert id / -1
    ComputeBuffer moeGroupTileRow0;   // [maxTiles] int32 tile start row
    ComputeBuffer moeGroupTileRows;   // [maxTiles] int32 tile row count
    ComputeBuffer moeGroupTileCount;  // [1]        int32 real tile count

    // M-Cuda.MoeGroup Sub-Step E-d — FP4-tensor-core grouped prefill scratch.
    // Per-BlockBuffers (i.e. per-slot / per-forward), so concurrent prefills
    // never share it — the precondition for MIMIRMIND_GROUPED_MOE=3 default-on.
    // Lazily grown on first TC use to maxPad = nRowsMax + nExperts*128 (padding
    // each expert to a 128-row tile); zero-sized on every non-TC path.
    ComputeBuffer moeTcPadOffset;    // [nExperts+1]        int32
    ComputeBuffer moeTcContigToPad;  // [nRowsMax]          int32
    ComputeBuffer moeTcPadAsn;       // [nRowsMax]          int32
    ComputeBuffer moeTcXPad;         // [maxPad, d_model]   f32
    ComputeBuffer moeTcGatePad;      // [maxPad, ffPerExpert] f32
    ComputeBuffer moeTcUpPad;        // [maxPad, ffPerExpert] f32
    ComputeBuffer moeTcDownPad;      // [maxPad, d_model]   f32
    ComputeBuffer moeTcABank;        // [maxPad, d_model/2] u8 (E2M1 nibbles)
    ComputeBuffer moeTcSfaBank;      // swizzled SFA over d_model
    ComputeBuffer moeTcABank2;       // [maxPad, ffPerExpert/2] u8
    ComputeBuffer moeTcSfaBank2;     // swizzled SFA over ffPerExpert
    ComputeBuffer moeTcBanksScratch; // CUTLASS per-group arrays + workspace

    // Track B: shared-expert FP4-TC dense GEMM scratch (single group, nExp=1).
    // At prefill (M>=64) each shexp projection runs through the CUTLASS grouped
    // GEMM as one padded group instead of the dense blocked kernel, whose
    // kGemmMaxM=16 loop re-streams the whole shexp weight ~M/16 times (the
    // measured stage-R prefill bottleneck). Lazily grown to padM=round_up(M,128)
    // and Kmax/Nmax over the gate/up (K=d_model,N=ffShexp) and down
    // (K=ffShexp,N=d_model) shapes. Zero-sized on the decode/blocked path.
    ComputeBuffer shexpTcOffsets;    // [4] int32 = {0, M, 0, padM}
    ComputeBuffer shexpTcABank;      // [padM, Kmax/2]  u8 (E2M1 nibbles)
    ComputeBuffer shexpTcSfaBank;    // swizzled SFA over Kmax
    ComputeBuffer shexpTcOutPad;     // [padM, Nmax]    f32
    ComputeBuffer shexpTcBanksScratch;
    std::int32_t  shexpTcOffM{-1};   // cached M in shexpTcOffsets (offsets are a
                                     // pure fn of M, constant across a forward's
                                     // blocks -> upload once per forward)

    // Q8_0 dp4a decode path (M-Q3N.4e): int8-quantized activation row +
    // per-row scale for xQuantI8Async -> matmulDp4aAsync.
    ComputeBuffer xqI8;      // [max(d_model, ffScratch)] int8
    ComputeBuffer xScaleI8;  // [maxSeq] f32

    // Qwen3-Next GatedDeltaNet linear-layer scratch (M-Q3N.3.2). Only
    // allocated for hybrid-recurrent models. Sizes derive from the SSM
    // hyperparameters: conv_dim = ssmConvDim(), value_dim = ssmInnerSize
    // (= H_v*S), H_v = ssmNumVHeads(), S = ssmStateSize, d_conv =
    // ssmConvKernel. These are all transient per-forward scratch; the
    // persistent recurrent state lives in a per-sequence SsmState and is
    // reached via the ssmStatePtr / ssmConvStatePtr pointers below.
    ComputeBuffer ssmQkvMixed;   // [maxT, conv_dim]  (also reused as conv out)
    ComputeBuffer ssmConvInput;  // [(d_conv-1)+maxT, conv_dim]
    ComputeBuffer ssmZ;          // [maxT, value_dim]  (output gate z)
    ComputeBuffer ssmQ;          // [maxT, value_dim]  (also reused as norm buf)
    ComputeBuffer ssmK;          // [maxT, value_dim]
    ComputeBuffer ssmV;          // [maxT, value_dim]
    ComputeBuffer ssmDeltaOut;   // [maxT, value_dim]  (delta-net output)
    ComputeBuffer ssmAlpha;      // [maxT, H_v]
    ComputeBuffer ssmBeta;       // [maxT, H_v]
    ComputeBuffer ssmGate;       // [maxT, H_v]        (gLog)
    // Persistent per-linear-layer recurrent state. This is NOT owned here
    // anymore — it lives in a per-sequence SsmState object (owned by
    // InferenceEngine, lifecycle like KvCache) so it survives BlockBuffers
    // reallocations and can become one-per-sequence for multi-tenant
    // serving. The engine binds these raw pointers after each BlockBuffers
    // (re)allocation. Indexed by blockIdx:
    //   ssmStatePtr[blockIdx * stateElemsPerLayer]      = delta-net [S,S] state
    //   ssmConvStatePtr[blockIdx * convStateElemsPerLayer] = rolling conv tail
    // nullptr for non-recurrent models (never dereferenced there).
    float* ssmStatePtr     = nullptr;  // -> SsmState::statePtr()
    float* ssmConvStatePtr = nullptr;  // -> SsmState::convStatePtr()
    // Allocated sequence-width of the SsmState slab bound above. The
    // per-layer stride is `blockIdx * (ssmSlabNSeq * elemsPerLayer)` — it
    // MUST use the slab's allocated nSeq, NOT the runtime batch nSeq, or a
    // continuous batch whose active count is smaller than the allocation
    // reads the wrong layer's state (M-Cuda.Batch D2e.2). Bound alongside
    // ssmStatePtr; defaults to 1 for the single-sequence path.
    std::size_t ssmSlabNSeq = 1;  // -> SsmState::nSeq()

    // Chunked-prefill (T>1) scratch (M-Q3N.4 integration): K0 cumgate output
    // gCum [maxT, H_v] and K1 triangular-inverse a0 [nChunks, H_v, C, C]
    // (C=64). Only used on the T>1 chunked path; the AR decode path ignores them.
    ComputeBuffer ssmGCum;
    ComputeBuffer ssmA0;

    // M-MoE.Fused-Decode — per-layer routing scratches for the fused-K
    // down kernel. The command queue records dispatches lazily, so the
    // caller cannot reuse a single K-sized scratch across layers (the
    // next layer's write would corrupt the earlier layer's captured
    // pointer). Stride is `expertUsedCount` per layer; total size is
    // `blockCount * expertUsedCount`. Only allocated for MoE models.
    ComputeBuffer moeExpIdxScratch;   // [blockCount, expertUsedCount] int32
    ComputeBuffer moeKwScratch;       // [blockCount, expertUsedCount] fp32
};

/// Allocate a single BlockBuffers. `qDimMax`/`kvDimMax` come from the
/// arch backend (`maxQKVDims()`); they must accommodate every layer's
/// Q/KV stride. MoE scratch is only allocated when the model is MoE.
BlockBuffers allocBlockBuffers(compute::ComputeOps&    ops,
                               const model::LlmConfig& config,
                               std::size_t             maxT,
                               std::size_t             maxSeq,
                               std::size_t             qDimMax,
                               std::size_t             kvDimMax,
                               bool                    withFusedQkv      = false,
                               bool                    withKvFp32Scratch = false,
                               bool                    withQGate         = false,
                               bool                    withSsm           = false,
                               bool                    perSeqConvInput   = false);

} // namespace mimirmind::runtime