#pragma once

#include "model/LlmConfig.hpp"
#include "runtime/UsmAllocator.hpp"
#include "runtime/UsmHandle.hpp"

#include <cstddef>

namespace mimirmind::runtime {

/**
 * Per-call transformer-block scratch in USM. Shared by every architecture
 * backend — the dense fields cover Qwen2 / Llama-family inference, the
 * trailing MoE fields are only allocated when the model is MoE
 * (currently Gemma 4 26B-A4B).
 *
 * `q_dim` / `kv_dim` are sized for the LARGEST layer the backend will
 * dispatch — Gemma 4's full-attention layers can need double the SWA
 * Q/K stride. Block buffers therefore use the backend-reported
 * `maxQKVDims()` instead of derived `headCount * headDim`.
 */
struct BlockBuffers {
    std::size_t maxT{0};
    std::size_t maxSeq{0};
    std::size_t d_model{0};
    std::size_t q_dim{0};
    std::size_t kv_dim{0};
    std::size_t ff_dim{0};

    UsmHandle qBuf;          // [maxT, q_dim]
    UsmHandle normBuf;       // [maxT, d_model]
    UsmHandle attnOut;       // [maxT, q_dim]
    UsmHandle projOut;       // [maxT, d_model]
    UsmHandle gateOut;       // [maxT, max(ff_dim, gate_up_per_expert)]
    UsmHandle upOut;         // [maxT, ff_dim]
    UsmHandle matmulScratch; // max(d_model, q_dim, ff_dim)
    UsmHandle scoreScratch;  // [maxSeq]

    // Fused QKV: staging output for the concatenated matmul, split by a
    // downstream kernel into qBuf / kSlot / vSlot. Sized for the widest
    // layer (Q + K + V heads). Only allocated when fused QKV is enabled.
    UsmHandle qkvFusedScratch; // [maxT, q_dim + 2*kv_dim]

    // M10.2 Phase 1a Commit 5 — persistent fp32 K/V staging for the Q8_0
    // KV cache path. rmsnorm_qkv + RoPE stay fp32 in registers/USM; then
    // `kv_quant_commit_q8_0` folds each row into a 32-elem Q8_0 block
    // (fp16 scale + 32 int8) inside the cache slot. Only allocated when
    // the engine is configured with KvDtype::Q8_0; zero-sized otherwise.
    // Layout: [maxT, kv_dim] fp32 each. Layer-lokal reused across the
    // forward pass (sequential runBlock), so 2 buffers cover every layer.
    UsmHandle kvKFp32Scratch;  // [maxT, kv_dim]
    UsmHandle kvVFp32Scratch;  // [maxT, kv_dim]

    // Gemma 4 Path B (MoE) scratch. Zero-sized for non-MoE blocks.
    UsmHandle moeAccumBuf;   // [maxT, d_model]   weighted sum of experts
    UsmHandle expertOutBuf;  // [maxT, d_model]   per-expert down output

    // Expert-grouping scratch (M5i.F). Compact buffers ordered by
    // expert: row i belongs to expert experts[i]. Sized for the
    // worst-case gather where every (token, top-k slot) fires:
    //   nRowsMax = maxT * expertUsedCount
    // Only allocated when the model has experts AND grouping is on.
    UsmHandle moeXCompact;    // [nRowsMax, d_model]
    UsmHandle moeGateCompact; // [nRowsMax, ffPerExpert]
    UsmHandle moeUpCompact;   // [nRowsMax, ffPerExpert]
    UsmHandle moeDownCompact; // [nRowsMax, d_model]
};

/// Allocate a single BlockBuffers. `qDimMax`/`kvDimMax` come from the
/// arch backend (`maxQKVDims()`); they must accommodate every layer's
/// Q/KV stride. MoE scratch is only allocated when the model is MoE.
BlockBuffers allocBlockBuffers(UsmAllocator&           allocator,
                               const model::LlmConfig& config,
                               std::size_t             maxT,
                               std::size_t             maxSeq,
                               std::size_t             qDimMax,
                               std::size_t             kvDimMax,
                               bool                    withFusedQkv     = false,
                               bool                    withKvFp32Scratch = false);

} // namespace mimirmind::runtime