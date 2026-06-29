#pragma once

#include "model/LlmConfig.hpp"
#include "runtime/UsmAllocator.hpp"
#include "runtime/UsmHandle.hpp"

#include <cstddef>

namespace mimirmind::runtime {

/**
 * Per-call transformer-block scratch in USM. Shared by every architecture
 * backend — the dense fields cover Qwen2 / Llama-family inference, the
 * trailing MoE fields are only allocated when `config.expertCount > 0`
 * (currently Gemma 4 26B-A4B).
 *
 * Sized for the largest expected T (prefill width) so the same buffer is
 * reused across prefill + every decode step without reallocation.
 */
struct BlockBuffers {
    std::size_t maxT{0};
    std::size_t maxSeq{0};
    std::size_t d_model{0};
    std::size_t q_dim{0};
    std::size_t ff_dim{0};

    UsmHandle qBuf;          // [maxT, q_dim]
    UsmHandle normBuf;       // [maxT, d_model]
    UsmHandle attnOut;       // [maxT, q_dim]
    UsmHandle projOut;       // [maxT, d_model]
    UsmHandle gateOut;       // [maxT, max(ff_dim, gate_up_per_expert)]
    UsmHandle upOut;         // [maxT, ff_dim]
    UsmHandle matmulScratch; // max(d_model, q_dim, ff_dim)
    UsmHandle scoreScratch;  // [maxSeq]

    // Gemma 4 Path B (MoE) scratch. Zero-sized for non-MoE blocks.
    UsmHandle moeAccumBuf;   // [maxT, d_model]   weighted sum of experts
    UsmHandle expertOutBuf;  // [maxT, d_model]   per-expert down output
};

/// Allocate a single BlockBuffers sized for `maxT` tokens and `maxSeq`
/// KV cache rows. The MoE scratch is only allocated when the model is
/// MoE (config.expertCount > 0) — dense models pay no overhead.
BlockBuffers allocBlockBuffers(UsmAllocator&           allocator,
                               const model::LlmConfig& config,
                               std::size_t             maxT,
                               std::size_t             maxSeq);

} // namespace mimirmind::runtime