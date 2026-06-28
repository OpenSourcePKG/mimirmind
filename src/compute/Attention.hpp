#pragma once

#include <cstddef>

namespace mimirmind::compute {

/**
 * Multi-head self-attention, GQA-aware, causal. Unified entry point for
 * both prefill and decode modes.
 *
 * Layouts (all row-major, contiguous):
 *   q:   [T_q, nHeads,    headDim]
 *   k:   [T_k, nKvHeads,  headDim]    — read-only, may be the KV cache
 *   v:   [T_k, nKvHeads,  headDim]    — read-only, may be the KV cache
 *   out: [T_q, nHeads,    headDim]
 *
 * Position math: query row p (0..T_q) corresponds to absolute position
 * `positionOffset + p`. It attends to keys at positions
 *   [0, min(positionOffset + p + 1, T_k))
 *
 * Modes by parameter combination:
 *   - Prefill:  T_q == T_k, positionOffset == 0
 *               (q,k,v all from this forward pass)
 *   - Decode:   T_q == 1, T_k == cache_length + 1,
 *               positionOffset == cache_length
 *               (q from this step; k,v from cache base pointer)
 *
 * scratch: T_k floats, reused per (head, query-position) for the score row.
 *
 * GQA: query head h_q reads from KV head (h_q * nKvHeads) / nHeads.
 *
 * Softmax is numerically stable (max-subtract). Scores are scaled by
 * 1/sqrt(headDim) before softmax.
 */
void multiHeadAttention(const float* q,
                        const float* k,
                        const float* v,
                        std::size_t  T_q,
                        std::size_t  T_k,
                        std::size_t  nHeads,
                        std::size_t  nKvHeads,
                        std::size_t  headDim,
                        std::size_t  positionOffset,
                        float*       scratch,
                        float*       out);

} // namespace mimirmind::compute