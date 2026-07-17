// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

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
 *   [kMin, min(positionOffset + p + 1, T_k))
 * where kMin = max(0, kMax - slidingWindow) if slidingWindow > 0, else 0.
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
 * `slidingWindow == 0` (default) = pure causal. `> 0` matches Gemma
 * SWA layers (each query sees only the last `slidingWindow` causal keys).
 *
 * Softmax is numerically stable (max-subtract). Scores are scaled by
 * `scale` before softmax; when `scale <= 0` the default 1/sqrt(headDim)
 * is used. Pass a positive value to override (e.g. Gemma 4 uses 1.0F
 * because Q was pre-scaled elsewhere).
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
                        float*       out,
                        std::size_t  slidingWindow = 0,
                        float        scale         = 0.0F);

} // namespace mimirmind::compute