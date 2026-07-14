// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstddef>
#include <cstdint>

namespace mimirmind::compute {

/**
 * MoE top-K expert routing: per-token softmax → partial sort → top-K
 * weight renormalisation. Mirrors llama.cpp's `LLM_FFN_GATING_SOFTMAX`
 * with `norm_topk_prob = true` (Gemma 4 26B-A4B uses this).
 *
 * Inputs:
 *   logits:   [T, nExperts]  row-major F32 expert scores (un-normalised).
 *   T:        number of tokens to route.
 *   nExperts: total experts available.
 *   topK:     experts kept per token; must satisfy 1 <= topK <= nExperts.
 *
 * Outputs (caller-allocated):
 *   outIdx:    [T, topK]  expert indices kept, descending probability.
 *   outWeight: [T, topK]  routing weights, **renormalised** so they sum
 *                         to ~1 per token (ties in probability may yield
 *                         a different but valid index order).
 *
 * Math per token t:
 *   p = softmax(logits[t])              (max-subtract, double sum for stability)
 *   keep = indices of the topK largest p (partial_sort, descending)
 *   w[k] = p[keep[k]] / sum_k p[keep[k]]
 *
 * Pure CPU. Allocates two small per-call vectors (size nExperts) on the
 * stack-equivalent heap for the working probability + index arrays.
 */
void moeTopKRoute(const float*  logits,
                  std::size_t   T,
                  std::size_t   nExperts,
                  std::size_t   topK,
                  std::int32_t* outIdx,
                  float*        outWeight);

} // namespace mimirmind::compute