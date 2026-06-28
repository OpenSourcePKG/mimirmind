#pragma once

#include <cstddef>

namespace mimirmind::compute {

/**
 * Multi-head self-attention in prefill mode (no KV-cache, all positions
 * processed at once with a causal mask). GQA-aware: Q has `nHeads`
 * heads, K and V have `nKvHeads`, and Q head `h` reads from KV head
 * `h * nKvHeads / nHeads`.
 *
 * Tensor layouts (all row-major, contiguous):
 *   q:   [seqLen, nHeads,    headDim]
 *   k:   [seqLen, nKvHeads,  headDim]
 *   v:   [seqLen, nKvHeads,  headDim]
 *   out: [seqLen, nHeads,    headDim]
 *
 * Scratch buffer: `seqLen` floats — reused per (head, query-position)
 * for the attention score row.
 *
 * Math per (query position p, query head h_q, dimension d):
 *
 *   scale = 1 / sqrt(headDim)
 *   h_kv  = (h_q * nKvHeads) / nHeads
 *
 *   for kk in 0..seqLen:
 *     scores[kk] = (kk <= p) ? dot(q[p, h_q], k[kk, h_kv]) * scale
 *                            : -inf      // (effectively masked)
 *   scores = softmax(scores)
 *
 *   out[p, h_q, d] = sum_{kk <= p} scores[kk] * v[kk, h_kv, d]
 */
void multiHeadAttentionPrefill(const float* q,
                               const float* k,
                               const float* v,
                               std::size_t  seqLen,
                               std::size_t  nHeads,
                               std::size_t  nKvHeads,
                               std::size_t  headDim,
                               float*       scratch,
                               float*       out);

} // namespace mimirmind::compute