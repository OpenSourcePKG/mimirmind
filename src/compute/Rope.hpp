#pragma once

#include <cstddef>

namespace mimirmind::compute {

/**
 * Rotary positional embedding (RoPE), llama.cpp "non-interleaved"
 * (split) layout used by every Llama / Qwen / Gemma family model.
 *
 * For each token at absolute position p in [startPos, startPos+seqLen),
 * each head, and each dim-pair (i, i + headDim/2) with i in [0, headDim/2):
 *
 *   theta = p * base^(-2i / headDim)
 *   c = cos(theta), s = sin(theta)
 *
 *   x'[i]              = x[i] * c  -  x[i + headDim/2] * s
 *   x'[i + headDim/2]  = x[i] * s  +  x[i + headDim/2] * c
 *
 * Apply to Q and K independently (V is not rotated). headDim must be
 * even. Operates in place on a `[seqLen, numHeads, headDim]` F32 buffer
 * in row-major order.
 *
 * `startPos` is the absolute position of the first row — set it to the
 * current KV-cache fill level once we have a cache.
 */
void applyRopeInPlace(float*        x,
                      std::size_t   seqLen,
                      std::size_t   numHeads,
                      std::size_t   headDim,
                      std::size_t   startPos,
                      float         base);

} // namespace mimirmind::compute