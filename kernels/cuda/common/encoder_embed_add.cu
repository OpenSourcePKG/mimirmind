// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Additive position + token-type embeddings for the BERT/RoBERTa/XLM-R
// embeddings block (EncoderRunner / cross-encoder reranker), in place:
//
//   x[t, d] += posTable[(t + posOffset) * hidden + d] + typeVec[d]
//
// No-padding case: position_ids = t + posOffset (XLM-R posOffset = 2). The
// single token-type row (type_vocab_size == 1) is broadcast to all positions.
// Word embeddings are already in x; LayerNorm follows. One thread per element.
//
// Launch: grid( ceil(T*hidden / LOCAL) ), block( LOCAL ).

#include <cuda_runtime.h>

#ifndef ENCODER_EMBED_ADD_LOCAL
#define ENCODER_EMBED_ADD_LOCAL 256
#endif

extern "C" __global__ __launch_bounds__(ENCODER_EMBED_ADD_LOCAL)
void encoder_embed_add(
          float* __restrict__ x,         // [T, hidden] in/out
    const float* __restrict__ posTable,  // [maxPos, hidden]
    const float* __restrict__ typeVec,   // [hidden]
    const int                 T,
    const int                 hidden,
    const int                 posOffset)
{
    const long idx   = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long total = static_cast<long>(T) * static_cast<long>(hidden);
    if (idx >= total) {
        return;
    }
    const int  t      = static_cast<int>(idx / hidden);
    const int  d      = static_cast<int>(idx % hidden);
    const long posRow = static_cast<long>(t + posOffset) * static_cast<long>(hidden);
    x[idx] += posTable[posRow + d] + typeVec[d];
}
