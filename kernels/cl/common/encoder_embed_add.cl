// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

// Additive position + token-type embeddings for the BERT/RoBERTa/XLM-R
// embeddings block (cross-encoder reranker / EncoderRunner), in place:
//   x[t, d] += posTable[(t + posOffset) * hidden + d] + typeVec[d]
// for the no-padding case (position_ids = t + posOffset; XLM-R posOffset = 2).
// typeVec is the single token-type embedding row (type_vocab_size == 1),
// broadcast to all positions. CPU reference: compute::encoderEmbedAdd.
// Launch: 1D global = T * hidden.

#ifndef ENCODER_EMBED_ADD_LOCAL
#define ENCODER_EMBED_ADD_LOCAL 256
#endif

__attribute__((reqd_work_group_size(ENCODER_EMBED_ADD_LOCAL, 1, 1)))
__kernel void encoder_embed_add(
    __global       float* x,         // [T, hidden]
    __global const float* posTable,  // [maxPos, hidden]
    __global const float* typeVec,   // [hidden]
    const int             T,
    const int             hidden,
    const int             posOffset)
{
    const int gid   = (int)get_global_id(0);
    const int total = T * hidden;
    if (gid >= total) {
        return;
    }
    const int t = gid / hidden;
    const int d = gid - t * hidden;
    const size_t posIdx = (size_t)(t + posOffset) * (size_t)hidden + (size_t)d;
    x[gid] += posTable[posIdx] + typeVec[d];
}
