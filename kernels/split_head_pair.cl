// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

// De-interleave a per-head pair of head-dim blocks into two contiguous
// buffers. Qwen3-Next's full-attention layers fuse the query projection
// and a per-head output gate into a single `attn_q` weight whose output
// is, per token, laid out as
//   [ Q_h0 | gate_h0 | Q_h1 | gate_h1 | ... ]   (head stride = 2*headDim)
// (llama.cpp qwen35moe build_layer_attn: Qcur/gate are strided views of
// Qcur_full at offset 0 and offset headDim, head stride 2*headDim).
//
// The downstream RoPE + attention kernels need Q contiguous, and the
// output gate is applied after attention, so we split the fused output
// into `a` (the first block of every head = Q) and `b` (the second
// block = gate), both contiguous [seqLen, numHeads, headDim].
//
// src layout : [seqLen, numHeads, 2, headDim] f32 row-major
// a   (out)  : [seqLen, numHeads,    headDim]  <- src[..., 0, ...]
// b   (out)  : [seqLen, numHeads,    headDim]  <- src[..., 1, ...]
//
// Launch: 1D global = seqLen * numHeads * headDim in groups of
// SPLIT_HEAD_PAIR_LOCAL.

#ifndef SPLIT_HEAD_PAIR_LOCAL
#define SPLIT_HEAD_PAIR_LOCAL 256
#endif

__attribute__((reqd_work_group_size(SPLIT_HEAD_PAIR_LOCAL, 1, 1)))
__kernel void split_head_pair(
    __global const float* src,
    __global       float* a,
    __global       float* b,
    const int             seqLen,
    const int             numHeads,
    const int             headDim)
{
    const int gid   = (int)get_global_id(0);
    const int total = seqLen * numHeads * headDim;
    if (gid >= total) {
        return;
    }

    // gid = ((p * numHeads) + h) * headDim + d
    const int d    = gid % headDim;
    const int hp   = gid / headDim;
    const int h    = hp % numHeads;
    const int p    = hp / numHeads;

    const size_t headPair = ((size_t)p * numHeads + h) * (size_t)(2 * headDim);
    a[gid] = src[headPair + d];
    b[gid] = src[headPair + headDim + d];
}