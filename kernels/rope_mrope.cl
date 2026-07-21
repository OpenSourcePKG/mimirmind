// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

// Interleaved multi-axis RoPE (IMRoPE), in-place. The rotary variant used
// by Qwen3-Next / Qwen3.5-VL (`LLM_ROPE_TYPE_IMROPE` in llama.cpp). Same
// "non-interleaved" (split) pair layout as rope_inplace.cl — the pair
// (x[i], x[i + halfDim]) is rotated by theta — but the *angle base* per
// pair is selected across four position axes (time / height / width /
// extra) following the interleaved-mRoPE sector rule.
//
// ggml reference (ggml/src/ggml-cpu/ops.cpp, ggml_mrope_cache_init, the
// `is_imrope` branch):
//   sect_dims = s0 + s1 + s2 + s3
//   sector    = (pair index) % sect_dims
//   if      (sector % 3 == 1 && sector < 3*s1) theta = pos_h * freq_i
//   else if (sector % 3 == 2 && sector < 3*s2) theta = pos_w * freq_i
//   else if (sector % 3 == 0 && sector < 3*s0) theta = pos_t * freq_i
//   else                                        theta = pos_e * freq_i
// with freq_i = base^(-2i/headDim).
//
// TEXT-ONLY POSITIONS: this engine feeds a single 1-D sequence position
// per token, so pos_t == pos_h == pos_w == pos_e == (startPos + p). With
// all four axis positions equal the sector selection collapses and the
// result is bit-identical to plain NeoX RoPE — as it must be, since
// llama.cpp assigns text tokens the same value on every mRoPE axis. The
// sector machinery below is the faithful IMRoPE algorithm and the single
// extension point for true multimodal support: replace the uniform
// `posAxis[4]` with per-token, per-axis position ids (a [seqLen, 4]
// input) and nothing else in the rotation changes.
//
// Layout: x is [seqLen, numHeads, headDim] f32 row-major.
// Launch: 1D global = seqLen * numHeads * halfDim in groups of ROPE_LOCAL.
//
// `startPos` arrives via a __global int slot (CLR replay, see
// rope_inplace.cl). `writeOffsetStride` shifts the base by
// startPos*stride inside the kernel (K-rope into the cache slot).

#ifndef ROPE_LOCAL
#define ROPE_LOCAL 256
#endif

__attribute__((reqd_work_group_size(ROPE_LOCAL, 1, 1)))
__kernel void rope_mrope(
    __global       float* x_base,
    const int             seqLen,
    const int             numHeads,
    const int             headDim,
    __global const int*   startPosPtr,
    const float           base,
    const int             writeOffsetStride,
    const int             sec0,
    const int             sec1,
    const int             sec2,
    const int             sec3)
{
    const int gid     = (int)get_global_id(0);
    const int halfDim = headDim / 2;
    const int total   = seqLen * numHeads * halfDim;
    if (gid >= total) {
        return;
    }

    // gid = ((p * numHeads) + h) * halfDim + i
    const int i  = gid % halfDim;
    const int hp = gid / halfDim;
    const int h  = hp % numHeads;
    const int p  = hp / numHeads;

    const int   startPos = startPosPtr[0];
    const float pos      = (float)(startPos + p);

    // Text-only: all four axis positions are the sequence position. The
    // multimodal extension replaces this with per-axis position ids.
    const float posAxis[4] = { pos, pos, pos, pos };

    const int sectDims = sec0 + sec1 + sec2 + sec3;

    // Select the position axis for this pair per the IMRoPE sector rule.
    // sectDims == 0 (no sections shipped) degenerates to plain RoPE.
    float posSel = posAxis[0];
    if (sectDims > 0) {
        const int sector = i % sectDims;
        if (sector % 3 == 1 && sector < 3 * sec1) {
            posSel = posAxis[1];        // height
        } else if (sector % 3 == 2 && sector < 3 * sec2) {
            posSel = posAxis[2];        // width
        } else if (sector % 3 == 0 && sector < 3 * sec0) {
            posSel = posAxis[0];        // time
        } else {
            posSel = posAxis[3];        // extra
        }
    }

    __global float* x  =
        x_base + (size_t)startPos * (size_t)writeOffsetStride;
    const float invDim = 1.0f / (float)headDim;
    const float freq   = pow(base, -(float)(2 * i) * invDim);
    const float theta  = posSel * freq;
    const float c      = cos(theta);
    const float s      = sin(theta);

    const int headBase = (p * numHeads + h) * headDim;
    const float a = x[headBase + i];
    const float b = x[headBase + i + halfDim];
    x[headBase + i]           = a * c - b * s;
    x[headBase + i + halfDim] = a * s + b * c;
}