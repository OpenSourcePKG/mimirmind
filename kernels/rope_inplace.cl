// Rotary positional embedding (RoPE), in-place. llama.cpp "non-interleaved"
// (split) layout used by Llama / Qwen / Gemma family models.
//
//   theta_i = (startPos + p) * base^(-2i / headDim)
//   c = cos(theta_i), s = sin(theta_i)
//   x'[i]           = x[i] * c - x[i + halfDim] * s
//   x'[i + halfDim] = x[i] * s + x[i + halfDim] * c
//
// Layout: x is [seqLen, numHeads, headDim] f32 row-major.
//
// Launch: 1D global = seqLen * numHeads * halfDim work-items in groups
// of ROPE_LOCAL. Each work-item owns one (p, h, i) and rotates the
// pair (head[i], head[i + halfDim]) atomically with respect to itself
// — both elements are read into registers before either is written.

#ifndef ROPE_LOCAL
#define ROPE_LOCAL 256
#endif

__attribute__((reqd_work_group_size(ROPE_LOCAL, 1, 1)))
__kernel void rope_inplace(
    __global       float* x,
    const int             seqLen,
    const int             numHeads,
    const int             headDim,
    const int             startPos,
    const float           base)
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

    const float pos    = (float)(startPos + p);
    const float invDim = 1.0f / (float)headDim;
    const float freq   = pow(base, -(float)(2 * i) * invDim);
    const float theta  = pos * freq;
    const float c      = cos(theta);
    const float s      = sin(theta);

    const int headBase = (p * numHeads + h) * headDim;
    const float a = x[headBase + i];
    const float b = x[headBase + i + halfDim];
    x[headBase + i]           = a * c - b * s;
    x[headBase + i + halfDim] = a * s + b * c;
}