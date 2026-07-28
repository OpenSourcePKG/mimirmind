// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

// In-place sigmoid gating: y[r,c] *= sigmoid(g[r, gcol]).
//
// Two Qwen3-Next call sites:
//   1. Full-attention output gate: y = attention output [T, q_dim], gate
//      g = [T, q_dim] (gateDim == dim) — per-element gate. llama.cpp:
//      `cur = ggml_mul(cur, ggml_sigmoid(gate))` before wo.
//   2. Shared-expert gate: y = shared-expert FFN output [T, d_model],
//      gate g = [T, 1] (gateDim == 1) — one scalar per token, broadcast
//      across the row. llama.cpp: `ffn_shexp *= sigmoid(gate_inp_shexp @ x)`.
//
// `gateDim` selects the mode: gateDim == dim → per-element; gateDim == 1
// → per-row broadcast. (Any 1 <= gateDim <= dim that divides the access
// is honoured, but only these two are used.)
//
// y layout: [rows, dim] f32 row-major. g layout: [rows, gateDim].
// Launch: 1D global = rows * dim in groups of SIGMOID_GATE_MUL_LOCAL.

#ifndef SIGMOID_GATE_MUL_LOCAL
#define SIGMOID_GATE_MUL_LOCAL 256
#endif

__attribute__((reqd_work_group_size(SIGMOID_GATE_MUL_LOCAL, 1, 1)))
__kernel void sigmoid_gate_mul(
    __global       float* y,
    __global const float* g,
    const int             rows,
    const int             dim,
    const int             gateDim)
{
    const int gid   = (int)get_global_id(0);
    const int total = rows * dim;
    if (gid >= total) {
        return;
    }

    const int c    = gid % dim;
    const int r    = gid / dim;
    const int gcol = (gateDim == 1) ? 0 : c;
    const float gv = g[(size_t)r * gateDim + gcol];
    const float sig = 1.0f / (1.0f + exp(-gv));
    y[gid] *= sig;
}