// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

// In-place exact (erf) GELU — the BERT/RoBERTa/XLM-R FFN activation
// (cross-encoder reranker / EncoderRunner):
//   y[i] = 0.5 * x * (1 + erf(x / sqrt(2)))
// NOT the tanh approximation (that is gelu_mul.cl). CPU reference:
// compute::geluErfInPlace. Launch: 1D global = n.

#ifndef GELU_ERF_LOCAL
#define GELU_ERF_LOCAL 256
#endif

__attribute__((reqd_work_group_size(GELU_ERF_LOCAL, 1, 1)))
__kernel void gelu_erf(
    __global float* x,
    const int       n)
{
    const int gid = (int)get_global_id(0);
    if (gid >= n) {
        return;
    }
    const float v = x[gid];
    x[gid] = 0.5f * v * (1.0f + erf(v * 0.70710678118654752440f));
}
