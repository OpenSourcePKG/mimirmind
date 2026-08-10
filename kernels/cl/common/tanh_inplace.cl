// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

// In-place tanh — the RoBERTa/XLM-R sequence-classification head activation
// (dense -> tanh -> out_proj) used by the cross-encoder reranker
// (EncoderRunner). CPU reference: compute::tanhInPlace. Launch: 1D global = n.

#ifndef TANH_INPLACE_LOCAL
#define TANH_INPLACE_LOCAL 256
#endif

__attribute__((reqd_work_group_size(TANH_INPLACE_LOCAL, 1, 1)))
__kernel void tanh_inplace(
    __global float* x,
    const int       n)
{
    const int gid = (int)get_global_id(0);
    if (gid >= n) {
        return;
    }
    x[gid] = tanh(x[gid]);
}
