// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/gguf/GgufTypes.hpp"

#include <cstddef>

namespace mimirmind::compute {

/**
 * Y = X · W^T, naive CPU triple loop with per-row on-the-fly dequant.
 *
 *   X: [M, K] F32 row-major  (typically M == sequence length)
 *   W: [N, K] in GGUF layout — innermost dim K stored as a single row
 *           of N contiguous elements of `weightType`. This is the
 *           llama.cpp convention for *.weight tensors that act as
 *           projection matrices (lm_head, attn_q.weight, ffn_down.weight,
 *           ...).
 *   Y: [M, N] F32 row-major
 *
 * scratch: caller-provided buffer of K F32 elements. We dequant each
 * row of W into this buffer once and reuse across all M rows of X — no
 * full-W dequant is ever materialised (lm_head would otherwise blow up
 * a Q6_K 22-GiB Gemma to ~85 GiB of F32).
 *
 * Dot product accumulates in double for numerical stability.
 *
 * Slow but correct. GPU kernels land at M5.
 */
void matmul(core::gguf::GgmlType weightType,
            const void*     W,
            std::size_t     N,
            std::size_t     K,
            const float*    X,
            std::size_t     M,
            float*          Y,
            float*          scratch);

/// In-place broadcast bias add: y[m,k] += bias[k] for all m in [0,M),
/// k in [0,K). Used after a projection matmul that has a per-feature
/// bias (Qwen2 has them on Q/K/V; Llama-family typically doesn't).
void addBias(float* y, std::size_t M, std::size_t K, const float* bias) noexcept;

/// In-place row-major residual add: y[i] += x[i] for i in [0, N).
void addResidual(float* y, const float* x, std::size_t N) noexcept;

} // namespace mimirmind::compute