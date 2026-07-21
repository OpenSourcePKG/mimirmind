// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstddef>

namespace mimirmind::compute {

/**
 * GatedDeltaNet — the linear-attention ("gated delta net") core of
 * Qwen3-Next / Qwen3.5-MoE (`qwen35moe`). These are the CPU *reference*
 * implementations: the golden truth the M-Q3N.3 CUDA kernels
 * (`gated_deltanet_ar` decode + `gated_deltanet_chunk` prefill) are
 * validated against, both for self-consistency and for llama.cpp parity.
 *
 * Reference derivation: research/qwen3next-gated-deltanet-recon-2026-07-21
 * (ggml `build_delta_net_autoregressive` / `build_conv_state`).
 *
 * All buffers are row-major, single sequence (n_seqs == 1 — the recurrence
 * is per-sequence and the GPU path loops sequences outside the kernel).
 * Heads are the *value* heads H_v; q/k must already be repeated from H_k to
 * H_v upstream (ggml does this before the delta net when H_k != H_v), so
 * every function here is fully head-parallel.
 */

/**
 * Autoregressive gated delta-rule recurrence. Processes `T` tokens
 * strictly in order, updating the per-head recurrent state in place and
 * writing the pre-output-norm result. This is the reference for BOTH the
 * decode (T==1) and prefill (T>1) GPU paths — the chunked prefill kernel
 * is an algebraic rearrangement that must reproduce this token-by-token
 * output bit-for-(near-)bit.
 *
 * Per token t, per head h, with head_dim S (== ssm_d_state):
 *   qs   = q[t,h,:] / sqrt(S)                 // query scale
 *   g_h  = exp(gLog[t,h])                     // decay in (0,1]
 *   s   *= g_h                                // decay the [S,S] state
 *   sk[j] = sum_i s[i,j] * k[t,h,i]           // s^T k
 *   d[j]  = (v[t,h,j] - sk[j]) * beta[t,h]    // gated prediction error
 *   s[i,j] += k[t,h,i] * d[j]                 // rank-1 update
 *   out[t,h,j] = sum_i s[i,j] * qs[i]         // s^T qs   (from updated s)
 *
 * Shapes:
 *   q, k, v : [T, H, S]  row-major
 *   gLog    : [T, H]     raw log-decay (exp applied here — do NOT pre-exp)
 *   beta    : [T, H]
 *   state   : [H, S, S]  in/out, s[h] laid out [i*S + j]
 *   out     : [T, H, S]
 */
void gatedDeltaNetRecurrent(const float* q,
                            const float* k,
                            const float* v,
                            const float* gLog,
                            const float* beta,
                            float*       state,
                            float*       out,
                            std::size_t  T,
                            std::size_t  H,
                            std::size_t  S);

/**
 * Causal depthwise 1-D convolution followed by SiLU, the Qwen3-Next
 * `ssm_conv1d` step. Per channel `c` the output at time `t` is a
 * kernelSize-tap causal correlation over the state-prepended input:
 *   y[t,c] = silu( sum_{k=0..K-1} convInput[t+k, c] * kernel[k, c] )
 *
 * `convInput` is the (kernelSize-1) rolling state rows concatenated with
 * the `T` new rows: shape [(kernelSize-1) + T, channels] row-major. The
 * GGUF `ssm_conv1d.weight` is [kernelSize, channels]; `kernel` here uses
 * the same tap-major layout kernel[k*channels + c]. Output is [T, channels].
 *
 * (The GGUF→this-layout mapping and the rolling-state write-back are the
 * loader/kernel's job; this reference fixes the numeric contract.)
 */
void causalConv1dSilu(const float* convInput,
                      const float* kernel,
                      float*       out,
                      std::size_t  T,
                      std::size_t  channels,
                      std::size_t  kernelSize);

/**
 * In-place L2 normalisation over the innermost `dim` (head_dim), matching
 * ggml `ggml_l2_norm` used on q/k in the linear layer:
 *   x[r, :] /= sqrt( sum_j x[r,j]^2 + eps )
 * `rows` is the number of length-`dim` vectors (== T * H). Note: L2, NOT
 * RMS — there is no learned weight and no 1/dim factor.
 */
void l2NormInPlace(float*      x,
                   std::size_t rows,
                   std::size_t dim,
                   float       eps);

} // namespace mimirmind::compute