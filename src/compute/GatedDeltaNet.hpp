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
 * Chunked ("WY / UT-transform") gated delta-rule forward — the parallel
 * prefill algebra that reproduces `gatedDeltaNetRecurrent` output and final
 * state exactly, but processes the sequence in chunks of `chunkSize` tokens
 * so the per-chunk work is matmul-shaped (the GPU M-Q3N.4 prefill kernel is
 * a direct port of this). Same tensor contract as the recurrent reference.
 *
 * Derivation (from the recurrent recurrence, per head, state S [S_k,S_v]):
 * unrolling one chunk of C tokens with cumulative log-decay
 * G_a = sum_{r<=a} gLog_r (within the chunk) yields the intra-chunk delta
 * vectors d_a as the solution of a unit lower-triangular system
 *   (I + A) d = r,   A[a,m] = beta_a (k_a . k_m) exp(G_a - G_m)   (m < a)
 *   r_a         = beta_a ( v_a - exp(G_a) (S0^T k_a) )
 * solved by forward substitution. The gate exp(G_a - G_m) is an EXACT
 * diagonal similarity of the ungated system (I + A) = D (I + A~) D^-1 with
 * D = diag(exp G), so it may equivalently be applied entrywise to the
 * ungated inverse (this is the FlashQLA factorisation — verified exact, not
 * an approximation). Output and the carried state are then
 *   o_a  = exp(G_a) (S0^T qs_a) + sum_{m<=a} exp(G_a - G_m)(k_m . qs_a) d_m
 *   S'   = exp(G_{C-1}) S0 + sum_m exp(G_{C-1} - G_m) k_m d_m^T
 * with qs = q / sqrt(S). exp arguments are <= 0 within a chunk (decay), so
 * no overflow. `state` is read at chunk start and overwritten with S' after
 * each chunk, chaining chunks and leaving the post-sequence state ready for
 * decode — identical to the recurrent path.
 *
 * Shapes identical to gatedDeltaNetRecurrent. `chunkSize` is a free tuning
 * knob (the result is chunk-size invariant); 64 is the default.
 */
void gatedDeltaNetChunk(const float* q,
                        const float* k,
                        const float* v,
                        const float* gLog,
                        const float* beta,
                        float*       state,
                        float*       out,
                        std::size_t  T,
                        std::size_t  H,
                        std::size_t  S,
                        std::size_t  chunkSize = 64);

// ---------------------------------------------------------------------------
// gatedDeltaNetChunk, decomposed into the three GPU-kernel-shaped stages
// (M-Q3N.4b ports each one-to-one). Running K0 -> K1 -> K2 reproduces the
// monolithic gatedDeltaNetChunk (and thus the recurrent reference) exactly;
// the intermediate tensors G and A0 are the kernel hand-off buffers.
// ---------------------------------------------------------------------------

/**
 * K0 — cumulative decay gate. Prefix-sum of `gLog` within each `chunkSize`
 * window (reset at every chunk boundary), per head:
 *   gCum[c*C + a, h] = sum_{r=0..a} gLog[c*C + r, h]
 * gLog, gCum are [T, H] row-major; the exp is applied by K2, not here.
 */
void deltanetChunkCumGate(const float* gLog,
                          float*       gCum,
                          std::size_t  T,
                          std::size_t  H,
                          std::size_t  chunkSize);

/**
 * K1 — per-chunk, per-head inverse of the UNGATED unit lower-triangular
 * system. With the strict-lower Gram A~[a,m] = beta_a (k_a . k_m) (m < a),
 * this writes A0 = (I + A~)^-1 (itself unit lower-triangular). The decay
 * gate is deliberately absent — K2 reintroduces it as the exact diagonal
 * similarity D A0 D^-1, D = diag(exp gCum) (see gatedDeltaNetChunk).
 *
 * a0 is [nChunks, H, C, C] row-major, nChunks = ceil(T/chunkSize),
 * C = chunkSize; element (c,h,a,m) at ((c*H + h)*C + a)*C + m. For a partial
 * trailing chunk only the leading cs x cs block (cs = tokens in chunk) is
 * meaningful; the remainder is zero-filled. k is [T,H,S], beta is [T,H].
 */
void deltanetKktSolveInverse(const float* k,
                             const float* beta,
                             float*       a0,
                             std::size_t  T,
                             std::size_t  H,
                             std::size_t  S,
                             std::size_t  chunkSize);

/**
 * K2 — chunk forward. Consumes gCum (K0) and a0 (K1) and, chunk by chunk,
 * carries the recurrent state while writing the output. Per chunk the intra-
 * chunk deltas are recovered without a solve:
 *   r_m   = beta_m ( v_m - exp(gCum_m) (S0^T k_m) )
 *   d_a   = exp(gCum_a) * sum_{m<=a} a0[a,m] exp(-gCum_m) r_m
 * then output and state carry exactly as in gatedDeltaNetChunk. Same tensor
 * contract; `state` is read at chunk start and overwritten with the carry.
 */
void deltanetChunkForward(const float* q,
                          const float* k,
                          const float* v,
                          const float* gCum,
                          const float* beta,
                          const float* a0,
                          float*       state,
                          float*       out,
                          std::size_t  T,
                          std::size_t  H,
                          std::size_t  S,
                          std::size_t  chunkSize);

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
 * GatedDeltaNet decay gate (Qwen3-Next linear layer). Per token t, per
 * value-head h:
 *   gLog[t,h] = -exp(ssmA[h]) * softplus(alpha[t,h] + ssmDt[h])
 * This is the raw log-decay fed to `gatedDeltaNetRecurrent` (which applies
 * exp internally). softplus uses the numerically-stable form. Reference for
 * ggml `-A_log.exp() * softplus(alpha + dt)`.
 *   alpha : [T, H]   ssmA, ssmDt : [H]   gLog (out) : [T, H]
 */
void deltanetGate(const float* alpha,
                  const float* ssmA,
                  const float* ssmDt,
                  float*       gLog,
                  std::size_t  T,
                  std::size_t  H);

/**
 * In-place logistic sigmoid: y[i] = 1 / (1 + exp(-y[i])). Used for the
 * GatedDeltaNet `beta` gate (sigmoid of the ssm_beta projection).
 */
void sigmoidInPlace(float* y, std::size_t n);

/**
 * Extract a per-head channel slice from a wide per-token buffer, with an
 * optional GQA-style head repeat — the single data-movement op that turns
 * the fused conv output into contiguous q / k / v for the delta net.
 *
 * The Qwen3-Next conv output is [T, convTotalWidth] token-major, with the
 * q / k / v blocks concatenated on the channel axis. For each the head
 * block starts at `offset` and spans `srcHeads * S` channels. This copies
 *   dst[t, hd, s] = src[t*convTotalWidth + offset + (hd % srcHeads)*S + s]
 * for hd in [0, dstHeads), s in [0, S). When `dstHeads == srcHeads` it is a
 * plain slice (v); when `dstHeads` is a multiple of `srcHeads` it also
 * repeats the source heads (q/k: H_k -> H_v, ggml_repeat modulo semantics).
 *
 * dst is [T, dstHeads, S] contiguous.
 */
void gatherHeadsFromChannels(const float* src,
                             float*       dst,
                             std::size_t  T,
                             std::size_t  offset,
                             std::size_t  srcHeads,
                             std::size_t  dstHeads,
                             std::size_t  S,
                             std::size_t  convTotalWidth);

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