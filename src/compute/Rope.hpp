// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstddef>
#include <cstdint>

namespace mimirmind::compute {

/**
 * Rotary positional embedding (RoPE), llama.cpp "non-interleaved"
 * (split) layout used by every Llama / Qwen / Gemma family model.
 *
 * For each token at absolute position p in [startPos, startPos+seqLen),
 * each head, and each dim-pair (i, i + headDim/2) with i in [0, headDim/2):
 *
 *   theta = p * base^(-2i / headDim)
 *   c = cos(theta), s = sin(theta)
 *
 *   x'[i]              = x[i] * c  -  x[i + headDim/2] * s
 *   x'[i + headDim/2]  = x[i] * s  +  x[i + headDim/2] * c
 *
 * Apply to Q and K independently (V is not rotated). headDim must be
 * even. Operates in place on a `[seqLen, numHeads, headDim]` F32 buffer
 * in row-major order.
 *
 * `startPos` is the absolute position of the first row — set it to the
 * current KV-cache fill level once we have a cache.
 */
void applyRopeInPlace(float*        x,
                      std::size_t   seqLen,
                      std::size_t   numHeads,
                      std::size_t   headDim,
                      std::size_t   startPos,
                      float         base);

/**
 * Proportional-RoPE variant with per-pair frequency factors. Matches
 * ggml's `ggml_rope_ext` extension used by Gemma 3 / 4 global-attention
 * layers: the per-pair angle becomes
 *
 *   theta_i = pos * base^(-2i/headDim) / freqFactors[i]
 *
 * `freqFactors` points at `headDim / 2` f32 values. Zero entries throw
 * (division by zero would silently NaN the head). Everything else
 * matches `applyRopeInPlace`.
 */
void applyRopeInPlaceWithFactors(float*        x,
                                 const float*  freqFactors,
                                 std::size_t   seqLen,
                                 std::size_t   numHeads,
                                 std::size_t   headDim,
                                 std::size_t   startPos,
                                 float         base);

/**
 * Interleaved ("GPT-J" / llama.cpp `LLAMA_ROPE_TYPE_NORM`) RoPE. Rotates
 * ADJACENT pairs (2i, 2i+1) rather than the split pairs (i, i+headDim/2)
 * of `applyRopeInPlace` (NEOX). This is the convention the `llama`
 * architecture uses (Llama-1/2/3, and Llama-3.2-derived checkpoints such
 * as Orpheus TTS) — as opposed to Qwen2/Qwen2.5/Gemma, which are NEOX.
 * Using NEOX on a `llama` checkpoint rotates the wrong coordinate pairs:
 * the self-attention score (relative position 0) still matches, but every
 * cross-position score is wrong and the error grows with |i-j|, which
 * silently corrupts generation into degenerate sub-word text.
 *
 *   theta_i = pos * base^(-2i/headDim) [ / freqFactors[i] when non-null ]
 *   x'[2i]     = x[2i] * c - x[2i+1] * s
 *   x'[2i+1]   = x[2i] * s + x[2i+1] * c
 *
 * `freqFactors` is optional: nullptr = plain interleaved RoPE; non-null
 * points at `headDim/2` f32 values and applies the Llama-3.1/3.2 "llama3"
 * proportional scaling exactly as `applyRopeInPlaceWithFactors` does for
 * the NEOX path. Zero factor entries throw. headDim must be even.
 */
void applyRopeInPlaceInterleaved(float*        x,
                                 const float*  freqFactors,
                                 std::size_t   seqLen,
                                 std::size_t   numHeads,
                                 std::size_t   headDim,
                                 std::size_t   startPos,
                                 float         base);

/**
 * Interleaved multi-axis RoPE (IMRoPE) — the rotary variant used by
 * Qwen3-Next / Qwen3.5-VL full-attention layers (`LLM_ROPE_TYPE_IMROPE`
 * in llama.cpp). Same split-pair rotation as `applyRopeInPlace`, but the
 * per-pair angle base is selected across four position axes (time /
 * height / width / extra) via the IMRoPE sector rule (ggml
 * `ggml_mrope_cache_init`, `is_imrope` branch):
 *
 *   sect_dims = sections[0..3] summed
 *   sector    = i % sect_dims             (i = pair index in [0, headDim/2))
 *   theta     = pos_axis(sector) * base^(-2i/headDim)
 *
 * `sections` points at 4 int32 dimension-section widths (GGUF
 * `<arch>.rope.dimension_sections`). This engine is text-only, so all
 * four axis positions equal the sequence position `startPos + p` and the
 * result is bit-identical to `applyRopeInPlace` — the sector machinery is
 * the faithful IMRoPE algorithm and the extension point for true
 * multimodal position ids. `sections == nullptr` or a zero sum
 * degenerates to plain RoPE.
 */
void applyMropeInPlace(float*              x,
                       std::size_t         seqLen,
                       std::size_t         numHeads,
                       std::size_t         headDim,
                       std::size_t         startPos,
                       float               base,
                       const std::int32_t* sections);

} // namespace mimirmind::compute