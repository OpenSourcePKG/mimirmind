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