// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace mimirmind::compute {
class ComputeOps;
class ComputeMatmul;
} // namespace mimirmind::compute

namespace mimirmind::runtime::encoder {

class EncoderModel;

/**
 * Bidirectional (non-causal) cross-encoder forward for a reranker
 * (bge-reranker-v2-m3 / XLM-RoBERTa). One whole-sequence pass, no KV cache,
 * no sampler:
 *
 *   embeddings(word + pos + type, LayerNorm)
 *   -> 24 x [ LN(x + SelfAttn(dense)) ; LN(x + FFN(erf-GELU)(dense)) ]  (post-LN)
 *   -> classifier head on the <s>/CLS token (dense -> tanh -> out_proj)
 *
 * Composes the five encoder primitives validated in cuda_encoder_layer0_parity
 * with the trusted matmul/addBias/residual path. Backend-neutral: all compute
 * goes through the abstract ComputeOps / ComputeMatmul, so the same runner
 * drives the CUDA (Spark) and Level-Zero (NUC) backends.
 *
 * Single sequence per call (no padding/batching yet). Not thread-safe.
 */
class EncoderRunner {
public:
    EncoderRunner(const EncoderModel& model,
                  compute::ComputeOps& ops,
                  compute::ComputeMatmul& matmul);

    /// Full forward → the `numLabels` classifier logits for one tokenized
    /// sequence (attention over the full range; caller must not pass padding).
    [[nodiscard]] std::vector<float>
    forwardLogits(std::span<const std::int32_t> inputIds);

    /// Batched full forward: B sequences in ONE GPU pass (padded to the max
    /// length, per-sequence attention masking). Returns B × numLabels logits.
    /// Far faster than B separate forwardLogits() for a rerank pool.
    [[nodiscard]] std::vector<std::vector<float>>
    forwardLogitsBatch(std::span<const std::vector<std::int32_t>> sequences);

    /// The single rerank relevance logit (numLabels == 1).
    [[nodiscard]] float score(std::span<const std::int32_t> inputIds);

private:
    const EncoderModel&     _m;
    compute::ComputeOps&    _ops;
    compute::ComputeMatmul& _mm;
};

} // namespace mimirmind::runtime::encoder
