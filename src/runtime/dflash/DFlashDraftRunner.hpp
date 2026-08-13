// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstddef>

namespace mimirmind::compute {
class ComputeOps;
class ComputeMatmul;
} // namespace mimirmind::compute

namespace mimirmind::runtime::dflash {

class DFlashDraftModel;

/**
 * Forward engine for the DFlash block-diffusion drafter (mirrors the
 * EncoderModel / EncoderRunner split: DFlashDraftModel holds the device
 * weights, this runs the math through the abstract ComputeOps / ComputeMatmul).
 *
 * Built incrementally against the z-lab `model.py` reference golden
 * (`tests/dflash_forward_parity_test`):
 *   P3.1 — materializeContext (fc -> hidden_norm)
 *   P3.2 — draftForward (6-layer non-causal block attn, host parity attn)
 *   P3.3 — draftForward block attention on-device (attentionEncoderCross)
 *
 * Not thread-safe. Holds references to the model / ops; they must outlive it.
 */
class DFlashDraftRunner {
public:
    DFlashDraftRunner(const DFlashDraftModel& model,
                      compute::ComputeOps&    ops,
                      compute::ComputeMatmul& matmul) noexcept
        : _m{model}, _ops{ops}, _mm{matmul} {}

    /// P3.1 — context materialization. Projects the concatenated 8-tap target
    /// hidden `target_hidden` [ctxLen, taps*hidden] (F32, device) through the
    /// fc [hidden, taps*hidden] (BF16) and hidden_norm RMSNorm into
    /// `out` [ctxLen, hidden] (F32, device). This is the `hidden_norm(fc(.))`
    /// step of the reference forward; the per-layer k/v_proj on this result and
    /// the draft-KV cache come in P3.2/P3.3. Async — call `ops.flush()` (or the
    /// matmul's sync) before a host readback of `out`.
    void materializeContext(const float* target_hidden,
                            std::size_t  ctxLen,
                            float*       out);

    /// P3.2 — full draft forward. Given block noise embeddings `noise` [bs,H]
    /// and the raw concatenated 8-tap hidden `target_hidden` [ctxLen, taps*H]
    /// (both F32, device), runs hidden_norm(fc(.)) + the 6-layer non-causal
    /// block transformer and writes the final-norm output `out` [bs,H] (F32,
    /// device). Rope positions match the reference golden: K over
    /// [0 .. ctxLen+bs-1], Q over [ctxLen .. ctxLen+bs-1]. The attention core
    /// (softmax(Q·Kᵀ·scale)·V, GQA, non-causal, full window) runs on-device via
    /// `attentionEncoderCrossAsync` (P3.3). Async on the stream; blocks once at
    /// the end (flush) before returning `out`.
    void draftForward(const float* noise,
                      const float* target_hidden,
                      std::size_t  bs,
                      std::size_t  ctxLen,
                      float*       out);

private:
    const DFlashDraftModel& _m;
    compute::ComputeOps&    _ops;
    compute::ComputeMatmul& _mm;
};

} // namespace mimirmind::runtime::dflash
