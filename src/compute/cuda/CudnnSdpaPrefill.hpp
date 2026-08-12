// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Thin wrapper around cuDNN 9's fused SDPA (flash attention) for the F32-KV
// prefill-attention path on GB10 / sm_121a. This is the win the hand-rolled
// tensor-core kernels could not reach: head_dim=256 exceeds the 99 KiB smem
// budget that caps every FlashAttention variant on consumer Blackwell, but
// cuDNN tiles internally. Feasibility + on-box validation:
// Synaipse decisions/2026-08-12-fmha-lib-feasibility-sm121.
//
// The wrapper owns a cuDNN handle, caches a cudnn-frontend graph per attention
// shape (build is expensive, execute is cheap), and keeps bf16 + workspace
// device scratch. Inputs/outputs are F32 (cast to/from bf16 internally) so the
// engine's F32 staging is untouched; parity vs the F32 kernel is bit-near (bf16).
//
// Only compiled when MIMIRMIND_ENABLE_CUDNN is set. Any cuDNN failure makes a
// call return false so the caller can fall back to the hand kernel.

#pragma once

namespace mimirmind::compute::cuda {

class CudnnSdpaPrefill {
public:
    CudnnSdpaPrefill();
    ~CudnnSdpaPrefill();
    CudnnSdpaPrefill(const CudnnSdpaPrefill&)            = delete;
    CudnnSdpaPrefill& operator=(const CudnnSdpaPrefill&) = delete;

    /// Causal GQA prefill attention over F32 device tensors, single forward
    /// (all query positions attend causally to key positions [0, T_q)).
    ///   q, out : [T_q, nHeads,   headDim] f32
    ///   k, v   : [T_q, nKvHeads, headDim] f32
    /// Runs on `stream` (a CUstream / cudaStream_t). Returns false on any cuDNN
    /// error (caller should fall back to the hand-written kernel).
    bool runF32Causal(void* stream,
                      const float* q, const float* k, const float* v, float* out,
                      int T_q, int nHeads, int nKvHeads, int headDim, float scale);

private:
    struct Impl;
    Impl* _impl;
};

}  // namespace mimirmind::compute::cuda
