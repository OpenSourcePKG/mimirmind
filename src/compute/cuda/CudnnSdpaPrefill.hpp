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

    /// Causal GQA prefill attention over F32 device tensors, POSITION-major
    /// layout ([position, head, dim] — matches the hand kernels / KV cache):
    ///   q, out : [T_q,  nHeads,   headDim] f32
    ///   k, v   : [T_kv, nKvHeads, headDim] f32   (full cache prefix + chunk)
    /// The T_q query positions occupy the LAST rows of the T_kv key range:
    /// query row i is at absolute position (T_kv - T_q + i) and attends keys
    /// [0, T_kv - T_q + i] — i.e. bottom-right-aligned causal. For a first
    /// chunk T_kv == T_q (plain causal). This covers chunked prefill where a
    /// continuation chunk (positionOffset>0) attends the cached prefix.
    /// Runs on `stream` (cudaStream_t). Returns false on any cuDNN error.
    bool runF32Causal(void* stream,
                      const float* q, const float* k, const float* v, float* out,
                      int T_q, int T_kv, int nHeads, int nKvHeads, int headDim,
                      float scale);

private:
    struct Impl;
    Impl* _impl;
};

}  // namespace mimirmind::compute::cuda
