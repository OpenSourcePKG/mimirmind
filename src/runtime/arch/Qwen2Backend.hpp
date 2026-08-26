// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "runtime/arch/ArchBackend.hpp"
#include "compute/ComputeBuffer.hpp"

#include <cstddef>
#include <string>

namespace mimirmind::compute {
class ComputeMatmul;
class ComputeOps;
} // namespace mimirmind::compute

namespace mimirmind::core::gguf {
class WeightsMap;
} // namespace mimirmind::core::gguf

namespace mimirmind::model {
class FusedQkvWeights;
struct LlmConfig;
} // namespace mimirmind::model

namespace mimirmind::runtime::arch {

/**
 * Qwen2 / Qwen2.5 / Llama-family decoder block.
 *
 *   x -> rmsNorm(attn_norm) -> Q/K/V proj (+bias) -> RoPE -> GQA attention
 *     -> O proj -> residual -> rmsNorm(ffn_norm) -> SwiGLU (silu(gate)*up)
 *     -> down -> residual
 */
class Qwen2Backend final : public ArchBackend {
public:
    Qwen2Backend(const model::LlmConfig&        config,
                 const core::gguf::WeightsMap&       weights,
                 const model::FusedQkvWeights*  fusedQkv,
                 compute::ComputeOps&               ops,
                 compute::ComputeMatmul&            gmm,
                 runtime::OpProfiler&           opProfiler);

    void runBlock(std::size_t   blockIdx,
                  float*        x,
                  std::size_t   T,
                  KvCache&      cache,
                  BlockBuffers& buffers,
                  bool          traceBlock0) override;

    [[nodiscard]] bool        scalesEmbedding() const noexcept override { return false; }
    [[nodiscard]] const char* name()            const noexcept override { return "qwen2"; }

    /// CLR-safe only when every block's QKV is fused (the split writes K/V
    /// via a stable base + device curLen). Mixed-quant QKV (e.g. Qwen2.5
    /// Q4_K_M: attn_v=Q6_K != attn_q/k=Q4_K) makes FusedQkvWeights skip the
    /// block, so runBlock falls to the unfused path that bakes a per-token
    /// K/V slot pointer into the recording — replay-unsafe. See
    /// ArchBackend::decodeQkvClrSafe.
    [[nodiscard]] bool decodeQkvClrSafe() const noexcept override;

    [[nodiscard]] std::vector<std::size_t>
        kvDimPerLayer() const override;
    [[nodiscard]] std::pair<std::size_t, std::size_t>
        maxQKVDims() const override;

    /// Enable per-stage parity dumps (M8.* tensor-parity workflow), mirroring
    /// GemmaBaseBackend. When set, runBlock writes `<prefix>-blk{N}-<stage>.bin`
    /// for the dense-Llama stages parity-diff checks (attn_norm, Qcur_pos,
    /// Kcur_pos, Vcur_normed, attn_out, ffn_mlp, l_out). Empty = disabled. This
    /// is what makes `mimirmind parity` work for the qwen2/llama forward, the
    /// oracle path used to localise the Llama-3.2 forward drift (step 8.13.2.1).
    void setParityDumpPrefix(const std::string& prefix) noexcept override {
        _parityDumpPrefix = prefix;
    }

private:
    /// Write a per-stage parity dump when `_parityDumpPrefix` is set. No-op
    /// otherwise. Sync-flushes the matmul queue, then writes a 3xu32 header
    /// {blockIdx, Trow, dim} followed by Trow*dim f32 — the exact layout
    /// llama-parity-dump emits and parity-diff.py consumes.
    void dumpStage(const char* stage,
                   std::size_t blockIdx,
                   const float* p,
                   std::size_t Trow,
                   std::size_t dim) const;

    const model::LlmConfig&        _config;
    const core::gguf::WeightsMap&       _weights;
    const model::FusedQkvWeights*  _fusedQkv{nullptr};
    compute::ComputeOps&               _ops;
    compute::ComputeMatmul&            _gmm;
    runtime::OpProfiler&           _op;  // held for parity with Gemma4; not instrumented yet

    // Proportional ("llama3") RoPE frequency factors, [headDim/2] f32 in USM,
    // or nullptr for plain RoPE. Points at the GGUF `rope_freqs.weight` tensor
    // when the checkpoint ships one — llama.cpp bakes the Llama-3.1/3.2 rope
    // scaling (rope_scaling.type == "llama3") into that tensor at conversion
    // time, and its semantics match ropeInPlaceWithFactorsAsync exactly
    // (theta_i = pos * base^(-2i/headDim) / freqFactors[i]). Qwen2/2.5 GGUFs
    // carry no such tensor, so they keep plain RoPE (no behaviour change).
    const float*                   _ropeFreqs{nullptr};

    // YaRN / rope_scaling long-context extension (roadmap 8.8). Backing store
    // for the synthesised YaRN/linear freq_factors (populated by loadRopeFreqs
    // only when the config carries a rope_scaling; otherwise empty and unused),
    // plus the attention temperature (mscale) applied to the softmax scale.
    // Dormant (byte-identical to today) for every checkpoint without scaling.
    compute::ComputeBuffer         _ropeFreqsComputed{};
    float                          _yarnMscale{1.0F};

    // True for the `llama` architecture (Llama / Orpheus), which uses
    // INTERLEAVED (GPT-J / LLAMA_ROPE_TYPE_NORM) RoPE — adjacent pairs
    // (2i, 2i+1) — instead of the split (NEOX) pairs (i, i+headDim/2) that
    // Qwen2/2.5 use. Resolved once from _config.architecture in the ctor.
    bool                           _interleavedRope{false};

    // Parity-dump file prefix carried by `diagnostics.parityDump`; empty =
    // disabled (the default). Set via setParityDumpPrefix() at prefill time.
    std::string                    _parityDumpPrefix{};
};

} // namespace mimirmind::runtime::arch