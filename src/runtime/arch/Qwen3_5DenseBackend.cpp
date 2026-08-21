// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/arch/Qwen3_5DenseBackend.hpp"

#include "compute/ComputeMatmul.hpp"
#include "compute/ComputeOps.hpp"
#include "core/gguf/GgufTypes.hpp"
#include "core/gguf/WeightsMap.hpp"
#include "runtime/BlockBuffers.hpp"

#include <stdexcept>
#include <string>

namespace mimirmind::runtime::arch {

namespace {

const core::gguf::GgufTensor& requireBlock(const core::gguf::WeightsMap& w,
                                           std::size_t                   blockIdx,
                                           std::string_view              suffix) {
    const auto* t = w.findBlock(blockIdx, suffix);
    if (t == nullptr) {
        throw std::runtime_error(
            "Qwen3_5DenseBackend: block " + std::to_string(blockIdx) +
            " missing tensor '" + std::string(suffix) + "'");
    }
    return *t;
}

} // namespace

void Qwen3_5DenseBackend::runFfn(std::size_t   blockIdx,
                                 const float*  ffnInput,
                                 std::size_t   T,
                                 BlockBuffers& s) {
    const auto& gateW = requireBlock(_weights, blockIdx, "ffn_gate.weight");
    const auto& upW   = requireBlock(_weights, blockIdx, "ffn_up.weight");
    const auto& downW = requireBlock(_weights, blockIdx, "ffn_down.weight");

    const std::size_t d_model = s.d_model;
    const std::size_t n_ff =
        gateW.dimensions.size() >= 2 ? gateW.dimensions[1] : 0;
    if (n_ff == 0) {
        throw std::runtime_error("Qwen3_5DenseBackend: ffn_gate has unexpected "
                                 "shape");
    }

    float* const gateOutBuf    = s.gateOut.as<float>();
    float* const upOutBuf      = s.upOut.as<float>();
    float* const matmulScratch = s.matmulScratch.as<float>();
    float* const moeAccumBuf   = s.moeAccumBuf.as<float>();

    // SwiGLU: gate/up @ ffnInput -> silu(gate)*up -> down. The down output IS
    // the whole FFN result (no routing / shared-expert), so it goes straight
    // into moeAccumBuf, which runFullAttentionBlock/runLinearBlock add to the
    // residual (same contract as the MoE runFfn). Buffers are sized for the
    // dense feedForwardLength (BlockBuffers ffScratch = max(ff_dim, expert ff)).
    {
        compute::UnorderedScope u{_ops};
        _gmm.matmulAsync(gateW.type, gateW.usmPtr, n_ff, d_model,
                         ffnInput, T, gateOutBuf, matmulScratch);
        _gmm.matmulAsync(upW.type, upW.usmPtr, n_ff, d_model,
                         ffnInput, T, upOutBuf, matmulScratch);
    }
    _ops.siluMulAsync(gateOutBuf, upOutBuf, T * n_ff);
    _gmm.matmulAsync(downW.type, downW.usmPtr, d_model, n_ff,
                     gateOutBuf, T, moeAccumBuf, matmulScratch);
}

} // namespace mimirmind::runtime::arch
