// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/arch/Kolibri1Backend.hpp"

#include "model/LlmConfig.hpp"

#include <algorithm>
#include <stdexcept>

namespace mimirmind::runtime::arch {

Kolibri1Backend::Kolibri1Backend(const model::LlmConfig&       config,
                                 const core::gguf::WeightsMap& weights,
                                 const model::FusedQkvWeights* fusedQkv,
                                 compute::ComputeOps&          ops,
                                 compute::ComputeMatmul&       gmm,
                                 runtime::OpProfiler&          opProfiler,
                                 bool                          moeGroupEnabled,
                                 bool                          moeFusedDownEnabled)
    : _config(config),
      _weights(weights),
      _fusedQkv(fusedQkv),
      _ops(ops),
      _gmm(gmm),
      _opProfiler(opProfiler),
      _moeGroupEnabled(moeGroupEnabled),
      _moeFusedDownEnabled(moeFusedDownEnabled) {}

void Kolibri1Backend::runBlock(std::size_t   blockIdx,
                               float*        x,
                               std::size_t   T,
                               KvCache&      cache,
                               BlockBuffers& buffers,
                               bool          traceBlock0) {
    (void)blockIdx; (void)x; (void)T; (void)cache; (void)buffers;
    (void)traceBlock0;
    // I-1 seam only. The forward lands across:
    //   - I-3 (5.34.4): FP8 128x128 block weight materialise,
    //   - I-4 (5.34.5): SWA(513)+full GQA attention, per-head QK-RMSNorm,
    //                   NoPE on full-attention layers (RoPE only on sliding),
    //   - I-5 (5.34.6): sigmoid top-6 router (expert_bias, no renorm) + shared
    //                   expert, over the 4-norm sandwich layout.
    throw std::runtime_error(
        "kolibri1: runBlock not implemented (I-1 seam; attention is 5.34.5 / "
        "MoE is 5.34.6)");
}

std::vector<std::size_t> Kolibri1Backend::kvDimPerLayer() const {
    std::vector<std::size_t> dims;
    dims.reserve(_config.blockCount);
    for (std::size_t l = 0; l < _config.blockCount; ++l) {
        dims.push_back(static_cast<std::size_t>(_config.headCountKvFor(l))
                       * _config.headDim(l));
    }
    return dims;
}

std::pair<std::size_t, std::size_t> Kolibri1Backend::maxQKVDims() const {
    const std::size_t qDim =
        static_cast<std::size_t>(_config.headCount) * _config.headDim();
    std::size_t kvDimMax = 0;
    for (std::size_t l = 0; l < _config.blockCount; ++l) {
        kvDimMax = std::max(kvDimMax,
                            static_cast<std::size_t>(_config.headCountKvFor(l))
                                * _config.headDim(l));
    }
    return {qDim, kvDimMax};
}

ArchCapabilities Kolibri1Backend::computeCapabilities() const {
    // Pure-attention MoE. Batched decode / CLR-safe MoE decode are not wired at
    // the I-1 seam; they turn on as the serving paths land (I-6+). The
    // ArchCapabilities defaults (everything false except decodeQkvClrSafe) are
    // the correct conservative starting point.
    return ArchCapabilities{};
}

} // namespace mimirmind::runtime::arch
