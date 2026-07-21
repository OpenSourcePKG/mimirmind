// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/arch/ArchBackend.hpp"

#include "runtime/arch/Gemma4Backend.hpp"
#include "runtime/arch/Qwen2Backend.hpp"
#include "runtime/arch/Qwen35MoeBackend.hpp"

namespace mimirmind::runtime::arch {

std::unique_ptr<ArchBackend>
createArchBackend(const std::string&             architecture,
                  const model::LlmConfig&        config,
                  const core::gguf::WeightsMap&       weights,
                  const model::FusedQkvWeights*  fusedQkv,
                  compute::ComputeOps&               ops,
                  compute::ComputeMatmul&            gmm,
                  OpProfiler&                    opProfiler,
                  bool                           moeGroupEnabled,
                  bool                           moeFusedDownEnabled) {
    if (architecture == "qwen2") {
        return std::make_unique<Qwen2Backend>(config, weights, fusedQkv,
                                              ops, gmm, opProfiler);
    }
    if (architecture == "gemma4") {
        return std::make_unique<Gemma4Backend>(config, weights, fusedQkv,
                                               ops, gmm, opProfiler,
                                               moeGroupEnabled,
                                               moeFusedDownEnabled);
    }
    if (architecture == "qwen35moe") {
        return std::make_unique<Qwen35MoeBackend>(config, weights, fusedQkv,
                                                  ops, gmm, opProfiler,
                                                  moeGroupEnabled,
                                                  moeFusedDownEnabled);
    }
    return nullptr;
}

} // namespace mimirmind::runtime::arch