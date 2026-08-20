// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/arch/ArchBackend.hpp"

#include "runtime/arch/Gemma4Backend.hpp"
#include "runtime/arch/Qwen2Backend.hpp"
#include "runtime/arch/Qwen3_5MoeBackend.hpp"

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
    // Qwen2Backend implements the Llama-family dense decoder — RMSNorm, RoPE,
    // SwiGLU MLP, GQA, and OPTIONAL QKV bias (added only when the bias tensors
    // are present). Llama-3.2 is exactly that shape minus the QKV bias, so it
    // routes through the same backend; the config reads its `llama.*` GGUF
    // metadata via the general.architecture prefix. (First consumer: the
    // Orpheus TTS acoustic model, roadmap 8.13.2.)
    if (architecture == "qwen2" || architecture == "llama") {
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
        return std::make_unique<Qwen3_5MoeBackend>(config, weights, fusedQkv,
                                                  ops, gmm, opProfiler,
                                                  moeGroupEnabled,
                                                  moeFusedDownEnabled);
    }
    return nullptr;
}

} // namespace mimirmind::runtime::arch