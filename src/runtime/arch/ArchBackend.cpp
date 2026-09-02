// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/arch/ArchBackend.hpp"

#include "model/LlmConfig.hpp"   // config.expertCount — pick dense vs MoE qwen3_5
#include "runtime/arch/Gemma4Backend.hpp"
#include "runtime/arch/Qwen2Backend.hpp"
#include "runtime/arch/Qwen3_5DenseBackend.hpp"
#include "runtime/arch/Qwen3_5MoeBackend.hpp"
#include "runtime/arch/Qwen4ExpBackend.hpp"

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
        // The qwen3_5 arch ships in two FFN flavours (HF model_type
        // qwen3_5_moe vs the dense qwen3_5 text tower). expertCount == 0 => the
        // dense SwiGLU variant (e.g. Qwen3.8-27B); otherwise routed experts.
        if (config.expertCount == 0) {
            return std::make_unique<Qwen3_5DenseBackend>(config, weights,
                                                         fusedQkv, ops, gmm,
                                                         opProfiler);
        }
        return std::make_unique<Qwen3_5MoeBackend>(config, weights, fusedQkv,
                                                  ops, gmm, opProfiler,
                                                  moeGroupEnabled,
                                                  moeFusedDownEnabled);
    }
    // 5.27 qwen4_exp (Qwen3.8-Flash-Next): same hybrid MoE backbone +
    // Hyper-Connections + PLE n-gram. Derives from Qwen3_5MoeBackend; the two
    // new components are stubbed at I-1 (behaves as the MoE backbone).
    if (architecture == "qwen4_exp") {
        return std::make_unique<Qwen4ExpBackend>(config, weights, fusedQkv,
                                                 ops, gmm, opProfiler,
                                                 moeGroupEnabled,
                                                 moeFusedDownEnabled);
    }
    return nullptr;
}

} // namespace mimirmind::runtime::arch