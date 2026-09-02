// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/arch/Qwen4ExpBackend.hpp"

namespace mimirmind::runtime::arch {

// I-1 seam: forward everything to the Qwen3_5MoeBackend backbone. Hyper-
// Connections (I-3) and PLE n-gram embeddings (I-4) hook in later by overriding
// the residual-add / layer-input seams in the shared Qwen3_5Backend::runBlock.
Qwen4ExpBackend::Qwen4ExpBackend(const model::LlmConfig&       config,
                                 const core::gguf::WeightsMap& weights,
                                 const model::FusedQkvWeights* fusedQkv,
                                 compute::ComputeOps&          ops,
                                 compute::ComputeMatmul&       gmm,
                                 runtime::OpProfiler&          opProfiler,
                                 bool                          moeGroupEnabled,
                                 bool                          moeFusedDownEnabled)
    : Qwen3_5MoeBackend(config, weights, fusedQkv, ops, gmm, opProfiler,
                        moeGroupEnabled, moeFusedDownEnabled) {}

} // namespace mimirmind::runtime::arch
