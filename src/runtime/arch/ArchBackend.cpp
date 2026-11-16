// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/arch/ArchBackend.hpp"

#include "model/LlmConfig.hpp"   // config.expertCount — pick dense vs MoE qwen3_5
#include "runtime/arch/Gemma4Backend.hpp"
#include "runtime/arch/Kolibri1Backend.hpp"
#include "runtime/arch/Qwen2Backend.hpp"
#include "runtime/arch/Qwen3_5DenseBackend.hpp"
#include "runtime/arch/Qwen3_5MoeBackend.hpp"
#include "runtime/arch/Qwen4ExpBackend.hpp"
#include "compute/ComputeMatmul.hpp"
#include "compute/ComputeOps.hpp"

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
    // 5.34 kolibri1 (Aleph-Alpha Kolibri-1): pure-attention SWA+full GQA MoE.
    // I-1 seam — selected here; the forward (attention 5.34.5 / MoE 5.34.6)
    // throws until implemented.
    if (architecture == "kolibri1") {
        return std::make_unique<Kolibri1Backend>(config, weights, fusedQkv,
                                                  ops, gmm, opProfiler,
                                                  moeGroupEnabled,
                                                  moeFusedDownEnabled);
    }
    return nullptr;
}

void ArchBackend::tapStage(compute::ComputeMatmul& gmm, compute::ComputeOps& ops,
                           std::string_view stage, std::size_t block,
                           const float* device, std::size_t rows, std::size_t cols) const {
    if (!stageTapWants(stage, block)) {
        return;
    }
    gmm.sync();   // the producing kernels may still be in flight
    std::vector<float> host(rows * cols);
    ops.readbackToHost(host.data(), device, host.size() * sizeof(float));
    _stageTap.onStage(stage, block, std::move(host), rows, cols);
}

void ArchBackend::tapStageMutable(compute::ComputeMatmul& gmm, compute::ComputeOps& ops,
                                  std::string_view stage, std::size_t block,
                                  float* device, std::size_t rows, std::size_t cols) const {
    if (!stageTapWants(stage, block)) {
        return;
    }
    gmm.sync();
    std::vector<float> host(rows * cols);
    ops.readbackToHost(host.data(), device, host.size() * sizeof(float));
    if (_stageTap.edit && _stageTap.edit(stage, block, host, rows, cols)) {
        ops.uploadHostBytes(device, host.data(), host.size() * sizeof(float));
    }
    _stageTap.onStage(stage, block, std::move(host), rows, cols);
}

void ArchBackend::tapStageHost(std::string_view stage, std::size_t block,
                               const float* host, std::size_t rows, std::size_t cols) const {
    if (!stageTapWants(stage, block)) {
        return;
    }
    _stageTap.onStage(stage, block, std::vector<float>(host, host + rows * cols), rows, cols);
}

void ArchBackend::tapStageI32(compute::ComputeMatmul& gmm, compute::ComputeOps& ops,
                              std::string_view stage, std::size_t block,
                              const std::int32_t* device, std::size_t rows,
                              std::size_t cols) const {
    if (!stageTapWants(stage, block)) {
        return;
    }
    gmm.sync();
    std::vector<std::int32_t> ids(rows * cols);
    ops.readbackToHost(ids.data(), device, ids.size() * sizeof(std::int32_t));
    tapStageHostI32(stage, block, ids.data(), rows, cols);
}

void ArchBackend::tapStageHostI32(std::string_view stage, std::size_t block,
                                  const std::int32_t* host, std::size_t rows,
                                  std::size_t cols) const {
    if (!stageTapWants(stage, block)) {
        return;
    }
    std::vector<float> v(rows * cols);
    for (std::size_t i = 0; i < v.size(); ++i) {
        v[i] = static_cast<float>(host[i]);
    }
    _stageTap.onStage(stage, block, std::move(v), rows, cols);
}

} // namespace mimirmind::runtime::arch