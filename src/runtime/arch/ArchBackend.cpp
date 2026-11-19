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
#include "compute/IRuneSteerOps.hpp"

#include <algorithm>

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

void ArchBackend::setResidualSteers(
    compute::ComputeOps& ops, std::size_t dModel,
    const std::vector<std::pair<std::size_t, const float*>>& steers) {
    _steerByBlock.clear();
    _steerOps    = nullptr;
    _steerDModel = dModel;
    for (const auto& [block, vec] : steers) {
        if (vec == nullptr) {
            continue;
        }
        if (_steerByBlock.size() <= block) {
            _steerByBlock.resize(block + 1, nullptr);
        }
        if (_steerByBlock[block] != nullptr) {
            throw std::runtime_error("setResidualSteers: two steer vectors for block " +
                                     std::to_string(block) + " (sum them first)");
        }
        _steerByBlock[block] = vec;
    }
    if (!_steerByBlock.empty()) {
        _steerOps = &ops;
    }
}

void ArchBackend::setExpertMasks(
    compute::ComputeOps& ops, std::size_t nExperts,
    const std::vector<std::pair<std::size_t, const float*>>& masks) {
    _maskByBlock.clear();
    _maskOps      = nullptr;
    _maskNExperts = nExperts;
    for (const auto& [block, bias] : masks) {
        if (bias == nullptr) {
            continue;
        }
        if (_maskByBlock.size() <= block) {
            _maskByBlock.resize(block + 1, nullptr);
        }
        if (_maskByBlock[block] != nullptr) {
            throw std::runtime_error("setExpertMasks: two expert masks for block " +
                                     std::to_string(block) + " (combine them first)");
        }
        _maskByBlock[block] = bias;
    }
    if (!_maskByBlock.empty()) {
        _maskOps = &ops;
    }
}

void ArchBackend::setLowRanks(compute::ComputeOps& ops, compute::ComputeMatmul& gmm,
                              std::vector<LowRankOverlay> overlays) {
    _lrByBlock.clear();
    _lrOps = nullptr;
    _lrGmm = nullptr;
    if (overlays.empty()) {
        return;
    }
    std::size_t maxRank = 0, maxOut = 0;
    for (auto& o : overlays) {
        if (o.a == nullptr || o.b == nullptr || o.rank == 0 || o.in == 0 || o.out == 0) {
            throw std::runtime_error("setLowRanks: incomplete overlay on block " +
                                     std::to_string(o.block) + " '" + o.tensor + "'");
        }
        if (_lrByBlock.size() <= o.block) {
            _lrByBlock.resize(o.block + 1);
        }
        for (const auto& prev : _lrByBlock[o.block]) {
            if (prev.tensor == o.tensor) {
                throw std::runtime_error("setLowRanks: two lowrank overlays on block " +
                                         std::to_string(o.block) + " '" + o.tensor +
                                         "' (stacking is not supported)");
            }
        }
        maxRank = std::max(maxRank, o.rank);
        maxOut  = std::max(maxOut, o.out);
        _lrByBlock[o.block].push_back(std::move(o));
    }
    // Fixed scratch, allocated once here (never during a forward / capture);
    // applyLowRank walks the rows in chunks of kLowRankChunk.
    _lrTmp = ops.allocate(kLowRankChunk * maxRank * sizeof(float));
    _lrOut = ops.allocate(kLowRankChunk * maxOut * sizeof(float));
    _lrOps = &ops;
    _lrGmm = &gmm;
}

bool ArchBackend::hasLowRank(std::size_t block, std::string_view tensor) const noexcept {
    if (_lrOps == nullptr || block >= _lrByBlock.size()) {
        return false;
    }
    for (const auto& o : _lrByBlock[block]) {
        if (o.tensor == tensor) {
            return true;
        }
    }
    return false;
}

bool ArchBackend::applyLowRank(std::size_t block, std::string_view tensor, const float* in,
                               std::size_t rows, float* out, float* matmulScratch) {
    if (_lrOps == nullptr || block >= _lrByBlock.size()) {
        return false;
    }
    for (const auto& o : _lrByBlock[block]) {
        if (o.tensor != tensor) {
            continue;
        }
        float* tmp  = _lrTmp.as<float>();
        float* outc = _lrOut.as<float>();
        for (std::size_t r0 = 0; r0 < rows; r0 += kLowRankChunk) {
            const std::size_t n = std::min(kLowRankChunk, rows - r0);
            _lrGmm->matmulAsync(core::gguf::GgmlType::F32, o.a, o.rank, o.in, in + r0 * o.in, n,
                                tmp, matmulScratch);
            _lrGmm->matmulAsync(core::gguf::GgmlType::F32, o.b, o.out, o.rank, tmp, n, outc,
                                matmulScratch);
            _lrOps->addResidualAsync(out + r0 * o.out, outc, n * o.out);
        }
        return true;
    }
    return false;
}

void ArchBackend::addExpertMask(const float* bias, float* logits, std::size_t rows) const {
    // logits[r, e] += bias[e]: 0 for kept experts, kExpertMaskedLogit for disabled.
    _maskOps->addBiasAsync(logits, rows, _maskNExperts, bias);
}

void ArchBackend::setConditionalSteers(compute::ComputeOps& ops, std::size_t dModel,
                                       std::vector<SteerCondition> conditions,
                                       std::vector<ConditionalSteer> steers) {
    clearConditionalSteers();
    if (steers.empty()) {
        if (!conditions.empty()) {
            throw std::runtime_error("setConditionalSteers: conditions without a steer");
        }
        return;
    }
    if (ops.runeSteerOps() == nullptr) {
        throw std::runtime_error("setConditionalSteers: this compute backend has no "
                                 "projection / conditional steer ops (IRuneSteerOps)");
    }
    std::size_t maxBlock = 0;
    for (const auto& c : conditions) {
        if (c.direction == nullptr) {
            throw std::runtime_error("setConditionalSteers: condition without a direction");
        }
        maxBlock = std::max(maxBlock, c.block);
    }
    for (const auto& s : steers) {
        if (s.values == nullptr) {
            throw std::runtime_error("setConditionalSteers: steer without values");
        }
        if (s.condition >= 0) {
            const auto ci = static_cast<std::size_t>(s.condition);
            if (ci >= conditions.size()) {
                throw std::runtime_error("setConditionalSteers: steer references condition " +
                                         std::to_string(s.condition) + " of " +
                                         std::to_string(conditions.size()));
            }
            if (conditions[ci].block > s.block) {
                throw std::runtime_error("setConditionalSteers: condition on block " +
                                         std::to_string(conditions[ci].block) +
                                         " is scored after the steered block " +
                                         std::to_string(s.block));
            }
        }
        maxBlock = std::max(maxBlock, s.block);
    }
    _condAtBlock.assign(maxBlock + 1, false);
    for (const auto& c : conditions) {
        _condAtBlock[c.block] = true;
    }
    for (const auto& s : steers) {
        _condAtBlock[s.block] = true;
    }
    _conditions = std::move(conditions);
    _condSteers = std::move(steers);
    _condDModel = dModel;
    _condOps    = &ops;
    // Covers every decode batch and typical prefill chunks, so no allocation
    // happens inside a captured graph.
    constexpr std::size_t kInitialRows = 65536;
    ensureConditionalCapacity(kInitialRows);
}

void ArchBackend::clearConditionalSteers() noexcept {
    _condOps = nullptr;
    _conditions.clear();
    _condSteers.clear();
    _condAtBlock.clear();
    _condScores.clear();
    _condProj = compute::ComputeBuffer{};
    _condRetired.clear();
    _condCapacity = 0;
}

void ArchBackend::ensureConditionalCapacity(std::size_t rows) const {
    if (rows <= _condCapacity) {
        return;
    }
    const std::size_t cap   = std::max(rows, _condCapacity * 2);
    const std::size_t bytes = cap * sizeof(float);
    for (auto& b : _condScores) {
        _condRetired.push_back(std::move(b));
    }
    _condScores.clear();
    for (std::size_t i = 0; i < _conditions.size(); ++i) {
        _condScores.push_back(_condOps->allocate(bytes));
    }
    if (_condProj) {
        _condRetired.push_back(std::move(_condProj));
    }
    _condProj     = _condOps->allocate(bytes);
    _condCapacity = cap;
}

void ArchBackend::applyConditionalSteers(std::size_t block, float* x, std::size_t rows) const {
    if (_condOps == nullptr || rows == 0) {
        return;
    }
    ensureConditionalCapacity(rows);
    compute::IRuneSteerOps& rs = *_condOps->runeSteerOps();
    for (std::size_t i = 0; i < _conditions.size(); ++i) {
        if (_conditions[i].block == block) {
            rs.rowDotAsync(x, rows, _condDModel, _conditions[i].direction,
                           _condScores[i].as<float>(), /*normalize=*/true);
        }
    }
    for (const auto& s : _condSteers) {
        if (s.block != block) {
            continue;
        }
        const float* gate      = nullptr;
        float        threshold = 0.0F;
        if (s.condition >= 0) {
            const auto ci = static_cast<std::size_t>(s.condition);
            gate      = _condScores[ci].as<float>();
            threshold = _conditions[ci].threshold;
        }
        if (s.project) {
            // x -= alpha (x.v) v, v unit: coefficient = x.v, scale = -alpha.
            rs.rowDotAsync(x, rows, _condDModel, s.values, _condProj.as<float>(),
                           /*normalize=*/false);
            rs.steerRowsAsync(x, rows, _condDModel, s.values, -s.alpha, gate, threshold,
                              _condProj.as<float>());
        } else {
            rs.steerRowsAsync(x, rows, _condDModel, s.values, 1.0F, gate, threshold, nullptr);
        }
    }
}

std::vector<float> ArchBackend::readConditionScores(std::size_t condition,
                                                    std::size_t rows) const {
    if (_condOps == nullptr || condition >= _conditions.size()) {
        throw std::runtime_error("readConditionScores: no condition " +
                                 std::to_string(condition));
    }
    rows = std::min(rows, _condCapacity);
    std::vector<float> out(rows);
    _condOps->readbackToHost(out.data(), _condScores[condition].as<float>(),
                             rows * sizeof(float));
    return out;
}

void ArchBackend::addResidualSteer(const float* steer, float* x, std::size_t rows) const {
    // Broadcast row add on the device stream: x[r, :] += steer for every row.
    _steerOps->addBiasAsync(x, rows, _steerDModel, steer);
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