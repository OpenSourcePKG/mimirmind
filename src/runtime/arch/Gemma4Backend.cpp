#include "runtime/arch/Gemma4Backend.hpp"

#include "runtime/arch/GemmaBaseBackend.hpp"
#include "runtime/arch/Gemma4DenseBackend.hpp"
#include "runtime/arch/Gemma4E4BBackend.hpp"
#include "runtime/arch/Gemma4MoeBackend.hpp"

#include "model/LlmConfig.hpp"
#include "model/WeightsMap.hpp"

namespace mimirmind::runtime::arch {

Gemma4Backend::Gemma4Backend(const model::LlmConfig&        config,
                             const model::WeightsMap&       weights,
                             const model::FusedQkvWeights*  fusedQkv,
                             compute::GpuOps&               ops,
                             compute::GpuMatmul&            gmm,
                             runtime::OpProfiler&           opProfiler,
                             bool                           moeGroupEnabled) {
    if (config.expertCount > 0) {
        _impl = std::make_unique<Gemma4MoeBackend>(
            config, weights, fusedQkv, ops, gmm, opProfiler,
            moeGroupEnabled);
    } else if (weights.findBlock(0, "inp_gate.weight") != nullptr) {
        // Gemma 4 E-series (E4B / E2B): MatFormer + Per-Layer Embeddings.
        // Detected by the presence of the per-block PLE gate tensor,
        // which the classic dense variants (4B / 12B) do not have.
        _impl = std::make_unique<Gemma4E4BBackend>(
            config, weights, fusedQkv, ops, gmm, opProfiler);
    } else {
        _impl = std::make_unique<Gemma4DenseBackend>(
            config, weights, fusedQkv, ops, gmm, opProfiler);
    }
}

// Out-of-line so `GemmaBaseBackend` needs only a forward-decl in the .hpp.
Gemma4Backend::~Gemma4Backend() = default;

void Gemma4Backend::runBlock(std::size_t   blockIdx,
                             float*        x,
                             std::size_t   T,
                             KvCache&      cache,
                             BlockBuffers& buffers,
                             bool          traceBlock0) {
    _impl->runBlock(blockIdx, x, T, cache, buffers, traceBlock0);
}

std::vector<std::size_t> Gemma4Backend::kvDimPerLayer() const {
    return _impl->kvDimPerLayer();
}

std::vector<std::size_t> Gemma4Backend::kvSourceLayerPerLayer() const {
    return _impl->kvSourceLayerPerLayer();
}

std::pair<std::size_t, std::size_t> Gemma4Backend::maxQKVDims() const {
    return _impl->maxQKVDims();
}

void Gemma4Backend::setParityDumpPrefix(const std::string& prefix) noexcept {
    _impl->setParityDumpPrefix(prefix);
}

void Gemma4Backend::prepareForward(std::span<const std::int32_t> tokIds,
                                   const float*                  hiddenStates,
                                   std::size_t                   T) {
    _impl->prepareForward(tokIds, hiddenStates, T);
}

} // namespace mimirmind::runtime::arch