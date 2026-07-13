#include "runtime/arch/ArchBackend.hpp"

#include "runtime/arch/Gemma4Backend.hpp"
#include "runtime/arch/Qwen2Backend.hpp"

namespace mimirmind::runtime::arch {

std::unique_ptr<ArchBackend>
createArchBackend(const std::string&             architecture,
                  const model::LlmConfig&        config,
                  const core::gguf::WeightsMap&       weights,
                  const model::FusedQkvWeights*  fusedQkv,
                  compute::GpuOps&               ops,
                  compute::GpuMatmul&            gmm,
                  OpProfiler&                    opProfiler,
                  bool                           moeGroupEnabled) {
    if (architecture == "qwen2") {
        return std::make_unique<Qwen2Backend>(config, weights, fusedQkv,
                                              ops, gmm, opProfiler);
    }
    if (architecture == "gemma4") {
        return std::make_unique<Gemma4Backend>(config, weights, fusedQkv,
                                               ops, gmm, opProfiler,
                                               moeGroupEnabled);
    }
    return nullptr;
}

} // namespace mimirmind::runtime::arch