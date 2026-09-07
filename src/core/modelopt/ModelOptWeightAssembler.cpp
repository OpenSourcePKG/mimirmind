// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/modelopt/ModelOptWeightAssembler.hpp"

#include "core/safetensors/SafetensorsModel.hpp"

#include <stdexcept>
#include <string>

namespace mimirmind::core::modelopt {

namespace {

using safetensors::SafetensorsTensor;

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("modelopt weight: " + msg);
}

} // namespace

bool ModelOptWeightAssembler::isQuantized(std::string_view module) const {
    const auto scheme = _config.schemeForTensor(module);
    if (!scheme.has_value()) {
        return false;
    }
    // Checkpoint truth wins over a uniform scheme. ModelOpt only quantises
    // nn.Linear modules; under a top-level `quant_algo` (e.g. the
    // Qwen3.5-122B-A10B NVFP4 checkpoint) implicitly-unquantised modules such
    // as embed_tokens are NOT listed in exclude_modules, so the scheme
    // resolves for them although the tensor is stored BF16. Treat a module as
    // quantised only if its weight actually carries the scheme's packed
    // dtype; a missing weight stays "quantised" so assemble() fails loudly.
    const SafetensorsTensor* w = _model.find(std::string(module) + ".weight");
    if (w == nullptr) {
        return true;
    }
    return w->dtype == schemeInfo(*scheme).weightDtype;
}

ModelOptWeight ModelOptWeightAssembler::assemble(std::string_view module) const {
    const auto mq = _config.resolve(module);
    if (!mq.has_value()) {
        fail(std::string(module) + " is not a quantised ModelOpt module");
    }
    const ModelOptSchemeInfo& info = schemeInfo(mq->scheme);
    const std::string base(module);

    const SafetensorsTensor* weight = _model.find(base + ".weight");
    if (weight == nullptr) {
        fail(base + " has no .weight tensor");
    }

    const SafetensorsTensor* blockScale  = nullptr;
    const SafetensorsTensor* globalScale = nullptr;
    const SafetensorsTensor* weightScale = nullptr;
    const SafetensorsTensor* inputScale  = nullptr;
    if (info.hasBlockScale) {
        blockScale = _model.find(base + ".weight_scale");
    }
    if (info.hasGlobalScale) {
        globalScale = _model.find(base + ".weight_scale_2");
    }
    if (info.hasTensorWeightScale) {
        weightScale = _model.find(base + ".weight_scale");
    }
    if (info.hasInputScale) {
        inputScale = _model.find(base + ".input_scale");
    }

    ModelOptWeight w;
    w.layout = validateWeightLayout(mq->scheme, mq->groupSize, *weight,
                                    blockScale, globalScale, weightScale, inputScale);

    w.packedWeight = _model.tensorBytes(base + ".weight");
    if (blockScale)  w.blockScale  = _model.tensorBytes(base + ".weight_scale");
    if (globalScale) w.globalScale = _model.tensorBytes(base + ".weight_scale_2");
    if (weightScale) w.weightScale = _model.tensorBytes(base + ".weight_scale");
    if (inputScale)  w.inputScale  = _model.tensorBytes(base + ".input_scale");

    return w;
}

} // namespace mimirmind::core::modelopt