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
    return _config.schemeForTensor(module).has_value();
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