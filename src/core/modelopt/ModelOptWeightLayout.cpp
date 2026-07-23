// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/modelopt/ModelOptWeightLayout.hpp"

#include <stdexcept>
#include <string>

namespace mimirmind::core::modelopt {

namespace {

using safetensors::SafetensorsDtype;
using safetensors::SafetensorsTensor;

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("modelopt weight: " + msg);
}

void requireScalar(const SafetensorsTensor* t, const char* role,
                   const std::string& module) {
    if (t == nullptr) {
        fail(module + " missing " + role);
    }
    if (t->dtype != SafetensorsDtype::F32) {
        fail(module + " " + role + " is not F32");
    }
    if (t->nelements != 1) {
        fail(module + " " + role + " is not a scalar (nelements="
             + std::to_string(t->nelements) + ")");
    }
}

} // namespace

ModelOptWeightLayout validateWeightLayout(
    ModelOptQuantScheme                   scheme,
    std::uint16_t                         groupSize,
    const SafetensorsTensor&              weight,
    const SafetensorsTensor*              blockScale,
    const SafetensorsTensor*              globalScale,
    const SafetensorsTensor*              weightScale,
    const SafetensorsTensor*              inputScale) {
    const ModelOptSchemeInfo& info = schemeInfo(scheme);
    const std::string mod = weight.name;

    // --- packed weight: must be 2D [out, packedIn] of the scheme's dtype ---
    if (weight.dtype != info.weightDtype) {
        fail(mod + " weight dtype mismatch (expected the scheme's weight dtype)");
    }
    if (weight.shape.size() != 2) {
        fail(mod + " weight is not 2-D");
    }
    const std::uint64_t out      = weight.shape[0];
    const std::uint64_t packedIn = weight.shape[1];
    if (out == 0 || packedIn == 0) {
        fail(mod + " weight has a zero dimension");
    }
    const std::uint64_t inFeatures = packedIn * info.weightPackFactor;

    // Byte size must agree with the scheme's packed-row math.
    const std::size_t rowBytes = packedRowBytes(scheme, inFeatures);
    if (rowBytes == 0 || weight.nbytes != rowBytes * out) {
        fail(mod + " weight byte size inconsistent with scheme packing");
    }

    ModelOptWeightLayout layout;
    layout.scheme      = scheme;
    layout.outFeatures = out;
    layout.inFeatures  = inFeatures;
    layout.groupSize   = info.blockScaleGroupSize;

    if (info.hasBlockScale) {
        // NVFP4: per-group block scale [out, in/group] of F8_E4M3.
        if (groupSize != 0 && groupSize != info.blockScaleGroupSize) {
            fail(mod + " config group_size " + std::to_string(groupSize)
                 + " != scheme group_size " + std::to_string(info.blockScaleGroupSize));
        }
        if (blockScale == nullptr) {
            fail(mod + " missing weight_scale (block scale)");
        }
        if (blockScale->dtype != info.blockScaleDtype) {
            fail(mod + " block scale dtype mismatch");
        }
        if (blockScale->shape.size() != 2 || blockScale->shape[0] != out) {
            fail(mod + " block scale shape does not match weight rows");
        }
        const std::size_t cols = blockScaleCols(scheme, inFeatures);
        if (cols == 0 || blockScale->shape[1] != cols) {
            fail(mod + " block scale columns != in/group_size");
        }
    }
    if (info.hasGlobalScale) {
        requireScalar(globalScale, "weight_scale_2 (global scale)", mod);
    }
    if (info.hasTensorWeightScale) {
        requireScalar(weightScale, "weight_scale", mod);
    }
    if (info.hasInputScale) {
        requireScalar(inputScale, "input_scale", mod);
    }

    return layout;
}

} // namespace mimirmind::core::modelopt