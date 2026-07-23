// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/modelopt/ModelOptQuant.hpp"

namespace mimirmind::core::modelopt {

const ModelOptSchemeInfo& schemeInfo(ModelOptQuantScheme s) noexcept {
    // group_size 16 verified against the on-disk index: gate_proj weight
    // U8 [512,1024] (real in = 2048) pairs with weight_scale [512,128] →
    // 2048/128 = 16; down_proj [2048,256] (in = 512) with [2048,32] →
    // 512/32 = 16.
    static constexpr ModelOptSchemeInfo kNvfp4{
        /* name                 */ "W4A16_NVFP4",
        /* weightDtype          */ SafetensorsDtype::U8,
        /* weightPackFactor     */ 2,
        /* blockScaleGroupSize  */ 16,
        /* blockScaleDtype      */ SafetensorsDtype::F8_E4M3,
        /* hasBlockScale        */ true,
        /* hasTensorWeightScale */ false,
        /* hasGlobalScale       */ true,
        /* hasInputScale        */ false,
    };
    static constexpr ModelOptSchemeInfo kFp8{
        /* name                 */ "FP8",
        /* weightDtype          */ SafetensorsDtype::F8_E4M3,
        /* weightPackFactor     */ 1,
        /* blockScaleGroupSize  */ 0,
        /* blockScaleDtype      */ SafetensorsDtype::Unknown,
        /* hasBlockScale        */ false,
        /* hasTensorWeightScale */ true,
        /* hasGlobalScale       */ false,
        /* hasInputScale        */ true,
    };

    switch (s) {
        case ModelOptQuantScheme::FP8_E4M3:         return kFp8;
        case ModelOptQuantScheme::NVFP4_E2M1_BLK16:
        default:                                    return kNvfp4;
    }
}

std::optional<ModelOptQuantScheme>
schemeFromQuantAlgo(std::string_view quantAlgo) noexcept {
    if (quantAlgo == "W4A16_NVFP4") return ModelOptQuantScheme::NVFP4_E2M1_BLK16;
    if (quantAlgo == "FP8")         return ModelOptQuantScheme::FP8_E4M3;
    return std::nullopt;
}

std::size_t packedRowBytes(ModelOptQuantScheme s, std::size_t inFeatures) noexcept {
    const auto& info = schemeInfo(s);
    if (info.weightPackFactor == 0 || (inFeatures % info.weightPackFactor) != 0) {
        return 0;
    }
    if (info.hasBlockScale && info.blockScaleGroupSize != 0
        && (inFeatures % info.blockScaleGroupSize) != 0) {
        return 0;
    }
    const std::size_t stored = inFeatures / info.weightPackFactor;
    return stored * safetensors::dtypeWidth(info.weightDtype);
}

std::size_t blockScaleCols(ModelOptQuantScheme s, std::size_t inFeatures) noexcept {
    const auto& info = schemeInfo(s);
    if (!info.hasBlockScale || info.blockScaleGroupSize == 0
        || (inFeatures % info.blockScaleGroupSize) != 0) {
        return 0;
    }
    return inFeatures / info.blockScaleGroupSize;
}

} // namespace mimirmind::core::modelopt