// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/nvfp4/NvFp4WeightsMap.hpp"

#include "core/gguf/GgufReader.hpp" // GgufTensor
#include "core/gguf/GgufTypes.hpp"

#include <cstddef>
#include <utility>

namespace mimirmind::runtime::nvfp4 {

core::gguf::WeightsMap buildBf16WeightsMap(std::vector<MaterializedTensor>& mats) {
    std::vector<core::gguf::GgufTensor> tensors;
    tensors.reserve(mats.size());

    for (MaterializedTensor& m : mats) {
        core::gguf::GgufTensor t{};
        t.name       = m.ggufName;
        t.type       = m.isF32 ? core::gguf::GgmlType::F32
                               : core::gguf::GgmlType::BF16;
        t.dimensions = m.ggufDims;
        t.nelements  = m.elems;
        t.nbytes     = static_cast<std::size_t>(m.elems) * (m.isF32 ? 4 : 2);
        t.usmPtr     = m.buffer.get();
        tensors.push_back(std::move(t));
    }

    return core::gguf::WeightsMap::fromOwnedTensors(std::move(tensors));
}

} // namespace mimirmind::runtime::nvfp4