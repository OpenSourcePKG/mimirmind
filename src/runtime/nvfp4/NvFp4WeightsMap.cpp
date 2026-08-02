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
        t.type       = m.isNvfp4Tc   ? core::gguf::GgmlType::NVFP4_TC
                     : m.isQ4K       ? core::gguf::GgmlType::Q4_K
                     : m.isQ6K       ? core::gguf::GgmlType::Q6_K
                     : m.isFp8       ? core::gguf::GgmlType::FP8_E4M3
                     : m.isNvfp4Blk  ? core::gguf::GgmlType::NVFP4_BLK
                     : m.isQ8_0      ? core::gguf::GgmlType::Q8_0
                     : m.isF32       ? core::gguf::GgmlType::F32
                                     : core::gguf::GgmlType::BF16;
        t.dimensions = m.ggufDims;
        t.nelements  = m.elems;
        // Block byte sizes: NVFP4_TC main = plain nibbles (elems/2), Q8_0/FP8
        // 34 B/32, NVFP4_BLK 20 B/32, Q4_K 144 B/256, Q6_K 210 B/256.
        t.nbytes     = m.isNvfp4Tc   ? (static_cast<std::size_t>(m.elems) / 2)
                     : m.isQ4K       ? (static_cast<std::size_t>(m.elems) / 256) * 144
                     : m.isQ6K       ? (static_cast<std::size_t>(m.elems) / 256) * 210
                     : m.isNvfp4Blk  ? (static_cast<std::size_t>(m.elems) / 32)  * 20
                     : (m.isQ8_0 || m.isFp8)
                                     ? (static_cast<std::size_t>(m.elems) / 32)  * 34
                     : static_cast<std::size_t>(m.elems) * (m.isF32 ? 4 : 2);
        t.usmPtr     = m.buffer.get();
        // E-d.5b: NVFP4_TC — the main buffer IS the plain-nibble bank; the
        // swizzled SFB + per-expert globals ride as side pointers, and
        // tcNibblePtr aliases usmPtr so the prefill grouped path reads the same.
        if (m.isNvfp4Tc) {
            t.tcNibblePtr  = m.buffer.get();
            t.tcSfbPtr     = m.tcSfbBank.get();
            t.tcGlobalsPtr = m.tcGlobalsBank.get();
        }
        tensors.push_back(std::move(t));
    }

    return core::gguf::WeightsMap::fromOwnedTensors(std::move(tensors));
}

} // namespace mimirmind::runtime::nvfp4