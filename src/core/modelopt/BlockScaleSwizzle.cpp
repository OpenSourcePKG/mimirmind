// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/modelopt/BlockScaleSwizzle.hpp"

#include <cstring>

namespace mimirmind::core::modelopt {

void swizzleBlockScale(const std::uint8_t* src, std::uint64_t rows,
                       std::uint64_t ksf, std::uint8_t* dst) noexcept {
    const std::uint64_t mTiles   = sfRowTiles(rows);
    const std::uint64_t ksfTiles = sfKTiles(ksf);
    std::memset(dst, 0, swizzledBlockScaleBytes(rows, ksf)); // zero the padding

    for (std::uint64_t m = 0; m < rows; ++m) {
        const std::uint8_t* row = src + m * ksf;
        for (std::uint64_t s = 0; s < ksf; ++s) {
            dst[swizzledScaleOffset(m, s, mTiles, ksfTiles)] = row[s];
        }
    }
}

} // namespace mimirmind::core::modelopt