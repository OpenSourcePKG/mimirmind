// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/serving/PreemptionPolicy.hpp"

#include <stdexcept>

namespace mimirmind::runtime::serving {

PreemptionPolicy::PreemptionPolicy(double freeBlockThreshold)
    : _threshold(freeBlockThreshold)
{
    if (!(freeBlockThreshold >= 0.0 && freeBlockThreshold <= 1.0)) {
        // Also catches NaN — the negation covers all non-in-range cases.
        throw std::invalid_argument{
            "PreemptionPolicy: freeBlockThreshold must be in [0.0, 1.0]"};
    }
}

bool PreemptionPolicy::shouldPreempt(std::size_t freeBlocks,
                                     std::size_t totalBlocks,
                                     std::size_t numActive) const noexcept
{
    if (totalBlocks == 0)  return false;
    if (numActive  <= 1)   return false;
    const double freeRatio =
        static_cast<double>(freeBlocks) / static_cast<double>(totalBlocks);
    return freeRatio < _threshold;
}

} // namespace mimirmind::runtime::serving