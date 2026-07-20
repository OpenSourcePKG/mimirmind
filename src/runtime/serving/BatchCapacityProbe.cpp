// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/serving/BatchCapacityProbe.hpp"

namespace mimirmind::runtime::serving {

std::size_t BatchCapacityProbe::roundToSchedulerStep(std::size_t raw) noexcept {
    if (raw >= 32) return 32;
    if (raw >= 16) return 16;
    if (raw >= 8)  return 8;
    if (raw >= 4)  return 4;
    if (raw >= 2)  return 2;
    return 1;
}

BatchCapacityEstimate BatchCapacityProbe::estimateConservativeFallback() noexcept {
    BatchCapacityEstimate est{};
    est.sustainableBatch        = 1;
    est.servingClassRecommended = false;
    est.reasoning               = "probe not yet implemented — assuming single-session";
    return est;
}

} // namespace mimirmind::runtime::serving
