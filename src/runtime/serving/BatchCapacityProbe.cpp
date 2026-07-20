// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/serving/BatchCapacityProbe.hpp"

#include <sstream>

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

BatchCapacityEstimate BatchCapacityProbe::estimate(
    std::size_t bandwidthGBps,
    std::size_t weightBytes,
    std::size_t kvBytesPerToken,
    std::size_t maxContext) noexcept
{
    BatchCapacityEstimate est{};
    est.bandwidthGBps    = bandwidthGBps;
    est.weightBytes      = weightBytes;
    est.kvBytesPerToken  = kvBytesPerToken;
    est.maxContext       = maxContext;

    // Any zero input means we don't have enough to compute a real
    // estimate — safest is single-session with the reason logged so
    // the operator sees which field was missing.
    if (bandwidthGBps == 0 || weightBytes == 0
        || kvBytesPerToken == 0 || maxContext == 0)
    {
        est.sustainableBatch        = 1;
        est.servingClassRecommended = false;
        std::ostringstream os;
        os << "insufficient probe data (bandwidth=" << bandwidthGBps
           << " GB/s, weight=" << weightBytes
           << " B, kv/tok=" << kvBytesPerToken
           << " B, ctx=" << maxContext
           << ") — assuming single-session";
        est.reasoning = os.str();
        return est;
    }

    // v1 heuristic: bandwidth-tier proxy for total device memory. A
    // per-device VRAM probe replaces this later.
    std::size_t rawBatch = 1;
    if      (bandwidthGBps <  80)  rawBatch = 1;
    else if (bandwidthGBps < 200)  rawBatch = 4;
    else if (bandwidthGBps < 400)  rawBatch = 16;
    else                           rawBatch = 32;

    est.sustainableBatch        = roundToSchedulerStep(rawBatch);
    est.servingClassRecommended = est.sustainableBatch >= kDefaultMinServingBatch;

    std::ostringstream os;
    os << "bandwidth=" << bandwidthGBps << " GB/s, weight=" << (weightBytes >> 20)
       << " MiB, kv/tok=" << kvBytesPerToken << " B, ctx=" << maxContext
       << " → sustainableBatch=" << est.sustainableBatch;
    est.reasoning = os.str();
    return est;
}

} // namespace mimirmind::runtime::serving
