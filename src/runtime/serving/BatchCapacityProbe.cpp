// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/serving/BatchCapacityProbe.hpp"

#include <sstream>

namespace mimirmind::runtime::serving {

std::size_t BatchCapacityProbe::roundToSchedulerStep(std::size_t raw) noexcept {
    if (raw >= 128) return 128;
    if (raw >= 64)  return 64;
    if (raw >= 32)  return 32;
    if (raw >= 16)  return 16;
    if (raw >= 8)   return 8;
    if (raw >= 4)   return 4;
    if (raw >= 2)   return 2;
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
    std::size_t maxContext,
    std::size_t freeMemoryBytes) noexcept
{
    BatchCapacityEstimate est{};
    est.bandwidthGBps    = bandwidthGBps;
    est.freeVramGB       = freeMemoryBytes >> 30;
    est.weightBytes      = weightBytes;
    est.kvBytesPerToken  = kvBytesPerToken;
    est.maxContext       = maxContext;

    // Any zero input means we don't have enough to compute a real
    // estimate — safest is single-session with the reason logged so
    // the operator sees which field was missing. (freeMemoryBytes is
    // optional — 0 selects the bandwidth-tier fallback, not an error.)
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

    // A backend below the serving-bandwidth floor (integrated iGPU /
    // CPU, e.g. Meteor-Lake Xe-LPG) is single-user regardless of how
    // much memory it has — never batch it up on a memory signal alone.
    if (bandwidthGBps < 80) {
        est.sustainableBatch        = 1;
        est.servingClassRecommended = false;
        std::ostringstream os;
        os << "bandwidth=" << bandwidthGBps << " GB/s below serving floor"
           << " — single-session tier";
        est.reasoning = os.str();
        return est;
    }

    std::size_t rawBatch = 1;
    std::ostringstream os;
    if (freeMemoryBytes > 0) {
        // v2: size the batch from real headroom. Each concurrent
        // sequence owns a paged-KV pool of kvBytesPerToken * maxContext;
        // budget a conservative reserve of free memory for those pools.
        const std::size_t kvPerSeq = kvBytesPerToken * maxContext;
        // Multiply-first is exact (no /100 truncation at pool boundaries);
        // free * 50 stays well within size_t for any real device memory.
        const std::size_t budget   = freeMemoryBytes * kMemoryReservePct / 100;
        rawBatch = kvPerSeq > 0 ? (budget / kvPerSeq) : 1;
        if (rawBatch < 1) rawBatch = 1;
        if (rawBatch > kMaxServingBatch) rawBatch = kMaxServingBatch;
        est.sustainableBatch = roundToSchedulerStep(rawBatch);
        os << "vram-aware: free=" << (freeMemoryBytes >> 30) << " GiB, reserve="
           << kMemoryReservePct << "%, kv/seq=" << (kvPerSeq >> 20)
           << " MiB (kv/tok=" << kvBytesPerToken << " B x ctx=" << maxContext
           << "), bw=" << bandwidthGBps << " GB/s → raw=" << rawBatch
           << " → sustainableBatch=" << est.sustainableBatch;
    } else {
        // v1 fallback: bandwidth-tier proxy (no memory signal).
        if      (bandwidthGBps < 200)  rawBatch = 4;
        else if (bandwidthGBps < 400)  rawBatch = 16;
        else                           rawBatch = 32;
        est.sustainableBatch = roundToSchedulerStep(rawBatch);
        os << "bandwidth-tier (no vram signal): bw=" << bandwidthGBps
           << " GB/s, weight=" << (weightBytes >> 20) << " MiB, kv/tok="
           << kvBytesPerToken << " B, ctx=" << maxContext
           << " → sustainableBatch=" << est.sustainableBatch;
    }

    est.servingClassRecommended = est.sustainableBatch >= kDefaultMinServingBatch;
    est.reasoning = os.str();
    return est;
}

} // namespace mimirmind::runtime::serving
