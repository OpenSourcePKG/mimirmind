// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstddef>
#include <functional>

namespace mimirmind::hugin::train {

/**
 * Minimal fork/join for the CPU train ops (7.4.3). Splits [0, n) into
 * contiguous chunks, one per worker thread. Uses half of the hardware threads
 * by default, so a training run on the Spark leaves the live server's host
 * threads (batcher, HTTP) room.
 */
class Parallel {
public:
    /// fn(begin, end) on disjoint chunks covering [0, n); returns after all ran.
    /// Runs inline when n < minChunk * 2 or one thread is configured.
    static void forRange(std::size_t n, std::size_t minChunk,
                         const std::function<void(std::size_t, std::size_t)>& fn);

    /// Worker count (>= 1). 0 = default (half the hardware threads).
    static void        setThreads(std::size_t n) noexcept;
    [[nodiscard]] static std::size_t threads() noexcept;
};

} // namespace mimirmind::hugin::train
