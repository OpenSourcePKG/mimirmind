// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "hugin/train/Parallel.hpp"

#include <algorithm>
#include <atomic>
#include <exception>
#include <thread>
#include <vector>

namespace mimirmind::hugin::train {

namespace {
std::atomic<std::size_t> g_threads{0};
} // namespace

void Parallel::setThreads(std::size_t n) noexcept { g_threads.store(n); }

std::size_t Parallel::threads() noexcept {
    const std::size_t n = g_threads.load();
    if (n > 0) return n;
    const std::size_t hw = std::thread::hardware_concurrency();
    return std::max<std::size_t>(1, hw / 2);
}

void Parallel::forRange(std::size_t n, std::size_t minChunk,
                        const std::function<void(std::size_t, std::size_t)>& fn) {
    if (n == 0) return;
    const std::size_t maxByWork = std::max<std::size_t>(1, n / std::max<std::size_t>(1, minChunk));
    const std::size_t workers   = std::min(threads(), maxByWork);
    if (workers <= 1) {
        fn(0, n);
        return;
    }
    const std::size_t chunk = (n + workers - 1) / workers;
    std::vector<std::thread>        pool;
    std::vector<std::exception_ptr> errors(workers);
    pool.reserve(workers - 1);
    for (std::size_t w = 1; w < workers; ++w) {
        const std::size_t b = w * chunk, e = std::min(n, b + chunk);
        if (b >= e) break;
        pool.emplace_back([&, w, b, e] {
            try {
                fn(b, e);
            } catch (...) {
                errors[w] = std::current_exception();
            }
        });
    }
    try {
        fn(0, std::min(n, chunk));
    } catch (...) {
        errors[0] = std::current_exception();
    }
    for (auto& t : pool) t.join();
    for (const auto& e : errors) {
        if (e) std::rethrow_exception(e);
    }
}

} // namespace mimirmind::hugin::train
