// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/OpProfiler.hpp"

#include "runtime/CommandQueue.hpp"
#include "core/log/Log.hpp"

#include <cstdio>
#include <string>

namespace mimirmind::runtime {

namespace {

constexpr const char* kCatNames[] = {
    "norm", "attn", "matmul", "act", "resid", "router",
};

} // namespace

OpProfiler::OpProfiler(CommandQueue& queue, bool enabled)
    : _enabled{enabled},
      _queue{queue},
      _dispatchBaseline{queue.dispatchCount()}
{
    if (_enabled) {
        MM_LOG_INFO("opprof",
                    "diagnostics.traceOpTimes=true — per-op-category timing "
                    "active (flushes serialise the pipeline, adds overhead). "
                    "Summary logged every {} tokens.", kDumpEvery);
    }
}

void OpProfiler::mark(Cat c) {
    if (!_enabled) return;
    _queue.flush();
    if (_active) {
        const auto dt = std::chrono::steady_clock::now() - _t0;
        const double ms =
            std::chrono::duration<double, std::milli>(dt).count();
        const std::size_t i = static_cast<std::size_t>(_cur);
        _ms[i] += ms;
        _n[i]  += 1;
    }
    _cur = c;
    _t0 = std::chrono::steady_clock::now();
    _active = true;
}

void OpProfiler::finish() {
    if (!_enabled || !_active) return;
    _queue.flush();
    const auto dt = std::chrono::steady_clock::now() - _t0;
    const double ms =
        std::chrono::duration<double, std::milli>(dt).count();
    const std::size_t i = static_cast<std::size_t>(_cur);
    _ms[i] += ms;
    _n[i]  += 1;
    _active = false;
}

void OpProfiler::maybeDumpAndReset(std::size_t tokenIdx) {
    if (!_enabled) return;
    ++_tokensSinceDump;
    if (_tokensSinceDump < kDumpEvery) return;

    double tot = 0.0;
    for (double m : _ms) tot += m;
    if (tot <= 0.0) {
        _tokensSinceDump  = 0;
        _dispatchBaseline = _queue.dispatchCount();
        return;
    }

    // Dispatch-per-token estimate — feeds the Command-List-Replay preflight
    // gate (M-CLR.0). Combined with the empirical Xe-LPG per-launch cost
    // (~12 µs, see the xelpg-dispatch-overhead lesson), disp/tok × 12 µs
    // is the current per-token overhead budget.
    const std::size_t dispNow = _queue.dispatchCount();
    const std::size_t dispDelta =
        (dispNow >= _dispatchBaseline) ? (dispNow - _dispatchBaseline) : 0;
    const double dispPerTok =
        static_cast<double>(dispDelta) / static_cast<double>(_tokensSinceDump);

    std::string summary;
    for (std::size_t i = 0; i < kNumCats; ++i) {
        char buf[96];
        std::snprintf(buf, sizeof(buf),
                      "%s=%.1f%%/%.2fms ",
                      kCatNames[i],
                      100.0 * _ms[i] / tot,
                      _ms[i] / static_cast<double>(_tokensSinceDump));
        summary += buf;
    }
    MM_LOG_INFO("opprof",
                "over last {} tokens [ending at tok={}]: {}"
                "disp/tok={:.0f} total={:.1f}ms/tok",
                _tokensSinceDump, tokenIdx, summary,
                dispPerTok,
                tot / static_cast<double>(_tokensSinceDump));

    _ms.fill(0.0);
    _n.fill(0);
    _tokensSinceDump  = 0;
    _dispatchBaseline = dispNow;
}

} // namespace mimirmind::runtime