// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/perf/OpProfiler.hpp"

#include "core/log/Log.hpp"

#include <cstdio>
#include <string>

// The L0 CommandQueue-tied timing pipeline is only reachable through
// the `OpProfiler(CommandQueue&, bool)` ctor, which lives in a
// TU-scope that only compiles when L0 is on. Without L0, the class
// keeps its default-ctor no-op behaviour: `mark/finish/maybeDumpAndReset`
// bail on `!_enabled`, and `_enabled` can only be set true by the L0
// ctor. HIP-only and CPU-only builds get a valid `OpProfiler` symbol
// via cheap stubs so arch backends can call the methods unconditionally.
#ifdef MIMIRMIND_HAVE_L0
#include "core/gpu/l0/CommandQueue.hpp"
#endif

namespace mimirmind::runtime {

namespace {

constexpr const char* kCatNames[] = {
    "norm", "attn", "matmul", "act", "resid", "router",
};

} // namespace

#ifdef MIMIRMIND_HAVE_L0

OpProfiler::OpProfiler(CommandQueue& queue, bool enabled)
    : _enabled{enabled},
      _queue{&queue},
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
    _queue->flush();
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
    _queue->flush();
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
        _dispatchBaseline = _queue->dispatchCount();
        return;
    }

    // Dispatch-per-token estimate — feeds the Command-List-Replay preflight
    // gate (M-CLR.0). Combined with the empirical Xe-LPG per-launch cost
    // (~12 µs, see the xelpg-dispatch-overhead lesson), disp/tok × 12 µs
    // is the current per-token overhead budget.
    const std::size_t dispNow = _queue->dispatchCount();
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

#else // MIMIRMIND_HAVE_L0

// Non-L0 build — every method a no-op. `_enabled` stays false so the
// arch backend's `if (!_enabled) return` early-return at the top of
// mark/finish/maybeDumpAndReset is the only path.
void OpProfiler::mark(Cat /*c*/) {}
void OpProfiler::finish() {}
void OpProfiler::maybeDumpAndReset(std::size_t /*tokenIdx*/) {}

#endif // MIMIRMIND_HAVE_L0

} // namespace mimirmind::runtime