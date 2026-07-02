#include "runtime/OpProfiler.hpp"

#include "runtime/CommandQueue.hpp"
#include "runtime/Log.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>

namespace mimirmind::runtime {

namespace {

bool envSet(const char* name) noexcept {
    const char* v = std::getenv(name);
    if (v == nullptr) return false;
    const std::string_view s{v};
    return !s.empty() && s != "0" && s != "false" && s != "off";
}

constexpr const char* kCatNames[] = {
    "norm", "attn", "matmul", "act", "resid", "router",
};

} // namespace

OpProfiler::OpProfiler(CommandQueue& queue)
    : _enabled{envSet("MIMIRMIND_TRACE_OP_TIMES")},
      _queue{queue}
{
    if (_enabled) {
        MM_LOG_INFO("opprof",
                    "MIMIRMIND_TRACE_OP_TIMES=on — per-op-category timing "
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
        _tokensSinceDump = 0;
        return;
    }

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
                "total={:.1f}ms/tok",
                _tokensSinceDump, tokenIdx, summary,
                tot / static_cast<double>(_tokensSinceDump));

    _ms.fill(0.0);
    _n.fill(0);
    _tokensSinceDump = 0;
}

} // namespace mimirmind::runtime