// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <array>
#include <chrono>
#include <cstddef>

namespace mimirmind::runtime {

class CommandQueue;

/**
 * M8.K.0 diagnostic — attribute per-token forward-time to op categories
 * (norm / attention / matmul / activation / residual / router) so we can
 * quantify how much of a decode-forward is spent where.
 *
 * OFF by default (`_enabled == false`): begin() / end() / maybeDumpAndReset()
 * are all no-ops with zero overhead. The backend keeps its normal
 * pipelined dispatch behaviour.
 *
 * ON when `diagnostics.traceOpTimes: true` in config.json: begin(cat)
 * flushes the shared command queue, starts a wall-clock, and remembers
 * `cat`. end() flushes again, adds the delta to the running total for
 * that category. Every kDumpEvery tokens the accumulated shares get
 * logged and cleared. The flushes serialise the pipeline — this is
 * diagnostic mode only, not for production traffic.
 *
 * Not thread-safe. The engine already serialises inference calls via
 * engineMutex; this class inherits that constraint.
 */
class OpProfiler {
public:
    enum class Cat {
        NORM,       // rmsnorm variants, add_bias
        ATTENTION,  // attention kernel, qkv_split, rope
        MATMUL,     // all matmuls incl. lm_head and MoE experts
        ACTIVATION, // silu/gelu/mul_scalar
        RESIDUAL,   // add_residual, scaled_add_residual
        ROUTER,     // MoE gate matmul + softmax (CPU + GPU parts)
        NUM
    };

    /// `enabled` maps to `diagnostics.traceOpTimes` in config.json.
    OpProfiler(CommandQueue& queue, bool enabled);

    [[nodiscard]] bool enabled() const noexcept { return _enabled; }

    /// Mark the start of a new phase. Implicitly closes any previously
    /// active phase (flushes + records its wall-time delta), then flushes
    /// and starts the timer for the new category. Instrumentation
    /// callers just insert `mark(Cat::X)` at each phase boundary; no
    /// matching close call required except finish() at the very end.
    void mark(Cat c);

    /// Close the currently active phase (if any). Called at the end of
    /// every forward so the last phase's time lands in the accumulator
    /// before maybeDumpAndReset() reports.
    void finish();

    /// Emit accumulated per-category shares to the log every
    /// kDumpEvery tokens, then reset. Cheap no-op when disabled.
    void maybeDumpAndReset(std::size_t tokenIdx);

private:
    static constexpr std::size_t kNumCats =
        static_cast<std::size_t>(Cat::NUM);
    static constexpr std::size_t kDumpEvery = 50;

    bool                                     _enabled;
    CommandQueue&                            _queue;
    Cat                                      _cur{Cat::NORM};
    bool                                     _active{false};
    std::chrono::steady_clock::time_point    _t0{};
    std::array<double,      kNumCats>        _ms{};
    std::array<std::size_t, kNumCats>        _n{};
    std::size_t                              _tokensSinceDump{0};
    // Baseline for `disp/tok` in the summary line. See maybeDumpAndReset().
    std::size_t                              _dispatchBaseline{0};
};

} // namespace mimirmind::runtime