#pragma once

#include <level_zero/ze_api.h>

#include <cstddef>
#include <cstdint>

namespace mimirmind::runtime {

class L0Context;
class GpuKernel;

/**
 * One compute command queue + a reusable command list. Submit-and-wait
 * style for now (each `dispatch` records, closes, executes, syncs, and
 * resets). Good enough for the first GPU kernel; batched async submission
 * comes later when we have multiple kernels per forward pass.
 *
 * Uses ordinal 0 (the device's first command queue group, which on Intel
 * iGPUs is the compute engine).
 */
class CommandQueue {
public:
    explicit CommandQueue(L0Context& ctx);
    ~CommandQueue();

    CommandQueue(const CommandQueue&)            = delete;
    CommandQueue& operator=(const CommandQueue&) = delete;
    CommandQueue(CommandQueue&&)                 = delete;
    CommandQueue& operator=(CommandQueue&&)      = delete;

    /**
     * Append a launch to the current command list. Does not execute or
     * sync — the launch sits on the open list until flush() is called.
     * Kernel arguments are captured at append time (per Level Zero spec),
     * so subsequent setArgumentValue calls on the same handle for the
     * *next* appendLaunch do not affect prior commands.
     */
    void appendLaunch(GpuKernel&    kernel,
                      std::uint32_t groupCountX,
                      std::uint32_t groupCountY = 1,
                      std::uint32_t groupCountZ = 1);

    /**
     * Close the list, execute on the queue, wait for completion, reset
     * the list. Idempotent and cheap when no work is pending.
     */
    void flush();

    /// True if appendLaunch has been called since the last flush().
    [[nodiscard]] bool hasPending() const noexcept { return _hasPending; }

    /**
     * Convenience for single-shot dispatch (one append + flush). Blocks
     * until completion. Equivalent to appendLaunch + flush.
     */
    void dispatch(GpuKernel&    kernel,
                  std::uint32_t groupCountX,
                  std::uint32_t groupCountY = 1,
                  std::uint32_t groupCountZ = 1);

    // -- M5f.4 — selective-barrier API ---------------------------------
    //
    // By default appendLaunch inserts a memory barrier after every kernel
    // launch so subsequent launches see its writes (Level Zero does not
    // do this implicitly — see M5f.1 + commit `40db230`). Within an
    // "unordered" scope the caller asserts that the launches inside it
    // write to DISJOINT memory regions and so can pipeline freely. The
    // pop() inserts a single barrier so the group's collective output is
    // visible to subsequent ordered launches. Use the RAII
    // UnorderedScope helper below in 99% of cases.

    /// Mark the start of an unordered launch group. While at depth > 0
    /// appendLaunch skips its post-launch barrier.
    void pushUnordered();

    /// End an unordered launch group. When depth drops to 0, a single
    /// barrier is appended so the group's writes are visible to the
    /// next ordered launch. Assert-fires if called more often than push.
    void popUnordered();

    /// Append a one-shot memory barrier. Useful when you need a barrier
    /// between two ordered launches that live in different functions.
    void appendBarrier();

    /// Test helper / sanity probe.
    [[nodiscard]] std::uint32_t unorderedDepth() const noexcept { return _unorderedDepth; }

    /// Monotonic count of `appendLaunch` calls since construction (or the
    /// last `resetDispatchCount`). Preflight signal for the Command-List-
    /// Replay milestone (M-CLR): dispatches × ~12 µs Xe-LPG launch overhead
    /// approximates the per-token overhead budget.
    [[nodiscard]] std::size_t dispatchCount() const noexcept { return _dispatchCount; }
    void resetDispatchCount() noexcept { _dispatchCount = 0; }

private:
    L0Context&                _ctx;
    ze_command_queue_handle_t _queue         {nullptr};
    ze_command_list_handle_t  _cmdList       {nullptr};
    std::uint32_t             _ordinal       {0};
    std::uint32_t             _unorderedDepth{0};
    bool                      _hasPending    {false};
    std::size_t               _dispatchCount {0};
};

/// RAII helper around pushUnordered() / popUnordered(). Construct a
/// scope around a sequence of provably-independent kernel launches; the
/// destructor inserts the trailing barrier.
class UnorderedScope {
public:
    explicit UnorderedScope(CommandQueue& q) : _q{q} { _q.pushUnordered(); }
    ~UnorderedScope() { _q.popUnordered(); }

    UnorderedScope(const UnorderedScope&)            = delete;
    UnorderedScope& operator=(const UnorderedScope&) = delete;
    UnorderedScope(UnorderedScope&&)                 = delete;
    UnorderedScope& operator=(UnorderedScope&&)      = delete;

private:
    CommandQueue& _q;
};

} // namespace mimirmind::runtime