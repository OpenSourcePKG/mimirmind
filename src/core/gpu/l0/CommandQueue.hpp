// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <level_zero/ze_api.h>

#include <cstddef>
#include <cstdint>

namespace mimirmind::core::l0 { class L0Context; }

namespace mimirmind::runtime {

using ::mimirmind::core::l0::L0Context;

class GpuKernel;

/**
 * One compute command queue + a reusable command list. Submit-and-wait
 * style for now (each `dispatch` records, closes, executes, syncs, and
 * resets). Good enough for the first GPU kernel; batched async submission
 * comes later when we have multiple kernels per forward pass.
 *
 * Uses ordinal 0 (the device's first command queue group, which on Intel
 * iGPUs is the compute engine).
 *
 * **Backend surface:** This class IS the Level-Zero impl of the compute
 * queue concept. Consumers that include this header pull in
 * `<level_zero/ze_api.h>` transitively and are implicitly L0-scoped.
 * A future HIP/CUDA backend would ship a parallel
 * `HipCommandQueue.hpp` / `CudaCommandQueue.hpp` alongside — not a
 * shared abstract base (see `[[MimirMind — HW-Abstraktions-Strategie
 * für Multi-Backend-Support]]` Schicht 3, ETA when second backend
 * work starts).
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

    /// Append a device-side memory copy into the current command list
    /// (recording or immediate). Same ordering + barrier semantics as
    /// appendLaunch. Prefer this over host `std::memcpy` for USM data
    /// that must be re-materialised on every command-list-replay pass —
    /// a host memcpy inside a recorded block runs exactly once (at
    /// record time) and stale values survive into every subsequent
    /// replay. Landed for the M10.2 Phase 1a Q8_0 altAttention V=K
    /// staging copy.
    void appendMemoryCopy(void*       dst,
                          const void* src,
                          std::size_t nBytes);

    /// Test helper / sanity probe.
    [[nodiscard]] std::uint32_t unorderedDepth() const noexcept { return _unorderedDepth; }

    /// Monotonic count of `appendLaunch` calls since construction (or the
    /// last `resetDispatchCount`). Preflight signal for the Command-List-
    /// Replay milestone (M-CLR): dispatches × ~12 µs Xe-LPG launch overhead
    /// approximates the per-token overhead budget.
    [[nodiscard]] std::size_t dispatchCount() const noexcept { return _dispatchCount; }
    void resetDispatchCount() noexcept { _dispatchCount = 0; }

    // -- M-CLR.3 — record / replay -------------------------------------
    //
    // Typical use for the Gemma4E4B decode loop:
    //
    //   engine.runFirstDecodeStep();          // immediate mode, warms cache
    //   queue.beginRecord();                  // arm the recording list
    //   engine.runOneDecodeStep();            // dispatches route into it
    //   queue.endRecord();
    //   for (each further decode step) {
    //     *ops.curLenSlot() = new_cur_len;    // one 4-byte host write
    //     queue.replay();                     // reuses the recorded list
    //   }
    //
    // Recording is exclusive with the immediate command list — the
    // implementation swaps which handle appendLaunch targets, so
    // profiler / diagnostic paths that try to flush() during a
    // recording just early-return (hasPending stays false on the
    // immediate list).

    /// Begin a fresh recording. Any prior recording is discarded. The
    /// immediate command list must be idle (call flush() first).
    void beginRecord();

    /// Close the recording — it is now replay-able.
    void endRecord();

    /// Execute the closed recording once. Synchronous.
    void replay();

    /// True once endRecord() has been called since the last beginRecord
    /// / resetRecording(). Immediate replay() is only valid then.
    [[nodiscard]] bool hasRecording() const noexcept { return _recordingReady; }

    /// True while beginRecord() is active. Used by callers that want to
    /// route diagnostics differently during recording.
    [[nodiscard]] bool isRecording() const noexcept { return _recording; }

    /// Free the recorded list handle and forget the recording. Called
    /// implicitly by the destructor. Idempotent.
    void resetRecording() noexcept;

private:
    L0Context&                _ctx;
    ze_command_queue_handle_t _queue         {nullptr};
    ze_command_list_handle_t  _cmdList       {nullptr};
    ze_command_list_handle_t  _recordList    {nullptr};
    std::uint32_t             _ordinal       {0};
    std::uint32_t             _unorderedDepth{0};
    bool                      _hasPending    {false};
    bool                      _recording     {false};
    bool                      _recordingReady{false};
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