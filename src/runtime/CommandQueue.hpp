#pragma once

#include <level_zero/ze_api.h>

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

private:
    L0Context&                _ctx;
    ze_command_queue_handle_t _queue     {nullptr};
    ze_command_list_handle_t  _cmdList   {nullptr};
    std::uint32_t             _ordinal   {0};
    bool                      _hasPending{false};
};

} // namespace mimirmind::runtime