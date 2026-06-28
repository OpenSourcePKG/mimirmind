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
     * Launch `kernel` with `groupCountX/Y/Z` workgroups. The per-workgroup
     * thread layout must be set on the kernel beforehand via
     * GpuKernel::setGroupSize.
     *
     * Blocks until completion (zeCommandQueueSynchronize).
     */
    void dispatch(GpuKernel& kernel,
                  std::uint32_t groupCountX,
                  std::uint32_t groupCountY = 1,
                  std::uint32_t groupCountZ = 1);

private:
    L0Context&                _ctx;
    ze_command_queue_handle_t _queue   {nullptr};
    ze_command_list_handle_t  _cmdList {nullptr};
    std::uint32_t             _ordinal {0};
};

} // namespace mimirmind::runtime