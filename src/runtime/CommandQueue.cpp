#include "runtime/CommandQueue.hpp"

#include "runtime/GpuKernel.hpp"
#include "runtime/L0Context.hpp"
#include "runtime/Log.hpp"

#include <limits>

namespace mimirmind::runtime {

namespace {

#define ZE_CHECK(call)                                                  \
    do {                                                                \
        const ze_result_t _r = (call);                                  \
        if (_r != ZE_RESULT_SUCCESS) {                                  \
            MM_LOG_ERROR("gpu", "{} -> {} (0x{:x})",                    \
                         #call,                                         \
                         L0Context::resultToString(_r),                 \
                         static_cast<unsigned>(_r));                    \
            throw L0Error(#call, _r);                                   \
        }                                                               \
    } while (false)

} // namespace

CommandQueue::CommandQueue(L0Context& ctx)
    : _ctx{ctx}
{
    ze_command_queue_desc_t qDesc{};
    qDesc.stype    = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    qDesc.ordinal  = _ordinal;
    qDesc.index    = 0;
    qDesc.flags    = 0;
    qDesc.mode     = ZE_COMMAND_QUEUE_MODE_DEFAULT;
    qDesc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    ZE_CHECK(zeCommandQueueCreate(_ctx.context(), _ctx.device(),
                                  &qDesc, &_queue));

    ze_command_list_desc_t lDesc{};
    lDesc.stype                          = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
    lDesc.commandQueueGroupOrdinal       = _ordinal;
    lDesc.flags                          = 0;
    ZE_CHECK(zeCommandListCreate(_ctx.context(), _ctx.device(),
                                 &lDesc, &_cmdList));

    MM_LOG_INFO("gpu",
                "CommandQueue ready — queue={} list={} ordinal={}",
                static_cast<const void*>(_queue),
                static_cast<const void*>(_cmdList),
                _ordinal);
}

CommandQueue::~CommandQueue() {
    if (_cmdList != nullptr) {
        zeCommandListDestroy(_cmdList);
        _cmdList = nullptr;
    }
    if (_queue != nullptr) {
        zeCommandQueueDestroy(_queue);
        _queue = nullptr;
    }
}

void CommandQueue::appendLaunch(GpuKernel&    kernel,
                                std::uint32_t groupCountX,
                                std::uint32_t groupCountY,
                                std::uint32_t groupCountZ) {
    ze_group_count_t groups{groupCountX, groupCountY, groupCountZ};
    ZE_CHECK(zeCommandListAppendLaunchKernel(
        _cmdList, kernel.handle(), &groups,
        nullptr, 0, nullptr));
    _hasPending = true;
}

void CommandQueue::flush() {
    if (!_hasPending) {
        return;
    }
    ZE_CHECK(zeCommandListClose(_cmdList));
    ZE_CHECK(zeCommandQueueExecuteCommandLists(
        _queue, 1, &_cmdList, nullptr));
    ZE_CHECK(zeCommandQueueSynchronize(
        _queue, std::numeric_limits<std::uint64_t>::max()));
    ZE_CHECK(zeCommandListReset(_cmdList));
    _hasPending = false;
    MM_LOG_DEBUG("gpu", "queue flushed");
}

void CommandQueue::dispatch(GpuKernel&    kernel,
                            std::uint32_t groupCountX,
                            std::uint32_t groupCountY,
                            std::uint32_t groupCountZ) {
    appendLaunch(kernel, groupCountX, groupCountY, groupCountZ);
    flush();
    MM_LOG_DEBUG("gpu",
                 "dispatch done — kernel={} groups=({},{},{})",
                 static_cast<const void*>(kernel.handle()),
                 groupCountX, groupCountY, groupCountZ);
}

} // namespace mimirmind::runtime