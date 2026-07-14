// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/CommandQueue.hpp"

#include "runtime/GpuKernel.hpp"
#include "core/gpu/l0/L0Context.hpp"
#include "core/log/Log.hpp"

#include <limits>
#include <stdexcept>

namespace mimirmind::runtime {

using ::mimirmind::core::l0::L0Error;

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
    resetRecording();
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
    // M-CLR.3: route into the recording list while beginRecord/endRecord
    // is active. The immediate list stays idle in that case (nothing
    // touches _hasPending) so a stray flush() during recording is a
    // no-op, which keeps profiler/diagnostic paths safe.
    ze_command_list_handle_t target =
        _recording ? _recordList : _cmdList;
    ZE_CHECK(zeCommandListAppendLaunchKernel(
        target, kernel.handle(), &groups,
        nullptr, 0, nullptr));
    ++_dispatchCount;

    // Level Zero does NOT insert an implicit memory barrier between
    // consecutive kernel launches — execution order is preserved but
    // memory writes from kernel N may not be visible to kernel N+1
    // without explicit synchronisation (see commit `40db230`). In the
    // default "ordered" mode we append a generic barrier here so the
    // caller doesn't have to track dependencies. Within an unordered
    // scope the caller has asserted that the writes are independent
    // and we skip the barrier; the matching popUnordered() inserts
    // exactly one barrier at the end of the group.
    if (_unorderedDepth == 0) {
        ZE_CHECK(zeCommandListAppendBarrier(target, nullptr, 0, nullptr));
    }

    if (!_recording) {
        _hasPending = true;
    }
}

void CommandQueue::pushUnordered() {
    ++_unorderedDepth;
}

void CommandQueue::popUnordered() {
    if (_unorderedDepth == 0) {
        MM_LOG_ERROR("gpu",
                     "CommandQueue::popUnordered called with depth=0");
        throw std::logic_error(
            "CommandQueue::popUnordered without matching pushUnordered");
    }
    --_unorderedDepth;
    // M-CLR.3: `_hasPending` is only tracked on the immediate list; a
    // group that ended in a recording emits its trailing barrier there.
    if (_unorderedDepth == 0) {
        if (_recording) {
            ZE_CHECK(zeCommandListAppendBarrier(_recordList, nullptr, 0, nullptr));
        } else if (_hasPending) {
            ZE_CHECK(zeCommandListAppendBarrier(_cmdList,    nullptr, 0, nullptr));
        }
    }
}

void CommandQueue::appendBarrier() {
    ze_command_list_handle_t target =
        _recording ? _recordList : _cmdList;
    ZE_CHECK(zeCommandListAppendBarrier(target, nullptr, 0, nullptr));
}

void CommandQueue::appendMemoryCopy(void*       dst,
                                    const void* src,
                                    std::size_t nBytes) {
    if (nBytes == 0) {
        return;
    }
    ze_command_list_handle_t target =
        _recording ? _recordList : _cmdList;
    ZE_CHECK(zeCommandListAppendMemoryCopy(
        target, dst, src, nBytes, nullptr, 0, nullptr));
    if (_unorderedDepth == 0) {
        ZE_CHECK(zeCommandListAppendBarrier(target, nullptr, 0, nullptr));
    }
    if (!_recording) {
        _hasPending = true;
    }
}

void CommandQueue::flush() {
    if (!_hasPending) {
        return;
    }
    if (_unorderedDepth != 0) {
        // Flushing inside an unordered scope drops the trailing
        // barrier that would have been inserted at pop. Whoever flushes
        // is reading results on the CPU after sync, which carries its
        // own ordering guarantee — but the scope is logically broken;
        // log loud.
        MM_LOG_ERROR("gpu",
                     "CommandQueue::flush while unorderedDepth={} — "
                     "missing popUnordered before flush", _unorderedDepth);
        _unorderedDepth = 0;
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

// -- M-CLR.3 — record / replay ------------------------------------------

void CommandQueue::beginRecord() {
    if (_recording) {
        throw std::logic_error(
            "CommandQueue::beginRecord already recording — call endRecord first");
    }
    if (_hasPending) {
        throw std::logic_error(
            "CommandQueue::beginRecord while immediate work is pending — "
            "call flush() first");
    }
    if (_recordList == nullptr) {
        ze_command_list_desc_t desc{};
        desc.stype                    = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
        desc.commandQueueGroupOrdinal = _ordinal;
        desc.flags                    = 0;
        ZE_CHECK(zeCommandListCreate(_ctx.context(), _ctx.device(),
                                     &desc, &_recordList));
    } else {
        ZE_CHECK(zeCommandListReset(_recordList));
    }
    _recording      = true;
    _recordingReady = false;
    MM_LOG_DEBUG("gpu", "CommandQueue: recording begin (list={})",
                 static_cast<const void*>(_recordList));
}

void CommandQueue::endRecord() {
    if (!_recording) {
        throw std::logic_error(
            "CommandQueue::endRecord called outside beginRecord");
    }
    if (_unorderedDepth != 0) {
        MM_LOG_ERROR("gpu",
                     "CommandQueue::endRecord with unorderedDepth={} — "
                     "missing popUnordered before endRecord", _unorderedDepth);
        _unorderedDepth = 0;
    }
    ZE_CHECK(zeCommandListClose(_recordList));
    _recording      = false;
    _recordingReady = true;
    MM_LOG_DEBUG("gpu", "CommandQueue: recording closed (list={})",
                 static_cast<const void*>(_recordList));
}

void CommandQueue::replay() {
    if (!_recordingReady) {
        throw std::logic_error(
            "CommandQueue::replay called with no closed recording — "
            "call beginRecord/endRecord first");
    }
    ZE_CHECK(zeCommandQueueExecuteCommandLists(
        _queue, 1, &_recordList, nullptr));
    ZE_CHECK(zeCommandQueueSynchronize(
        _queue, std::numeric_limits<std::uint64_t>::max()));
}

void CommandQueue::resetRecording() noexcept {
    _recording      = false;
    _recordingReady = false;
    if (_recordList != nullptr) {
        zeCommandListDestroy(_recordList);
        _recordList = nullptr;
    }
}

} // namespace mimirmind::runtime