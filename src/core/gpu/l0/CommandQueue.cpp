// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/gpu/l0/CommandQueue.hpp"

#include "core/gpu/l0/GpuKernel.hpp"
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
    // M8.L double-buffer teardown: any in-flight submission must complete
    // before its list/event is destroyed.
    for (int i = 0; i < 2; ++i) {
        if (_dbInFlight[i] && _dbEvent[i] != nullptr) {
            zeEventHostSynchronize(_dbEvent[i],
                                   std::numeric_limits<std::uint64_t>::max());
            _dbInFlight[i] = false;
        }
        if (_dbEvent[i] != nullptr) { zeEventDestroy(_dbEvent[i]); _dbEvent[i] = nullptr; }
        if (_dbList[i]  != nullptr) { zeCommandListDestroy(_dbList[i]); _dbList[i] = nullptr; }
    }
    if (_dbPool != nullptr) { zeEventPoolDestroy(_dbPool); _dbPool = nullptr; }
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
    // Route into the active double-buffer list (M8.L) when armed, else the
    // recording list while beginRecord/endRecord is active (M-CLR.3), else the
    // immediate list. The immediate list stays idle during recording/db so a
    // stray flush() is a no-op, which keeps profiler/diagnostic paths safe.
    ze_command_list_handle_t target = currentList();
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

    if (_dbActive) {
        _dbHasPending = true;
    } else if (!_recording) {
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
    // The trailing barrier goes onto whichever list is active: the db list
    // (M8.L), else the recording list (M-CLR.3), else the immediate list.
    if (_unorderedDepth == 0) {
        if (_dbActive) {
            if (_dbHasPending) {
                ZE_CHECK(zeCommandListAppendBarrier(_dbList[_dbCur], nullptr, 0, nullptr));
            }
        } else if (_recording) {
            ZE_CHECK(zeCommandListAppendBarrier(_recordList, nullptr, 0, nullptr));
        } else if (_hasPending) {
            ZE_CHECK(zeCommandListAppendBarrier(_cmdList,    nullptr, 0, nullptr));
        }
    }
}

void CommandQueue::appendBarrier() {
    ZE_CHECK(zeCommandListAppendBarrier(currentList(), nullptr, 0, nullptr));
}

void CommandQueue::appendMemoryCopy(void*       dst,
                                    const void* src,
                                    std::size_t nBytes) {
    if (nBytes == 0) {
        return;
    }
    ze_command_list_handle_t target = currentList();
    ZE_CHECK(zeCommandListAppendMemoryCopy(
        target, dst, src, nBytes, nullptr, 0, nullptr));
    if (_unorderedDepth == 0) {
        ZE_CHECK(zeCommandListAppendBarrier(target, nullptr, 0, nullptr));
    }
    if (_dbActive) {
        _dbHasPending = true;
    } else if (!_recording) {
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

// -- M8.L (4.5.5) — double-buffered chunked submit ----------------------

void CommandQueue::ensureDbResources() {
    if (_dbPool != nullptr) {
        return;
    }
    ze_event_pool_desc_t pDesc{};
    pDesc.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
    pDesc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE; // host waits on completion
    pDesc.count = 2;                               // one event per db list
    ze_device_handle_t dev = _ctx.device();
    ZE_CHECK(zeEventPoolCreate(_ctx.context(), &pDesc, 1, &dev, &_dbPool));

    for (int i = 0; i < 2; ++i) {
        ze_event_desc_t eDesc{};
        eDesc.stype  = ZE_STRUCTURE_TYPE_EVENT_DESC;
        eDesc.index  = static_cast<std::uint32_t>(i);
        eDesc.signal = ZE_EVENT_SCOPE_FLAG_HOST; // signalled to the host
        eDesc.wait   = ZE_EVENT_SCOPE_FLAG_HOST;
        ZE_CHECK(zeEventCreate(_dbPool, &eDesc, &_dbEvent[i]));

        ze_command_list_desc_t lDesc{};
        lDesc.stype                    = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
        lDesc.commandQueueGroupOrdinal = _ordinal;
        lDesc.flags                    = 0;
        ZE_CHECK(zeCommandListCreate(_ctx.context(), _ctx.device(),
                                     &lDesc, &_dbList[i]));
    }
    MM_LOG_INFO("gpu",
                "CommandQueue: double-buffer armed (lists={}/{}, host-visible "
                "event pool)",
                static_cast<const void*>(_dbList[0]),
                static_cast<const void*>(_dbList[1]));
}

void CommandQueue::beginDoubleBuffered() {
    if (_dbActive) {
        throw std::logic_error(
            "CommandQueue::beginDoubleBuffered already armed — call "
            "endDoubleBuffered first");
    }
    if (_recording) {
        throw std::logic_error(
            "CommandQueue::beginDoubleBuffered while recording — exclusive with "
            "beginRecord");
    }
    if (_hasPending) {
        throw std::logic_error(
            "CommandQueue::beginDoubleBuffered while immediate work is pending — "
            "call flush() first");
    }
    ensureDbResources();
    _dbCur        = 0;
    _dbHasPending = false;
    _dbInFlight[0] = _dbInFlight[1] = false;
    _dbActive     = true;
}

void CommandQueue::checkpoint() {
    if (!_dbActive) {
        throw std::logic_error(
            "CommandQueue::checkpoint outside beginDoubleBuffered");
    }
    const int cur = _dbCur;
    if (_dbHasPending) {
        // A trailing barrier that SIGNALS this list's completion event: it both
        // makes all writes globally visible and fires the event when the list
        // finishes. Then submit async (no fence, no queue sync).
        ZE_CHECK(zeCommandListAppendBarrier(_dbList[cur], _dbEvent[cur], 0, nullptr));
        ZE_CHECK(zeCommandListClose(_dbList[cur]));
        ZE_CHECK(zeCommandQueueExecuteCommandLists(_queue, 1, &_dbList[cur], nullptr));
        _dbInFlight[cur] = true;
        _dbHasPending    = false;
    }
    // Swap to the other list. If it still has an outstanding submission, wait on
    // ONLY its event (never the whole queue) before recycling its buffer — this
    // is what bounds in-flight work to two lists without the pathological
    // mid-workload zeCommandQueueSynchronize.
    const int nxt = 1 - cur;
    if (_dbInFlight[nxt]) {
        ZE_CHECK(zeEventHostSynchronize(_dbEvent[nxt],
                                        std::numeric_limits<std::uint64_t>::max()));
        ZE_CHECK(zeEventHostReset(_dbEvent[nxt]));
        ZE_CHECK(zeCommandListReset(_dbList[nxt]));
        _dbInFlight[nxt] = false;
    }
    _dbCur = nxt;
}

void CommandQueue::endDoubleBuffered() {
    if (!_dbActive) {
        return;
    }
    if (_dbHasPending) {
        const int cur = _dbCur;
        ZE_CHECK(zeCommandListAppendBarrier(_dbList[cur], _dbEvent[cur], 0, nullptr));
        ZE_CHECK(zeCommandListClose(_dbList[cur]));
        ZE_CHECK(zeCommandQueueExecuteCommandLists(_queue, 1, &_dbList[cur], nullptr));
        _dbInFlight[cur] = true;
        _dbHasPending    = false;
    }
    // One final drain of the ≤2 in-flight lists.
    for (int i = 0; i < 2; ++i) {
        if (_dbInFlight[i]) {
            ZE_CHECK(zeEventHostSynchronize(_dbEvent[i],
                                            std::numeric_limits<std::uint64_t>::max()));
            ZE_CHECK(zeEventHostReset(_dbEvent[i]));
            ZE_CHECK(zeCommandListReset(_dbList[i]));
            _dbInFlight[i] = false;
        }
    }
    _dbActive     = false;
    _dbCur        = 0;
    _dbHasPending = false;
}

} // namespace mimirmind::runtime