// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/gpu/hip/HipEvent.hpp"

#include "core/log/Log.hpp"

namespace mimirmind::core::hip {

namespace {

inline void hipCheck(const char* call, hipError_t code) {
    if (code != hipSuccess) {
        throw HipError(call, code);
    }
}

unsigned int flagsFor(HipEventKind k) noexcept {
    switch (k) {
        case HipEventKind::Timing:   return hipEventDefault;
        case HipEventKind::SyncOnly: return hipEventDisableTiming;
    }
    return hipEventDefault;
}

} // namespace

HipEvent::HipEvent(HipContext& ctx, HipEventKind kind)
    : _ctx(&ctx), _kind(kind) {
    hipCheck("hipEventCreateWithFlags",
             hipEventCreateWithFlags(&_event, flagsFor(kind)));
}

HipEvent::~HipEvent() noexcept {
    destroy();
}

HipEvent::HipEvent(HipEvent&& other) noexcept
    : _ctx  (other._ctx),
      _event(other._event),
      _kind (other._kind) {
    other._ctx   = nullptr;
    other._event = nullptr;
}

HipEvent& HipEvent::operator=(HipEvent&& other) noexcept {
    if (this != &other) {
        destroy();
        _ctx   = other._ctx;
        _event = other._event;
        _kind  = other._kind;
        other._ctx   = nullptr;
        other._event = nullptr;
    }
    return *this;
}

void HipEvent::destroy() noexcept {
    if (_event != nullptr) {
        const hipError_t rc = hipEventDestroy(_event);
        if (rc != hipSuccess) {
            MM_LOG_WARN("HipEvent", "hipEventDestroy failed: {}",
                        hipGetErrorString(rc));
        }
        _event = nullptr;
    }
    _ctx = nullptr;
}

void HipEvent::record(HipStream& stream) {
    hipCheck("hipEventRecord", hipEventRecord(_event, stream.handle()));
}

void HipEvent::synchronize() {
    if (_event == nullptr) return;
    hipCheck("hipEventSynchronize", hipEventSynchronize(_event));
}

bool HipEvent::query() const noexcept {
    if (_event == nullptr) return true;
    return hipEventQuery(_event) == hipSuccess;
}

float HipEvent::elapsedMs(const HipEvent& start) const {
    float ms = 0.0f;
    hipCheck("hipEventElapsedTime",
             hipEventElapsedTime(&ms, start._event, _event));
    return ms;
}

} // namespace mimirmind::core::hip