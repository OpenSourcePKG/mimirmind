// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/gpu/cuda/CudaEvent.hpp"

#include "core/log/Log.hpp"

namespace mimirmind::core::cuda {

namespace {

inline void cudaCheck(const char* call, cudaError_t code) {
    if (code != cudaSuccess) {
        throw CudaError(call, code);
    }
}

unsigned int flagsFor(CudaEventKind k) noexcept {
    switch (k) {
        case CudaEventKind::Timing:   return cudaEventDefault;
        case CudaEventKind::SyncOnly: return cudaEventDisableTiming;
    }
    return cudaEventDefault;
}

} // namespace

CudaEvent::CudaEvent(CudaContext& ctx, CudaEventKind kind)
    : _ctx(&ctx), _kind(kind) {
    cudaCheck("cudaEventCreateWithFlags",
              cudaEventCreateWithFlags(&_event, flagsFor(kind)));
}

CudaEvent::~CudaEvent() noexcept {
    destroy();
}

CudaEvent::CudaEvent(CudaEvent&& other) noexcept
    : _ctx  (other._ctx),
      _event(other._event),
      _kind (other._kind) {
    other._ctx   = nullptr;
    other._event = nullptr;
}

CudaEvent& CudaEvent::operator=(CudaEvent&& other) noexcept {
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

void CudaEvent::destroy() noexcept {
    if (_event != nullptr) {
        const cudaError_t rc = cudaEventDestroy(_event);
        if (rc != cudaSuccess) {
            MM_LOG_WARN("CudaEvent", "cudaEventDestroy failed: {}",
                        cudaGetErrorString(rc));
        }
        _event = nullptr;
    }
    _ctx = nullptr;
}

void CudaEvent::record(CudaStream& stream) {
    cudaCheck("cudaEventRecord", cudaEventRecord(_event, stream.handle()));
}

void CudaEvent::synchronize() {
    if (_event == nullptr) return;
    cudaCheck("cudaEventSynchronize", cudaEventSynchronize(_event));
}

bool CudaEvent::query() const noexcept {
    if (_event == nullptr) return true;
    return cudaEventQuery(_event) == cudaSuccess;
}

float CudaEvent::elapsedMs(const CudaEvent& start) const {
    float ms = 0.0f;
    cudaCheck("cudaEventElapsedTime",
              cudaEventElapsedTime(&ms, start._event, _event));
    return ms;
}

} // namespace mimirmind::core::cuda