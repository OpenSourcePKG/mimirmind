// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/gpu/hip/HipStream.hpp"

#include "core/log/Log.hpp"

#include <utility>

namespace mimirmind::core::hip {

namespace {

inline void hipCheck(const char* call, hipError_t code) {
    if (code != hipSuccess) {
        throw HipError(call, code);
    }
}

const char* kindName(HipStreamKind k) noexcept {
    switch (k) {
        case HipStreamKind::BlockingDefault: return "BlockingDefault";
        case HipStreamKind::NonBlocking:     return "NonBlocking";
    }
    return "Unknown";
}

} // namespace

HipStream::HipStream(HipContext& ctx, HipStreamKind kind)
    : _ctx(&ctx), _kind(kind) {
    // hipSetDevice is per-thread state; assume the caller wired
    // HipContext ctor already (which it does). No re-select here to
    // keep behaviour predictable when a caller holds two HipContexts.
    switch (kind) {
        case HipStreamKind::BlockingDefault:
            hipCheck("hipStreamCreate", hipStreamCreate(&_stream));
            break;
        case HipStreamKind::NonBlocking:
            hipCheck("hipStreamCreateWithFlags",
                     hipStreamCreateWithFlags(&_stream, hipStreamNonBlocking));
            break;
    }
    MM_LOG_DEBUG("HipStream", "created ({}) on device #{}",
                 kindName(_kind), _ctx->hipDeviceIndex());
}

HipStream::~HipStream() noexcept {
    destroy();
}

HipStream::HipStream(HipStream&& other) noexcept
    : _ctx   (other._ctx),
      _stream(other._stream),
      _kind  (other._kind) {
    other._ctx    = nullptr;
    other._stream = nullptr;
}

HipStream& HipStream::operator=(HipStream&& other) noexcept {
    if (this != &other) {
        destroy();
        _ctx    = other._ctx;
        _stream = other._stream;
        _kind   = other._kind;
        other._ctx    = nullptr;
        other._stream = nullptr;
    }
    return *this;
}

void HipStream::destroy() noexcept {
    if (_stream != nullptr) {
        const hipError_t rc = hipStreamDestroy(_stream);
        if (rc != hipSuccess) {
            MM_LOG_WARN("HipStream", "hipStreamDestroy failed: {}",
                        hipGetErrorString(rc));
        }
        _stream = nullptr;
    }
    _ctx = nullptr;
}

void HipStream::synchronize() {
    if (_stream == nullptr) return;
    hipCheck("hipStreamSynchronize", hipStreamSynchronize(_stream));
}

bool HipStream::query() const noexcept {
    if (_stream == nullptr) return true;    // empty stream is "done"
    return hipStreamQuery(_stream) == hipSuccess;
}

} // namespace mimirmind::core::hip