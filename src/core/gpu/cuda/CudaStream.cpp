// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/gpu/cuda/CudaStream.hpp"

#include "core/log/Log.hpp"

#include <utility>

namespace mimirmind::core::cuda {

namespace {

inline void cudaCheck(const char* call, cudaError_t code) {
    if (code != cudaSuccess) {
        throw CudaError(call, code);
    }
}

const char* kindName(CudaStreamKind k) noexcept {
    switch (k) {
        case CudaStreamKind::BlockingDefault: return "BlockingDefault";
        case CudaStreamKind::NonBlocking:     return "NonBlocking";
    }
    return "Unknown";
}

} // namespace

CudaStream::CudaStream(CudaContext& ctx, CudaStreamKind kind)
    : _ctx(&ctx), _kind(kind) {
    // cudaSetDevice is per-thread state; assume the caller wired
    // CudaContext ctor already. No re-select here to keep behaviour
    // predictable when a caller holds two CudaContexts.
    switch (kind) {
        case CudaStreamKind::BlockingDefault:
            cudaCheck("cudaStreamCreate", cudaStreamCreate(&_stream));
            break;
        case CudaStreamKind::NonBlocking:
            cudaCheck("cudaStreamCreateWithFlags",
                      cudaStreamCreateWithFlags(&_stream, cudaStreamNonBlocking));
            break;
    }
    MM_LOG_DEBUG("CudaStream", "created ({}) on device #{}",
                 kindName(_kind), _ctx->cudaDeviceIndex());
}

CudaStream::~CudaStream() noexcept {
    destroy();
}

CudaStream::CudaStream(CudaStream&& other) noexcept
    : _ctx   (other._ctx),
      _stream(other._stream),
      _kind  (other._kind) {
    other._ctx    = nullptr;
    other._stream = nullptr;
}

CudaStream& CudaStream::operator=(CudaStream&& other) noexcept {
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

void CudaStream::destroy() noexcept {
    if (_stream != nullptr) {
        const cudaError_t rc = cudaStreamDestroy(_stream);
        if (rc != cudaSuccess) {
            MM_LOG_WARN("CudaStream", "cudaStreamDestroy failed: {}",
                        cudaGetErrorString(rc));
        }
        _stream = nullptr;
    }
    _ctx = nullptr;
}

void CudaStream::synchronize() {
    if (_stream == nullptr) return;
    cudaCheck("cudaStreamSynchronize", cudaStreamSynchronize(_stream));
}

bool CudaStream::query() const noexcept {
    if (_stream == nullptr) return true;
    return cudaStreamQuery(_stream) == cudaSuccess;
}

} // namespace mimirmind::core::cuda