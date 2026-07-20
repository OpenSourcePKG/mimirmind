// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/gpu/cuda/CudaKernel.hpp"

#include "core/gpu/cuda/CudaModule.hpp"

#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace mimirmind::core::cuda {

namespace {

inline void cudaDriverCheck(const char* call, CUresult code) {
    if (code != CUDA_SUCCESS) {
        throw CudaDriverError(call, code);
    }
}

} // namespace

// ---------- CudaDriverError ------------------------------------------------

CudaDriverError::CudaDriverError(const std::string& call, CUresult code)
    : std::runtime_error(std::string{}), _code(code) {
    const char* msg = nullptr;
    if (cuGetErrorString(code, &msg) != CUDA_SUCCESS || msg == nullptr) {
        msg = "unknown CUDA driver error";
    }
    // Rebuild the what() string; base ctor took an empty string above.
    static_cast<std::runtime_error&>(*this) =
        std::runtime_error(call + ": " + msg);
}

// ---------- CudaKernel -----------------------------------------------------

CudaKernel::CudaKernel(CudaModule& /*parent*/, CUfunction fn, std::string name)
    : _fn(fn), _name(std::move(name)) {
    // Prime argPtrs table once — each slot points at a fixed location
    // in _argStorage, so the cuLaunchKernel `void**` at launch time is
    // just `_argPtrs.data()` with no per-launch fixup.
    for (std::size_t i = 0; i < kMaxArgs; ++i) {
        _argPtrs[i] = _argStorage[i].data();
    }
}

void CudaKernel::setPtr(std::uint32_t index, const void* devicePtr) {
    setRaw(index, sizeof(devicePtr), &devicePtr);
}

void CudaKernel::setRaw(std::uint32_t index, std::size_t bytes, const void* value) {
    if (index >= kMaxArgs) {
        throw std::runtime_error(
            std::string{"CudaKernel::setRaw("} + _name +
            "): arg index " + std::to_string(index) +
            " exceeds kMaxArgs=" + std::to_string(kMaxArgs));
    }
    if (bytes > kMaxArgBytes) {
        throw std::runtime_error(
            std::string{"CudaKernel::setRaw("} + _name +
            "): arg size " + std::to_string(bytes) +
            " exceeds kMaxArgBytes=" + std::to_string(kMaxArgBytes));
    }
    std::memcpy(_argStorage[index].data(), value, bytes);
    if (index + 1 > _argCount) {
        _argCount = index + 1;
    }
}

void CudaKernel::clearArgs() noexcept {
    _argCount = 0;
}

void CudaKernel::launch(CudaStream&   stream,
                        std::uint32_t gridX,
                        std::uint32_t gridY,
                        std::uint32_t gridZ,
                        std::uint32_t blockX,
                        std::uint32_t blockY,
                        std::uint32_t blockZ,
                        std::size_t   sharedMemBytes) {
    // No allocation. `_argPtrs.data()` was primed at construction and
    // points at `_argStorage[i].data()` for each slot; the driver reads
    // the value pointed to at queue-submit time.
    cudaDriverCheck("cuLaunchKernel",
                    cuLaunchKernel(
                        _fn,
                        gridX, gridY, gridZ,
                        blockX, blockY, blockZ,
                        static_cast<unsigned int>(sharedMemBytes),
                        stream.handle(),
                        _argCount == 0 ? nullptr : _argPtrs.data(),
                        /*extra=*/nullptr));
}

} // namespace mimirmind::core::cuda