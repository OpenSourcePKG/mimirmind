// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/gpu/hip/HipKernel.hpp"

#include "core/gpu/hip/HipModule.hpp"

#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace mimirmind::core::hip {

namespace {

inline void hipCheck(const char* call, hipError_t code) {
    if (code != hipSuccess) {
        throw HipError(call, code);
    }
}

} // namespace

HipKernel::HipKernel(HipModule& /*parent*/, hipFunction_t fn, std::string name)
    : _fn(fn), _name(std::move(name)) {
    // Prime the argPtrs table once. Each slot points at a fixed
    // location in _argStorage — pointers never invalidate, so the
    // hipModuleLaunchKernel `void**` at launch time is the raw
    // `_argPtrs.data()` without any per-launch fixup.
    for (std::size_t i = 0; i < kMaxArgs; ++i) {
        _argPtrs[i] = _argStorage[i].data();
    }
}

void HipKernel::setPtr(std::uint32_t index, const void* devicePtr) {
    setRaw(index, sizeof(devicePtr), &devicePtr);
}

void HipKernel::setRaw(std::uint32_t index, std::size_t bytes, const void* value) {
    if (index >= kMaxArgs) {
        throw std::runtime_error(
            std::string{"HipKernel::setRaw("} + _name +
            "): arg index " + std::to_string(index) +
            " exceeds kMaxArgs=" + std::to_string(kMaxArgs));
    }
    if (bytes > kMaxArgBytes) {
        throw std::runtime_error(
            std::string{"HipKernel::setRaw("} + _name +
            "): arg size " + std::to_string(bytes) +
            " exceeds kMaxArgBytes=" + std::to_string(kMaxArgBytes));
    }
    std::memcpy(_argStorage[index].data(), value, bytes);
    if (index + 1 > _argCount) {
        _argCount = index + 1;
    }
}

void HipKernel::clearArgs() noexcept {
    _argCount = 0;
}

void HipKernel::launch(HipStream&    stream,
                       std::uint32_t gridX,
                       std::uint32_t gridY,
                       std::uint32_t gridZ,
                       std::uint32_t blockX,
                       std::uint32_t blockY,
                       std::uint32_t blockZ,
                       std::size_t   sharedMemBytes) {
    // No allocation. `_argPtrs.data()` was primed at construction and
    // points at `_argStorage[i].data()` for each slot; HIP reads the
    // value pointed to at queue-submit time.
    hipCheck("hipModuleLaunchKernel",
             hipModuleLaunchKernel(
                 _fn,
                 gridX, gridY, gridZ,
                 blockX, blockY, blockZ,
                 static_cast<unsigned int>(sharedMemBytes),
                 stream.handle(),
                 _argCount == 0 ? nullptr : _argPtrs.data(),
                 /*extra=*/nullptr));
}

} // namespace mimirmind::core::hip