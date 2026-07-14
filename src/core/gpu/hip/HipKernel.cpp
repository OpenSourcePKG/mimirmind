// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/gpu/hip/HipKernel.hpp"

#include "core/gpu/hip/HipModule.hpp"

#include <cstring>
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
    : _fn(fn), _name(std::move(name)) {}

void HipKernel::setPtr(std::uint32_t index, const void* devicePtr) {
    setRaw(index, sizeof(devicePtr), &devicePtr);
}

void HipKernel::setRaw(std::uint32_t index, std::size_t bytes, const void* value) {
    if (index >= _argStorage.size()) {
        _argStorage.resize(index + 1);
    }
    auto& slot = _argStorage[index];
    slot.assign(bytes, std::byte{0});
    std::memcpy(slot.data(), value, bytes);
}

void HipKernel::clearArgs() noexcept {
    _argStorage.clear();
}

void HipKernel::launch(HipStream&    stream,
                       std::uint32_t gridX,
                       std::uint32_t gridY,
                       std::uint32_t gridZ,
                       std::uint32_t blockX,
                       std::uint32_t blockY,
                       std::uint32_t blockZ,
                       std::size_t   sharedMemBytes) {
    // Build the void** pointer-array HIP expects. Each entry points to
    // the corresponding slot in _argStorage — HIP reads the value at
    // launch queue-time, not at return, so the storage must outlive
    // the launch call (it does — it's owned by the kernel).
    std::vector<void*> argPtrs;
    argPtrs.reserve(_argStorage.size());
    for (auto& slot : _argStorage) {
        argPtrs.push_back(slot.data());
    }

    hipCheck("hipModuleLaunchKernel",
             hipModuleLaunchKernel(
                 _fn,
                 gridX, gridY, gridZ,
                 blockX, blockY, blockZ,
                 static_cast<unsigned int>(sharedMemBytes),
                 stream.handle(),
                 argPtrs.empty() ? nullptr : argPtrs.data(),
                 /*extra=*/nullptr));
}

} // namespace mimirmind::core::hip