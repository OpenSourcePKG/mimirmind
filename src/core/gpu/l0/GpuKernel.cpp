// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/gpu/l0/GpuKernel.hpp"

#include "core/gpu/l0/L0Context.hpp"
#include "core/log/Log.hpp"

namespace mimirmind::runtime {

using ::mimirmind::core::l0::L0Context;
using ::mimirmind::core::l0::L0Error;

void GpuKernel::setPtr(std::uint32_t index, const void* ptr) {
    const ze_result_t r = zeKernelSetArgumentValue(
        _h, index, sizeof(void*), &ptr);
    if (r != ZE_RESULT_SUCCESS) {
        MM_LOG_ERROR("gpu",
                     "zeKernelSetArgumentValue(ptr) idx={} -> {} (0x{:x})",
                     index, L0Context::resultToString(r),
                     static_cast<unsigned>(r));
        throw L0Error("zeKernelSetArgumentValue", r);
    }
}

void GpuKernel::setRaw(std::uint32_t index, std::size_t bytes, const void* data) {
    const ze_result_t r = zeKernelSetArgumentValue(_h, index, bytes, data);
    if (r != ZE_RESULT_SUCCESS) {
        MM_LOG_ERROR("gpu",
                     "zeKernelSetArgumentValue(raw {}B) idx={} -> {} (0x{:x})",
                     bytes, index, L0Context::resultToString(r),
                     static_cast<unsigned>(r));
        throw L0Error("zeKernelSetArgumentValue", r);
    }
}

void GpuKernel::setGroupSize(std::uint32_t x, std::uint32_t y, std::uint32_t z) {
    const ze_result_t r = zeKernelSetGroupSize(_h, x, y, z);
    if (r != ZE_RESULT_SUCCESS) {
        MM_LOG_ERROR("gpu",
                     "zeKernelSetGroupSize({},{},{}) -> {} (0x{:x})",
                     x, y, z, L0Context::resultToString(r),
                     static_cast<unsigned>(r));
        throw L0Error("zeKernelSetGroupSize", r);
    }
}

} // namespace mimirmind::runtime