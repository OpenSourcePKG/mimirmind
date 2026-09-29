// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

// Shared kernel-launch argument guards for the GPU backends (8.30.2). The L0,
// CUDA and HIP GpuOps each pushed size_t kernel arg counts through int32
// setValue slots and derived workgroup counts from the same ceil-div, and each
// carried a byte-identical copy of these two guards in its private anonymous
// namespace. Centralise the logic here; every backend keeps a one-line
// file-local wrapper that binds its own diagnostic label, so all call sites and
// error messages stay exactly as they were.
namespace mimirmind::compute::detail {

/// Narrowing cast std::size_t -> std::int32_t with an overflow guard. `tag`
/// names the kernel argument for the diagnostic; `backendLabel` names the
/// backend ("GpuOps", "cuda::GpuOps", "hip::GpuOps") so the thrown message is
/// byte-identical to each backend's former private copy.
[[nodiscard]] inline std::int32_t toInt32(std::size_t v, const char* tag,
                                          const char* backendLabel) {
    if (v > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::runtime_error(
            std::string{backendLabel} + ": " + tag +
            " overflows int32 (" + std::to_string(v) + ")");
    }
    return static_cast<std::int32_t>(v);
}

/// ceil(n / local) workgroup count with a uint32 overflow guard. `backendLabel`
/// keeps the diagnostic byte-identical to each backend's former copy.
[[nodiscard]] inline std::uint32_t groupsForN(std::size_t n, std::uint32_t local,
                                              const char* backendLabel) {
    const std::size_t g = (n + local - 1) / local;
    if (g > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error(
            std::string{backendLabel} + ": workgroup count overflows uint32");
    }
    return static_cast<std::uint32_t>(g);
}

} // namespace mimirmind::compute::detail
