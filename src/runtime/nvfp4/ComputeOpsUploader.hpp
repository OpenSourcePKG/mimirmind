// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "compute/ComputeOps.hpp"
#include "runtime/nvfp4/NvFp4Model.hpp"

#include <cstddef>

namespace mimirmind::runtime::nvfp4 {

/**
 * Adapts a `compute::ComputeOps` (any backend — CUDA/L0/HIP/CPU) to the
 * loader's narrow `DeviceUploader` seam. This is the one place that couples
 * the NVFP4 loader to the compute backend; the loader itself and its tests
 * stay backend-free. Include only where the real load path is wired.
 */
class ComputeOpsUploader final : public DeviceUploader {
public:
    explicit ComputeOpsUploader(compute::ComputeOps& ops) noexcept : _ops(ops) {}

    [[nodiscard]] compute::ComputeBuffer allocate(std::size_t bytes) override {
        return _ops.allocate(bytes);
    }
    void uploadHostBytes(void* deviceDst, const void* hostSrc, std::size_t bytes) override {
        _ops.uploadHostBytes(deviceDst, hostSrc, bytes);
    }

private:
    compute::ComputeOps& _ops;
};

} // namespace mimirmind::runtime::nvfp4