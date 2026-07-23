// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "compute/ComputeBuffer.hpp"
#include "core/modelopt/Qwen35MoeMaterializer.hpp"
#include "runtime/nvfp4/NvFp4Model.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mimirmind::runtime::nvfp4 {

/**
 * The device capability the materializer executor needs: allocate BF16
 * output buffers, run the two dequant kernels, device-copy the BF16
 * passthrough tensors, and read a scalar scale back to the host. A CUDA
 * adapter over `GpuOps` + `CudaModule` satisfies this; a host test double
 * (CPU oracle + memcpy) does too, so the executor is exercisable without a
 * GPU. All device pointers come from the source `NvFp4Model`'s uploads and
 * the buffers this allocates.
 */
struct MaterializerDeviceOps {
    virtual ~MaterializerDeviceOps() = default;

    [[nodiscard]] virtual compute::ComputeBuffer allocate(std::size_t bytes) = 0;

    /// packed U8 + F8_E4M3 block scale + F32 global -> BF16 [rows*in] at dst.
    virtual void dequantNvfp4(const void* packed, const void* blockScale, float global,
                              std::uint64_t rows, std::uint64_t in, void* dstBf16) = 0;

    /// F8_E4M3 weight + F32 per-tensor scale -> BF16 [n] at dst.
    virtual void dequantFp8(const void* weight, float scale,
                            std::uint64_t n, void* dstBf16) = 0;

    /// Device-to-device copy of already-BF16 bytes (passthrough tensors).
    virtual void copyBytes(void* dst, const void* src, std::size_t bytes) = 0;

    /// Widen a half-width source (`srcDtype` BF16 or F16) to F32 [n] at dst,
    /// or D2D-copy an F32 source verbatim. Used for unquantised passthrough
    /// tensors the runtime reads as `const float*` (norms, ssm scalars,
    /// conv1d, biases, router, embed).
    virtual void widenToF32(void* dstF32, const void* src,
                            safetensors::SafetensorsDtype srcDtype,
                            std::uint64_t n) = 0;

    /// Read a 4-byte F32 scalar from device memory to the host.
    [[nodiscard]] virtual float readF32(const void* devPtr) = 0;

    /// In-place element-wise y = -exp(y) over F32 [n] device memory. Turns the
    /// HF `A_log` into the GGUF `ssm_a` (= -exp(A_log)) the DeltaNet decay gate
    /// consumes directly. Only invoked on F32 passthrough outputs.
    virtual void negExpInPlaceF32(void* f32, std::uint64_t n) = 0;
};

/// One materialised GGUF tensor: BF16 on device, named + dimensioned in GGUF
/// convention, ready to wrap in a WeightsMap. Owns its device buffer.
struct MaterializedTensor {
    std::string                ggufName;
    compute::ComputeBuffer     buffer;   ///< device memory: F32 if isF32 else BF16
    std::vector<std::uint64_t> ggufDims; ///< ne-order
    std::uint64_t              elems{0};
    bool                       isF32{false}; ///< true for widened passthrough tensors
};

/**
 * Execute a materialization plan: for each step allocate a BF16 device
 * buffer and fill it from the plan's HF sources — dequant NVFP4/FP8 weights
 * (their packed data + scales resolved by name from `src`) or copy BF16
 * passthrough tensors — writing each source at its `dstElemOffset`
 * (expert stacking). Returns the BF16 tensors keyed by GGUF name.
 *
 * Throws std::runtime_error if a source tensor or its scale sidecar is
 * absent from `src`.
 */
[[nodiscard]] std::vector<MaterializedTensor>
executeMaterialization(const std::vector<core::modelopt::MaterializationStep>& steps,
                       const NvFp4Model&        src,
                       MaterializerDeviceOps&   ops);

} // namespace mimirmind::runtime::nvfp4