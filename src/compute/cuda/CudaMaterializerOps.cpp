// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "compute/cuda/CudaMaterializerOps.hpp"

#include "compute/ComputeOps.hpp"
#include "core/gpu/cuda/CudaComputeContext.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>

namespace mimirmind::compute::cuda {

namespace {

// Resolve "<name>.ptx" the same way GpuOps / MoeTopKRouteDevice do.
std::filesystem::path resolvePtx(std::string_view name) {
    const std::string filename = std::string{name} + ".ptx";
    if (const char* env = std::getenv("MIMIRMIND_HSACO_DIR")) {
        const std::filesystem::path p = std::filesystem::path{env} / filename;
        if (std::filesystem::exists(p)) return p;
    }
    {
        const std::filesystem::path p =
            std::filesystem::path{"/usr/local/share/mimirmind/ptx"} / filename;
        if (std::filesystem::exists(p)) return p;
    }
    for (const auto* rel : std::array<const char*, 5>{
             "build/ptx", "build-both/ptx", "../build/ptx", "../build-both/ptx", "ptx"}) {
        const std::filesystem::path p = std::filesystem::path{rel} / filename;
        if (std::filesystem::exists(p)) return p;
    }
    throw std::runtime_error("CudaMaterializerOps: cannot find " + filename +
                             " — set MIMIRMIND_HSACO_DIR or install to "
                             "/usr/local/share/mimirmind/ptx");
}

core::cuda::CudaModule loadModule(core::cuda::CudaContext& ctx, std::string_view name) {
    return core::cuda::CudaModule::fromFile(ctx, resolvePtx(name).string());
}

constexpr std::uint32_t kBlock = 256; // must match DEQUANT_*_LOCAL

std::uint32_t gridFor(std::uint64_t total) {
    return static_cast<std::uint32_t>((total + kBlock - 1) / kBlock);
}

} // namespace

CudaMaterializerOps::CudaMaterializerOps(core::cuda::CudaComputeContext& ctx, ComputeOps& ops)
    : _ctx{ctx},
      _ops{ops},
      _nvfp4Module{loadModule(ctx.cudaContext(), "dequant_nvfp4")},
      _fp8Module{loadModule(ctx.cudaContext(), "dequant_fp8")},
      _dqNvfp4{_nvfp4Module.getFunction("dequant_nvfp4")},
      _dqFp8{_fp8Module.getFunction("dequant_fp8")} {}

ComputeBuffer CudaMaterializerOps::allocate(std::size_t bytes) {
    return _ops.allocate(bytes);
}

void CudaMaterializerOps::dequantNvfp4(const void* packed, const void* blockScale,
                                       float global, std::uint64_t rows,
                                       std::uint64_t in, void* dstBf16) {
    // Kernel: (packed U8, block_scale F8, float global, bf16* out, int rows, int in)
    _dqNvfp4.clearArgs();
    _dqNvfp4.setPtr  (0, packed);
    _dqNvfp4.setPtr  (1, blockScale);
    _dqNvfp4.setValue(2, global);
    _dqNvfp4.setPtr  (3, dstBf16);
    _dqNvfp4.setValue(4, static_cast<std::int32_t>(rows));
    _dqNvfp4.setValue(5, static_cast<std::int32_t>(in));
    _dqNvfp4.launch(_ctx.stream(), gridFor(rows * in), 1, 1, kBlock, 1, 1);
}

void CudaMaterializerOps::dequantFp8(const void* weight, float scale,
                                     std::uint64_t n, void* dstBf16) {
    // Kernel: (weight F8, float scale, bf16* out, long n)
    _dqFp8.clearArgs();
    _dqFp8.setPtr  (0, weight);
    _dqFp8.setValue(1, scale);
    _dqFp8.setPtr  (2, dstBf16);
    _dqFp8.setValue(3, static_cast<std::int64_t>(n));
    _dqFp8.launch(_ctx.stream(), gridFor(n), 1, 1, kBlock, 1, 1);
}

void CudaMaterializerOps::copyBytes(void* dst, const void* src, std::size_t bytes) {
    _ops.appendMemoryCopy(dst, src, bytes); // async D2D on the context stream
}

float CudaMaterializerOps::readF32(const void* devPtr) {
    float host = 0.0F;
    const cudaError_t rc = cudaMemcpy(&host, devPtr, sizeof(float), cudaMemcpyDeviceToHost);
    if (rc != cudaSuccess) {
        throw std::runtime_error(std::string("CudaMaterializerOps::readF32: cudaMemcpy failed: ")
                                 + cudaGetErrorString(rc));
    }
    return host;
}

} // namespace mimirmind::compute::cuda