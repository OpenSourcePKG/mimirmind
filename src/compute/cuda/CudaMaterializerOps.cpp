// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "compute/cuda/CudaMaterializerOps.hpp"

#include "compute/ComputeOps.hpp"
#include "core/gpu/cuda/CudaComputeContext.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

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
      _castModule{loadModule(ctx.cudaContext(), "cast_to_f32")},
      _negExpModule{loadModule(ctx.cudaContext(), "neg_exp")},
      _addOneModule{loadModule(ctx.cudaContext(), "add_one")},
      _quantQ8Module{loadModule(ctx.cudaContext(), "quantize_bf16_to_q8_0")},
      _quantQ4KModule{loadModule(ctx.cudaContext(), "quantize_bf16_to_q4k")},
      _quantQ6KModule{loadModule(ctx.cudaContext(), "quantize_bf16_to_q6k")},
      _quantFp8Module{loadModule(ctx.cudaContext(), "quantize_bf16_to_fp8")},
      _dqNvfp4{_nvfp4Module.getFunction("dequant_nvfp4")},
      _dqFp8{_fp8Module.getFunction("dequant_fp8")},
      _castBf16{_castModule.getFunction("cast_bf16_to_f32")},
      _castF16{_castModule.getFunction("cast_f16_to_f32")},
      _negExp{_negExpModule.getFunction("neg_exp_f32")},
      _addOne{_addOneModule.getFunction("add_one_f32")},
      _quantQ8{_quantQ8Module.getFunction("quantize_bf16_to_q8_0")},
      _quantQ4K{_quantQ4KModule.getFunction("quantize_bf16_to_q4k")},
      _quantQ6K{_quantQ6KModule.getFunction("quantize_bf16_to_q6k")},
      _quantFp8{_quantFp8Module.getFunction("quantize_bf16_to_fp8")} {}

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

void CudaMaterializerOps::widenToF32(void* dstF32, const void* src,
                                     core::safetensors::SafetensorsDtype srcDtype,
                                     std::uint64_t n) {
    using Dt = core::safetensors::SafetensorsDtype;
    switch (srcDtype) {
        case Dt::F32:
            // Already F32; straight D2D copy on the context stream.
            _ops.appendMemoryCopy(dstF32, src, static_cast<std::size_t>(n) * 4);
            return;
        case Dt::BF16: {
            _castBf16.clearArgs();
            _castBf16.setPtr  (0, src);
            _castBf16.setPtr  (1, dstF32);
            _castBf16.setValue(2, static_cast<std::int64_t>(n));
            _castBf16.launch(_ctx.stream(), gridFor(n), 1, 1, kBlock, 1, 1);
            return;
        }
        case Dt::F16: {
            _castF16.clearArgs();
            _castF16.setPtr  (0, src);
            _castF16.setPtr  (1, dstF32);
            _castF16.setValue(2, static_cast<std::int64_t>(n));
            _castF16.launch(_ctx.stream(), gridFor(n), 1, 1, kBlock, 1, 1);
            return;
        }
        default:
            throw std::runtime_error(
                "CudaMaterializerOps::widenToF32: unsupported passthrough source "
                "dtype '" + std::string{core::safetensors::dtypeName(srcDtype)} +
                "' — expected F32, F16 or BF16");
    }
}

void CudaMaterializerOps::negExpInPlaceF32(void* f32, std::uint64_t n) {
    // Kernel: (float* x, long n) -> x = -exp(x), in place.
    _negExp.clearArgs();
    _negExp.setPtr  (0, f32);
    _negExp.setValue(1, static_cast<std::int64_t>(n));
    _negExp.launch(_ctx.stream(), gridFor(n), 1, 1, kBlock, 1, 1);
}

void CudaMaterializerOps::addOneInPlaceF32(void* f32, std::uint64_t n) {
    // Kernel: (float* x, long n) -> x = x + 1, in place.
    _addOne.clearArgs();
    _addOne.setPtr  (0, f32);
    _addOne.setValue(1, static_cast<std::int64_t>(n));
    _addOne.launch(_ctx.stream(), gridFor(n), 1, 1, kBlock, 1, 1);
}

void CudaMaterializerOps::quantizeBf16ToQ8_0(void* dstQ8_0, const void* srcBf16,
                                             std::uint64_t rows, std::uint64_t K) {
    // Kernel: (bf16* src, u8* dst, int K); grid (rows, K/32), block 32.
    _quantQ8.clearArgs();
    _quantQ8.setPtr  (0, srcBf16);
    _quantQ8.setPtr  (1, dstQ8_0);
    _quantQ8.setValue(2, static_cast<std::int32_t>(K));
    _quantQ8.launch(_ctx.stream(),
                    static_cast<std::uint32_t>(rows),
                    static_cast<std::uint32_t>(K / 32), 1,
                    32, 1, 1);
}

void CudaMaterializerOps::quantizeBf16ToQ4K(void* dstQ4K, const void* srcBf16,
                                            std::uint64_t totalElems) {
    // Kernel: (bf16* src, u8* dst); grid (totalElems/256), block 256.
    _quantQ4K.clearArgs();
    _quantQ4K.setPtr(0, srcBf16);
    _quantQ4K.setPtr(1, dstQ4K);
    _quantQ4K.launch(_ctx.stream(),
                     static_cast<std::uint32_t>(totalElems / 256), 1, 1,
                     256, 1, 1);
}

void CudaMaterializerOps::quantizeBf16ToQ6K(void* dstQ6K, const void* srcBf16,
                                            std::uint64_t totalElems) {
    // Kernel: (bf16* src, u8* dst); grid (totalElems/256), block 256.
    _quantQ6K.clearArgs();
    _quantQ6K.setPtr(0, srcBf16);
    _quantQ6K.setPtr(1, dstQ6K);
    _quantQ6K.launch(_ctx.stream(),
                     static_cast<std::uint32_t>(totalElems / 256), 1, 1,
                     256, 1, 1);
}

void CudaMaterializerOps::quantizeBf16ToFp8(void* dstFp8, const void* srcBf16,
                                            std::uint64_t rows, std::uint64_t K) {
    // Per-tensor E4M3 scale: readback the BF16 weights, take absmax on host,
    // scale = absmax / 448 (E4M3 max). This equals the checkpoint's original
    // FP8 weight_scale, so e4m3(BF16/scale) recovers the original e4m3 grid
    // near-losslessly (no double-quant). One-time at load.
    const std::uint64_t n = rows * K;
    std::vector<std::uint16_t> hostBf16(n);
    _ops.readbackToHost(hostBf16.data(), srcBf16, n * sizeof(std::uint16_t));
    _ops.flush();
    float absMax = 0.0f;
    for (std::uint64_t i = 0; i < n; ++i) {
        const std::uint32_t bits = static_cast<std::uint32_t>(hostBf16[i]) << 16;
        float f;
        std::memcpy(&f, &bits, sizeof(f));
        absMax = std::fmax(absMax, std::fabs(f));
    }
    const float scale    = (absMax > 0.0f) ? absMax * (1.0f / 448.0f) : 1.0f;
    const float invScale = (absMax > 0.0f) ? (448.0f / absMax) : 0.0f;

    // Kernel: (bf16* src, u8* dst, int K, float scale, float invScale).
    _quantFp8.clearArgs();
    _quantFp8.setPtr  (0, srcBf16);
    _quantFp8.setPtr  (1, dstFp8);
    _quantFp8.setValue(2, static_cast<std::int32_t>(K));
    _quantFp8.setValue(3, scale);
    _quantFp8.setValue(4, invScale);
    _quantFp8.launch(_ctx.stream(),
                     static_cast<std::uint32_t>(rows),
                     static_cast<std::uint32_t>(K / 32), 1,
                     32, 1, 1);
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