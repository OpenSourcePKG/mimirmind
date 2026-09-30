// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// 8.30.11.4 — the 11 FP4-tensor-core grouped-MoE ops, carved out of the GpuOps
// god-class into their own IMoeGroupedOps implementation. Bodies are the former
// GpuOps::moe*Async methods verbatim (only GpuOps:: -> CudaMoeGroupedOps:: and
// _pimpl->_moe* -> _moe* member access changed). The module loader + toInt32 are
// copied from GpuOps.cpp (anon-namespace; small, self-contained). The
// availability + scratch-size queries stay on GpuOps (consulted first).

#include "compute/cuda/CudaMoeGroupedOps.hpp"

#include "compute/detail/LaunchGuards.hpp"
#include "core/gpu/cuda/CudaComputeContext.hpp"
#include "core/gpu/cuda/CudaKernel.hpp"
#include "core/gpu/cuda/CudaModule.hpp"
#include "core/gpu/cuda/CudaStream.hpp"
#include "core/log/Log.hpp"

#include "MoeGroupedGemmNvfp4Tc.hpp"

#include <cuda_runtime.h>

#include <array>
#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

namespace mimirmind::compute::cuda {

namespace {

constexpr const char* kDefaultPtxDir = "/usr/local/share/mimirmind/ptx";

std::filesystem::path resolveHsacoPath(std::string_view name) {
    const std::string filename = std::string{name} + ".ptx";
    if (const char* env = std::getenv("MIMIRMIND_HSACO_DIR")) {
        if (env[0] != '\0') {
            const std::filesystem::path p =
                std::filesystem::path{env} / filename;
            if (std::filesystem::exists(p)) {
                return p;
            }
        }
    }
    {
        const std::filesystem::path p =
            std::filesystem::path{kDefaultPtxDir} / filename;
        if (std::filesystem::exists(p)) {
            return p;
        }
    }
    for (auto rel : std::array<const char*, 5>{
             "build/ptx", "build-both/ptx",
             "../build/ptx", "../build-both/ptx",
             "ptx"}) {
        const std::filesystem::path p =
            std::filesystem::path{rel} / filename;
        if (std::filesystem::exists(p)) {
            return p;
        }
    }
    throw std::runtime_error(
        "cuda::MoeGroupedOps: cannot find " + filename +
        " — set MIMIRMIND_HSACO_DIR or install to " + kDefaultPtxDir);
}

core::cuda::CudaModule loadCudaModule(core::cuda::CudaContext& ctx,
                                      std::string_view       name) {
    const auto path = resolveHsacoPath(name);
    MM_LOG_INFO("cuda::MoeGroupedOps", "loading module '{}' from {}",
                std::string{name}, path.string());
    return core::cuda::CudaModule::fromFile(ctx, path.string());
}

std::int32_t toInt32(std::size_t v, const char* tag) {
    return detail::toInt32(v, tag, "cuda::MoeGroupedOps");
}

} // namespace

CudaMoeGroupedOps::CudaMoeGroupedOps(core::cuda::CudaComputeContext& ctx)
    : _ctx{ctx},
      _moePadModule{loadCudaModule(ctx.cudaContext(), "moe_pad")},
      _moeActQuantModule{loadCudaModule(ctx.cudaContext(), "moe_act_quant_nvfp4")},
      _moePadOffsetsKernel{_moePadModule.getFunction("moe_pad_offsets")},
      _moeContigToPadKernel{_moePadModule.getFunction("moe_contig_to_pad")},
      _moeRowsScatterKernel{_moePadModule.getFunction("moe_rows_scatter_f32")},
      _moeIndexGatherKernel{_moePadModule.getFunction("moe_index_gather_i32")},
      _moeActQuantKernel{_moeActQuantModule.getFunction("moe_act_quant_nvfp4")},
      _moeActQuantRowsKernel{
          _moeActQuantModule.getFunction("moe_act_quant_nvfp4_rows")},
      _moeActQuantGatherRowsKernel{
          _moeActQuantModule.getFunction("moe_act_quant_nvfp4_gather_rows")},
      _moeSiluMulQuantRowsKernel{
          _moeActQuantModule.getFunction("moe_silu_mul_quant_nvfp4_rows")} {}

void CudaMoeGroupedOps::moeZeroBytesAsync(void* dst, std::size_t bytes) {
    if (bytes == 0) return;
    const cudaError_t rc = cudaMemsetAsync(dst, 0, bytes, _ctx.stream().handle());
    if (rc != cudaSuccess) {
        throw std::runtime_error(std::string("moeZeroBytesAsync: cudaMemsetAsync failed: ")
                                 + cudaGetErrorString(rc));
    }
}

void CudaMoeGroupedOps::moePadOffsetsAsync(const std::int32_t* expOffset,
                                std::int32_t* padOffset, std::size_t nExperts) {
    if (nExperts == 0) return;
    auto& k = _moePadOffsetsKernel;
    k.setPtr  (0, expOffset);
    k.setPtr  (1, padOffset);
    k.setValue(2, toInt32(nExperts, "moePadOffsets nExperts"));
    k.launch(_ctx.stream(), 1, 1, 1, 1, 1, 1);
}

void CudaMoeGroupedOps::moeContigToPadAsync(const std::int32_t* expOffset,
                                 const std::int32_t* padOffset,
                                 std::int32_t* contigToPad,
                                 std::size_t nExperts, std::size_t R) {
    if (R == 0) return;
    auto& k = _moeContigToPadKernel;
    k.setPtr  (0, expOffset);
    k.setPtr  (1, padOffset);
    k.setPtr  (2, contigToPad);
    k.setValue(3, toInt32(nExperts, "moeContigToPad nExperts"));
    k.setValue(4, toInt32(R, "moeContigToPad R"));
    k.launch(_ctx.stream(), static_cast<std::uint32_t>((R + 127) / 128), 1, 1, 128, 1, 1);
}

void CudaMoeGroupedOps::moeRowsScatterF32Async(const float* src, const std::int32_t* idxMap,
                                    float* dst, std::size_t nRows, std::size_t dim) {
    if (nRows == 0 || dim == 0) return;
    auto& k = _moeRowsScatterKernel;
    k.setPtr  (0, src);
    k.setPtr  (1, idxMap);
    k.setPtr  (2, dst);
    k.setValue(3, toInt32(nRows, "moeRowsScatter nRows"));
    k.setValue(4, toInt32(dim, "moeRowsScatter dim"));
    const std::uint32_t gy = static_cast<std::uint32_t>((dim + 255) / 256);
    k.launch(_ctx.stream(), static_cast<std::uint32_t>(nRows), gy, 1, 256, 1, 1);
}

void CudaMoeGroupedOps::moeIndexGatherI32Async(const std::int32_t* src,
                                    const std::int32_t* idxMap,
                                    std::int32_t* dst, std::size_t n) {
    if (n == 0) return;
    auto& k = _moeIndexGatherKernel;
    k.setPtr  (0, src);
    k.setPtr  (1, idxMap);
    k.setPtr  (2, dst);
    k.setValue(3, toInt32(n, "moeIndexGather n"));
    k.launch(_ctx.stream(), static_cast<std::uint32_t>((n + 127) / 128), 1, 1, 128, 1, 1);
}

void CudaMoeGroupedOps::moeActQuantNvfp4Async(const float* in, unsigned char* outNib,
                                   unsigned char* outSf, float gscale,
                                   std::size_t M, std::size_t K) {
    if (M == 0 || K == 0) return;
    auto& k = _moeActQuantKernel;
    k.setPtr  (0, in);
    k.setPtr  (1, outNib);
    k.setPtr  (2, outSf);
    k.setValue(3, gscale);
    k.setValue(4, toInt32(M, "moeActQuant M"));
    k.setValue(5, toInt32(K, "moeActQuant K"));
    const std::uint32_t gy = static_cast<std::uint32_t>(((K / 16) + 255) / 256);
    k.launch(_ctx.stream(), static_cast<std::uint32_t>(M), gy, 1, 256, 1, 1);
}

void CudaMoeGroupedOps::moeActQuantNvfp4RowsAsync(const float* in, unsigned char* outNib,
                                       unsigned char* outSf, float gscale,
                                       const std::int32_t* rowMap,
                                       std::size_t nRows, std::size_t K) {
    if (nRows == 0 || K == 0) return;
    auto& k = _moeActQuantRowsKernel;
    k.setPtr  (0, in);
    k.setPtr  (1, outNib);
    k.setPtr  (2, outSf);
    k.setValue(3, gscale);
    k.setPtr  (4, rowMap);
    k.setValue(5, toInt32(nRows, "moeActQuantRows nRows"));
    k.setValue(6, toInt32(K, "moeActQuantRows K"));
    const std::uint32_t gy = static_cast<std::uint32_t>(((K / 16) + 255) / 256);
    k.launch(_ctx.stream(), static_cast<std::uint32_t>(nRows), gy, 1, 256, 1, 1);
}

void CudaMoeGroupedOps::moeActQuantNvfp4GatherRowsAsync(const float* in, unsigned char* outNib,
                                             unsigned char* outSf, float gscale,
                                             const std::int32_t* rowMap,
                                             std::size_t nRows, std::size_t K,
                                             const std::int32_t* srcMap) {
    if (nRows == 0 || K == 0) return;
    // 5.21.10: fused gather + quant — reads COMPACT rows, writes padded slots.
    // 5.18.21: srcMap != nullptr additionally fuses the per-expert gather — reads
    // the UNGATHERED source `in` at srcMap[logical] (rowSrcTok), so the separate
    // moeGatherRowsAsync + xComp intermediate are skipped on the TC path.
    auto& k = _moeActQuantGatherRowsKernel;
    k.setPtr  (0, in);
    k.setPtr  (1, outNib);
    k.setPtr  (2, outSf);
    k.setValue(3, gscale);
    k.setPtr  (4, rowMap);
    k.setValue(5, toInt32(nRows, "moeActQuantGatherRows nRows"));
    k.setValue(6, toInt32(K, "moeActQuantGatherRows K"));
    k.setPtr  (7, srcMap);
    const std::uint32_t gy = static_cast<std::uint32_t>(((K / 16) + 255) / 256);
    k.launch(_ctx.stream(), static_cast<std::uint32_t>(nRows), gy, 1, 256, 1, 1);
}

void CudaMoeGroupedOps::moeSiluMulQuantNvfp4RowsAsync(const float* gate, const float* up,
                                           unsigned char* outNib, unsigned char* outSf,
                                           float gscale, const std::int32_t* rowMap,
                                           std::size_t nRows, std::size_t K) {
    if (nRows == 0 || K == 0) return;
    auto& k = _moeSiluMulQuantRowsKernel;
    k.setPtr  (0, gate);
    k.setPtr  (1, up);
    k.setPtr  (2, outNib);
    k.setPtr  (3, outSf);
    k.setValue(4, gscale);
    k.setPtr  (5, rowMap);
    k.setValue(6, toInt32(nRows, "moeSiluMulQuant nRows"));
    k.setValue(7, toInt32(K, "moeSiluMulQuant K"));
    const std::uint32_t gy = static_cast<std::uint32_t>(((K / 16) + 255) / 256);
    k.launch(_ctx.stream(), static_cast<std::uint32_t>(nRows), gy, 1, 256, 1, 1);
}

void CudaMoeGroupedOps::moeGroupedGemmNvfp4TcBanksAsync(
    std::size_t nExperts, std::size_t N, std::size_t K,
    const std::int32_t* expOffset, const std::int32_t* padOffset,
    const void* aBank, const void* sfaBank,
    const void* bBank, const void* sfbBank,
    const float* globalsBank, void* dBank,
    void* scratch, std::size_t scratchBytes) {
#ifdef MIMIRMIND_HAVE_CUTLASS_MOE
    // 5.27.10 diagnostic (env MIMIRMIND_BANKS_DIAG): the FIRST process-wide call
    // into the CUTLASS NVFP4-TC grouped GEMM lazily initializes it. If that
    // first call lands on a serve worker thread that never ran cudaSetDevice,
    // gemm.initialize() returns kErrorInternal and poisons the context (5.27.10).
    // Log the calling thread + current CUDA device ONCE so smoke (main thread)
    // vs serve (worker thread) can be compared, and to verify the warmup fix.
    static std::atomic<bool> firstBanksCall{true};
    if (std::getenv("MIMIRMIND_BANKS_DIAG") != nullptr &&
        firstBanksCall.exchange(false)) {
        int dev = -999;
        const cudaError_t drc = cudaGetDevice(&dev);
        std::ostringstream tid;
        tid << std::this_thread::get_id();
        MM_LOG_INFO("nvfp4-tc-banks",
                    "first banks call: thread={} cudaGetDevice={} (getDeviceRc={}) "
                    "nExperts={} N={} K={}",
                    tid.str(), dev, static_cast<int>(drc), nExperts, N, K);
    }
    // Scratch is caller-owned (per-slot BlockBuffers) — no shared GpuOps state,
    // so concurrent prefills never collide on it.
    const int rc = kernels::cutlassmoe::runGroupedNvfp4TcF32Banks(
        static_cast<int>(nExperts), static_cast<int>(N), static_cast<int>(K),
        expOffset, padOffset, aBank, sfaBank, bBank, sfbBank, globalsBank, dBank,
        scratch, scratchBytes, _ctx.stream().handle());
    if (rc != 0) {
        throw std::runtime_error(
            "moeGroupedGemmNvfp4TcBanksAsync: CUTLASS grouped GEMM failed rc="
            + std::to_string(rc));
    }
#else
    (void)nExperts; (void)N; (void)K; (void)expOffset; (void)padOffset;
    (void)aBank; (void)sfaBank; (void)bBank; (void)sfbBank; (void)globalsBank;
    (void)dBank; (void)scratch; (void)scratchBytes;
    throw std::runtime_error(
        "moeGroupedGemmNvfp4TcBanksAsync: CUTLASS not linked in this build");
#endif
}

void CudaMoeGroupedOps::moeGroupedGemmNvfp4TcBanksGateUpAsync(
    std::size_t nExperts, std::size_t N, std::size_t K,
    const std::int32_t* expOffset, const std::int32_t* padOffset,
    const void* aBank, const void* sfaBank,
    const void* gateBBank, const void* gateSfbBank,
    const float* gateGlobalsBank, void* gateDBank,
    const void* upBBank, const void* upSfbBank,
    const float* upGlobalsBank, void* upDBank,
    void* scratch, std::size_t scratchBytes) {
#ifdef MIMIRMIND_HAVE_CUTLASS_MOE
    // Scratch is caller-owned (per-slot BlockBuffers) — concurrent prefills
    // never collide. 5.18.21: gate+up in one grouped GEMM (2*nExperts groups).
    const int rc = kernels::cutlassmoe::runGroupedNvfp4TcF32BanksGateUp(
        static_cast<int>(nExperts), static_cast<int>(N), static_cast<int>(K),
        expOffset, padOffset, aBank, sfaBank,
        gateBBank, gateSfbBank, gateGlobalsBank, gateDBank,
        upBBank, upSfbBank, upGlobalsBank, upDBank,
        scratch, scratchBytes, _ctx.stream().handle());
    if (rc != 0) {
        throw std::runtime_error(
            "moeGroupedGemmNvfp4TcBanksGateUpAsync: CUTLASS grouped GEMM failed rc="
            + std::to_string(rc));
    }
#else
    (void)nExperts; (void)N; (void)K; (void)expOffset; (void)padOffset;
    (void)aBank; (void)sfaBank; (void)gateBBank; (void)gateSfbBank;
    (void)gateGlobalsBank; (void)gateDBank; (void)upBBank; (void)upSfbBank;
    (void)upGlobalsBank; (void)upDBank; (void)scratch; (void)scratchBytes;
    throw std::runtime_error(
        "moeGroupedGemmNvfp4TcBanksGateUpAsync: CUTLASS not linked in this build");
#endif
}

} // namespace mimirmind::compute::cuda
