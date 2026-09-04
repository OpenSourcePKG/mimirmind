// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "compute/cuda/GpuMatmul.hpp"

#include "compute/Matmul.hpp"
#include "compute/QuantType.hpp"
#include "compute/QuantTypeRegistry.hpp"
#include "compute/cuda/GpuOps.hpp"
#include "core/config/Config.hpp"
#include "core/gguf/GgufTypes.hpp"
#include "core/gpu/cuda/CudaComputeContext.hpp"
#include "core/gpu/cuda/CudaKernel.hpp"
#include "core/gpu/cuda/CudaMemoryAllocator.hpp"
#include "core/gpu/cuda/CudaModule.hpp"
#include "core/gpu/cuda/CudaStream.hpp"
#include "core/log/Log.hpp"

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <unordered_map>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mimirmind::compute::cuda {

namespace {

constexpr const char* kDefaultPtxDir = "/usr/local/share/mimirmind/ptx";

// Same resolver as HipGpuOps.cpp — env override, prod install, three
// build-tree fallbacks. Kept as a local duplicate rather than lifted to
// a shared helper because these two files are the only consumers and
// their lookups are independent (a common header would drag CudaModule
// deps into a place that doesn't need them).
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
        "compute::cuda::GpuMatmul: cannot find " + filename +
        " — set MIMIRMIND_HSACO_DIR or install to " + kDefaultPtxDir);
}

::mimirmind::core::cuda::CudaModule loadCudaModule(
    ::mimirmind::core::cuda::CudaContext& ctx, std::string_view name) {
    const auto path = resolveHsacoPath(name);
    MM_LOG_INFO("hip::GpuMatmul", "loading module '{}' from {}",
                std::string{name}, path.string());
    return ::mimirmind::core::cuda::CudaModule::fromFile(ctx, path.string());
}

[[noreturn]] void throwNotImplemented(const char* method) {
    throw std::runtime_error(
        std::string{"compute::cuda::GpuMatmul::"} + method +
        ": not yet implemented (Schritt 3b sub-F skeleton — kernel-launch "
        "impl lands in follow-up commits)");
}

// Fill a raw Q8_0-encoded weight buffer with reproducible pseudo-random
// bytes plus a tiny fp16 per-block scale (~0.02). Same shape as the L0
// autotune's `fillSyntheticWeights` for GgmlType::Q8_0 — random int8
// quants would land the matmul-output magnitude near zero and starve
// the autotune of signal, so we pin the scale to a small constant.
void fillSyntheticQ8_0(std::uint8_t* dst, std::size_t nbytes) {
    std::mt19937 rng{0xC0FFEEU};
    std::uniform_int_distribution<int> distByte(0, 255);
    for (std::size_t i = 0; i < nbytes; ++i) {
        dst[i] = static_cast<std::uint8_t>(distByte(rng));
    }
    // fp16 tiny positive constant used as per-block scale (~0.02).
    constexpr std::uint16_t kHalfTiny = 0x2400U;
    constexpr std::size_t   kBlockBytes = 34;
    for (std::size_t off = 0; off + kBlockBytes <= nbytes; off += kBlockBytes) {
        std::memcpy(dst + off, &kHalfTiny, 2);
    }
}

// Median of a sample. Same shape as the L0 helper; used to squash the
// occasional outlier iteration (thermal / OS jitter) without giving
// away the wall-clock signal a mean would.
double medianMs(std::vector<double> xs) {
    if (xs.empty()) return 0.0;
    std::sort(xs.begin(), xs.end());
    const std::size_t mid = xs.size() / 2;
    return xs.size() % 2 == 1
        ? xs[mid]
        : 0.5 * (xs[mid - 1] + xs[mid]);
}

} // namespace

// Pimpl body — Q8_0 matmul family (5 kernels) + Q5_0 vec + Q6_K vec
// + Q4_K vec + Q5_K vec. Full K-quant coverage (Q4_K / Q5_K / Q6_K)
// plus the Q5_0 / Q8_0 fixed-block quants used across Qwen 2.5 and
// Gemma 4. matmulAsync's CPU-fallback branch stays as a safety net
// for exotic types with a QuantTypeRegistry entry that lack a native
// kernel (e.g. F16 / BF16 / F32 direct matmul).
struct GpuMatmul::Impl {
    ::mimirmind::core::cuda::CudaModule _matmulQ8_0VecModule;
    ::mimirmind::core::cuda::CudaKernel _matmulQ8_0VecKernel;
    ::mimirmind::core::cuda::CudaModule _matmulQ8_0GemmModule;
    ::mimirmind::core::cuda::CudaKernel _matmulQ8_0GemmKernel;
    ::mimirmind::core::cuda::CudaModule _matmulQ8_0GemmV2Module;
    ::mimirmind::core::cuda::CudaKernel _matmulQ8_0GemmV2Kernel;
    ::mimirmind::core::cuda::CudaModule _matmulQ8_0MmqModule;
    ::mimirmind::core::cuda::CudaKernel _matmulQ8_0MmqKernel;
    ::mimirmind::core::cuda::CudaModule _matmulQ8_0MmqTcModule;
    ::mimirmind::core::cuda::CudaKernel _matmulQ8_0MmqTcKernel;
    ::mimirmind::core::cuda::CudaModule _matmulQ4KMmqModule;
    ::mimirmind::core::cuda::CudaKernel _matmulQ4KMmqKernel;
    ::mimirmind::core::cuda::CudaModule _matmulQ5KMmqModule;
    ::mimirmind::core::cuda::CudaKernel _matmulQ5KMmqKernel;
    ::mimirmind::core::cuda::CudaModule _matmulQ8_0VecDp4aModule;
    ::mimirmind::core::cuda::CudaKernel _matmulQ8_0VecDp4aKernel;
    ::mimirmind::core::cuda::CudaModule _matmulQ6KVecDp4aModule;
    ::mimirmind::core::cuda::CudaKernel _matmulQ6KVecDp4aKernel;
    ::mimirmind::core::cuda::CudaModule _moeDownFusedKQ8_0Module;
    ::mimirmind::core::cuda::CudaKernel _moeDownFusedKQ8_0Kernel;
    ::mimirmind::core::cuda::CudaModule _moeDownFusedKQ6KModule;
    ::mimirmind::core::cuda::CudaKernel _moeDownFusedKQ6KKernel;
    ::mimirmind::core::cuda::CudaModule _moeDownFusedKQ5KModule;
    ::mimirmind::core::cuda::CudaKernel _moeDownFusedKQ5KKernel;
    ::mimirmind::core::cuda::CudaModule _moeDownFusedKQ5KBatchedModule;
    ::mimirmind::core::cuda::CudaKernel _moeDownFusedKQ5KBatchedKernel;
    ::mimirmind::core::cuda::CudaModule _moeDownFusedKQ6KBatchedModule;
    ::mimirmind::core::cuda::CudaKernel _moeDownFusedKQ6KBatchedKernel;
    ::mimirmind::core::cuda::CudaModule _moeGateUpFusedKQ4KModule;
    ::mimirmind::core::cuda::CudaKernel _moeGateUpFusedKQ4KKernel;
    ::mimirmind::core::cuda::CudaModule _moeGateUpFusedKQ4KBatchedModule;
    ::mimirmind::core::cuda::CudaKernel _moeGateUpFusedKQ4KBatchedKernel;
    ::mimirmind::core::cuda::CudaModule _moeGateUpFusedKNvblkBatchedModule;
    ::mimirmind::core::cuda::CudaKernel _moeGateUpFusedKNvblkBatchedKernel;
    ::mimirmind::core::cuda::CudaModule _moeDownFusedKNvblkBatchedModule;
    ::mimirmind::core::cuda::CudaKernel _moeDownFusedKNvblkBatchedKernel;
    // E-d.5 FP4-TC decode kernels.
    ::mimirmind::core::cuda::CudaModule _moeGateUpFusedKTcModule;
    ::mimirmind::core::cuda::CudaKernel _moeGateUpFusedKTcKernel;
    ::mimirmind::core::cuda::CudaModule _moeDownFusedKTcModule;
    ::mimirmind::core::cuda::CudaKernel _moeDownFusedKTcKernel;
    ::mimirmind::core::cuda::CudaModule _moeGateUpFusedKBf16BatchedModule;
    ::mimirmind::core::cuda::CudaKernel _moeGateUpFusedKBf16BatchedKernel;
    ::mimirmind::core::cuda::CudaModule _moeDownFusedKBf16BatchedModule;
    ::mimirmind::core::cuda::CudaKernel _moeDownFusedKBf16BatchedKernel;
    ::mimirmind::core::cuda::CudaModule _ffnGateUpFusedQ8Module;
    ::mimirmind::core::cuda::CudaKernel _ffnGateUpFusedQ8Kernel;
    ::mimirmind::core::cuda::CudaModule _matmulQ5_0VecModule;
    ::mimirmind::core::cuda::CudaKernel _matmulQ5_0VecKernel;
    ::mimirmind::core::cuda::CudaModule _matmulQ6KVecModule;
    ::mimirmind::core::cuda::CudaKernel _matmulQ6KVecKernel;
    ::mimirmind::core::cuda::CudaModule _matmulQ3KVecModule;
    ::mimirmind::core::cuda::CudaKernel _matmulQ3KVecKernel;
    ::mimirmind::core::cuda::CudaModule _matmulQ4KVecModule;
    ::mimirmind::core::cuda::CudaKernel _matmulQ4KVecKernel;
    ::mimirmind::core::cuda::CudaModule _matmulQ5KVecModule;
    ::mimirmind::core::cuda::CudaKernel _matmulQ5KVecKernel;
    ::mimirmind::core::cuda::CudaModule _matmulF32VecModule;
    ::mimirmind::core::cuda::CudaKernel _matmulF32VecKernel;
    ::mimirmind::core::cuda::CudaModule _matmulBf16VecModule;
    ::mimirmind::core::cuda::CudaKernel _matmulBf16VecKernel;
    ::mimirmind::core::cuda::CudaModule _matmulBf16GemmModule;
    ::mimirmind::core::cuda::CudaKernel _matmulBf16GemmKernel;
    ::mimirmind::core::cuda::CudaModule _matmulBf16GemmTcModule;
    ::mimirmind::core::cuda::CudaKernel _matmulBf16GemmTcKernel;
    ::mimirmind::core::cuda::CudaModule _matmulBf16GemmTf32TcModule;
    ::mimirmind::core::cuda::CudaKernel _matmulBf16GemmTf32TcKernel;
    ::mimirmind::core::cuda::CudaModule _matmulFp8VecModule;
    ::mimirmind::core::cuda::CudaKernel _matmulFp8VecKernel;
    ::mimirmind::core::cuda::CudaModule _matmulFp8GemmModule;
    ::mimirmind::core::cuda::CudaKernel _matmulFp8GemmKernel;
    ::mimirmind::core::cuda::CudaModule _matmulNvblkVecModule;
    ::mimirmind::core::cuda::CudaKernel _matmulNvblkVecKernel;
    ::mimirmind::core::cuda::CudaModule _matmulNvblkGemmModule;
    ::mimirmind::core::cuda::CudaKernel _matmulNvblkGemmKernel;
    // De-interleaved uint4 NVFP4 vec (shexp / full-attn decode) + its helper,
    // plus a by-pointer cache of the de-interleaved (nib, scale) buffers.
    ::mimirmind::core::cuda::CudaModule _nvblkDeintModule;
    ::mimirmind::core::cuda::CudaKernel _nvblkDeinterleaveKernel;
    ::mimirmind::core::cuda::CudaKernel _nvblkDeintVecKernel;
    // Inc 5: dense register-x + uint4 deint lmhead GEMV (no smem), selected when
    // MIMIRMIND_DENSE_DEINT_REG=1 (bit-identical to matmul_nvfp4blk_vec).
    ::mimirmind::core::cuda::CudaKernel _nvblkDeintRegVecKernel;
    struct NvblkDeint { void* nib{nullptr}; void* scale{nullptr}; };
    std::unordered_map<const void*, NvblkDeint> _nvblkDeintCache;
    // cuBLASLt dense BF16 matmul (opt-in). The cast_f32_to_bf16 kernel stages
    // the F32 activation into `_xBf16` before the tensor-core GEMM; the handle
    // + workspace are created lazily on first use in cublasBf16Matmul().
    ::mimirmind::core::cuda::CudaModule _castToF32Module;
    ::mimirmind::core::cuda::CudaKernel _castF32ToBf16Kernel;
    cublasLtHandle_t _ltHandle{nullptr};
    void*            _ltWorkspace{nullptr};
    std::size_t      _ltWorkspaceBytes{0};
    void*            _xBf16{nullptr};   // staged BF16 activations (grows on demand)
    std::size_t      _xBf16Bytes{0};

    // cuBLASLt per-tensor FP8 path (opt-in). Quant kernels + a by-pointer cache
    // of E4M3 weights (each with a device [scale,invScale] pair), plus per-call
    // E4M3-activation staging and amax/scale scratch.
    ::mimirmind::core::cuda::CudaModule _cublasFp8Module;
    ::mimirmind::core::cuda::CudaKernel _amaxF32Kernel;
    ::mimirmind::core::cuda::CudaKernel _amaxBf16Kernel;
    ::mimirmind::core::cuda::CudaKernel _scaleFromAmaxKernel;
    ::mimirmind::core::cuda::CudaKernel _castF32E4m3Kernel;
    ::mimirmind::core::cuda::CudaKernel _castBf16E4m3Kernel;
    struct Fp8W { void* data{nullptr}; void* scale{nullptr}; }; // E4M3 + [scale,inv]
    std::unordered_map<const void*, Fp8W> _fp8WeightCache;
    // F32 weight -> BF16 copy, cached by source pointer (stable loaded weights).
    // Lets small M>1 F32 GEMMs (e.g. the MoE router) reuse the BF16 tensor-core
    // kernel instead of the per-row F32 vec launches.
    std::unordered_map<const void*, void*> _bf16FromF32Cache;
    void*  _xFp8{nullptr};        // staged E4M3 activations (grows on demand)
    std::size_t _xFp8Bytes{0};
    void*  _xScaleDev{nullptr};   // 2 floats [scale, invScale] for X
    void*  _amaxDev{nullptr};     // 1 float amax scratch (kept zeroed by
                                  // scale_from_amax's consume-and-reset)
    // 5.18.10.4: cached cuBLASLt plans keyed by packed (N, M, K). The FP8
    // decode/prefill path issues 72-144 narrow GEMMs per forward; recreating
    // the matmul desc + three layouts (and letting Lt re-run its internal
    // heuristic) per call is pure host time that starves the stream. Static
    // attributes (transposes, scale modes, the FIXED X-scale pointer) are set
    // once at plan build; only the per-weight A_SCALE_POINTER is updated per
    // call. Numerics are untouched — bit-identical.
    struct LtPlan {
        cublasLtMatmulDesc_t   op{nullptr};
        cublasLtMatrixLayout_t aL{nullptr};
        cublasLtMatrixLayout_t bL{nullptr};
        cublasLtMatrixLayout_t cL{nullptr};
    };
    std::unordered_map<std::uint64_t, LtPlan> _ltPlanCache;

    ~Impl() {
        for (auto& kv : _fp8WeightCache) {
            if (kv.second.data  != nullptr) { cudaFree(kv.second.data); }
            if (kv.second.scale != nullptr) { cudaFree(kv.second.scale); }
        }
        for (auto& kv : _nvblkDeintCache) {
            if (kv.second.nib   != nullptr) { cudaFree(kv.second.nib); }
            if (kv.second.scale != nullptr) { cudaFree(kv.second.scale); }
        }
        for (auto& kv : _bf16FromF32Cache) {
            if (kv.second != nullptr) { cudaFree(kv.second); }
        }
        for (auto& kv : _ltPlanCache) {
            if (kv.second.cL != nullptr) { cublasLtMatrixLayoutDestroy(kv.second.cL); }
            if (kv.second.bL != nullptr) { cublasLtMatrixLayoutDestroy(kv.second.bL); }
            if (kv.second.aL != nullptr) { cublasLtMatrixLayoutDestroy(kv.second.aL); }
            if (kv.second.op != nullptr) { cublasLtMatmulDescDestroy(kv.second.op); }
        }
        if (_amaxDev != nullptr)     { cudaFree(_amaxDev); }
        if (_xScaleDev != nullptr)   { cudaFree(_xScaleDev); }
        if (_xFp8 != nullptr)        { cudaFree(_xFp8); }
        if (_xBf16 != nullptr)       { cudaFree(_xBf16); }
        if (_ltWorkspace != nullptr) { cudaFree(_ltWorkspace); }
        if (_ltHandle != nullptr)    { cublasLtDestroy(_ltHandle); }
    }

    explicit Impl(::mimirmind::core::cuda::CudaContext& ctx)
        : _matmulQ8_0VecModule    {loadCudaModule(ctx, "matmul_q8_0_vec")},
          _matmulQ8_0VecKernel    {
              _matmulQ8_0VecModule.getFunction("matmul_q8_0_vec")},
          _matmulQ8_0GemmModule   {loadCudaModule(ctx, "matmul_q8_0_gemm")},
          _matmulQ8_0GemmKernel   {
              _matmulQ8_0GemmModule.getFunction("matmul_q8_0_gemm")},
          _matmulQ8_0GemmV2Module {loadCudaModule(ctx, "matmul_q8_0_gemm_v2")},
          _matmulQ8_0GemmV2Kernel {
              _matmulQ8_0GemmV2Module.getFunction("matmul_q8_0_gemm_v2")},
          _matmulQ8_0MmqModule    {loadCudaModule(ctx, "matmul_q8_0_mmq")},
          _matmulQ8_0MmqKernel    {
              _matmulQ8_0MmqModule.getFunction("matmul_q8_0_mmq")},
          _matmulQ8_0MmqTcModule  {loadCudaModule(ctx, "matmul_q8_0_mmq_tc")},
          _matmulQ8_0MmqTcKernel  {
              _matmulQ8_0MmqTcModule.getFunction("matmul_q8_0_mmq_tc")},
          _matmulQ4KMmqModule     {loadCudaModule(ctx, "matmul_q4k_mmq")},
          _matmulQ4KMmqKernel     {
              _matmulQ4KMmqModule.getFunction("matmul_q4k_mmq")},
          _matmulQ5KMmqModule     {loadCudaModule(ctx, "matmul_q5k_mmq")},
          _matmulQ5KMmqKernel     {
              _matmulQ5KMmqModule.getFunction("matmul_q5k_mmq")},
          _matmulQ8_0VecDp4aModule{loadCudaModule(ctx, "matmul_q8_0_vec_dp4a")},
          _matmulQ8_0VecDp4aKernel{
              _matmulQ8_0VecDp4aModule.getFunction("matmul_q8_0_vec_dp4a")},
          _matmulQ6KVecDp4aModule {loadCudaModule(ctx, "matmul_q6k_vec_dp4a")},
          _matmulQ6KVecDp4aKernel {
              _matmulQ6KVecDp4aModule.getFunction("matmul_q6k_vec_dp4a")},
          _moeDownFusedKQ8_0Module{loadCudaModule(ctx, "moe_down_fused_k_q8_0")},
          _moeDownFusedKQ8_0Kernel{
              _moeDownFusedKQ8_0Module.getFunction("moe_down_fused_k_q8_0")},
          _moeDownFusedKQ6KModule {loadCudaModule(ctx, "moe_down_fused_k_q6k")},
          _moeDownFusedKQ6KKernel {
              _moeDownFusedKQ6KModule.getFunction("moe_down_fused_k_q6k")},
          _moeDownFusedKQ5KModule {loadCudaModule(ctx, "moe_down_fused_k_q5k")},
          _moeDownFusedKQ5KKernel {
              _moeDownFusedKQ5KModule.getFunction("moe_down_fused_k_q5k")},
          _moeDownFusedKQ5KBatchedModule{loadCudaModule(ctx, "moe_down_fused_k_q5k_batched")},
          _moeDownFusedKQ5KBatchedKernel{
              _moeDownFusedKQ5KBatchedModule.getFunction("moe_down_fused_k_q5k_batched")},
          _moeDownFusedKQ6KBatchedModule{loadCudaModule(ctx, "moe_down_fused_k_q6k_batched")},
          _moeDownFusedKQ6KBatchedKernel{
              _moeDownFusedKQ6KBatchedModule.getFunction("moe_down_fused_k_q6k_batched")},
          _moeGateUpFusedKQ4KModule{loadCudaModule(ctx, "moe_gate_up_fused_k_q4k")},
          _moeGateUpFusedKQ4KKernel{
              _moeGateUpFusedKQ4KModule.getFunction("moe_gate_up_fused_k_q4k")},
          _moeGateUpFusedKQ4KBatchedModule{loadCudaModule(ctx, "moe_gate_up_fused_k_q4k_batched")},
          _moeGateUpFusedKQ4KBatchedKernel{
              _moeGateUpFusedKQ4KBatchedModule.getFunction("moe_gate_up_fused_k_q4k_batched")},
          _moeGateUpFusedKNvblkBatchedModule{loadCudaModule(ctx, "moe_gate_up_fused_k_nvfp4blk_batched")},
          _moeGateUpFusedKNvblkBatchedKernel{
              _moeGateUpFusedKNvblkBatchedModule.getFunction("moe_gate_up_fused_k_nvfp4blk_batched")},
          _moeDownFusedKNvblkBatchedModule{loadCudaModule(ctx, "moe_down_fused_k_nvfp4blk_batched")},
          _moeDownFusedKNvblkBatchedKernel{
              _moeDownFusedKNvblkBatchedModule.getFunction("moe_down_fused_k_nvfp4blk_batched")},
          _moeGateUpFusedKTcModule{loadCudaModule(ctx, "moe_gate_up_fused_k_nvfp4tc_batched")},
          _moeGateUpFusedKTcKernel{
              _moeGateUpFusedKTcModule.getFunction("moe_gate_up_fused_k_nvfp4tc_batched")},
          _moeDownFusedKTcModule{loadCudaModule(ctx, "moe_down_fused_k_nvfp4tc_batched")},
          _moeDownFusedKTcKernel{
              _moeDownFusedKTcModule.getFunction("moe_down_fused_k_nvfp4tc_batched")},
          _moeGateUpFusedKBf16BatchedModule{loadCudaModule(ctx, "moe_gate_up_fused_k_bf16_batched")},
          _moeGateUpFusedKBf16BatchedKernel{
              _moeGateUpFusedKBf16BatchedModule.getFunction("moe_gate_up_fused_k_bf16_batched")},
          _moeDownFusedKBf16BatchedModule{loadCudaModule(ctx, "moe_down_fused_k_bf16_batched")},
          _moeDownFusedKBf16BatchedKernel{
              _moeDownFusedKBf16BatchedModule.getFunction("moe_down_fused_k_bf16_batched")},
          _ffnGateUpFusedQ8Module{loadCudaModule(ctx, "ffn_gate_up_fused_q8_0")},
          _ffnGateUpFusedQ8Kernel{
              _ffnGateUpFusedQ8Module.getFunction("ffn_gate_up_fused_q8_0")},
          _matmulQ5_0VecModule    {loadCudaModule(ctx, "matmul_q5_0_vec")},
          _matmulQ5_0VecKernel    {
              _matmulQ5_0VecModule.getFunction("matmul_q5_0_vec")},
          _matmulQ6KVecModule     {loadCudaModule(ctx, "matmul_q6k_vec")},
          _matmulQ6KVecKernel     {
              _matmulQ6KVecModule.getFunction("matmul_q6k_vec")},
          _matmulQ3KVecModule     {loadCudaModule(ctx, "matmul_q3k_vec")},
          _matmulQ3KVecKernel     {
              _matmulQ3KVecModule.getFunction("matmul_q3k_vec")},
          _matmulQ4KVecModule     {loadCudaModule(ctx, "matmul_q4k_vec")},
          _matmulQ4KVecKernel     {
              _matmulQ4KVecModule.getFunction("matmul_q4k_vec")},
          _matmulQ5KVecModule     {loadCudaModule(ctx, "matmul_q5k_vec")},
          _matmulQ5KVecKernel     {
              _matmulQ5KVecModule.getFunction("matmul_q5k_vec")},
          _matmulF32VecModule     {loadCudaModule(ctx, "matmul_f32_vec")},
          _matmulF32VecKernel     {
              _matmulF32VecModule.getFunction("matmul_f32_vec")},
          _matmulBf16VecModule    {loadCudaModule(ctx, "matmul_bf16_vec")},
          _matmulBf16VecKernel    {
              _matmulBf16VecModule.getFunction("matmul_bf16_vec")},
          _matmulBf16GemmModule   {loadCudaModule(ctx, "matmul_bf16_gemm")},
          _matmulBf16GemmKernel   {
              _matmulBf16GemmModule.getFunction("matmul_bf16_gemm")},
          _matmulBf16GemmTcModule {loadCudaModule(ctx, "matmul_bf16_gemm_tc")},
          _matmulBf16GemmTcKernel {
              _matmulBf16GemmTcModule.getFunction("matmul_bf16_gemm_tc")},
          _matmulBf16GemmTf32TcModule {
              loadCudaModule(ctx, "matmul_bf16_gemm_tf32_tc")},
          _matmulBf16GemmTf32TcKernel {
              _matmulBf16GemmTf32TcModule.getFunction("matmul_bf16_gemm_tf32_tc")},
          _matmulFp8VecModule     {loadCudaModule(ctx, "matmul_fp8_vec")},
          _matmulFp8VecKernel     {
              _matmulFp8VecModule.getFunction("matmul_fp8_vec")},
          _matmulFp8GemmModule    {loadCudaModule(ctx, "matmul_fp8_gemm")},
          _matmulFp8GemmKernel    {
              _matmulFp8GemmModule.getFunction("matmul_fp8_gemm")},
          _matmulNvblkVecModule   {loadCudaModule(ctx, "matmul_nvfp4blk_vec")},
          _matmulNvblkVecKernel   {
              _matmulNvblkVecModule.getFunction("matmul_nvfp4blk_vec")},
          _matmulNvblkGemmModule  {loadCudaModule(ctx, "matmul_nvfp4blk_gemm")},
          _matmulNvblkGemmKernel  {
              _matmulNvblkGemmModule.getFunction("matmul_nvfp4blk_gemm")},
          _nvblkDeintModule       {loadCudaModule(ctx, "matmul_nvfp4blk_deint_vec")},
          _nvblkDeinterleaveKernel{
              _nvblkDeintModule.getFunction("nvfp4blk_deinterleave")},
          _nvblkDeintVecKernel    {
              _nvblkDeintModule.getFunction("matmul_nvfp4blk_deint_vec")},
          _nvblkDeintRegVecKernel {
              _nvblkDeintModule.getFunction("matmul_nvfp4blk_deint_reg_vec")},
          _castToF32Module        {loadCudaModule(ctx, "cast_to_f32")},
          _castF32ToBf16Kernel    {
              _castToF32Module.getFunction("cast_f32_to_bf16")},
          _cublasFp8Module        {loadCudaModule(ctx, "cublas_fp8_quant")},
          _amaxF32Kernel          {_cublasFp8Module.getFunction("amax_f32")},
          _amaxBf16Kernel         {_cublasFp8Module.getFunction("amax_bf16")},
          _scaleFromAmaxKernel    {
              _cublasFp8Module.getFunction("scale_from_amax")},
          _castF32E4m3Kernel      {
              _cublasFp8Module.getFunction("cast_f32_to_e4m3")},
          _castBf16E4m3Kernel     {
              _cublasFp8Module.getFunction("cast_bf16_to_e4m3")}
    {}
};

GpuMatmul::GpuMatmul(::mimirmind::core::cuda::CudaComputeContext& ctx,
                     GpuOps& ops)
    : _ctx{ctx},
      _ops{ops},
      _pimpl{std::make_unique<Impl>(ctx.cudaContext())}
{
    // Value-aware so `env MIMIRMIND_MMQ=` (empty) reads as OFF — a bare
    // presence check turns an empty A/B-baseline var into an accidental ON.
    if (const char* mmq = std::getenv("MIMIRMIND_MMQ")) {
        _mmqEnabled = (mmq[0] != '\0' && !(mmq[0] == '0' && mmq[1] == '\0'));
        _mmqEnvSet  = true;   // env is explicit — Layer-2 profile must not override
    }
    if (const char* tc = std::getenv("MIMIRMIND_MMQ_TC")) {
        _mmqTc     = (tc[0] != '\0' && !(tc[0] == '0' && tc[1] == '\0'));
        _mmqEnvSet = true;    // env is explicit — Layer-2 profile must not override
    }
    if (const char* mx = std::getenv("MIMIRMIND_MMQ_MAX_N")) {
        const long v = std::strtol(mx, nullptr, 10);
        if (v > 0) _mmqMaxN = static_cast<std::size_t>(v);
    }
    if (const char* bt = std::getenv("MIMIRMIND_BF16_TC")) {
        _bf16Tc = (bt[0] != '\0' && !(bt[0] == '0' && bt[1] == '\0'));
    }
    if (const char* tt = std::getenv("MIMIRMIND_TF32_TC")) {
        _tf32Tc = (tt[0] != '\0' && !(tt[0] == '0' && tt[1] == '\0'));
    }
    if (const char* cb = std::getenv("MIMIRMIND_CUBLAS")) {
        _useCublas = (cb[0] != '\0' && !(cb[0] == '0' && cb[1] == '\0'));
    }
    if (_useCublas) {
        MM_LOG_INFO("hip::GpuMatmul",
                    "cuBLASLt dense BF16 matmul enabled (MIMIRMIND_CUBLAS=1) — "
                    "decode GEMV (M==1) and batched GEMM (M>1) route through "
                    "cublasLtMatmul; hand kernels remain the fallback");
    }
    if (const char* cf = std::getenv("MIMIRMIND_CUBLAS_FP8")) {
        _useCublasFp8 = (cf[0] != '\0' && !(cf[0] == '0' && cf[1] == '\0'));
    }
    if (const char* cfp = std::getenv("MIMIRMIND_CUBLAS_FP8_PREFILL")) {
        _useCublasFp8Prefill = (cfp[0] != '\0' && !(cfp[0] == '0' && cfp[1] == '\0'));
        _cublasFp8PrefillEnvSet = true;   // env is explicit — Layer-2 profile must not override
    }
    if (const char* f32t = std::getenv("MIMIRMIND_F32_TC_PREFILL")) {
        _useF32TcPrefill = (f32t[0] != '\0' && !(f32t[0] == '0' && f32t[1] == '\0'));
        _f32TcPrefillEnvSet = true;       // env is explicit — Layer-2 profile must not override
    }
    if (const char* dv = std::getenv("MIMIRMIND_NVFP4_DEINT")) {
        _useDeintVec = (dv[0] == '1' && dv[1] == '\0');
    }
    if (_useCublasFp8) {
        MM_LOG_INFO("hip::GpuMatmul",
                    "cuBLASLt per-tensor FP8 (E4M3) dense matmul enabled "
                    "(MIMIRMIND_CUBLAS_FP8=1) — BF16 weights quantised to E4M3 "
                    "(cached) + per-call X quant; hand kernel is the fallback");
    }
    if (_useF32TcPrefill) {
        MM_LOG_INFO("hip::GpuMatmul",
                    "F32 batched (M>1) prefill GEMM -> BF16 tensor cores enabled "
                    "(MIMIRMIND_F32_TC_PREFILL=1) — small F32 weights (MoE router, "
                    "GDN ssm_beta/alpha, shexp router) leave the per-row F32 vec "
                    "path; F32 vec is the fallback (encoder path unaffected)");
    }
    if (_useCublasFp8Prefill) {
        MM_LOG_INFO("hip::GpuMatmul",
                    "cuBLASLt FP8 batched (M>1) prefill GEMM enabled "
                    "(MIMIRMIND_CUBLAS_FP8_PREFILL=1) — dense prefill projections "
                    "run on FP8 tensor cores (RMSNorm inputs bound the per-tensor "
                    "activation scale); BF16/wmma is the fallback");
    }
    if (_tf32Tc) {
        MM_LOG_INFO("hip::GpuMatmul",
                    "E-FP4.5 TF32 tensor-core GEMM enabled for batched (M>1) "
                    "dense matmuls — fidelity variant (MIMIRMIND_TF32_TC=0 to "
                    "disable, falls back to BF16-TC)");
    } else if (_bf16Tc) {
        MM_LOG_INFO("hip::GpuMatmul",
                    "E-FP4.3 BF16 tensor-core GEMM enabled for batched (M>1) "
                    "dense matmuls (MIMIRMIND_BF16_TC=0 to disable)");
    }
    if (_mmqEnabled) {
        MM_LOG_INFO("hip::GpuMatmul",
                    "M-Cuda.MMQ enabled — Q8_0 prefill (M>1) uses int8 {} MMQ "
                    "for N<={} (lm_head/vocab stays fp32)",
                    _mmqTc ? "tensor-core" : "dp4a", _mmqMaxN);
    }
    MM_LOG_INFO("hip::GpuMatmul",
                "compute::cuda::GpuMatmul ready — 13 kernels loaded "
                "(Q8_0: vec / gemm / gemm_v2 / vec_dp4a / moe_down_fused_k; "
                "Q6_K: vec / vec_dp4a / moe_down_fused_k; Q5_0: vec; "
                "Q4_K: vec; Q5_K: vec; Q3_K: vec; F32: vec)");
}

GpuMatmul::~GpuMatmul() = default;

// ---- Real (non-stub) implementations --------------------------------

bool GpuMatmul::supports(::mimirmind::core::gguf::GgmlType type)
    const noexcept {
    // Only Q8_0 quantised weights are dispatched on the HIP side today.
    // Q4_K / Q5_K / Q6_K HIP kernels haven't been ported yet — the
    // dispatcher returns false so callers can fall back to CPU (or
    // the engine refuses the model at load time).
    return type == ::mimirmind::core::gguf::GgmlType::Q8_0;
}

bool GpuMatmul::dp4aAvailable() const noexcept {
    // The kernel is compiled in whenever this class exists (see the
    // ctor's Impl init list), so availability is unconditional on the
    // HIP side. gfx1101 has native `v_dot4_i32_iu8` — the kernel
    // uses that via the manual dot4 helper in matmul_q8_0_vec_dp4a.hip
    // (there is no pure signed-signed sdot4 on RDNA3).
    return true;
}

bool GpuMatmul::dp4aAvailable(::mimirmind::core::gguf::GgmlType type)
    const noexcept {
    return type == ::mimirmind::core::gguf::GgmlType::Q8_0
        || type == ::mimirmind::core::gguf::GgmlType::Q6_K;
}

bool GpuMatmul::moeDownFusedKAvailable() const noexcept {
    return true;   // Q8_0 variant is always loaded.
}

bool GpuMatmul::moeDownFusedKAvailable(::mimirmind::core::gguf::GgmlType type)
    const noexcept {
    return type == ::mimirmind::core::gguf::GgmlType::Q8_0
        || type == ::mimirmind::core::gguf::GgmlType::Q6_K
        || type == ::mimirmind::core::gguf::GgmlType::Q5_K
        || type == ::mimirmind::core::gguf::GgmlType::NVFP4_BLK
        || type == ::mimirmind::core::gguf::GgmlType::NVFP4_TC
        || type == ::mimirmind::core::gguf::GgmlType::BF16;
}

bool GpuMatmul::moeGateUpFusedKAvailable(::mimirmind::core::gguf::GgmlType type)
    const noexcept {
    return type == ::mimirmind::core::gguf::GgmlType::Q4_K
        || type == ::mimirmind::core::gguf::GgmlType::NVFP4_BLK
        || type == ::mimirmind::core::gguf::GgmlType::NVFP4_TC
        || type == ::mimirmind::core::gguf::GgmlType::BF16;
}

bool GpuMatmul::ffnGateUpFusedQ8Available() const noexcept {
    return true;
}

void GpuMatmul::sync() {
    _ctx.stream().synchronize();
}

std::vector<::mimirmind::compute::AutotuneReport>
GpuMatmul::autotuneReport() const {
    // Populated from the autotune-side state members. `gemmPicked` is
    // true whenever the crossover threshold is finite (i.e. GEMM won
    // at some bucket AND wasn't disabled). Timings are 0 pre-autotune
    // — SystemStatusBuilder renders those as "not measured" already.
    ::mimirmind::compute::AutotuneReport r{};
    r.name             = "Q8_0";
    r.gemmAvailable    = true;
    r.gemmPicked       = (_gemmMinM != kGemmMinMNever);
    // Legacy fields — the M=16 bucket, kept so existing consumers don't
    // break. Autotune fills these too (bucket 0 == M=16).
    r.vecMs            = _vecMsAtM[0];
    r.gemmMs           = _gemmMsAtM[0];
    r.mBuckets         = ::mimirmind::compute::kAutotuneMBuckets;
    r.vecMsAtM         = _vecMsAtM;
    r.gemmMsAtM        = _gemmMsAtM;
    // 0 in the report is rendered as null; kGemmMinMNever means "never
    // GEMM" (matvec-loop always wins). Neither is a real threshold.
    r.gemmMinM         = (_gemmMinM == kGemmMinMNever) ? 0 : _gemmMinM;
    r.dp4aAvailable    = true;
    r.dp4aPicked       = _useDp4a;
    r.dp4aMs           = _dp4aMs;
    r.gemmV2Available  = true;
    r.gemmV2Picked     = _useGemmV2 && (_gemmMinM != kGemmMinMNever);
    r.gemmV2MsAtM      = _gemmV2MsAtM;
    r.source           = _autotuneSource;
    return {r};
}

void GpuMatmul::autotune(
    ::mimirmind::core::cuda::CudaMemoryAllocator&       alloc,
    std::size_t                                       hiddenDim,
    const ::mimirmind::core::config::FeatureSettings& features) {

    using ::mimirmind::core::config::TriState;
    using ::mimirmind::compute::kAutotuneMBuckets;

    const bool forceDisable    = features.gemm == TriState::Disable;
    const bool forceEnable     = features.gemm == TriState::Force;
    const bool forceDisableDp4a = features.dp4a == TriState::Disable;
    const bool forceEnableDp4a  = features.dp4a == TriState::Force;
    const std::size_t envMinM  = features.gemmMinM.value_or(std::size_t{0});
    _useGemmV2                 = features.gemmV2;

    // features.gemmMinM — pin the crossover threshold and skip the bench.
    // Highest-priority override; features.gemm loses to it.
    if (envMinM > 0) {
        _gemmMinM       = envMinM;
        _autotuneSource = "cfg_gemm_min_m";
        MM_LOG_INFO("hip::GpuMatmul",
                    "autotune: features.gemmMinM={} — Q8_0 pinned, bench "
                    "skipped", envMinM);
        return;
    }
    if (forceDisable) {
        _gemmMinM       = kGemmMinMNever;
        _autotuneSource = "cfg_disable_gemm";
        MM_LOG_INFO("hip::GpuMatmul",
                    "autotune: features.gemm=disable — Q8_0 pinned to "
                    "matvec-loop");
        return;
    }
    if (forceEnable) {
        _gemmMinM       = 2;   // GEMM whenever M > 1
        _autotuneSource = "cfg_force_gemm";
        MM_LOG_INFO("hip::GpuMatmul",
                    "autotune: features.gemm=force — Q8_0 pinned to GEMM "
                    "(gemmMinM=2)");
        return;
    }
    if (forceEnableDp4a && !forceDisableDp4a) {
        _useDp4a        = true;
        _autotuneSource = "cfg_force_dp4a";
        MM_LOG_INFO("hip::GpuMatmul",
                    "autotune: features.dp4a=force — Q8_0 pinned to DP4A "
                    "path (matmulAsync currently requires pre-quantised "
                    "input; auto-from-float wiring lands in a follow-up)");
        return;
    }

    // ---- Bench-driven path -------------------------------------------

    // Round N=K to 256-aligned so any future Q4_K/Q6_K add-in shares the
    // same synthetic shape. Q8_0 only needs K % 32 == 0 today.
    const std::size_t K    = ((hiddenDim + 255) / 256) * 256;
    const std::size_t N    = K;
    const std::size_t Mmax = kAutotuneMBuckets.back();

    // Shared X / Y / W USM sized for the largest bucket. Q8_0 weight
    // size: (N * K/32) blocks × 34 bytes.
    constexpr std::size_t kBlockElts  = 32;
    constexpr std::size_t kBlockBytes = 34;
    const std::size_t nBlocksPerRow  = K / kBlockElts;
    const std::size_t wBytes = N * nBlocksPerRow * kBlockBytes;
    const std::size_t xBytes = Mmax * K * sizeof(float);
    const std::size_t yBytes = Mmax * N * sizeof(float);
    const std::size_t sBytes = K * sizeof(float);   // unused on the HIP path

    void* wDev = alloc.allocate(wBytes);
    void* xDev = alloc.allocate(xBytes);
    void* yDev = alloc.allocate(yBytes);
    void* sDev = alloc.allocate(sBytes);

    // Fill synthetic W (Q8_0 with tiny per-block scale) + X on host,
    // copy H2D. Deterministic seed matches the L0 bench so timings are
    // comparable across backends.
    {
        std::vector<std::uint8_t> wHost(wBytes);
        fillSyntheticQ8_0(wHost.data(), wBytes);
        alloc.copyH2D(wDev, wHost.data(), wBytes);
    }
    {
        std::vector<float> xHost(Mmax * K);
        std::mt19937 rng{0xB00BB00BU};
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto& v : xHost) v = dist(rng);
        alloc.copyH2D(xDev, xHost.data(), xBytes);
    }

    constexpr int nWarmup = 2;
    constexpr int nTimed  = 5;
    using clk = std::chrono::steady_clock;

    // Warmup at Mmax — primes both kernel dispatch paths + fills any
    // driver-side JIT / arg-buffer state that would otherwise skew the
    // first timed iteration.
    for (int i = 0; i < nWarmup; ++i) {
        _gemmMinM = kGemmMinMNever;   // force matvec-loop
        matmulAsync(::mimirmind::core::gguf::GgmlType::Q8_0,
                    wDev, N, K,
                    static_cast<const float*>(xDev), Mmax,
                    static_cast<float*>(yDev),
                    static_cast<float*>(sDev));
        sync();
        _gemmMinM = 2;                // force GEMM
        matmulAsync(::mimirmind::core::gguf::GgmlType::Q8_0,
                    wDev, N, K,
                    static_cast<const float*>(xDev), Mmax,
                    static_cast<float*>(yDev),
                    static_cast<float*>(sDev));
        sync();
    }

    // Per-M timing loop. Wall-clock median over `nTimed` iterations
    // per bucket, per kernel — same shape as the L0 bench so numbers
    // are directly comparable at report time.
    for (std::size_t bi = 0; bi < kAutotuneMBuckets.size(); ++bi) {
        const std::size_t Mb = kAutotuneMBuckets[bi];

        _gemmMinM = kGemmMinMNever;
        std::vector<double> vecMs;
        vecMs.reserve(nTimed);
        for (int it = 0; it < nTimed; ++it) {
            const auto t0 = clk::now();
            matmulAsync(::mimirmind::core::gguf::GgmlType::Q8_0,
                        wDev, N, K,
                        static_cast<const float*>(xDev), Mb,
                        static_cast<float*>(yDev),
                        static_cast<float*>(sDev));
            sync();
            const auto t1 = clk::now();
            vecMs.push_back(
                std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        _vecMsAtM[bi] = medianMs(std::move(vecMs));

        _gemmMinM = 2;
        std::vector<double> gemmMs;
        gemmMs.reserve(nTimed);
        for (int it = 0; it < nTimed; ++it) {
            const auto t0 = clk::now();
            matmulAsync(::mimirmind::core::gguf::GgmlType::Q8_0,
                        wDev, N, K,
                        static_cast<const float*>(xDev), Mb,
                        static_cast<float*>(yDev),
                        static_cast<float*>(sDev));
            sync();
            const auto t1 = clk::now();
            gemmMs.push_back(
                std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        _gemmMsAtM[bi] = medianMs(std::move(gemmMs));
    }

    // Derive gemmMinM: smallest bucket-M where gemm × 1.05 < vec. 5 %
    // margin is the noise floor between iGPU runs; below that we stick
    // with matvec-loop as the conservative default. M=1 (decode) never
    // triggers GEMM because bucket 0 starts at M=16.
    _gemmMinM       = kGemmMinMNever;
    _autotuneSource = "bench";
    for (std::size_t bi = 0; bi < kAutotuneMBuckets.size(); ++bi) {
        if (_gemmMsAtM[bi] > 0.0 &&
            _vecMsAtM[bi] > 0.0 &&
            _gemmMsAtM[bi] * 1.05 < _vecMsAtM[bi]) {
            _gemmMinM = kAutotuneMBuckets[bi];
            break;
        }
    }

    // Release bench scratch.
    alloc.deallocate(sDev, sBytes, ::mimirmind::core::cuda::CudaAllocKind::Device);
    alloc.deallocate(yDev, yBytes, ::mimirmind::core::cuda::CudaAllocKind::Device);
    alloc.deallocate(xDev, xBytes, ::mimirmind::core::cuda::CudaAllocKind::Device);
    alloc.deallocate(wDev, wBytes, ::mimirmind::core::cuda::CudaAllocKind::Device);

    std::ostringstream summary;
    summary << "autotune Q8_0: gemmMinM=";
    if (_gemmMinM == kGemmMinMNever) {
        summary << "NEVER";
    } else {
        summary << _gemmMinM;
    }
    for (std::size_t bi = 0; bi < kAutotuneMBuckets.size(); ++bi) {
        summary << " | M=" << kAutotuneMBuckets[bi]
                << " vec=" << _vecMsAtM[bi] << "ms"
                << " gemm=" << _gemmMsAtM[bi] << "ms";
    }
    MM_LOG_INFO("hip::GpuMatmul", "{}", summary.str());
}

// ---- Stubbed matmul-launch overrides --------------------------------

void GpuMatmul::matmul(::mimirmind::core::gguf::GgmlType type,
                       const void*     W,
                       std::size_t     N,
                       std::size_t     K,
                       const float*    X,
                       std::size_t     M,
                       float*          Y,
                       float*          scratch) {
    matmulAsync(type, W, N, K, X, M, Y, scratch);
    sync();
}

bool GpuMatmul::cublasBf16Matmul(const void*  W,
                                 std::size_t  N,
                                 std::size_t  K,
                                 const float* X,
                                 std::size_t  M,
                                 float*       Y) {
    // Lazily create the cuBLASLt handle + a fixed workspace on first use.
    if (_pimpl->_ltHandle == nullptr) {
        if (cublasLtCreate(&_pimpl->_ltHandle) != CUBLAS_STATUS_SUCCESS) {
            _pimpl->_ltHandle = nullptr;
            return false;
        }
        _pimpl->_ltWorkspaceBytes = std::size_t{32} * 1024 * 1024;   // 32 MiB
        if (cudaMalloc(&_pimpl->_ltWorkspace, _pimpl->_ltWorkspaceBytes)
                != cudaSuccess) {
            _pimpl->_ltWorkspace      = nullptr;
            _pimpl->_ltWorkspaceBytes = 0;   // matmul still runs, no scratch
        }
    }

    // Stage the F32 activation X[M*K] -> BF16 (grow the buffer on demand).
    const std::size_t elems    = M * K;
    const std::size_t needBytes = elems * sizeof(std::uint16_t);   // BF16 = 2B
    if (needBytes > _pimpl->_xBf16Bytes) {
        if (_pimpl->_xBf16 != nullptr) {
            cudaFree(_pimpl->_xBf16);
            _pimpl->_xBf16 = nullptr;
        }
        if (cudaMalloc(&_pimpl->_xBf16, needBytes) != cudaSuccess) {
            _pimpl->_xBf16      = nullptr;
            _pimpl->_xBf16Bytes = 0;
            return false;
        }
        _pimpl->_xBf16Bytes = needBytes;
    }
    {
        auto& kern = _pimpl->_castF32ToBf16Kernel;
        constexpr std::uint32_t kCastLocal = 256;
        const std::uint32_t grid = static_cast<std::uint32_t>(
            (elems + kCastLocal - 1) / kCastLocal);
        kern.setPtr  (0, X);
        kern.setPtr  (1, _pimpl->_xBf16);
        kern.setValue(2, static_cast<std::int64_t>(elems));
        kern.launch(_ctx.stream(), grid, 1, 1, kCastLocal, 1, 1);
    }

    // Column-major cuBLASLt. The logical (row-major) op is
    //   Y[m,n] = sum_k X[m,k] * W[n,k]   (Y = X @ W^T).
    // Row-major W[N,K] read as column-major is a [K,N] matrix (Aw[k,n]=W[n,k]),
    // ld=K; row-major X[M,K] read column-major is [K,M] (ld=K); Y as row-major
    // [M,N] is column-major [N,M], ld=N. So D[N,M] = op_T(Aw[K,N]) @ Bx[K,M].
    const int nI = static_cast<int>(N);
    const int mI = static_cast<int>(M);
    const int kI = static_cast<int>(K);

    cublasLtMatmulDesc_t   op = nullptr;
    cublasLtMatrixLayout_t aL = nullptr, bL = nullptr, cL = nullptr;
    const cublasOperation_t opT = CUBLAS_OP_T;
    const cublasOperation_t opN = CUBLAS_OP_N;

    bool ok =
        cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F)
            == CUBLAS_STATUS_SUCCESS
        && cublasLtMatmulDescSetAttribute(
               op, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT))
            == CUBLAS_STATUS_SUCCESS
        && cublasLtMatmulDescSetAttribute(
               op, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN))
            == CUBLAS_STATUS_SUCCESS
        && cublasLtMatrixLayoutCreate(&aL, CUDA_R_16BF, kI, nI, kI)
            == CUBLAS_STATUS_SUCCESS   // A = W  [K,N] col-major, op=T
        && cublasLtMatrixLayoutCreate(&bL, CUDA_R_16BF, kI, mI, kI)
            == CUBLAS_STATUS_SUCCESS   // B = Xbf16 [K,M] col-major, op=N
        && cublasLtMatrixLayoutCreate(&cL, CUDA_R_32F, nI, mI, nI)
            == CUBLAS_STATUS_SUCCESS;  // D = Y  [N,M] col-major

    if (ok) {
        const float alpha = 1.0f;
        const float beta  = 0.0f;
        const cublasStatus_t st = cublasLtMatmul(
            _pimpl->_ltHandle, op,
            &alpha,
            W,              aL,
            _pimpl->_xBf16, bL,
            &beta,
            Y,              cL,
            Y,              cL,
            nullptr,        // let cuBLASLt heuristically pick the algo
            _pimpl->_ltWorkspace, _pimpl->_ltWorkspaceBytes,
            _ctx.stream().handle());
        ok = (st == CUBLAS_STATUS_SUCCESS);
    }

    if (cL != nullptr) { cublasLtMatrixLayoutDestroy(cL); }
    if (bL != nullptr) { cublasLtMatrixLayoutDestroy(bL); }
    if (aL != nullptr) { cublasLtMatrixLayoutDestroy(aL); }
    if (op != nullptr) { cublasLtMatmulDescDestroy(op); }
    return ok;
}

void GpuMatmul::nvblkDeintVec(const void* W, std::size_t N, std::size_t K,
                             const float* X, float* Y) {
    const std::size_t totalSupers = N * (K / 32);
    auto it = _pimpl->_nvblkDeintCache.find(W);
    if (it == _pimpl->_nvblkDeintCache.end()) {
        Impl::NvblkDeint c;
        if (cudaMalloc(&c.nib, totalSupers * 16) != cudaSuccess) {
            return;
        }
        if (cudaMalloc(&c.scale, totalSupers * 4) != cudaSuccess) {
            cudaFree(c.nib);
            return;
        }
        auto& dk = _pimpl->_nvblkDeinterleaveKernel;
        dk.setPtr  (0, W);
        dk.setPtr  (1, c.nib);
        dk.setPtr  (2, c.scale);
        dk.setValue(3, static_cast<std::int64_t>(totalSupers));
        const std::uint32_t g =
            static_cast<std::uint32_t>((totalSupers + 255) / 256);
        dk.launch(_ctx.stream(), g, 1, 1, 256, 1, 1);
        it = _pimpl->_nvblkDeintCache.emplace(W, c).first;
    }
    // Inc 5 A/B: MIMIRMIND_DENSE_DEINT_REG=1 selects the register-x variant
    // (no shared memory: activation coalesced into registers, uint4 broadcast),
    // removing the smem-x MIO stall that made the v1 dense deint neutral.
    static const bool useReg = []() {
        const char* r = std::getenv("MIMIRMIND_DENSE_DEINT_REG");
        return r != nullptr && r[0] == '1' && r[1] == '\0';
    }();
    auto& k = useReg ? _pimpl->_nvblkDeintRegVecKernel
                     : _pimpl->_nvblkDeintVecKernel;
    k.setPtr  (0, X);
    k.setPtr  (1, it->second.nib);
    k.setPtr  (2, it->second.scale);
    k.setPtr  (3, Y);
    k.setValue(4, static_cast<std::int32_t>(N));
    k.setValue(5, static_cast<std::int32_t>(K));
    const std::uint32_t nGroups = static_cast<std::uint32_t>((N + 3) / 4);
    // Register variant uses no shared memory; the v1 variant stages x (K floats).
    const std::size_t smemBytes = useReg ? 0 : K * sizeof(float);
    k.launch(_ctx.stream(), nGroups, 1, 1, 128, 1, 1, smemBytes);
}

bool GpuMatmul::cublasFp8Matmul(const void*  W,
                                std::size_t  N,
                                std::size_t  K,
                                const float* X,
                                std::size_t  M,
                                float*       Y) {
    if (_pimpl->_ltHandle == nullptr) {
        if (cublasLtCreate(&_pimpl->_ltHandle) != CUBLAS_STATUS_SUCCESS) {
            _pimpl->_ltHandle = nullptr;
            return false;
        }
        _pimpl->_ltWorkspaceBytes = std::size_t{32} * 1024 * 1024;
        if (cudaMalloc(&_pimpl->_ltWorkspace, _pimpl->_ltWorkspaceBytes)
                != cudaSuccess) {
            _pimpl->_ltWorkspace      = nullptr;
            _pimpl->_ltWorkspaceBytes = 0;
        }
    }
    if (_pimpl->_xScaleDev == nullptr
            && cudaMalloc(&_pimpl->_xScaleDev, 2 * sizeof(float)) != cudaSuccess) {
        _pimpl->_xScaleDev = nullptr;
        return false;
    }
    if (_pimpl->_amaxDev == nullptr) {
        if (cudaMalloc(&_pimpl->_amaxDev, sizeof(float)) != cudaSuccess) {
            _pimpl->_amaxDev = nullptr;
            return false;
        }
        // Zero ONCE — scale_from_amax consume-and-resets it after every use,
        // so the per-call cudaMemsetAsync launches are gone (5.18.10.4).
        cudaMemsetAsync(_pimpl->_amaxDev, 0, sizeof(float),
                        _ctx.stream().handle());
    }

    cudaStream_t stream = _ctx.stream().handle();
    constexpr std::uint32_t kLocal = 256;
    auto gridOf = [](std::size_t n) {
        return static_cast<std::uint32_t>((n + kLocal - 1) / kLocal);
    };

    // ---- W: quantise BF16 -> per-tensor E4M3 once, cache by pointer ----------
    auto wit = _pimpl->_fp8WeightCache.find(W);
    if (wit == _pimpl->_fp8WeightCache.end()) {
        const std::size_t wElems = N * K;
        void* wData  = nullptr;
        void* wScale = nullptr;
        if (cudaMalloc(&wData, wElems) != cudaSuccess) {
            return false;
        }
        if (cudaMalloc(&wScale, 2 * sizeof(float)) != cudaSuccess) {
            cudaFree(wData);
            return false;
        }
        void* wInv = static_cast<char*>(wScale) + sizeof(float);
        _pimpl->_amaxBf16Kernel.setPtr  (0, W);
        _pimpl->_amaxBf16Kernel.setPtr  (1, _pimpl->_amaxDev);
        _pimpl->_amaxBf16Kernel.setValue(2, static_cast<std::int64_t>(wElems));
        _pimpl->_amaxBf16Kernel.launch(_ctx.stream(), gridOf(wElems), 1, 1,
                                       kLocal, 1, 1);
        _pimpl->_scaleFromAmaxKernel.setPtr(0, _pimpl->_amaxDev);
        _pimpl->_scaleFromAmaxKernel.setPtr(1, wScale);
        _pimpl->_scaleFromAmaxKernel.setPtr(2, wInv);
        _pimpl->_scaleFromAmaxKernel.launch(_ctx.stream(), 1, 1, 1, 1, 1, 1);
        _pimpl->_castBf16E4m3Kernel.setPtr  (0, W);
        _pimpl->_castBf16E4m3Kernel.setPtr  (1, wData);
        _pimpl->_castBf16E4m3Kernel.setPtr  (2, wInv);
        _pimpl->_castBf16E4m3Kernel.setValue(3, static_cast<std::int64_t>(wElems));
        _pimpl->_castBf16E4m3Kernel.launch(_ctx.stream(), gridOf(wElems), 1, 1,
                                           kLocal, 1, 1);
        wit = _pimpl->_fp8WeightCache.emplace(
            W, Impl::Fp8W{wData, wScale}).first;
    }
    void* wData  = wit->second.data;
    void* wScale = wit->second.scale;

    // ---- X: quantise F32 -> per-tensor E4M3 per call -------------------------
    const std::size_t xElems = M * K;
    if (xElems > _pimpl->_xFp8Bytes) {
        if (_pimpl->_xFp8 != nullptr) {
            cudaFree(_pimpl->_xFp8);
            _pimpl->_xFp8 = nullptr;
        }
        if (cudaMalloc(&_pimpl->_xFp8, xElems) != cudaSuccess) {
            _pimpl->_xFp8Bytes = 0;
            return false;
        }
        _pimpl->_xFp8Bytes = xElems;
    }
    void* xInv = static_cast<char*>(_pimpl->_xScaleDev) + sizeof(float);
    _pimpl->_amaxF32Kernel.setPtr  (0, X);
    _pimpl->_amaxF32Kernel.setPtr  (1, _pimpl->_amaxDev);
    _pimpl->_amaxF32Kernel.setValue(2, static_cast<std::int64_t>(xElems));
    _pimpl->_amaxF32Kernel.launch(_ctx.stream(), gridOf(xElems), 1, 1,
                                  kLocal, 1, 1);
    _pimpl->_scaleFromAmaxKernel.setPtr(0, _pimpl->_amaxDev);
    _pimpl->_scaleFromAmaxKernel.setPtr(1, _pimpl->_xScaleDev);
    _pimpl->_scaleFromAmaxKernel.setPtr(2, xInv);
    _pimpl->_scaleFromAmaxKernel.launch(_ctx.stream(), 1, 1, 1, 1, 1, 1);
    _pimpl->_castF32E4m3Kernel.setPtr  (0, X);
    _pimpl->_castF32E4m3Kernel.setPtr  (1, _pimpl->_xFp8);
    _pimpl->_castF32E4m3Kernel.setPtr  (2, xInv);
    _pimpl->_castF32E4m3Kernel.setValue(3, static_cast<std::int64_t>(xElems));
    _pimpl->_castF32E4m3Kernel.launch(_ctx.stream(), gridOf(xElems), 1, 1,
                                      kLocal, 1, 1);

    // ---- cuBLASLt per-tensor FP8 matmul (same col-major layout as BF16) ------
    // 5.18.10.4: plan (desc + layouts) cached by (N, M, K) — see Impl::LtPlan.
    // Only the per-weight A scale pointer changes between calls.
    const int nI = static_cast<int>(N);
    const int mI = static_cast<int>(M);
    const int kI = static_cast<int>(K);
    const std::uint64_t planKey = (static_cast<std::uint64_t>(N & 0xFFFFFu) << 40)
                                | (static_cast<std::uint64_t>(M & 0xFFFFFu) << 20)
                                |  static_cast<std::uint64_t>(K & 0xFFFFFu);
    auto pit = _pimpl->_ltPlanCache.find(planKey);
    if (pit == _pimpl->_ltPlanCache.end()) {
        Impl::LtPlan p{};
        const cublasOperation_t opT = CUBLAS_OP_T;
        const cublasOperation_t opN = CUBLAS_OP_N;
        const std::int32_t scaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
        const bool built =
            cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F)
                == CUBLAS_STATUS_SUCCESS
            && cublasLtMatmulDescSetAttribute(
                   p.op, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT))
                == CUBLAS_STATUS_SUCCESS
            && cublasLtMatmulDescSetAttribute(
                   p.op, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN))
                == CUBLAS_STATUS_SUCCESS
            && cublasLtMatmulDescSetAttribute(
                   p.op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleMode,
                   sizeof(scaleMode))
                == CUBLAS_STATUS_SUCCESS
            && cublasLtMatmulDescSetAttribute(
                   p.op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaleMode,
                   sizeof(scaleMode))
                == CUBLAS_STATUS_SUCCESS
            && cublasLtMatmulDescSetAttribute(
                   p.op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                   &_pimpl->_xScaleDev, sizeof(void*))
                == CUBLAS_STATUS_SUCCESS
            && cublasLtMatrixLayoutCreate(&p.aL, CUDA_R_8F_E4M3, kI, nI, kI)
                == CUBLAS_STATUS_SUCCESS
            && cublasLtMatrixLayoutCreate(&p.bL, CUDA_R_8F_E4M3, kI, mI, kI)
                == CUBLAS_STATUS_SUCCESS
            && cublasLtMatrixLayoutCreate(&p.cL, CUDA_R_32F, nI, mI, nI)
                == CUBLAS_STATUS_SUCCESS;
        if (!built) {
            if (p.cL != nullptr) { cublasLtMatrixLayoutDestroy(p.cL); }
            if (p.bL != nullptr) { cublasLtMatrixLayoutDestroy(p.bL); }
            if (p.aL != nullptr) { cublasLtMatrixLayoutDestroy(p.aL); }
            if (p.op != nullptr) { cublasLtMatmulDescDestroy(p.op); }
            return false;
        }
        pit = _pimpl->_ltPlanCache.emplace(planKey, p).first;
    }
    Impl::LtPlan& plan = pit->second;
    if (cublasLtMatmulDescSetAttribute(
            plan.op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &wScale,
            sizeof(wScale)) != CUBLAS_STATUS_SUCCESS) {
        return false;
    }

    const float alpha = 1.0f;
    const float beta  = 0.0f;
    const cublasStatus_t st = cublasLtMatmul(
        _pimpl->_ltHandle, plan.op,
        &alpha,
        wData,         plan.aL,
        _pimpl->_xFp8, plan.bL,
        &beta,
        Y,             plan.cL,
        Y,             plan.cL,
        nullptr,
        _pimpl->_ltWorkspace, _pimpl->_ltWorkspaceBytes,
        stream);
    return st == CUBLAS_STATUS_SUCCESS;
}

void GpuMatmul::matmulAsync(::mimirmind::core::gguf::GgmlType type,
                            const void*     W,
                            std::size_t     N,
                            std::size_t     K,
                            const float*    X,
                            std::size_t     M,
                            float*          Y,
                            float*          scratch) {
    // The caller-provided `scratch` is a device pointer for HIP and
    // therefore not host-writable. The CPU fallback below allocates
    // its own host scratch; on the Q8_0 GPU path scratch is unused
    // entirely (Q8_0 kernels don't need dequant workspace).
    (void)scratch;

    if (M == 0 || N == 0 || K == 0) {
        return;
    }

    if (type == ::mimirmind::core::gguf::GgmlType::Q5_0) {
        // Native Q5_0 vec kernel — same launch geometry as the Q8_0
        // vec path (128 threads = 4 warps × 32 lanes, one warp per
        // output row, MATMUL_Q5_0_OUTPUTS_PER_GROUP = 4). Q5_0 blocks
        // are 32 elements / 22 bytes with fp16 scale + u32 high-bits
        // + 16 packed nibbles; the kernel handles the bit-extraction
        // per lane. Per-row matvec loop over M like Q8_0's fallback.
        constexpr std::size_t kBlockElts = 32;
        if (K % kBlockElts != 0) {
            throw std::runtime_error(
                "compute::cuda::GpuMatmul::matmulAsync: K=" +
                std::to_string(K) + " is not a multiple of Q5_0 "
                "blockElements=" + std::to_string(kBlockElts));
        }
        const std::uint32_t nGroups = static_cast<std::uint32_t>(
            (N + kOutputsPerGroup - 1) / kOutputsPerGroup);

        auto& kern = _pimpl->_matmulQ5_0VecKernel;
        for (std::size_t m = 0; m < M; ++m) {
            const float* xRow = X + m * K;
            float*       yRow = Y + m * N;

            kern.setPtr  (0, xRow);
            kern.setPtr  (1, W);
            kern.setPtr  (2, yRow);
            kern.setValue(3, static_cast<std::int32_t>(K));
            kern.setValue(4, static_cast<std::int32_t>(N));

            kern.launch(_ctx.stream(),
                        nGroups, 1, 1,
                        kVecLocalSize, 1, 1);
        }
        return;
    }

    if (type == ::mimirmind::core::gguf::GgmlType::Q6_K) {
        // Native Q6_K vec kernel — Q5_0-shape launch (128 threads =
        // 4 warps × 32 lanes, MATMUL_Q6K_OUTPUTS_PER_GROUP = 4), but
        // block is 256 elements / 210 bytes with ql/qh/sc/d fields.
        // K must be a multiple of the super-block size.
        constexpr std::size_t kBlockElts = 256;
        if (K % kBlockElts != 0) {
            throw std::runtime_error(
                "compute::cuda::GpuMatmul::matmulAsync: K=" +
                std::to_string(K) + " is not a multiple of Q6_K "
                "blockElements=" + std::to_string(kBlockElts));
        }
        const std::uint32_t nGroups = static_cast<std::uint32_t>(
            (N + kOutputsPerGroup - 1) / kOutputsPerGroup);

        auto& kern = _pimpl->_matmulQ6KVecKernel;
        for (std::size_t m = 0; m < M; ++m) {
            const float* xRow = X + m * K;
            float*       yRow = Y + m * N;

            kern.setPtr  (0, xRow);
            kern.setPtr  (1, W);
            kern.setPtr  (2, yRow);
            kern.setValue(3, static_cast<std::int32_t>(K));
            kern.setValue(4, static_cast<std::int32_t>(N));

            kern.launch(_ctx.stream(),
                        nGroups, 1, 1,
                        kVecLocalSize, 1, 1);
        }
        return;
    }

    if (type == ::mimirmind::core::gguf::GgmlType::Q3_K) {
        // Native Q3_K vec kernel — same launch shape as Q6_K but block
        // is 256 elements / 110 bytes with hmask[32] / qs[64] /
        // scales[12] (packed 16 x 6-bit) / fp16 d. Signed 3-bit quant:
        // value = d * (scale - 32) * (low_2 - (high_1 ? 0 : 4)).
        // K must be a multiple of the super-block size.
        constexpr std::size_t kBlockElts = 256;
        if (K % kBlockElts != 0) {
            throw std::runtime_error(
                "compute::cuda::GpuMatmul::matmulAsync: K=" +
                std::to_string(K) + " is not a multiple of Q3_K "
                "blockElements=" + std::to_string(kBlockElts));
        }
        const std::uint32_t nGroups = static_cast<std::uint32_t>(
            (N + kOutputsPerGroup - 1) / kOutputsPerGroup);

        auto& kern = _pimpl->_matmulQ3KVecKernel;
        for (std::size_t m = 0; m < M; ++m) {
            const float* xRow = X + m * K;
            float*       yRow = Y + m * N;

            kern.setPtr  (0, xRow);
            kern.setPtr  (1, W);
            kern.setPtr  (2, yRow);
            kern.setValue(3, static_cast<std::int32_t>(K));
            kern.setValue(4, static_cast<std::int32_t>(N));

            kern.launch(_ctx.stream(),
                        nGroups, 1, 1,
                        kVecLocalSize, 1, 1);
        }
        return;
    }

    if (type == ::mimirmind::core::gguf::GgmlType::Q4_K) {
        // Native Q4_K vec kernel — same launch as Q6_K but block is
        // 256 elements / 144 bytes with d/dmin/scales[12]/qs[128]
        // layout. Asymmetric quant: value = d*scale*nibble - dmin*min
        // per sub-block. K must be a multiple of the super-block size.
        constexpr std::size_t kBlockElts = 256;
        if (K % kBlockElts != 0) {
            throw std::runtime_error(
                "compute::cuda::GpuMatmul::matmulAsync: K=" +
                std::to_string(K) + " is not a multiple of Q4_K "
                "blockElements=" + std::to_string(kBlockElts));
        }
        const std::uint32_t nGroups = static_cast<std::uint32_t>(
            (N + kOutputsPerGroup - 1) / kOutputsPerGroup);

        auto& kern = _pimpl->_matmulQ4KVecKernel;
        for (std::size_t m = 0; m < M; ++m) {
            const float* xRow = X + m * K;
            float*       yRow = Y + m * N;

            kern.setPtr  (0, xRow);
            kern.setPtr  (1, W);
            kern.setPtr  (2, yRow);
            kern.setValue(3, static_cast<std::int32_t>(K));
            kern.setValue(4, static_cast<std::int32_t>(N));

            kern.launch(_ctx.stream(),
                        nGroups, 1, 1,
                        kVecLocalSize, 1, 1);
        }
        return;
    }

    if (type == ::mimirmind::core::gguf::GgmlType::F32) {
        // Batched (M>1) F32 GEMM: the per-row vec path below launches M kernels
        // with no weight reuse or tensor cores, so a small-N F32 weight (the MoE
        // router ffn_gate_inp: N=nExperts, K=d_model) costs ~100x more than it
        // should at prefill M. Cast the weight to BF16 once (cached by pointer;
        // loaded weights are stable) and run it through the TF32/BF16 tensor-core
        // GEMM — the same kernel the BF16 dense projections use, F32 activations
        // staged in-kernel. Default on with _tf32Tc/_bf16Tc (E-FP4.3/.5).
        if (M > 1 && _useF32TcPrefill && _f32TcAllowed && (_tf32Tc || _bf16Tc)) {
            auto cit = _pimpl->_bf16FromF32Cache.find(W);
            if (cit == _pimpl->_bf16FromF32Cache.end()) {
                const std::size_t wElems = N * K;
                void* wBf16 = nullptr;
                if (cudaMalloc(&wBf16, wElems * sizeof(std::uint16_t))
                        == cudaSuccess) {
                    _pimpl->_castF32ToBf16Kernel.setPtr  (0, W);
                    _pimpl->_castF32ToBf16Kernel.setPtr  (1, wBf16);
                    _pimpl->_castF32ToBf16Kernel.setValue(2,
                        static_cast<std::int64_t>(wElems));
                    _pimpl->_castF32ToBf16Kernel.launch(
                        _ctx.stream(),
                        static_cast<std::uint32_t>((wElems + 255) / 256), 1, 1,
                        256, 1, 1);
                    cit = _pimpl->_bf16FromF32Cache.emplace(W, wBf16).first;
                }
            }
            if (cit != _pimpl->_bf16FromF32Cache.end()) {
                auto& tk = _tf32Tc ? _pimpl->_matmulBf16GemmTf32TcKernel
                                   : _pimpl->_matmulBf16GemmTcKernel;
                tk.setPtr  (0, X);
                tk.setPtr  (1, cit->second);
                tk.setPtr  (2, Y);
                tk.setValue(3, static_cast<std::int32_t>(K));
                tk.setValue(4, static_cast<std::int32_t>(N));
                tk.setValue(5, static_cast<std::int32_t>(M));
                const std::uint32_t gx = static_cast<std::uint32_t>((N + 15) / 16);
                const std::uint32_t gy = static_cast<std::uint32_t>((M + 15) / 16);
                tk.launch(_ctx.stream(), gx, gy, 1, 32, 1, 1);
                return;
            }
        }

        // Native F32 vec kernel — no dequant, straight fp32 dot. Same
        // launch shape as the K-quant vec kernels. Motivated by Gemma 4
        // E4B: inp_gate.weight + proj.weight per layer are F32 (~2.6 MiB
        // each, 84 dispatches per decode-step). Native F32 closes the
        // last non-K-quant CPU-fallback for that model.
        const std::uint32_t nGroups = static_cast<std::uint32_t>(
            (N + kOutputsPerGroup - 1) / kOutputsPerGroup);

        auto& kern = _pimpl->_matmulF32VecKernel;
        for (std::size_t m = 0; m < M; ++m) {
            const float* xRow = X + m * K;
            float*       yRow = Y + m * N;

            kern.setPtr  (0, xRow);
            kern.setPtr  (1, W);
            kern.setPtr  (2, yRow);
            kern.setValue(3, static_cast<std::int32_t>(K));
            kern.setValue(4, static_cast<std::int32_t>(N));

            kern.launch(_ctx.stream(),
                        nGroups, 1, 1,
                        kVecLocalSize, 1, 1);
        }
        return;
    }

    if (type == ::mimirmind::core::gguf::GgmlType::BF16) {
        // Native BF16 matmul — reads 2-byte BF16 weights and widens each to
        // fp32 in the FMA. The NVFP4 checkpoints materialise their
        // (dequantised) weights to BF16, notably every MoE expert bank;
        // without this the whole qwen35moe decode fell through to the host
        // CPU-fallback below (~seconds per token).
        const std::uint32_t nGroups = static_cast<std::uint32_t>(
            (N + kOutputsPerGroup - 1) / kOutputsPerGroup);

        // Opt-in cuBLASLt paths (decode GEMV + batched GEMM). FP8 first (half
        // the weight bytes), then BF16. Both fall through to the hand kernels
        // below if disabled or the shape is unsupported.
        // FP8 is gated to M==1 (decode): the per-tensor E4M3 activation quant is
        // validated for the single-token GEMV, but a large prefill activation
        // (M>1) can hold outliers that make the amax/448 scale crush precision,
        // so prefill stays on the higher-fidelity BF16 path below.
        if (_useCublasFp8 && M == 1 && cublasFp8Matmul(W, N, K, X, M, Y)) {
            return;
        }
        // Batched prefill (M>1) on FP8 tensor cores — opt-in, takes precedence
        // over the BF16 path. Dense prefill projections read RMSNorm outputs, so
        // the per-tensor activation scale stays well-conditioned.
        if (_useCublasFp8Prefill && M > 1 && cublasFp8Matmul(W, N, K, X, M, Y)) {
            return;
        }
        if (_useCublas && cublasBf16Matmul(W, N, K, X, M, Y)) {
            return;
        }

        if (M == 1) {
            // Single-token decode: GEMV, one weight read per output.
            auto& kern = _pimpl->_matmulBf16VecKernel;
            kern.setPtr  (0, X);
            kern.setPtr  (1, W);
            kern.setPtr  (2, Y);
            kern.setValue(3, static_cast<std::int32_t>(K));
            kern.setValue(4, static_cast<std::int32_t>(N));
            kern.launch(_ctx.stream(), nGroups, 1, 1, kVecLocalSize, 1, 1);
            return;
        }

        // Batched (M > 1): GEMM that reads each weight row ONCE and reuses
        // it across all M activation rows — the serving-throughput path.
        // M is chunked to the kernel's fixed accumulator bound (GEMM_MAX_M);
        // weights are re-read once per chunk, not once per row.
        // E-FP4.3: BF16 tensor-core GEMM (wmma). One launch tiles M and N by
        // 16; activations are rounded F32->BF16 on stage (not bit-exact vs the
        // scalar path). Default on (MIMIRMIND_BF16_TC=0 disables).
        // E-FP4.5: TF32 fidelity variant — same tile/launch, activations round
        // to 10-bit (TF32) instead of 7-bit (BF16). Takes precedence over the
        // BF16-TC kernel when MIMIRMIND_TF32_TC=1.
        if (_tf32Tc || _bf16Tc) {
            auto& tk = _tf32Tc ? _pimpl->_matmulBf16GemmTf32TcKernel
                               : _pimpl->_matmulBf16GemmTcKernel;
            tk.setPtr  (0, X);
            tk.setPtr  (1, W);
            tk.setPtr  (2, Y);
            tk.setValue(3, static_cast<std::int32_t>(K));
            tk.setValue(4, static_cast<std::int32_t>(N));
            tk.setValue(5, static_cast<std::int32_t>(M));
            const std::uint32_t gx = static_cast<std::uint32_t>((N + 15) / 16);
            const std::uint32_t gy = static_cast<std::uint32_t>((M + 15) / 16);
            tk.launch(_ctx.stream(), gx, gy, 1, 32, 1, 1);
            return;
        }

        constexpr std::size_t kGemmMaxM = 16;   // == GEMM_MAX_M in the .cu
        auto& kern = _pimpl->_matmulBf16GemmKernel;
        for (std::size_t m0 = 0; m0 < M; m0 += kGemmMaxM) {
            const std::size_t mChunk = std::min(kGemmMaxM, M - m0);
            const float* xChunk = X + m0 * K;
            float*       yChunk = Y + m0 * N;

            kern.setPtr  (0, xChunk);
            kern.setPtr  (1, W);
            kern.setPtr  (2, yChunk);
            kern.setValue(3, static_cast<std::int32_t>(K));
            kern.setValue(4, static_cast<std::int32_t>(N));
            kern.setValue(5, static_cast<std::int32_t>(mChunk));

            kern.launch(_ctx.stream(),
                        nGroups, 1, 1,
                        kVecLocalSize, 1, 1);
        }
        return;
    }

    if (type == ::mimirmind::core::gguf::GgmlType::FP8_E4M3) {
        // Blocked-FP8 (E4M3) matmul — 1-byte log-format weights (34 B / 32).
        // Same warp layout as BF16 (LOCAL=128, 4 outputs/group); the scale is
        // embedded per block so no scale plumbing is needed. Used for the
        // attention projections the NVFP4 loader keeps blocked-FP8.
        const std::uint32_t nGroups = static_cast<std::uint32_t>(
            (N + kOutputsPerGroup - 1) / kOutputsPerGroup);

        if (M == 1) {
            auto& kern = _pimpl->_matmulFp8VecKernel;
            kern.setPtr  (0, X);
            kern.setPtr  (1, W);
            kern.setPtr  (2, Y);
            kern.setValue(3, static_cast<std::int32_t>(K));
            kern.setValue(4, static_cast<std::int32_t>(N));
            kern.launch(_ctx.stream(), nGroups, 1, 1, kVecLocalSize, 1, 1);
            return;
        }

        constexpr std::size_t kGemmMaxM = 16;   // == GEMM_MAX_M in the .cu
        auto& kern = _pimpl->_matmulFp8GemmKernel;
        for (std::size_t m0 = 0; m0 < M; m0 += kGemmMaxM) {
            const std::size_t mChunk = std::min(kGemmMaxM, M - m0);
            kern.setPtr  (0, X + m0 * K);
            kern.setPtr  (1, W);
            kern.setPtr  (2, Y + m0 * N);
            kern.setValue(3, static_cast<std::int32_t>(K));
            kern.setValue(4, static_cast<std::int32_t>(N));
            kern.setValue(5, static_cast<std::int32_t>(mChunk));
            kern.launch(_ctx.stream(), nGroups, 1, 1, kVecLocalSize, 1, 1);
        }
        return;
    }

    if (type == ::mimirmind::core::gguf::GgmlType::NVFP4_BLK) {
        // Blocked-NVFP4 (E2M1) matmul — native 4-bit weights kept resident
        // (20 B / 32 elems). Same warp layout as BF16; scales embedded per
        // super-block, so no plumbing. Full-attention projections.
        const std::uint32_t nGroups = static_cast<std::uint32_t>(
            (N + kOutputsPerGroup - 1) / kOutputsPerGroup);

        if (M == 1) {
            if (_useDeintVec) {
                nvblkDeintVec(W, N, K, X, Y);
                return;
            }
            auto& kern = _pimpl->_matmulNvblkVecKernel;
            kern.setPtr  (0, X);
            kern.setPtr  (1, W);
            kern.setPtr  (2, Y);
            kern.setValue(3, static_cast<std::int32_t>(K));
            kern.setValue(4, static_cast<std::int32_t>(N));
            kern.launch(_ctx.stream(), nGroups, 1, 1, kVecLocalSize, 1, 1);
            return;
        }

        constexpr std::size_t kGemmMaxM = 16;
        auto& kern = _pimpl->_matmulNvblkGemmKernel;
        for (std::size_t m0 = 0; m0 < M; m0 += kGemmMaxM) {
            const std::size_t mChunk = std::min(kGemmMaxM, M - m0);
            kern.setPtr  (0, X + m0 * K);
            kern.setPtr  (1, W);
            kern.setPtr  (2, Y + m0 * N);
            kern.setValue(3, static_cast<std::int32_t>(K));
            kern.setValue(4, static_cast<std::int32_t>(N));
            kern.setValue(5, static_cast<std::int32_t>(mChunk));
            kern.launch(_ctx.stream(), nGroups, 1, 1, kVecLocalSize, 1, 1);
        }
        return;
    }

    if (type == ::mimirmind::core::gguf::GgmlType::Q5_K) {
        // Native Q5_K vec kernel — Q4_K-shape launch, block is 256
        // elements / 176 bytes: d/dmin/scales[12] identical to Q4_K
        // plus qh[32] (one high-bit per element, 2-bit shift per pair)
        // + qs[128] nibbles. Q5_K appears in Gemma 4 Q4_K_M for
        // attn_k / attn_output — hitting this on every draft step of
        // M9.11 speculative decoding.
        constexpr std::size_t kBlockElts = 256;
        if (K % kBlockElts != 0) {
            throw std::runtime_error(
                "compute::cuda::GpuMatmul::matmulAsync: K=" +
                std::to_string(K) + " is not a multiple of Q5_K "
                "blockElements=" + std::to_string(kBlockElts));
        }
        const std::uint32_t nGroups = static_cast<std::uint32_t>(
            (N + kOutputsPerGroup - 1) / kOutputsPerGroup);

        auto& kern = _pimpl->_matmulQ5KVecKernel;
        for (std::size_t m = 0; m < M; ++m) {
            const float* xRow = X + m * K;
            float*       yRow = Y + m * N;

            kern.setPtr  (0, xRow);
            kern.setPtr  (1, W);
            kern.setPtr  (2, yRow);
            kern.setValue(3, static_cast<std::int32_t>(K));
            kern.setValue(4, static_cast<std::int32_t>(N));

            kern.launch(_ctx.stream(),
                        nGroups, 1, 1,
                        kVecLocalSize, 1, 1);
        }
        return;
    }

    if (type != ::mimirmind::core::gguf::GgmlType::Q8_0) {
        // CPU fallback for weight types without a HIP matmul kernel
        // today (Q4_K / Q5_K / Q6_K / F16 / BF16 / F32, or any other
        // type present in compute::QuantTypeRegistry).
        //
        // Shape mirrors the L0 side (l0::GpuMatmul::matmulAsync →
        // _queue.flush() + compute::matmul), with one extra step:
        // L0's USM is host-visible so the CPU code reads W/X and
        // writes Y directly; HIP device pointers aren't, so we stage
        // through host buffers. This is a correctness path — perf is
        // O(slow) and expected to be replaced by native HIP kernels
        // per weight type over time.
        //
        // Q5_0 currently still throws inside compute::matmul →
        // dequantToF32 because QuantTypeRegistry has no Q5_0 entry
        // (Option D on the shelf). Once Q5_0 lands there, this path
        // dispatches it too without further changes here.

        const ::mimirmind::compute::QuantType* qt =
            ::mimirmind::compute::quantType(type);
        if (qt == nullptr) {
            throw std::runtime_error(
                std::string{"compute::cuda::GpuMatmul::matmulAsync: no CPU "
                            "dequant impl for ggml type '"} +
                std::string{::mimirmind::core::gguf::typeInfo(type).name} +
                "' — HIP CPU-fallback cannot proceed. Register the type "
                "in compute::QuantTypeRegistry to unblock.");
        }

        if (!_cpuFallbackLogged) {
            _cpuFallbackLogged = true;
            MM_LOG_INFO("hip::GpuMatmul",
                        "CPU fallback active — dispatching '{}' (and any "
                        "other non-Q8_0 type this session) through "
                        "compute::matmul on the host. W/X are copied D2H, "
                        "Y is copied H2D. Slow by design; native HIP "
                        "kernels replace this per type. Further "
                        "per-dispatch logs suppressed.",
                        qt->name());
        }

        // Preceding matmulAsync / GpuOps launches may still be running
        // and their outputs may be inputs here (X, or W in some
        // configurations). Sync the stream so the D2H copies observe
        // the completed state.
        sync();

        const std::size_t nBlocksPerRow = K / qt->blockElements();
        const std::size_t wBytes        = N * nBlocksPerRow * qt->blockBytes();
        const std::size_t xBytes        = M * K * sizeof(float);
        const std::size_t yBytes        = M * N * sizeof(float);

        std::vector<std::uint8_t> wHost(wBytes);
        std::vector<float>        xHost(M * K);
        std::vector<float>        yHost(M * N);
        std::vector<float>        cpuScratch(K);

        auto& alloc = _ctx.allocator();
        alloc.copyD2H(wHost.data(), W, wBytes);
        alloc.copyD2H(xHost.data(), X, xBytes);

        ::mimirmind::compute::matmul(type,
                                     wHost.data(), N, K,
                                     xHost.data(), M,
                                     yHost.data(),
                                     cpuScratch.data());

        alloc.copyH2D(Y, yHost.data(), yBytes);
        return;
    }

    // Dispatch by autotune state:
    //   1. useDp4a set (autotune / features.dp4a=Force) → quant + DP4A
    //   2. else M >= gemmMinM AND GEMM available → batched GEMM (V2 if
    //      useGemmV2 pinned)
    //   3. else per-row matvec loop
    //
    // Pre-autotune defaults leave `_gemmMinM = kGemmMinMNever` and
    // `_useDp4a = false`, so an untuned dispatcher takes the safe
    // matvec-loop path even at Mmax. Same behaviour as L0.

    if (_mmqEnabled && type == ::mimirmind::core::gguf::GgmlType::Q8_0
        && M > 1 && N <= _mmqMaxN) {
        // M-Cuda.MMQ C1 — int8 MMQ GEMM for Q8_0 prefill (env-gated). Decode
        // (M==1) falls through to the launch-bound-friendly GEMV path.
        // MIMIRMIND_MMQ_TC picks the tensor-core (wmma) kernel over dp4a.
        // Alternative F: N>_mmqMaxN (the vocab-sized lm_head) stays fp32 so the
        // greedy argmax is decided by exact final logits.
        if (_mmqTc) {
            matmulQ8_0MmqTcAsync(W, N, K, X, M, Y);
        } else {
            matmulQ8_0MmqAsync(W, N, K, X, M, Y);
        }
        return;
    }

    if (_useDp4a) {
        // Quantise X → int8 into engine-owned scratch (allocated by the
        // caller through _ops), then DP4A-matvec. K must be a multiple
        // of 32 — matmulDp4aAsync enforces the same guard.
        //
        // Sub-F.4 scope: DP4A auto-scratch not yet wired (the L0 side
        // owns `_dp4aXqUsm` / `_dp4aScaleUsm` on GpuMatmul); a full
        // DP4A-from-float dispatch lands together with the DP4A
        // auto-pick bench. Until then `_useDp4a=true` requires the
        // caller to have quantised upfront.
        throw std::runtime_error(
            "compute::cuda::GpuMatmul::matmulAsync: _useDp4a=true but the "
            "DP4A-from-float scratch is not owned by this class yet — "
            "call matmulDp4aAsync with pre-quantised Xq/Xscale (from "
            "GpuOps::xQuantI8Async) instead. Auto-DP4A-from-float lands "
            "in a follow-up commit.");
    }

    if (M >= _gemmMinM) {
        // Batched GEMM path. V1 and V2 have identical arg layouts —
        // only the M_TILE differs, so the mGroups division changes.
        auto& kern = _useGemmV2
            ? _pimpl->_matmulQ8_0GemmV2Kernel
            : _pimpl->_matmulQ8_0GemmKernel;
        const std::size_t mTile = _useGemmV2 ? kGemmV2MTile : kGemmMTile;

        kern.setPtr  (0, X);
        kern.setPtr  (1, W);
        kern.setPtr  (2, Y);
        kern.setValue(3, static_cast<std::int32_t>(K));
        kern.setValue(4, static_cast<std::int32_t>(N));
        kern.setValue(5, static_cast<std::int32_t>(M));

        const std::uint32_t nGroups = static_cast<std::uint32_t>(
            (N + kOutputsPerGroup - 1) / kOutputsPerGroup);
        const std::uint32_t mGroups = static_cast<std::uint32_t>(
            (M + mTile - 1) / mTile);
        kern.launch(_ctx.stream(),
                    nGroups, mGroups, 1,
                    kLocalSize, 1, 1);
        return;
    }

    // Matvec-loop fallback: one launch per row of X. The vec kernel
    // (matmul_q8_0_vec.hip) needs 4 warps × 32 lanes = 128 threads on
    // RDNA3 (one warp per output row, MATMUL_Q8_0_OUTPUTS_PER_GROUP=4).
    // Launching with the gemm-path's kLocalSize=64 would only give 2
    // warps — warps 2/3 never execute and their output slots contain
    // stale memory (2026-07-17 modulo-4 attn_v garbage on Qwen 2.5).
    const std::uint32_t nGroups = static_cast<std::uint32_t>(
        (N + kOutputsPerGroup - 1) / kOutputsPerGroup);

    auto& kern = _pimpl->_matmulQ8_0VecKernel;
    for (std::size_t m = 0; m < M; ++m) {
        const float* xRow = X + m * K;
        float*       yRow = Y + m * N;

        kern.setPtr  (0, xRow);
        kern.setPtr  (1, W);
        kern.setPtr  (2, yRow);
        kern.setValue(3, static_cast<std::int32_t>(K));
        kern.setValue(4, static_cast<std::int32_t>(N));

        kern.launch(_ctx.stream(),
                    nGroups, 1, 1,
                    kVecLocalSize, 1, 1);
    }
}

void GpuMatmul::matmulDp4aAsync(::mimirmind::core::gguf::GgmlType type,
                                const std::int8_t* Xq,
                                const float*       Xscale,
                                const void*        W,
                                std::size_t        N,
                                std::size_t        K,
                                std::size_t        M,
                                float*             Y) {
    const bool isQ8_0 = type == ::mimirmind::core::gguf::GgmlType::Q8_0;
    const bool isQ6K  = type == ::mimirmind::core::gguf::GgmlType::Q6_K;
    if (!isQ8_0 && !isQ6K) {
        throw std::runtime_error(
            "compute::cuda::GpuMatmul::matmulDp4aAsync: only Q8_0/Q6_K have "
            "a DP4A kernel on the CUDA side — check dp4aAvailable(type) "
            "before dispatching");
    }
    if (M == 0 || N == 0 || K == 0) {
        return;
    }

    // Q8_0 blocks are 32 elements; Q6_K super-blocks are 256. `Xscale` is
    // one float per 32-wide activation quant block either way (produced by
    // x_quant_q8_1_blocks) — Q6_K's kernel reads 8 of those per
    // super-block internally.
    const std::size_t blockElts = isQ8_0 ? 32 : 256;
    if (K % blockElts != 0) {
        throw std::runtime_error(
            "compute::cuda::GpuMatmul::matmulDp4aAsync: K=" +
            std::to_string(K) + " is not a multiple of blockElements=" +
            std::to_string(blockElts));
    }
    const std::size_t xScaleBlocksPerRow = K / 32;

    auto& kern = isQ8_0 ? _pimpl->_matmulQ8_0VecDp4aKernel
                        : _pimpl->_matmulQ6KVecDp4aKernel;

    // warp32 geometry (kDp4aOutputsPerGroup outputs per WG — see
    // matmul_q8_0_vec_dp4a.cu / matmul_q6k_vec_dp4a.cu). WG count in the
    // N dim only; per-row loop iterates over X in the M dim.
    const std::uint32_t nGroups = static_cast<std::uint32_t>(
        (N + kDp4aOutputsPerGroup - 1) / kDp4aOutputsPerGroup);

    // Constant args across all rows — K and N are set once outside
    // the loop, W stays constant, only Xq/Xscale/Y advance per m.
    kern.setValue(4, static_cast<std::int32_t>(K));
    kern.setValue(5, static_cast<std::int32_t>(N));

    for (std::size_t m = 0; m < M; ++m) {
        const std::int8_t* xqRow = Xq + m * K;
        const float*       xsRow = Xscale + m * xScaleBlocksPerRow;
        float*             yRow  = Y + m * N;

        kern.setPtr(0, xqRow);
        kern.setPtr(1, xsRow);
        kern.setPtr(2, W);
        kern.setPtr(3, yRow);

        kern.launch(_ctx.stream(),
                    nGroups, 1, 1,
                    kDp4aLocalSize, 1, 1);
    }
}

void GpuMatmul::matmulQ8_0MmqAsync(const void*  W,
                                   std::size_t  N,
                                   std::size_t  K,
                                   const float* X,
                                   std::size_t  M,
                                   float*       Y) {
    if (M == 0 || N == 0 || K == 0) {
        return;
    }
    // Same launch geometry as the Q8_0 GEMM path (4 output cols/WG, M_TILE
    // rows/WG, 64 threads). Kernel arg order: X, W, Y, K, N, M.
    auto& kern = _pimpl->_matmulQ8_0MmqKernel;
    kern.setPtr  (0, X);
    kern.setPtr  (1, W);
    kern.setPtr  (2, Y);
    kern.setValue(3, static_cast<std::int32_t>(K));
    kern.setValue(4, static_cast<std::int32_t>(N));
    kern.setValue(5, static_cast<std::int32_t>(M));

    const std::uint32_t nGroups = static_cast<std::uint32_t>(
        (N + kOutputsPerGroup - 1) / kOutputsPerGroup);
    const std::uint32_t mGroups = static_cast<std::uint32_t>(
        (M + kGemmV2MTile - 1) / kGemmV2MTile);
    kern.launch(_ctx.stream(),
                nGroups, mGroups, 1,
                kLocalSize, 1, 1);
}

void GpuMatmul::matmulQ8_0MmqTcAsync(const void*  W,
                                     std::size_t  N,
                                     std::size_t  K,
                                     const float* X,
                                     std::size_t  M,
                                     float*       Y) {
    if (M == 0 || N == 0 || K == 0) {
        return;
    }
    // One warp (32 threads) per 16x16 output tile. Kernel args: X, W, Y, K, N, M.
    auto& kern = _pimpl->_matmulQ8_0MmqTcKernel;
    kern.setPtr  (0, X);
    kern.setPtr  (1, W);
    kern.setPtr  (2, Y);
    kern.setValue(3, static_cast<std::int32_t>(K));
    kern.setValue(4, static_cast<std::int32_t>(N));
    kern.setValue(5, static_cast<std::int32_t>(M));

    const std::uint32_t nGroups = static_cast<std::uint32_t>((N + 15) / 16);
    const std::uint32_t mGroups = static_cast<std::uint32_t>((M + 15) / 16);
    kern.launch(_ctx.stream(),
                nGroups, mGroups, 1,
                32, 1, 1);
}

void GpuMatmul::matmulQ4KMmqAsync(const void*  W,
                                  std::size_t  N,
                                  std::size_t  K,
                                  const float* X,
                                  std::size_t  M,
                                  float*       Y) {
    if (M == 0 || N == 0 || K == 0) {
        return;
    }
    // Same launch geometry as the Q8_0 MMQ / GEMM path. Kernel args: X,W,Y,K,N,M.
    auto& kern = _pimpl->_matmulQ4KMmqKernel;
    kern.setPtr  (0, X);
    kern.setPtr  (1, W);
    kern.setPtr  (2, Y);
    kern.setValue(3, static_cast<std::int32_t>(K));
    kern.setValue(4, static_cast<std::int32_t>(N));
    kern.setValue(5, static_cast<std::int32_t>(M));

    const std::uint32_t nGroups = static_cast<std::uint32_t>(
        (N + kOutputsPerGroup - 1) / kOutputsPerGroup);
    const std::uint32_t mGroups = static_cast<std::uint32_t>(
        (M + kGemmV2MTile - 1) / kGemmV2MTile);
    kern.launch(_ctx.stream(),
                nGroups, mGroups, 1,
                kLocalSize, 1, 1);
}

void GpuMatmul::matmulQ5KMmqAsync(const void*  W,
                                  std::size_t  N,
                                  std::size_t  K,
                                  const float* X,
                                  std::size_t  M,
                                  float*       Y) {
    if (M == 0 || N == 0 || K == 0) {
        return;
    }
    auto& kern = _pimpl->_matmulQ5KMmqKernel;
    kern.setPtr  (0, X);
    kern.setPtr  (1, W);
    kern.setPtr  (2, Y);
    kern.setValue(3, static_cast<std::int32_t>(K));
    kern.setValue(4, static_cast<std::int32_t>(N));
    kern.setValue(5, static_cast<std::int32_t>(M));

    const std::uint32_t nGroups = static_cast<std::uint32_t>(
        (N + kOutputsPerGroup - 1) / kOutputsPerGroup);
    const std::uint32_t mGroups = static_cast<std::uint32_t>(
        (M + kGemmV2MTile - 1) / kGemmV2MTile);
    kern.launch(_ctx.stream(),
                nGroups, mGroups, 1,
                kLocalSize, 1, 1);
}

void GpuMatmul::moeDownFusedKAsync(::mimirmind::core::gguf::GgmlType type,
                                   const float*         gateAct,
                                   const void*          W,
                                   const std::int32_t*  expIdx,
                                   const float*         kw,
                                   float*               accum,
                                   std::size_t          ffPer,
                                   std::size_t          dModel,
                                   std::size_t          kActive,
                                   std::size_t          expertBytes) {
    if (kActive == 0 || dModel == 0 || ffPer == 0) {
        return;
    }

    if (type == ::mimirmind::core::gguf::GgmlType::NVFP4_BLK) {
        // Single-token = the batched NVFP4 down kernel with nSeq=1 (grid.y=1).
        if (ffPer % 32 != 0) {
            throw std::runtime_error(
                "moeDownFusedKAsync: ffPer not a multiple of 32 (NVFP4_BLK)");
        }
        auto& kern = _pimpl->_moeDownFusedKNvblkBatchedKernel;
        kern.setPtr  (0, gateAct);
        kern.setPtr  (1, W);
        kern.setPtr  (2, expIdx);
        kern.setPtr  (3, kw);
        kern.setPtr  (4, accum);
        kern.setValue(5, static_cast<std::int32_t>(ffPer));
        kern.setValue(6, static_cast<std::int32_t>(dModel));
        kern.setValue(7, static_cast<std::int32_t>(kActive));
        kern.setValue(8, static_cast<std::int32_t>(expertBytes));
        const std::uint32_t nGroups = static_cast<std::uint32_t>(
            (dModel + kMoeDownOutputsPerGroup - 1) / kMoeDownOutputsPerGroup);
        kern.launch(_ctx.stream(), nGroups, 1, 1, kMoeDownLocalSize, 1, 1);
        return;
    }

    // Select kernel + block-element alignment guard per Q-type. Q6_K
    // super-blocks are 256 elements; Q8_0 blocks are 32. A partial
    // block at the tail corrupts the dequant accumulator.
    ::mimirmind::core::cuda::CudaKernel* kernPtr = nullptr;
    std::size_t                        kBlockElts = 0;
    const char*                        kBlockName = "";
    switch (type) {
        case ::mimirmind::core::gguf::GgmlType::Q8_0:
            kernPtr    = &_pimpl->_moeDownFusedKQ8_0Kernel;
            kBlockElts = 32;
            kBlockName = "Q8_0";
            break;
        case ::mimirmind::core::gguf::GgmlType::Q6_K:
            kernPtr    = &_pimpl->_moeDownFusedKQ6KKernel;
            kBlockElts = 256;
            kBlockName = "Q6_K";
            break;
        case ::mimirmind::core::gguf::GgmlType::Q5_K:
            kernPtr    = &_pimpl->_moeDownFusedKQ5KKernel;
            kBlockElts = 256;
            kBlockName = "Q5_K";
            break;
        default:
            throw std::runtime_error(
                "compute::cuda::GpuMatmul::moeDownFusedKAsync: type not "
                "supported — check moeDownFusedKAvailable"
                "(type) before dispatching. Supported: Q8_0, Q6_K, Q5_K.");
    }
    if (ffPer % kBlockElts != 0) {
        throw std::runtime_error(
            std::string("compute::cuda::GpuMatmul::moeDownFusedKAsync: ffPer=") +
            std::to_string(ffPer) + " is not a multiple of " + kBlockName +
            " blockElements=" + std::to_string(kBlockElts));
    }

    auto& kern = *kernPtr;
    kern.setPtr  (0, gateAct);
    kern.setPtr  (1, W);
    kern.setPtr  (2, expIdx);
    kern.setPtr  (3, kw);
    kern.setPtr  (4, accum);
    kern.setValue(5, static_cast<std::int32_t>(ffPer));
    kern.setValue(6, static_cast<std::int32_t>(dModel));
    kern.setValue(7, static_cast<std::int32_t>(kActive));
    kern.setValue(8, static_cast<std::int32_t>(expertBytes));

    // 4 outputs per workgroup — geometry shared across Q8_0 and Q6_K
    // (MOE_DOWN_LOCAL=64, SG=16 in the .hip sources).
    const std::uint32_t nGroups = static_cast<std::uint32_t>(
        (dModel + kMoeDownOutputsPerGroup - 1) / kMoeDownOutputsPerGroup);

    // Read-modify-write on `accum`: the kernel adds the fused
    // sum-over-K into the caller-supplied accumulator without
    // pre-zeroing. Caller is responsible for the seed value (matches
    // Gemma 4 MoE per-expert loop's expectation).
    kern.launch(_ctx.stream(),
                nGroups, 1, 1,
                kMoeDownLocalSize, 1, 1);
}

void GpuMatmul::moeDownFusedKBatchedAsync(
        ::mimirmind::core::gguf::GgmlType type, const float* gateAct,
        const void* W, const std::int32_t* expIdx, const float* kw,
        float* accum, std::size_t nSeq, std::size_t ffPer, std::size_t dModel,
        std::size_t kActive, std::size_t expertBytes) {
    if (nSeq == 0 || kActive == 0 || dModel == 0 || ffPer == 0) {
        return;
    }
    if (type == ::mimirmind::core::gguf::GgmlType::BF16) {
        // Dense BF16 experts. Element strides derive from dims; expertBytes
        // is unused. Kernel block = 128 (4 warps -> 4 outputs/group).
        (void)expertBytes;
        auto& kern = _pimpl->_moeDownFusedKBf16BatchedKernel;
        kern.setPtr  (0, gateAct);
        kern.setPtr  (1, W);
        kern.setPtr  (2, expIdx);
        kern.setPtr  (3, kw);
        kern.setPtr  (4, accum);
        kern.setValue(5, static_cast<std::int32_t>(ffPer));
        kern.setValue(6, static_cast<std::int32_t>(dModel));
        kern.setValue(7, static_cast<std::int32_t>(kActive));
        const std::uint32_t nGroups = static_cast<std::uint32_t>(
            (dModel + kMoeDownOutputsPerGroup - 1) / kMoeDownOutputsPerGroup);
        kern.launch(_ctx.stream(),
                    nGroups, static_cast<std::uint32_t>(nSeq), 1,
                    kMoeGateUpLocalSize, 1, 1);
        return;
    }
    if (type == ::mimirmind::core::gguf::GgmlType::NVFP4_BLK) {
        // Native blocked-NVFP4 down experts (32-element supers). Same launch
        // geometry as the K-quant path; ffPer must be a multiple of 32.
        if (ffPer % 32 != 0) {
            throw std::runtime_error(
                "moeDownFusedKBatchedAsync: ffPer not a multiple of 32 (NVFP4_BLK)");
        }
        auto& kern = _pimpl->_moeDownFusedKNvblkBatchedKernel;
        kern.setPtr  (0, gateAct);
        kern.setPtr  (1, W);
        kern.setPtr  (2, expIdx);
        kern.setPtr  (3, kw);
        kern.setPtr  (4, accum);
        kern.setValue(5, static_cast<std::int32_t>(ffPer));
        kern.setValue(6, static_cast<std::int32_t>(dModel));
        kern.setValue(7, static_cast<std::int32_t>(kActive));
        kern.setValue(8, static_cast<std::int32_t>(expertBytes));
        const std::uint32_t nGroups = static_cast<std::uint32_t>(
            (dModel + kMoeDownOutputsPerGroup - 1) / kMoeDownOutputsPerGroup);
        kern.launch(_ctx.stream(),
                    nGroups, static_cast<std::uint32_t>(nSeq), 1,
                    kMoeDownLocalSize, 1, 1);
        return;
    }
    const bool isQ5K = (type == ::mimirmind::core::gguf::GgmlType::Q5_K);
    const bool isQ6K = (type == ::mimirmind::core::gguf::GgmlType::Q6_K);
    if (!isQ5K && !isQ6K) {
        throw std::runtime_error(
            "compute::cuda::GpuMatmul::moeDownFusedKBatchedAsync: only Q5_K/Q6_K/"
            "BF16 batched today — Q8_0 batched variant not yet ported.");
    }
    if (ffPer % 256 != 0) {
        throw std::runtime_error(
            "compute::cuda::GpuMatmul::moeDownFusedKBatchedAsync: ffPer=" +
            std::to_string(ffPer) +
            " is not a multiple of K-quant blockElements=256");
    }
    // grid = (nGroups, nSeq); each seq has its own gateAct/expIdx/kw/accum
    // (per-token routing + RMW accumulator). Expert bank shared. Math
    // byte-identical to nSeq single moeDownFusedKAsync (Cat B). Q5_K and Q6_K
    // share the launch geometry (MOE_DOWN_LOCAL=64, 4 outputs/group).
    auto& kern = isQ6K ? _pimpl->_moeDownFusedKQ6KBatchedKernel
                       : _pimpl->_moeDownFusedKQ5KBatchedKernel;
    kern.setPtr  (0, gateAct);
    kern.setPtr  (1, W);
    kern.setPtr  (2, expIdx);
    kern.setPtr  (3, kw);
    kern.setPtr  (4, accum);
    kern.setValue(5, static_cast<std::int32_t>(ffPer));
    kern.setValue(6, static_cast<std::int32_t>(dModel));
    kern.setValue(7, static_cast<std::int32_t>(kActive));
    kern.setValue(8, static_cast<std::int32_t>(expertBytes));

    const std::uint32_t nGroups = static_cast<std::uint32_t>(
        (dModel + kMoeDownOutputsPerGroup - 1) / kMoeDownOutputsPerGroup);
    kern.launch(_ctx.stream(),
                nGroups, static_cast<std::uint32_t>(nSeq), 1,
                kMoeDownLocalSize, 1, 1);
}

void GpuMatmul::moeGateUpFusedKAsync(::mimirmind::core::gguf::GgmlType type,
                                     const float*         x,
                                     const void*          Wg,
                                     const void*          Wu,
                                     const std::int32_t*  expIdx,
                                     float*               gateActOut,
                                     std::size_t          dModel,
                                     std::size_t          nFf,
                                     std::size_t          kActive,
                                     std::size_t          expertBytesGate,
                                     std::size_t          expertBytesUp) {
    if (kActive == 0 || dModel == 0 || nFf == 0) {
        return;
    }
    if (type == ::mimirmind::core::gguf::GgmlType::NVFP4_BLK) {
        // Single-token = the batched NVFP4 kernel with nSeq=1 (grid.y=1).
        if (dModel % 32 != 0) {
            throw std::runtime_error(
                "moeGateUpFusedKAsync: dModel not a multiple of 32 (NVFP4_BLK)");
        }
        auto& kern = _pimpl->_moeGateUpFusedKNvblkBatchedKernel;
        kern.setPtr  (0, x);
        kern.setPtr  (1, Wg);
        kern.setPtr  (2, Wu);
        kern.setPtr  (3, expIdx);
        kern.setPtr  (4, gateActOut);
        kern.setValue(5, static_cast<std::int32_t>(dModel));
        kern.setValue(6, static_cast<std::int32_t>(nFf));
        kern.setValue(7, static_cast<std::int32_t>(kActive));
        kern.setValue(8, static_cast<std::int32_t>(expertBytesGate));
        kern.setValue(9, static_cast<std::int32_t>(expertBytesUp));
        const std::uint32_t nOutputs = static_cast<std::uint32_t>(kActive * nFf);
        const std::uint32_t nGroups  =
            (nOutputs + kMoeGateUpOutputsPerGroup - 1) / kMoeGateUpOutputsPerGroup;
        kern.launch(_ctx.stream(), nGroups, 1, 1, kMoeGateUpLocalSize, 1, 1);
        return;
    }
    if (type != ::mimirmind::core::gguf::GgmlType::Q4_K) {
        throw std::runtime_error(
            "compute::cuda::GpuMatmul::moeGateUpFusedKAsync: only Q4_K "
            "supported — check moeGateUpFusedKAvailable(type) first.");
    }
    // Both GEMVs reduce over dModel in Q4_K super-blocks of 256; require an
    // exact multiple so no partial block corrupts the dequant.
    if (dModel % 256 != 0) {
        throw std::runtime_error(
            "compute::cuda::GpuMatmul::moeGateUpFusedKAsync: dModel=" +
            std::to_string(dModel) +
            " is not a multiple of Q4_K blockElements=256");
    }

    auto& kern = _pimpl->_moeGateUpFusedKQ4KKernel;
    kern.setPtr  (0, x);
    kern.setPtr  (1, Wg);
    kern.setPtr  (2, Wu);
    kern.setPtr  (3, expIdx);
    kern.setPtr  (4, gateActOut);
    kern.setValue(5, static_cast<std::int32_t>(dModel));
    kern.setValue(6, static_cast<std::int32_t>(nFf));
    kern.setValue(7, static_cast<std::int32_t>(kActive));
    kern.setValue(8, static_cast<std::int32_t>(expertBytesGate));
    kern.setValue(9, static_cast<std::int32_t>(expertBytesUp));

    // One warp per (k, f) output; kMoeGateUpOutputsPerGroup warps per WG.
    const std::uint32_t nOutputs = static_cast<std::uint32_t>(kActive * nFf);
    const std::uint32_t nGroups  =
        (nOutputs + kMoeGateUpOutputsPerGroup - 1) / kMoeGateUpOutputsPerGroup;

    kern.launch(_ctx.stream(),
                nGroups, 1, 1,
                kMoeGateUpLocalSize, 1, 1);
}

void GpuMatmul::moeGateUpFusedKBatchedAsync(
        ::mimirmind::core::gguf::GgmlType type, const float* x, const void* Wg,
        const void* Wu, const std::int32_t* expIdx, float* gateActOut,
        std::size_t nSeq, std::size_t dModel, std::size_t nFf,
        std::size_t kActive, std::size_t expertBytesGate,
        std::size_t expertBytesUp) {
    if (nSeq == 0 || kActive == 0 || dModel == 0 || nFf == 0) {
        return;
    }
    if (type == ::mimirmind::core::gguf::GgmlType::BF16) {
        // Dense BF16 experts (NVFP4->BF16). Element strides derive from
        // dims; the byte-stride args are unused.
        (void)expertBytesGate; (void)expertBytesUp;
        auto& kern = _pimpl->_moeGateUpFusedKBf16BatchedKernel;
        kern.setPtr  (0, x);
        kern.setPtr  (1, Wg);
        kern.setPtr  (2, Wu);
        kern.setPtr  (3, expIdx);
        kern.setPtr  (4, gateActOut);
        kern.setValue(5, static_cast<std::int32_t>(dModel));
        kern.setValue(6, static_cast<std::int32_t>(nFf));
        kern.setValue(7, static_cast<std::int32_t>(kActive));
        const std::uint32_t nOutputs = static_cast<std::uint32_t>(kActive * nFf);
        const std::uint32_t nGroups  =
            (nOutputs + kMoeGateUpOutputsPerGroup - 1) / kMoeGateUpOutputsPerGroup;
        kern.launch(_ctx.stream(),
                    nGroups, static_cast<std::uint32_t>(nSeq), 1,
                    kMoeGateUpLocalSize, 1, 1);
        return;
    }
    if (type == ::mimirmind::core::gguf::GgmlType::NVFP4_BLK) {
        // Native blocked-NVFP4 gate/up experts (32-element supers).
        if (dModel % 32 != 0) {
            throw std::runtime_error(
                "moeGateUpFusedKBatchedAsync: dModel not a multiple of 32 (NVFP4_BLK)");
        }
        auto& kern = _pimpl->_moeGateUpFusedKNvblkBatchedKernel;
        kern.setPtr  (0, x);
        kern.setPtr  (1, Wg);
        kern.setPtr  (2, Wu);
        kern.setPtr  (3, expIdx);
        kern.setPtr  (4, gateActOut);
        kern.setValue(5, static_cast<std::int32_t>(dModel));
        kern.setValue(6, static_cast<std::int32_t>(nFf));
        kern.setValue(7, static_cast<std::int32_t>(kActive));
        kern.setValue(8, static_cast<std::int32_t>(expertBytesGate));
        kern.setValue(9, static_cast<std::int32_t>(expertBytesUp));
        const std::uint32_t nOutputs = static_cast<std::uint32_t>(kActive * nFf);
        const std::uint32_t nGroups  =
            (nOutputs + kMoeGateUpOutputsPerGroup - 1) / kMoeGateUpOutputsPerGroup;
        kern.launch(_ctx.stream(),
                    nGroups, static_cast<std::uint32_t>(nSeq), 1,
                    kMoeGateUpLocalSize, 1, 1);
        return;
    }
    if (type != ::mimirmind::core::gguf::GgmlType::Q4_K) {
        throw std::runtime_error(
            "compute::cuda::GpuMatmul::moeGateUpFusedKBatchedAsync: only "
            "Q4_K supported — check moeGateUpFusedKAvailable(type) first.");
    }
    if (dModel % 256 != 0) {
        throw std::runtime_error(
            "compute::cuda::GpuMatmul::moeGateUpFusedKBatchedAsync: dModel=" +
            std::to_string(dModel) +
            " is not a multiple of Q4_K blockElements=256");
    }
    // grid = (nGroups, nSeq); each seq has its own x + expIdx (per-token
    // routing) writing its own gateActOut slice. Expert banks shared.
    // Math byte-identical to nSeq single moeGateUpFusedKAsync (Cat B).
    auto& kern = _pimpl->_moeGateUpFusedKQ4KBatchedKernel;
    kern.setPtr  (0, x);
    kern.setPtr  (1, Wg);
    kern.setPtr  (2, Wu);
    kern.setPtr  (3, expIdx);
    kern.setPtr  (4, gateActOut);
    kern.setValue(5, static_cast<std::int32_t>(dModel));
    kern.setValue(6, static_cast<std::int32_t>(nFf));
    kern.setValue(7, static_cast<std::int32_t>(kActive));
    kern.setValue(8, static_cast<std::int32_t>(expertBytesGate));
    kern.setValue(9, static_cast<std::int32_t>(expertBytesUp));

    const std::uint32_t nOutputs = static_cast<std::uint32_t>(kActive * nFf);
    const std::uint32_t nGroups  =
        (nOutputs + kMoeGateUpOutputsPerGroup - 1) / kMoeGateUpOutputsPerGroup;

    kern.launch(_ctx.stream(),
                nGroups, static_cast<std::uint32_t>(nSeq), 1,
                kMoeGateUpLocalSize, 1, 1);
}

void GpuMatmul::moeGateUpFusedKTcBatchedAsync(
        const float* x, const void* WgNib, const void* WuNib,
        const void* WgSfb, const void* WuSfb,
        const float* WgGlobal, const float* WuGlobal,
        const std::int32_t* expIdx, float* gateActOut,
        std::size_t nSeq, std::size_t dModel, std::size_t nFf,
        std::size_t kActive, std::size_t sfbStride) {
    auto& kern = _pimpl->_moeGateUpFusedKTcKernel;
    kern.setPtr  (0, x);
    kern.setPtr  (1, WgNib);
    kern.setPtr  (2, WuNib);
    kern.setPtr  (3, WgSfb);
    kern.setPtr  (4, WuSfb);
    kern.setPtr  (5, WgGlobal);
    kern.setPtr  (6, WuGlobal);
    kern.setPtr  (7, expIdx);
    kern.setPtr  (8, gateActOut);
    kern.setValue(9,  static_cast<std::int32_t>(dModel));
    kern.setValue(10, static_cast<std::int32_t>(nFf));
    kern.setValue(11, static_cast<std::int32_t>(kActive));
    kern.setValue(12, static_cast<std::int32_t>(sfbStride));
    const std::uint32_t nOutputs = static_cast<std::uint32_t>(kActive * nFf);
    const std::uint32_t nGroups  =
        (nOutputs + kMoeGateUpOutputsPerGroup - 1) / kMoeGateUpOutputsPerGroup;
    kern.launch(_ctx.stream(), nGroups, static_cast<std::uint32_t>(nSeq), 1,
                kMoeGateUpLocalSize, 1, 1);
}

void GpuMatmul::moeDownFusedKTcBatchedAsync(
        const float* gateAct, const void* WNib, const void* WSfb,
        const float* WGlobal, const std::int32_t* expIdx, const float* kw,
        float* accum, std::size_t nSeq, std::size_t ffPer, std::size_t dModel,
        std::size_t kActive, std::size_t sfbStride) {
    auto& kern = _pimpl->_moeDownFusedKTcKernel;
    kern.setPtr  (0, gateAct);
    kern.setPtr  (1, WNib);
    kern.setPtr  (2, WSfb);
    kern.setPtr  (3, WGlobal);
    kern.setPtr  (4, expIdx);
    kern.setPtr  (5, kw);
    kern.setPtr  (6, accum);
    kern.setValue(7,  static_cast<std::int32_t>(ffPer));
    kern.setValue(8,  static_cast<std::int32_t>(dModel));
    kern.setValue(9,  static_cast<std::int32_t>(kActive));
    kern.setValue(10, static_cast<std::int32_t>(sfbStride));
    const std::uint32_t nGroups = static_cast<std::uint32_t>(
        (dModel + kMoeDownOutputsPerGroup - 1) / kMoeDownOutputsPerGroup);
    kern.launch(_ctx.stream(), nGroups, static_cast<std::uint32_t>(nSeq), 1,
                kMoeDownLocalSize, 1, 1);
}

void GpuMatmul::ffnGateUpFusedQ8Async(const float* x,
                                      const void*  Wg,
                                      const void*  Wu,
                                      float*       Y,
                                      std::size_t  dModel,
                                      std::size_t  nFf) {
    if (dModel == 0 || nFf == 0) {
        return;
    }
    if (dModel % 32 != 0) {
        throw std::runtime_error(
            "compute::cuda::GpuMatmul::ffnGateUpFusedQ8Async: dModel=" +
            std::to_string(dModel) +
            " is not a multiple of Q8_0 blockElements=32");
    }

    auto& kern = _pimpl->_ffnGateUpFusedQ8Kernel;
    kern.setPtr  (0, x);
    kern.setPtr  (1, Wg);
    kern.setPtr  (2, Wu);
    kern.setPtr  (3, Y);
    kern.setValue(4, static_cast<std::int32_t>(dModel));   // K
    kern.setValue(5, static_cast<std::int32_t>(nFf));      // N

    const std::uint32_t nGroups = static_cast<std::uint32_t>(
        (nFf + kFfnGuQ8OutputsPerGroup - 1) / kFfnGuQ8OutputsPerGroup);

    kern.launch(_ctx.stream(),
                nGroups, 1, 1,
                kFfnGuQ8LocalSize, 1, 1);
}

} // namespace mimirmind::compute::cuda