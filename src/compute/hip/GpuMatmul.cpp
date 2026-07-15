// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "compute/hip/GpuMatmul.hpp"

#include "compute/hip/GpuOps.hpp"
#include "core/gpu/hip/HipComputeContext.hpp"
#include "core/gpu/hip/HipKernel.hpp"
#include "core/gpu/hip/HipMemoryAllocator.hpp"
#include "core/gpu/hip/HipModule.hpp"
#include "core/gpu/hip/HipStream.hpp"
#include "core/log/Log.hpp"

#include <array>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace mimirmind::compute::hip {

namespace {

constexpr const char* kDefaultHsacoDir = "/usr/local/share/mimirmind/hsaco";

// Same resolver as HipGpuOps.cpp — env override, prod install, three
// build-tree fallbacks. Kept as a local duplicate rather than lifted to
// a shared helper because these two files are the only consumers and
// their lookups are independent (a common header would drag HipModule
// deps into a place that doesn't need them).
std::filesystem::path resolveHsacoPath(std::string_view name) {
    const std::string filename = std::string{name} + ".hsaco";

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
            std::filesystem::path{kDefaultHsacoDir} / filename;
        if (std::filesystem::exists(p)) {
            return p;
        }
    }

    for (auto rel : std::array<const char*, 3>{
             "build/hsaco", "../build/hsaco", "hsaco"}) {
        const std::filesystem::path p =
            std::filesystem::path{rel} / filename;
        if (std::filesystem::exists(p)) {
            return p;
        }
    }

    throw std::runtime_error(
        "compute::hip::GpuMatmul: cannot find " + filename +
        " — set MIMIRMIND_HSACO_DIR or install to " + kDefaultHsacoDir);
}

::mimirmind::core::hip::HipModule loadHipModule(
    ::mimirmind::core::hip::HipContext& ctx, std::string_view name) {
    const auto path = resolveHsacoPath(name);
    MM_LOG_INFO("hip::GpuMatmul", "loading module '{}' from {}",
                std::string{name}, path.string());
    return ::mimirmind::core::hip::HipModule::fromFile(ctx, path.string());
}

[[noreturn]] void throwNotImplemented(const char* method) {
    throw std::runtime_error(
        std::string{"compute::hip::GpuMatmul::"} + method +
        ": not yet implemented (Schritt 3b sub-F skeleton — kernel-launch "
        "impl lands in follow-up commits)");
}

} // namespace

// Pimpl body — 5 Q8_0 matmul-family kernels. Q4_K / Q5_K / Q6_K
// matmul kernels aren't ported to HIP yet; the dispatcher's
// `supports()` returns false for those types so callers know to
// fall back (or the HIP backend refuses to load models with
// unsupported quant weights).
struct GpuMatmul::Impl {
    ::mimirmind::core::hip::HipModule _matmulQ8_0VecModule;
    ::mimirmind::core::hip::HipKernel _matmulQ8_0VecKernel;
    ::mimirmind::core::hip::HipModule _matmulQ8_0GemmModule;
    ::mimirmind::core::hip::HipKernel _matmulQ8_0GemmKernel;
    ::mimirmind::core::hip::HipModule _matmulQ8_0GemmV2Module;
    ::mimirmind::core::hip::HipKernel _matmulQ8_0GemmV2Kernel;
    ::mimirmind::core::hip::HipModule _matmulQ8_0VecDp4aModule;
    ::mimirmind::core::hip::HipKernel _matmulQ8_0VecDp4aKernel;
    ::mimirmind::core::hip::HipModule _moeDownFusedKQ8_0Module;
    ::mimirmind::core::hip::HipKernel _moeDownFusedKQ8_0Kernel;

    explicit Impl(::mimirmind::core::hip::HipContext& ctx)
        : _matmulQ8_0VecModule    {loadHipModule(ctx, "matmul_q8_0_vec")},
          _matmulQ8_0VecKernel    {
              _matmulQ8_0VecModule.getKernel("matmul_q8_0_vec")},
          _matmulQ8_0GemmModule   {loadHipModule(ctx, "matmul_q8_0_gemm")},
          _matmulQ8_0GemmKernel   {
              _matmulQ8_0GemmModule.getKernel("matmul_q8_0_gemm")},
          _matmulQ8_0GemmV2Module {loadHipModule(ctx, "matmul_q8_0_gemm_v2")},
          _matmulQ8_0GemmV2Kernel {
              _matmulQ8_0GemmV2Module.getKernel("matmul_q8_0_gemm_v2")},
          _matmulQ8_0VecDp4aModule{loadHipModule(ctx, "matmul_q8_0_vec_dp4a")},
          _matmulQ8_0VecDp4aKernel{
              _matmulQ8_0VecDp4aModule.getKernel("matmul_q8_0_vec_dp4a")},
          _moeDownFusedKQ8_0Module{loadHipModule(ctx, "moe_down_fused_k_q8_0")},
          _moeDownFusedKQ8_0Kernel{
              _moeDownFusedKQ8_0Module.getKernel("moe_down_fused_k_q8_0")}
    {}
};

GpuMatmul::GpuMatmul(::mimirmind::core::hip::HipComputeContext& ctx,
                     GpuOps& ops)
    : _ctx{ctx},
      _ops{ops},
      _pimpl{std::make_unique<Impl>(ctx.hipContext())}
{
    MM_LOG_INFO("hip::GpuMatmul",
                "compute::hip::GpuMatmul ready — 5 Q8_0 kernels loaded "
                "(vec / gemm / gemm_v2 / vec_dp4a / moe_down_fused_k)");
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
    return type == ::mimirmind::core::gguf::GgmlType::Q8_0;
}

bool GpuMatmul::moeDownFusedKAvailable() const noexcept {
    return true;   // Q8_0 variant is always loaded.
}

bool GpuMatmul::moeDownFusedKAvailable(::mimirmind::core::gguf::GgmlType type)
    const noexcept {
    return type == ::mimirmind::core::gguf::GgmlType::Q8_0;
}

void GpuMatmul::sync() {
    _ctx.stream().synchronize();
}

std::vector<::mimirmind::compute::AutotuneReport>
GpuMatmul::autotuneReport() const {
    // Skeleton: return a single Q8_0 entry with the availability flags
    // set (vec + gemm + dp4a all compiled in) but every timing at 0.
    // Follow-up commit fills in real bench-time populated timings and
    // the gemmMinM cross-over pick. Consumers (SystemStatusBuilder)
    // render 0-ms timings as "not measured" already.
    ::mimirmind::compute::AutotuneReport r{};
    r.name             = "Q8_0";
    r.gemmAvailable    = true;
    r.gemmPicked       = false;
    r.vecMs            = 0.0;
    r.gemmMs           = 0.0;
    r.mBuckets         = ::mimirmind::compute::kAutotuneMBuckets;
    r.gemmMinM         = 0;         // rendered as null in JSON — no threshold
    r.dp4aAvailable    = true;
    r.dp4aPicked       = false;
    r.dp4aMs           = 0.0;
    r.gemmV2Available  = true;
    r.gemmV2Picked     = false;
    r.source           = "pending (hip skeleton)";
    return {r};
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

void GpuMatmul::matmulAsync(::mimirmind::core::gguf::GgmlType type,
                            const void*     W,
                            std::size_t     N,
                            std::size_t     K,
                            const float*    X,
                            std::size_t     M,
                            float*          Y,
                            float*          scratch) {
    (void)scratch;   // Only used by the L0 side's CPU fallback path.

    if (type != ::mimirmind::core::gguf::GgmlType::Q8_0) {
        // Unlike the L0 side, the HIP backend has no CPU fallback wired
        // in yet — only Q8_0 has a HIP kernel. Callers must have gated
        // on `supports(type)` before dispatching here.
        throw std::runtime_error(
            "compute::hip::GpuMatmul::matmulAsync: only Q8_0 is supported "
            "on the HIP backend today — Q4_K / Q5_K / Q6_K HIP kernels "
            "are not ported. Check supports(type) before dispatching.");
    }
    if (M == 0 || N == 0 || K == 0) {
        return;
    }

    // Skeleton scope: always route through the plain vec kernel — one
    // launch per row of X. GEMM / GEMM_V2 / DP4A picks + the M-based
    // gemmMinM crossover come with sub-F.2..F.4. This matches how the
    // L0 side behaves pre-autotune (`gemmMinM = kGemmMinMNever` until
    // `autotune()` runs, so matvec-loop is the safe default).
    //
    // 4 outputs per workgroup (4 subgroups × 16 threads on Xe / RDNA3
    // — matches MATMUL_Q8_0_LOCAL / _SG in matmul_q8_0_vec.hip).
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
                    kLocalSize, 1, 1);
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
    if (type != ::mimirmind::core::gguf::GgmlType::Q8_0) {
        throw std::runtime_error(
            "compute::hip::GpuMatmul::matmulDp4aAsync: only Q8_0 has a "
            "DP4A kernel on the HIP side — check dp4aAvailable(type) "
            "before dispatching");
    }
    if (M == 0 || N == 0 || K == 0) {
        return;
    }

    // Q8_0 block size = 32 elements. The DP4A kernel processes weights
    // block-by-block via `v_dot4_i32_iu8` (gfx1101) / manual byte
    // expansion (any RDNA3 without pure sdot4); K that isn't a
    // multiple of 32 would leave a partial block at the tail, which
    // the kernel can't handle. Same guard as L0 GpuMatmul.
    constexpr std::size_t kBlockElts = 32;
    if (K % kBlockElts != 0) {
        throw std::runtime_error(
            "compute::hip::GpuMatmul::matmulDp4aAsync: K=" +
            std::to_string(K) + " is not a multiple of Q8_0 blockElements="
            + std::to_string(kBlockElts));
    }

    auto& kern = _pimpl->_matmulQ8_0VecDp4aKernel;

    // 4 outputs per workgroup, local=64 — same as matmul_q8_0_vec. WG
    // count in the N dim only; per-row loop iterates over X in the M
    // dim (same shape as L0 side).
    const std::uint32_t nGroups = static_cast<std::uint32_t>(
        (N + kDp4aOutputsPerGroup - 1) / kDp4aOutputsPerGroup);

    // Constant args across all rows — K and N are set once outside
    // the loop, W stays constant, only Xq/Xscale/Y advance per m.
    kern.setValue(4, static_cast<std::int32_t>(K));
    kern.setValue(5, static_cast<std::int32_t>(N));

    for (std::size_t m = 0; m < M; ++m) {
        const std::int8_t* xqRow = Xq + m * K;
        const float*       xsRow = Xscale + m;
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
    if (type != ::mimirmind::core::gguf::GgmlType::Q8_0) {
        throw std::runtime_error(
            "compute::hip::GpuMatmul::moeDownFusedKAsync: only Q8_0 has a "
            "fused-K kernel on the HIP side — check moeDownFusedKAvailable"
            "(type) before dispatching");
    }
    if (kActive == 0 || dModel == 0 || ffPer == 0) {
        return;
    }
    // Q8_0 block = 32 elements. `ffPer` is the inner dim swept per
    // expert; a partial block at the tail would corrupt the
    // dequant accumulator. Same guard as L0 GpuMatmul.
    constexpr std::size_t kBlockElts = 32;
    if (ffPer % kBlockElts != 0) {
        throw std::runtime_error(
            "compute::hip::GpuMatmul::moeDownFusedKAsync: ffPer=" +
            std::to_string(ffPer) + " is not a multiple of Q8_0 "
            "blockElements=" + std::to_string(kBlockElts));
    }

    auto& kern = _pimpl->_moeDownFusedKQ8_0Kernel;
    kern.setPtr  (0, gateAct);
    kern.setPtr  (1, W);
    kern.setPtr  (2, expIdx);
    kern.setPtr  (3, kw);
    kern.setPtr  (4, accum);
    kern.setValue(5, static_cast<std::int32_t>(ffPer));
    kern.setValue(6, static_cast<std::int32_t>(dModel));
    kern.setValue(7, static_cast<std::int32_t>(kActive));
    kern.setValue(8, static_cast<std::int32_t>(expertBytes));

    // 4 outputs per workgroup — same geometry as the plain Q8_0 vec
    // kernel by coincidence (MOE_DOWN_LOCAL=64, SG=16 in the .hip).
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

} // namespace mimirmind::compute::hip