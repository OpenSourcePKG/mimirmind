// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// M-Cuda.MoeGroup Sub-Step E-d.3 — CUTLASS block-scaled NVFP4 grouped GEMM
// (f32 output) for Blackwell sm_120/sm_121 (GB10). See the header for the
// operand contract. The type stack is CUTLASS example 79d's Sm120 NVFP4
// grouped mainloop with example 75's plain F32 LinearCombination epilogue
// (no scale-factor output) — i.e. vLLM's run_fp4_blockwise_scaled_group_mm_sm120
// shape. Compiled as real sm_121a SASS in its own static library.

#include "MoeGroupedGemmNvfp4Tc.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

namespace mimirmind::kernels::cutlassmoe {

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

namespace {

using namespace cute;

using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
using ElementInput = cutlass::float_e2m1_t;

// A / B: NVFP4 (E2M1 + block scale), 32-element FP4 alignment.
using ElementA   = cutlass::nv_float4_t<ElementInput>;
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 32;

using ElementB   = cutlass::nv_float4_t<ElementInput>;
using LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 32;

// C / D: F32, plain LinearCombination epilogue (D = alpha * acc). No SFD.
// F32 output keeps the MoE pipeline (siluMul / act-quant / scatter) single-dtype
// on the device — no bf16<->f32 casts around the grouped GEMM.
using ElementC    = float;
using ElementD    = float;
using LayoutCTag  = cutlass::layout::RowMajor;
using LayoutDTag  = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using ElementAccumulator = float;
using ElementCompute     = float;
using ArchTag            = cutlass::arch::Sm120;
using OperatorClass      = cutlass::arch::OpClassBlockScaledTensorOp;

using ThreadBlockShape = Shape<_128, _128, _128>;
using ClusterShape     = Shape<_1, _1, _1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutCTag*, AlignmentC,
    ElementD, LayoutDTag*, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag*, AlignmentA,
    ElementB, LayoutBTag*, AlignmentB,
    ElementAccumulator,
    ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA   = typename Gemm::GemmKernel::InternalStrideA;
using StrideB   = typename Gemm::GemmKernel::InternalStrideB;
using StrideC   = typename Gemm::GemmKernel::InternalStrideC;
using StrideD   = typename Gemm::GemmKernel::InternalStrideD;
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
using Sm1xxBlkScaledConfig =
    typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

using ElementSF = typename Gemm::GemmKernel::CollectiveMainloop::ElementSF;
using ElementDOut = typename Gemm::EpilogueOutputOp::ElementOutput;
using UnderlyingProblem = typename ProblemShape::UnderlyingProblemShape;

// The collective's operand element is the raw FP4 element (float_e2m1_t)
// exposed as Gemm::ElementA/B, NOT the nv_float4_t<> block-scaled alias.
using GemmElementA = typename Gemm::ElementA;
using GemmElementB = typename Gemm::ElementB;

// RAII device scratch — one cudaFree per allocation, freed after the stream
// sync at the end of the run.
template <typename T>
struct DevBuf {
    T* p{nullptr};
    explicit DevBuf(std::size_t n) {
        if (n) cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(T));
    }
    ~DevBuf() { if (p) cudaFree(p); }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    T* get() const { return p; }
};

template <typename T>
void uploadAsync(T* dst, const T* src, std::size_t n, cudaStream_t s) {
    cudaMemcpyAsync(dst, src, n * sizeof(T), cudaMemcpyHostToDevice, s);
}

// Device builder (E-d.3b): one thread per group builds the per-group
// problem_sizes, strides, and SF layouts from the device expOffset — the only
// M-dependent arrays (M_e = expOffset[e+1]-expOffset[e]). Strides are actually
// M-independent (RowMajor A/D row stride is K/N) and the SFB layout depends
// only on N,K, but building all of them here keeps the host free of M and needs
// no D2H. make_cute_packed_stride and tile_atom_to_shape_SFA/B are
// CUTLASS_HOST_DEVICE, so they lower into the kernel.
__global__ void buildGroupArraysKernel(
    const int* __restrict__ expOffset, int G, int N, int K,
    UnderlyingProblem* __restrict__ problems,
    StrideA* __restrict__ strideA, StrideB* __restrict__ strideB,
    StrideC* __restrict__ strideC, StrideD* __restrict__ strideD,
    LayoutSFA* __restrict__ layoutSFA, LayoutSFB* __restrict__ layoutSFB) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= G) return;
    const int M = expOffset[e + 1] - expOffset[e];
    problems[e]  = UnderlyingProblem{M, N, K};
    strideA[e]   = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    strideB[e]   = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    strideC[e]   = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    strideD[e]   = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});
    layoutSFA[e] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
    layoutSFB[e] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));
}

// Banks builder (E-d.4): additionally computes every per-group operand pointer
// from the contiguous banks + the padded row offsets — nothing on the host.
//   A/SFA/D:  padded row offset padOffset[e] (each expert padded to 128 rows so
//             its SFA sub-tensor is tile-aligned in the one big act-quant SFA).
//   B/SFB:    per-expert index e (weight shape shared, so stride is constant).
//   alpha:    &globalsBank[e] (one F32 weight global per expert).
__global__ void buildGroupArraysBanksKernel(
    const int* __restrict__ expOffset, const int* __restrict__ padOffset,
    int G, int N, int K,
    const unsigned char* __restrict__ aBank, const unsigned char* __restrict__ sfaBank,
    const unsigned char* __restrict__ bBank, const unsigned char* __restrict__ sfbBank,
    const float* __restrict__ globalsBank, unsigned char* __restrict__ dBank,
    UnderlyingProblem* __restrict__ problems,
    const GemmElementA** __restrict__ ptrA, const GemmElementB** __restrict__ ptrB,
    const ElementSF** __restrict__ ptrSFA, const ElementSF** __restrict__ ptrSFB,
    ElementDOut** __restrict__ ptrD, const float** __restrict__ ptrAlpha,
    StrideA* __restrict__ strideA, StrideB* __restrict__ strideB,
    StrideC* __restrict__ strideC, StrideD* __restrict__ strideD,
    LayoutSFA* __restrict__ layoutSFA, LayoutSFB* __restrict__ layoutSFB) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= G) return;
    const int  M       = expOffset[e + 1] - expOffset[e];
    const long padRow  = padOffset[e];
    const long ksfTiles = ((static_cast<long>(K) / 16) + 3) / 4;
    const long nMTiles  = (N + 127) / 128;
    const long sfbStride = 512L * nMTiles * ksfTiles;          // moeSwizzledScaleStride(N,K/16)
    const long sfaTileOff = 512L * ksfTiles * (padRow / 128);  // big-SFA tile of expert e

    problems[e]  = UnderlyingProblem{M, N, K};
    ptrA[e]      = reinterpret_cast<const GemmElementA*>(aBank + padRow * (static_cast<long>(K) / 2));
    ptrSFA[e]    = reinterpret_cast<const ElementSF*>(sfaBank + sfaTileOff);
    ptrB[e]      = reinterpret_cast<const GemmElementB*>(bBank + static_cast<long>(e) * N * (static_cast<long>(K) / 2));
    ptrSFB[e]    = reinterpret_cast<const ElementSF*>(sfbBank + static_cast<long>(e) * sfbStride);
    ptrD[e]      = reinterpret_cast<ElementDOut*>(dBank + padRow * N * static_cast<long>(sizeof(ElementDOut)));
    ptrAlpha[e]  = globalsBank + e;
    strideA[e]   = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    strideB[e]   = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    strideC[e]   = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    strideD[e]   = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});
    layoutSFA[e] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
    layoutSFB[e] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));
}

// --- E-d.4c: pre-allocated scratch layout for the banks GEMM ---------------
// The per-group CUTLASS arrays + the CUTLASS workspace live in one caller-owned
// buffer so the banks GEMM does no per-call cudaMalloc and never syncs (the
// caller's stream ordering + persistent scratch keep it host-sync-free — the
// whole point of the FP4-TC path).
inline std::size_t alignUp(std::size_t x, std::size_t a) { return (x + a - 1) / a * a; }

struct BanksScratch {
    UnderlyingProblem*  problems{nullptr};
    const GemmElementA** pA{nullptr};
    const GemmElementB** pB{nullptr};
    const ElementSF**   pSFA{nullptr};
    const ElementSF**   pSFB{nullptr};
    ElementDOut**       pD{nullptr};
    const float**       pAlpha{nullptr};
    StrideA*   sA{nullptr};
    StrideB*   sB{nullptr};
    StrideC*   sC{nullptr};
    StrideD*   sD{nullptr};
    LayoutSFA* lSFA{nullptr};
    LayoutSFB* lSFB{nullptr};
    std::uint8_t* workspace{nullptr};
    std::size_t total{0};
};

// Assigns sub-pointers into `base` (or, with base==nullptr, only measures
// `total`). `groups` and `wsBytes` fully determine the layout.
inline BanksScratch layoutBanksScratch(void* base, int groups, std::size_t wsBytes) {
    BanksScratch s{};
    std::size_t off = 0;
    auto* const b = static_cast<std::uint8_t*>(base);
    auto take = [&](std::size_t bytes) -> void* {
        off = alignUp(off, 256);
        void* p = b ? (b + off) : nullptr;
        off += bytes;
        return p;
    };
    const std::size_t g = static_cast<std::size_t>(groups);
    s.problems = static_cast<UnderlyingProblem*>(take(g * sizeof(UnderlyingProblem)));
    s.pA       = static_cast<const GemmElementA**>(take(g * sizeof(void*)));
    s.pB       = static_cast<const GemmElementB**>(take(g * sizeof(void*)));
    s.pSFA     = static_cast<const ElementSF**>(take(g * sizeof(void*)));
    s.pSFB     = static_cast<const ElementSF**>(take(g * sizeof(void*)));
    s.pD       = static_cast<ElementDOut**>(take(g * sizeof(void*)));
    s.pAlpha   = static_cast<const float**>(take(g * sizeof(void*)));
    s.sA       = static_cast<StrideA*>(take(g * sizeof(StrideA)));
    s.sB       = static_cast<StrideB*>(take(g * sizeof(StrideB)));
    s.sC       = static_cast<StrideC*>(take(g * sizeof(StrideC)));
    s.sD       = static_cast<StrideD*>(take(g * sizeof(StrideD)));
    s.lSFA     = static_cast<LayoutSFA*>(take(g * sizeof(LayoutSFA)));
    s.lSFB     = static_cast<LayoutSFB*>(take(g * sizeof(LayoutSFB)));
    s.workspace = static_cast<std::uint8_t*>(take(wsBytes ? wsBytes : 1));
    s.total    = alignUp(off, 256);
    return s;
}

// CUTLASS grouped workspace bytes for `groups` (host-only; group count +
// sm_count, does not read device data — safe with null/stub pointers).
inline std::size_t banksWorkspaceBytes(int groups) {
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count  = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
    typename Gemm::Arguments args;
    decltype(args.epilogue.thread) fusion_args;
    fusion_args.alpha = 0; fusion_args.beta = 0;
    fusion_args.alpha_ptr = nullptr; fusion_args.beta_ptr = nullptr;
    fusion_args.alpha_ptr_array = nullptr; fusion_args.beta_ptr_array = nullptr;
    fusion_args.dAlpha = {_0{}, _0{}, 1}; fusion_args.dBeta = {_0{}, _0{}, 0};
    typename Gemm::GemmKernel::TileSchedulerArguments scheduler;
    args = typename Gemm::Arguments{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {groups, nullptr, nullptr},
        {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr},
        {fusion_args, nullptr, nullptr, nullptr, nullptr},
        hw_info, scheduler};
    return Gemm::get_workspace_size(args);
}

} // namespace

bool nvfp4TcAvailable() noexcept { return true; }

int runGroupedNvfp4TcF32(
    int                  groups,
    const int*           mHost,
    const int*           nHost,
    const int*           kHost,
    const void* const*   dPtrA,
    const void* const*   dPtrSFA,
    const void* const*   dPtrB,
    const void* const*   dPtrSFB,
    const float* const*  dPtrAlpha,
    void* const*         dPtrD,
    CUstream_st*         streamRaw) {

    if (groups <= 0) return 0;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(streamRaw);

    // Per-group host problem shapes + strides + SF layouts (needs M,N,K).
    std::vector<UnderlyingProblem> problems(groups);
    std::vector<StrideA>   sA(groups);
    std::vector<StrideB>   sB(groups);
    std::vector<StrideC>   sC(groups);
    std::vector<StrideD>   sD(groups);
    std::vector<LayoutSFA> lSFA(groups);
    std::vector<LayoutSFB> lSFB(groups);

    for (int e = 0; e < groups; ++e) {
        const int M = mHost[e], N = nHost[e], K = kHost[e];
        problems[e] = UnderlyingProblem{M, N, K};
        sA[e] = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
        sB[e] = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
        sC[e] = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
        sD[e] = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});
        lSFA[e] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
        lSFB[e] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));
    }

    // Device arrays consumed by the grouped kernel.
    DevBuf<UnderlyingProblem>   dProblems(groups);
    DevBuf<const GemmElementA*> dA(groups);
    DevBuf<const GemmElementB*> dB(groups);
    DevBuf<const ElementSF*>    dSFA(groups);
    DevBuf<const ElementSF*>    dSFB(groups);
    DevBuf<ElementDOut*>        dD(groups);
    DevBuf<StrideA>             dsA(groups);
    DevBuf<StrideB>             dsB(groups);
    DevBuf<StrideC>             dsC(groups);
    DevBuf<StrideD>             dsD(groups);
    DevBuf<LayoutSFA>           dlSFA(groups);
    DevBuf<LayoutSFB>           dlSFB(groups);

    const auto n = static_cast<std::size_t>(groups);
    uploadAsync(dProblems.get(), problems.data(), n, stream);
    uploadAsync(dsA.get(), sA.data(), n, stream);
    uploadAsync(dsB.get(), sB.data(), n, stream);
    uploadAsync(dsC.get(), sC.data(), n, stream);
    uploadAsync(dsD.get(), sD.data(), n, stream);
    uploadAsync(dlSFA.get(), lSFA.data(), n, stream);
    uploadAsync(dlSFB.get(), lSFB.data(), n, stream);
    // The caller-supplied dPtr* arrays are device arrays of device pointers,
    // bit-identical to the CUTLASS pointer-of-pointer operands.
    cudaMemcpyAsync(dA.get(),   dPtrA,   n * sizeof(void*), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(dB.get(),   dPtrB,   n * sizeof(void*), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(dSFA.get(), dPtrSFA, n * sizeof(void*), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(dSFB.get(), dPtrSFB, n * sizeof(void*), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(dD.get(),   dPtrD,   n * sizeof(void*), cudaMemcpyDeviceToDevice, stream);

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count  = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);

    typename Gemm::Arguments arguments;
    decltype(arguments.epilogue.thread) fusion_args;
    fusion_args.alpha = 0;
    fusion_args.beta  = 0;
    fusion_args.alpha_ptr = nullptr;
    fusion_args.beta_ptr  = nullptr;
    fusion_args.alpha_ptr_array = dPtrAlpha;      // one F32 alpha per group
    fusion_args.beta_ptr_array  = nullptr;
    fusion_args.dAlpha = {_0{}, _0{}, 1};
    fusion_args.dBeta  = {_0{}, _0{}, 0};

    typename Gemm::GemmKernel::TileSchedulerArguments scheduler;

    arguments = typename Gemm::Arguments{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {groups, dProblems.get(), problems.data()},
        {dA.get(), dsA.get(), dB.get(), dsB.get(),
         dSFA.get(), dlSFA.get(), dSFB.get(), dlSFB.get()},
        {fusion_args, nullptr, dsC.get(), dD.get(), dsD.get()},
        hw_info, scheduler};

    Gemm gemm;
    const std::size_t wsBytes = Gemm::get_workspace_size(arguments);
    DevBuf<std::uint8_t> workspace(wsBytes ? wsBytes : 1);

    cutlass::Status st = gemm.can_implement(arguments);
    if (st != cutlass::Status::kSuccess) {
        std::fprintf(stderr, "[nvfp4-tc] can_implement: %s\n",
                     cutlassGetStatusString(st));
        cudaStreamSynchronize(stream);
        return 1;
    }
    st = gemm.initialize(arguments, workspace.get(), stream);
    if (st != cutlass::Status::kSuccess) {
        std::fprintf(stderr, "[nvfp4-tc] initialize: %s\n", cutlassGetStatusString(st));
        cudaStreamSynchronize(stream);
        return 2;
    }
    st = gemm.run(stream);
    if (st != cutlass::Status::kSuccess) {
        std::fprintf(stderr, "[nvfp4-tc] run: %s\n", cutlassGetStatusString(st));
        cudaStreamSynchronize(stream);
        return 3;
    }
    // Sync so the per-group scratch DevBufs are safe to free.
    cudaStreamSynchronize(stream);
    return 0;
}

int runGroupedNvfp4TcF32DeviceDriven(
    int                  groups,
    int                  N,
    int                  K,
    const int*           dExpOffset,
    const void* const*   dPtrA,
    const void* const*   dPtrSFA,
    const void* const*   dPtrB,
    const void* const*   dPtrSFB,
    const float* const*  dPtrAlpha,
    void* const*         dPtrD,
    CUstream_st*         streamRaw) {

    if (groups <= 0) return 0;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(streamRaw);

    // Per-group problem_sizes / strides / SF layouts, built on the device from
    // dExpOffset — nothing crosses to the host.
    DevBuf<UnderlyingProblem> dProblems(groups);
    DevBuf<StrideA>   dsA(groups);
    DevBuf<StrideB>   dsB(groups);
    DevBuf<StrideC>   dsC(groups);
    DevBuf<StrideD>   dsD(groups);
    DevBuf<LayoutSFA> dlSFA(groups);
    DevBuf<LayoutSFB> dlSFB(groups);

    const int threads = 64;
    const int blocks  = (groups + threads - 1) / threads;
    buildGroupArraysKernel<<<blocks, threads, 0, stream>>>(
        dExpOffset, groups, N, K,
        dProblems.get(), dsA.get(), dsB.get(), dsC.get(), dsD.get(),
        dlSFA.get(), dlSFB.get());

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count  = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);

    typename Gemm::Arguments arguments;
    decltype(arguments.epilogue.thread) fusion_args;
    fusion_args.alpha = 0;
    fusion_args.beta  = 0;
    fusion_args.alpha_ptr = nullptr;
    fusion_args.beta_ptr  = nullptr;
    fusion_args.alpha_ptr_array = dPtrAlpha;
    fusion_args.beta_ptr_array  = nullptr;
    fusion_args.dAlpha = {_0{}, _0{}, 1};
    fusion_args.dBeta  = {_0{}, _0{}, 0};

    typename Gemm::GemmKernel::TileSchedulerArguments scheduler;

    // C-style casts: the operand arrays are device arrays of device pointers,
    // bit-identical to the CUTLASS pointer-of-pointer operands; combine the
    // const strip + reinterpret in one (reinterpret_cast alone can't strip).
    auto ptrA   = (const GemmElementA**)(dPtrA);
    auto ptrB   = (const GemmElementB**)(dPtrB);
    auto ptrSFA = (const ElementSF**)(dPtrSFA);
    auto ptrSFB = (const ElementSF**)(dPtrSFB);
    auto ptrD   = (ElementDOut**)(dPtrD);

    arguments = typename Gemm::Arguments{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {groups, dProblems.get(), nullptr},   // host shapes unavailable -> device
        {ptrA, dsA.get(), ptrB, dsB.get(),
         ptrSFA, dlSFA.get(), ptrSFB, dlSFB.get()},
        {fusion_args, nullptr, dsC.get(), ptrD, dsD.get()},
        hw_info, scheduler};

    Gemm gemm;
    const std::size_t wsBytes = Gemm::get_workspace_size(arguments);
    DevBuf<std::uint8_t> workspace(wsBytes ? wsBytes : 1);

    cutlass::Status st = gemm.can_implement(arguments);
    if (st != cutlass::Status::kSuccess) {
        std::fprintf(stderr, "[nvfp4-tc-dev] can_implement: %s\n",
                     cutlassGetStatusString(st));
        cudaStreamSynchronize(stream);
        return 1;
    }
    st = gemm.initialize(arguments, workspace.get(), stream);
    if (st != cutlass::Status::kSuccess) {
        std::fprintf(stderr, "[nvfp4-tc-dev] initialize: %s\n", cutlassGetStatusString(st));
        cudaStreamSynchronize(stream);
        return 2;
    }
    st = gemm.run(stream);
    if (st != cutlass::Status::kSuccess) {
        std::fprintf(stderr, "[nvfp4-tc-dev] run: %s\n", cutlassGetStatusString(st));
        cudaStreamSynchronize(stream);
        return 3;
    }
    // Sync so the device scratch DevBufs are safe to free. E-d.4 will use a
    // pre-allocated scratch + no sync (the true host-sync-free runtime path).
    cudaStreamSynchronize(stream);
    return 0;
}

int runGroupedNvfp4TcF32Banks(
    int                  groups,
    int                  N,
    int                  K,
    const int*           dExpOffset,
    const int*           dPadOffset,
    const void*          aBank,
    const void*          sfaBank,
    const void*          bBank,
    const void*          sfbBank,
    const float*         globalsBank,
    void*                dBank,
    void*                scratch,
    std::size_t          scratchBytes,
    CUstream_st*         streamRaw) {

    if (groups <= 0) return 0;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(streamRaw);

    const std::size_t wsBytes = banksWorkspaceBytes(groups);
    BanksScratch sc = layoutBanksScratch(scratch, groups, wsBytes);
    if (scratch == nullptr || scratchBytes < sc.total) {
        std::fprintf(stderr, "[nvfp4-tc-banks] scratch too small: have %zu need %zu\n",
                     scratchBytes, sc.total);
        return 4;
    }

    const int threads = 64;
    const int blocks  = (groups + threads - 1) / threads;
    buildGroupArraysBanksKernel<<<blocks, threads, 0, stream>>>(
        dExpOffset, dPadOffset, groups, N, K,
        static_cast<const unsigned char*>(aBank), static_cast<const unsigned char*>(sfaBank),
        static_cast<const unsigned char*>(bBank), static_cast<const unsigned char*>(sfbBank),
        globalsBank, static_cast<unsigned char*>(dBank),
        sc.problems, sc.pA, sc.pB, sc.pSFA, sc.pSFB, sc.pD, sc.pAlpha,
        sc.sA, sc.sB, sc.sC, sc.sD, sc.lSFA, sc.lSFB);

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count  = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);

    typename Gemm::Arguments arguments;
    decltype(arguments.epilogue.thread) fusion_args;
    fusion_args.alpha = 0;
    fusion_args.beta  = 0;
    fusion_args.alpha_ptr = nullptr;
    fusion_args.beta_ptr  = nullptr;
    fusion_args.alpha_ptr_array = sc.pAlpha;
    fusion_args.beta_ptr_array  = nullptr;
    fusion_args.dAlpha = {_0{}, _0{}, 1};
    fusion_args.dBeta  = {_0{}, _0{}, 0};

    typename Gemm::GemmKernel::TileSchedulerArguments scheduler;

    arguments = typename Gemm::Arguments{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {groups, sc.problems, nullptr},
        {sc.pA, sc.sA, sc.pB, sc.sB,
         sc.pSFA, sc.lSFA, sc.pSFB, sc.lSFB},
        {fusion_args, nullptr, sc.sC, sc.pD, sc.sD},
        hw_info, scheduler};

    Gemm gemm;
    // No per-call cudaMalloc, no sync: the scratch + workspace persist across
    // calls (caller-owned) and the stream ordering serialises the builder ->
    // GEMM chain against the surrounding MoE ops.
    cutlass::Status st = gemm.can_implement(arguments);
    if (st != cutlass::Status::kSuccess) {
        std::fprintf(stderr, "[nvfp4-tc-banks] can_implement: %s\n", cutlassGetStatusString(st));
        return 1;
    }
    st = gemm.initialize(arguments, sc.workspace, stream);
    if (st != cutlass::Status::kSuccess) {
        std::fprintf(stderr, "[nvfp4-tc-banks] initialize: %s\n", cutlassGetStatusString(st));
        return 2;
    }
    st = gemm.run(stream);
    if (st != cutlass::Status::kSuccess) {
        std::fprintf(stderr, "[nvfp4-tc-banks] run: %s\n", cutlassGetStatusString(st));
        return 3;
    }
    return 0;
}

std::size_t groupedNvfp4TcBanksScratchBytes(int groups) {
    if (groups <= 0) return 0;
    return layoutBanksScratch(nullptr, groups, banksWorkspaceBytes(groups)).total;
}

#else // no SM120/SM121 support in this build

bool nvfp4TcAvailable() noexcept { return false; }

int runGroupedNvfp4TcF32(int, const int*, const int*, const int*,
                          const void* const*, const void* const*,
                          const void* const*, const void* const*,
                          const float* const*, void* const*,
                          CUstream_st*) {
    return -1;
}

int runGroupedNvfp4TcF32DeviceDriven(int, int, int, const int*,
                                      const void* const*, const void* const*,
                                      const void* const*, const void* const*,
                                      const float* const*, void* const*,
                                      CUstream_st*) {
    return -1;
}

int runGroupedNvfp4TcF32Banks(int, int, int, const int*, const int*,
                               const void*, const void*, const void*,
                               const void*, const float*, void*,
                               void*, std::size_t, CUstream_st*) {
    return -1;
}

std::size_t groupedNvfp4TcBanksScratchBytes(int) { return 0; }

#endif

} // namespace mimirmind::kernels::cutlassmoe
