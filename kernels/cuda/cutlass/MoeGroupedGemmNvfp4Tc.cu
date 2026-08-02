// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// M-Cuda.MoeGroup Sub-Step E-d.3 — CUTLASS block-scaled NVFP4 grouped GEMM
// (bf16 output) for Blackwell sm_120/sm_121 (GB10). See the header for the
// operand contract. The type stack is CUTLASS example 79d's Sm120 NVFP4
// grouped mainloop with example 75's plain BF16 LinearCombination epilogue
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

// C / D: BF16, plain LinearCombination epilogue (D = alpha * acc). No SFD.
using ElementC    = cutlass::bfloat16_t;
using ElementD    = cutlass::bfloat16_t;
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

} // namespace

bool nvfp4TcAvailable() noexcept { return true; }

int runGroupedNvfp4TcBf16(
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

int runGroupedNvfp4TcBf16DeviceDriven(
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

#else // no SM120/SM121 support in this build

bool nvfp4TcAvailable() noexcept { return false; }

int runGroupedNvfp4TcBf16(int, const int*, const int*, const int*,
                          const void* const*, const void* const*,
                          const void* const*, const void* const*,
                          const float* const*, void* const*,
                          CUstream_st*) {
    return -1;
}

int runGroupedNvfp4TcBf16DeviceDriven(int, int, int, const int*,
                                      const void* const*, const void* const*,
                                      const void* const*, const void* const*,
                                      const float* const*, void* const*,
                                      CUstream_st*) {
    return -1;
}

#endif

} // namespace mimirmind::kernels::cutlassmoe
