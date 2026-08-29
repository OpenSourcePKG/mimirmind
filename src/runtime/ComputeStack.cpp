// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/ComputeStack.hpp"

#include "compute/ComputeMatmul.hpp"
#include "compute/ComputeOps.hpp"
#include "compute/cpu/GpuMatmul.hpp"
#include "compute/cpu/GpuOps.hpp"
#include "core/config/Config.hpp"
#include "core/cpu/CpuContext.hpp"

#include <stdexcept>
#include <string>

#ifdef MIMIRMIND_HAVE_L0
#include "compute/l0/GpuMatmul.hpp"
#include "compute/l0/GpuOps.hpp"
#include "core/gpu/l0/L0ComputeContext.hpp"
#endif

#ifdef MIMIRMIND_HAVE_HIP
#include "compute/hip/GpuMatmul.hpp"
#include "compute/hip/GpuOps.hpp"
#include "core/gpu/hip/HipComputeContext.hpp"
#endif

#ifdef MIMIRMIND_HAVE_CUDA
#include "compute/cuda/GpuMatmul.hpp"
#include "compute/cuda/GpuOps.hpp"
#include "core/gpu/cuda/CudaComputeContext.hpp"
#endif

namespace mimirmind::runtime {

namespace {

std::unique_ptr<core::backend::ComputeContext>
makeContext(const core::config::Config& cfg, core::backend::BackendKind kind) {
    switch (kind) {
        case core::backend::BackendKind::LevelZero:
#ifdef MIMIRMIND_HAVE_L0
            return std::make_unique<core::l0::L0ComputeContext>(
                core::l0::L0ComputeContext::Options{
                    .spvDirOverride   = std::string{cfg.runtime.spvDir.value_or("")},
                    .usmProbeTotalGiB = cfg.runtime.usmProbeTotalGib,
                    .usmKindOverride  = std::nullopt});
#else
            break;
#endif
        case core::backend::BackendKind::Hip:
#ifdef MIMIRMIND_HAVE_HIP
            (void)cfg;
            return std::make_unique<core::hip::HipComputeContext>();
#else
            break;
#endif
        case core::backend::BackendKind::Cuda:
#ifdef MIMIRMIND_HAVE_CUDA
            (void)cfg;
            return std::make_unique<core::cuda::CudaComputeContext>();
#else
            break;
#endif
        case core::backend::BackendKind::Cpu:
            (void)cfg;
            return std::make_unique<core::cpu::CpuContext>();
        default:
            break;
    }
    throw std::runtime_error{
        "makeComputeStack: backend kind not compiled in for this build"};
}

std::unique_ptr<compute::ComputeOps>
makeOps(core::backend::ComputeContext& ctx,
        const core::config::FeatureSettings& features) {
    switch (ctx.kind()) {
        case core::backend::BackendKind::LevelZero:
#ifdef MIMIRMIND_HAVE_L0
            return std::make_unique<compute::l0::GpuOps>(
                static_cast<core::l0::L0ComputeContext&>(ctx),
                features.flashPrefill, features.flashPrefillGqaQ8,
                features.flashPrefillKTileQ8, features.q8_0Reorder);
#else
            break;
#endif
        case core::backend::BackendKind::Hip:
#ifdef MIMIRMIND_HAVE_HIP
            return std::make_unique<compute::hip::GpuOps>(
                static_cast<core::hip::HipComputeContext&>(ctx),
                features.flashPrefill, features.flashPrefillGqaQ8,
                features.flashPrefillKTileQ8, features.q8_0Reorder);
#else
            break;
#endif
        case core::backend::BackendKind::Cuda:
#ifdef MIMIRMIND_HAVE_CUDA
            return std::make_unique<compute::cuda::GpuOps>(
                static_cast<core::cuda::CudaComputeContext&>(ctx),
                features.flashPrefill, features.flashPrefillGqaQ8,
                features.flashPrefillGqaQ8Bq,
                features.flashPrefillKTileQ8, features.q8_0Reorder);
#else
            break;
#endif
        case core::backend::BackendKind::Cpu:
            (void)features;
            return std::make_unique<compute::cpu::GpuOps>(
                static_cast<core::cpu::CpuContext&>(ctx));
        default:
            break;
    }
    throw std::runtime_error{"makeComputeStack: no ComputeOps for backend kind"};
}

std::unique_ptr<compute::ComputeMatmul>
makeMatmul(core::backend::ComputeContext& ctx, compute::ComputeOps& ops) {
    switch (ctx.kind()) {
        case core::backend::BackendKind::LevelZero:
#ifdef MIMIRMIND_HAVE_L0
            return std::make_unique<compute::l0::GpuMatmul>(
                static_cast<core::l0::L0ComputeContext&>(ctx),
                static_cast<compute::l0::GpuOps&>(ops));
#else
            break;
#endif
        case core::backend::BackendKind::Hip:
#ifdef MIMIRMIND_HAVE_HIP
            return std::make_unique<compute::hip::GpuMatmul>(
                static_cast<core::hip::HipComputeContext&>(ctx),
                static_cast<compute::hip::GpuOps&>(ops));
#else
            break;
#endif
        case core::backend::BackendKind::Cuda:
#ifdef MIMIRMIND_HAVE_CUDA
            return std::make_unique<compute::cuda::GpuMatmul>(
                static_cast<core::cuda::CudaComputeContext&>(ctx),
                static_cast<compute::cuda::GpuOps&>(ops));
#else
            break;
#endif
        case core::backend::BackendKind::Cpu:
            (void)ops;
            return std::make_unique<compute::cpu::GpuMatmul>(
                static_cast<core::cpu::CpuContext&>(ctx));
        default:
            break;
    }
    throw std::runtime_error{"makeComputeStack: no ComputeMatmul for backend kind"};
}

} // namespace

ComputeStack makeComputeStack(const core::config::Config& cfg,
                              core::backend::BackendKind  kind) {
    ComputeStack s{};
    s.context = makeContext(cfg, kind);
    s.ops     = makeOps(*s.context, cfg.features);
    s.matmul  = makeMatmul(*s.context, *s.ops);
    return s;
}

} // namespace mimirmind::runtime
