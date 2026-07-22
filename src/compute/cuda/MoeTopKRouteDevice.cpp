// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "compute/cuda/MoeTopKRouteDevice.hpp"

#include "core/gpu/cuda/CudaComputeContext.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>

namespace mimirmind::compute::cuda {
namespace {

// Resolve a "<name>.ptx" the same way GpuOps/GpuMatmul do: env override, the
// install dir, then the in-tree build dirs. Kept local so this module needs
// no shared helper (GpuOps' resolveHsacoPath is in an anonymous namespace).
std::filesystem::path resolvePtx(std::string_view name) {
    const std::string filename = std::string{name} + ".ptx";

    if (const char* env = std::getenv("MIMIRMIND_HSACO_DIR")) {
        const std::filesystem::path p = std::filesystem::path{env} / filename;
        if (std::filesystem::exists(p)) {
            return p;
        }
    }
    {
        const std::filesystem::path p =
            std::filesystem::path{"/usr/local/share/mimirmind/ptx"} / filename;
        if (std::filesystem::exists(p)) {
            return p;
        }
    }
    for (const auto* rel : std::array<const char*, 5>{
             "build/ptx", "build-both/ptx", "../build/ptx",
             "../build-both/ptx", "ptx"}) {
        const std::filesystem::path p = std::filesystem::path{rel} / filename;
        if (std::filesystem::exists(p)) {
            return p;
        }
    }
    throw std::runtime_error(
        "compute::cuda::MoeTopKRouteDevice: cannot find " + filename +
        " — set MIMIRMIND_HSACO_DIR or install to "
        "/usr/local/share/mimirmind/ptx");
}

core::cuda::CudaModule loadModule(core::cuda::CudaContext& ctx,
                                  std::string_view name) {
    return core::cuda::CudaModule::fromFile(ctx, resolvePtx(name).string());
}

} // namespace

MoeTopKRouteDevice::MoeTopKRouteDevice(core::cuda::CudaComputeContext& ctx)
    : _ctx{ctx},
      _module{loadModule(ctx.cudaContext(), "moe_topk")},
      _kernel{_module.getFunction("moe_topk")} {}

void MoeTopKRouteDevice::launch(const float*  logits,
                                std::int32_t* outIdx,
                                float*        outWeight,
                                std::size_t   T,
                                std::size_t   nExperts,
                                std::size_t   K,
                                float         wScale) {
    if (T == 0 || nExperts == 0 || K == 0) {
        return;
    }
    if (nExperts > kMaxExperts || K > kMaxK) {
        throw std::runtime_error(
            "compute::cuda::MoeTopKRouteDevice::launch: nExperts/K exceed the "
            "kernel ceilings (nExperts=" + std::to_string(nExperts) + " max " +
            std::to_string(kMaxExperts) + ", K=" + std::to_string(K) +
            " max " + std::to_string(kMaxK) + ") — bump both moe_topk.cu and "
            "kMaxExperts/kMaxK together");
    }

    // Args match the moe_topk kernel signature exactly.
    _kernel.setPtr  (0, logits);
    _kernel.setPtr  (1, outIdx);
    _kernel.setPtr  (2, outWeight);
    _kernel.setValue(3, static_cast<std::int32_t>(nExperts));
    _kernel.setValue(4, static_cast<std::int32_t>(K));
    _kernel.setValue(5, wScale);

    // Grid: one block per token. Block: 32 (warp-aligned; only thread 0
    // computes in the v1 kernel). No dynamic shared memory.
    _kernel.launch(_ctx.stream(),
                   static_cast<std::uint32_t>(T), 1, 1,
                   32, 1, 1);
}

} // namespace mimirmind::compute::cuda