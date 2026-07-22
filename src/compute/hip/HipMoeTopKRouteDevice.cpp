// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "compute/hip/HipMoeTopKRouteDevice.hpp"

#include "core/gpu/hip/HipComputeContext.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>

namespace mimirmind::compute::hip {
namespace {

// Resolve "<name>.hsaco" the same three-tier way hip::GpuOps does: env
// override, install dir, then in-tree build fallbacks. Kept local so this
// module needs no shared helper (GpuOps' resolveHsacoPath is anonymous).
std::filesystem::path resolveHsaco(std::string_view name) {
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
            std::filesystem::path{"/usr/local/share/mimirmind/hsaco"} / filename;
        if (std::filesystem::exists(p)) {
            return p;
        }
    }
    for (const auto* rel : std::array<const char*, 5>{
             "build/hsaco", "build-both/hsaco", "../build/hsaco",
             "../build-both/hsaco", "hsaco"}) {
        const std::filesystem::path p = std::filesystem::path{rel} / filename;
        if (std::filesystem::exists(p)) {
            return p;
        }
    }
    throw std::runtime_error(
        "compute::hip::HipMoeTopKRouteDevice: cannot find " + filename +
        " — set MIMIRMIND_HSACO_DIR or install to "
        "/usr/local/share/mimirmind/hsaco");
}

core::hip::HipModule loadModule(core::hip::HipContext& ctx,
                                std::string_view name) {
    return core::hip::HipModule::fromFile(ctx, resolveHsaco(name).string());
}

} // namespace

HipMoeTopKRouteDevice::HipMoeTopKRouteDevice(core::hip::HipComputeContext& ctx)
    : _ctx{ctx},
      _module{loadModule(ctx.hipContext(), "moe_topk")},
      _kernel{_module.getKernel("moe_topk")} {}

void HipMoeTopKRouteDevice::launch(const float*  logits,
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
            "compute::hip::HipMoeTopKRouteDevice::launch: nExperts/K exceed "
            "the kernel ceilings (nExperts=" + std::to_string(nExperts) +
            " max " + std::to_string(kMaxExperts) + ", K=" +
            std::to_string(K) + " max " + std::to_string(kMaxK) + ") — bump "
            "both moe_topk.hip and kMaxExperts/kMaxK together");
    }

    // Args match the moe_topk kernel signature exactly.
    _kernel.setPtr  (0, logits);
    _kernel.setPtr  (1, outIdx);
    _kernel.setPtr  (2, outWeight);
    _kernel.setValue(3, static_cast<std::int32_t>(nExperts));
    _kernel.setValue(4, static_cast<std::int32_t>(K));
    _kernel.setValue(5, wScale);

    // Grid: one work-group per token. Block: 32 (warp-aligned; only thread 0
    // computes in the v1 kernel). No dynamic shared memory.
    _kernel.launch(_ctx.stream(),
                   static_cast<std::uint32_t>(T), 1, 1,
                   32, 1, 1, /*shared=*/0);
}

} // namespace mimirmind::compute::hip