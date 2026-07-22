// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// CUDA GPU-vs-CPU parity test for the M-Q3N.5 device-side MoE top-K router
// (kernels_cuda/moe_topk.cu). Runs the `moe_topk` kernel against the CPU
// golden reference `compute::moeTopKRoute` on random router logits and
// asserts the kept expert indices match exactly and the renormalised weights
// match within fp tolerance.
//
// This is the correctness gate before the kernel gets wired into
// Qwen35MoeBackend (replacing the host top-K + the host->USM copy loop).
// The kernel is launched directly through the driver API (CudaModule /
// CudaKernel) so this test needs no new GpuOps / GpuMatmul entry point --
// it only uses GpuOps for the generic allocate / upload / readback path.
//
// Built and run only on a CUDA host (MIMIRMIND_ENABLE_CUDA).
//
// Tie-break note: `moeTopKRoute` leaves equal-probability ties in
// implementation-defined order; `moe_topk` fixes them to lowest-index-wins.
// With continuous random logits an exact tie has ~zero probability, so index
// parity holds here regardless. Aligning the CPU reference to the same fixed
// rule is a separate change that lands with the backend wiring.

#include "TestFramework.hpp"

#include "compute/MoeRouting.hpp"
#include "compute/cuda/GpuOps.hpp"
#include "core/gpu/cuda/CudaComputeContext.hpp"
#include "core/gpu/cuda/CudaKernel.hpp"
#include "core/gpu/cuda/CudaModule.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ::mimirmind::compute::cuda::GpuOps;
using ::mimirmind::core::cuda::CudaComputeContext;
using ::mimirmind::core::cuda::CudaKernel;
using ::mimirmind::core::cuda::CudaModule;

// Deterministic pseudo-random in [-1, 1) -- no Date/rand, reproducible.
struct Lcg {
    std::uint32_t s;
    float next() {
        s = s * 1664525u + 1013904223u;
        return static_cast<float>((s >> 8) & 0xFFFFu) / 32768.0f - 1.0f;
    }
};

std::vector<float> randVec(std::size_t n, std::uint32_t seed) {
    Lcg g{seed};
    std::vector<float> v(n);
    for (auto& x : v) {
        x = g.next();
    }
    return v;
}

// Locate moe_topk.ptx the same way GpuMatmul's resolveHsacoPath does.
std::string resolvePtx() {
    const char* env = std::getenv("MIMIRMIND_HSACO_DIR");
    std::array<std::string, 6> dirs{
        env ? std::string{env} : std::string{},
        "build/ptx", "build-both/ptx", "../build/ptx", "../build-both/ptx",
        "ptx"};
    for (const auto& d : dirs) {
        if (d.empty()) {
            continue;
        }
        std::filesystem::path p = std::filesystem::path{d} / "moe_topk.ptx";
        if (std::filesystem::exists(p)) {
            return p.string();
        }
    }
    throw std::runtime_error(
        "cuda_moe_topk_parity: cannot find moe_topk.ptx -- build the CUDA "
        "kernels or set MIMIRMIND_HSACO_DIR");
}

// Run the device kernel for one [T, nExperts] logit batch; return
// (idx[T*K], weight[T*K]).
struct DeviceResult {
    std::vector<std::int32_t> idx;
    std::vector<float>        weight;
};

DeviceResult runDevice(CudaComputeContext& ctx, GpuOps& ops, CudaKernel& kern,
                       const std::vector<float>& logits,
                       std::size_t T, std::size_t nExperts, std::size_t K,
                       float wScale) {
    auto dLogits = ops.allocate(logits.size() * sizeof(float));
    ops.uploadHostBytes(dLogits.get(), logits.data(),
                        logits.size() * sizeof(float));
    auto dIdx = ops.allocate(T * K * sizeof(std::int32_t));
    auto dW   = ops.allocate(T * K * sizeof(float));

    kern.setPtr  (0, dLogits.get());
    kern.setPtr  (1, dIdx.get());
    kern.setPtr  (2, dW.get());
    kern.setValue(3, static_cast<std::int32_t>(nExperts));
    kern.setValue(4, static_cast<std::int32_t>(K));
    kern.setValue(5, wScale);
    kern.launch(ctx.stream(),
                static_cast<std::uint32_t>(T), 1, 1,   // grid: one block/token
                32, 1, 1);                             // block: warp-aligned
    ops.flush();

    DeviceResult r;
    r.idx.resize(T * K);
    r.weight.resize(T * K);
    ops.readbackToHost(r.idx.data(), dIdx.get(), T * K * sizeof(std::int32_t));
    ops.readbackToHost(r.weight.data(), dW.get(), T * K * sizeof(float));
    return r;
}

void checkParity(std::size_t T, std::size_t nExperts, std::size_t K,
                 float wScale, std::uint32_t seed) {
    CudaComputeContext ctx{};
    GpuOps ops{ctx};
    CudaModule mod = CudaModule::fromFile(ctx.cudaContext(), resolvePtx());
    CudaKernel kern = mod.getFunction("moe_topk");

    const auto logits = randVec(T * nExperts, seed);

    // CPU golden reference (weights renormalise to 1, no wScale applied).
    std::vector<std::int32_t> refIdx(T * K);
    std::vector<float>        refW(T * K);
    ::mimirmind::compute::moeTopKRoute(logits.data(), T, nExperts, K,
                                       refIdx.data(), refW.data());

    const auto got = runDevice(ctx, ops, kern, logits, T, nExperts, K, wScale);

    for (std::size_t i = 0; i < T * K; ++i) {
        EXPECT_EQ(got.idx[i], refIdx[i]);
        EXPECT_NEAR(got.weight[i], refW[i] * wScale, 1e-4f);
    }
}

} // namespace

TEST(cuda_moe_topk_parity_k8_scale1) {
    // 128 experts, top-8, several tokens, no router scale.
    checkParity(/*T=*/6, /*nExperts=*/128, /*K=*/8, /*wScale=*/1.0f, 0xA11Cu);
}

TEST(cuda_moe_topk_parity_k4_scaled) {
    // Different geometry + a non-unit router weight scale.
    checkParity(/*T=*/4, /*nExperts=*/64, /*K=*/4, /*wScale=*/2.5f, 0xBEEFu);
}

TEST(cuda_moe_topk_parity_single_token) {
    // The decode-critical shape: T == 1.
    checkParity(/*T=*/1, /*nExperts=*/128, /*K=*/8, /*wScale=*/1.0f, 0x5EEDu);
}

int main() {
    return mm::test::run();
}