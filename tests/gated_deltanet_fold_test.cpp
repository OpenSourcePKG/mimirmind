// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// GDN ReplaySSM fold-kernel golden (Phase-4 Teil 1). Proves gated_deltanet_fold
// lands the SAME committed recurrent state as (a) the CPU golden truth
// compute::gatedDeltaNetRecurrent and (b) the device gated_deltanet_ar kernel —
// so a DFlash/MTP partial accept can replay the accepted prefix instead of
// re-forwarding the trunk. Also covers a partial-accept window (acceptLen < L)
// and the acceptLen == 0 no-op.
//
// Build + run (CUDA box):
//   cmake --build build-cuda --target gated_deltanet_fold_test &&
//   MIMIRMIND_HSACO_DIR=build-cuda/ptx ./build-cuda/gated_deltanet_fold_test

#include "TestFramework.hpp"

#include "compute/GatedDeltaNet.hpp"
#include "compute/cuda/GpuOps.hpp"
#include "core/gpu/cuda/CudaComputeContext.hpp"
#include "core/gpu/cuda/CudaKernel.hpp"
#include "core/gpu/cuda/CudaModule.hpp"

#include <array>
#include <cmath>
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

std::string resolvePtx(const std::string& name) {
    const char* env = std::getenv("MIMIRMIND_HSACO_DIR");
    std::array<std::string, 6> dirs{
        env ? std::string{env} : std::string{},
        "build/ptx", "build-cuda/ptx", "../build/ptx", "../build-cuda/ptx",
        "ptx"};
    for (const auto& d : dirs) {
        if (d.empty()) {
            continue;
        }
        std::filesystem::path p = std::filesystem::path{d} / (name + ".ptx");
        if (std::filesystem::exists(p)) {
            return p.string();
        }
    }
    throw std::runtime_error("gated_deltanet_fold_test: cannot find " + name +
                             ".ptx -- build the CUDA kernels or set "
                             "MIMIRMIND_HSACO_DIR");
}

// Replay `acceptLen` tokens' state via gated_deltanet_fold on the device;
// returns the folded state [H*S*S].
std::vector<float> runFold(CudaComputeContext& ctx, GpuOps& ops, CudaKernel& kern,
                           const std::vector<float>& state0,
                           const std::vector<float>& k, const std::vector<float>& v,
                           const std::vector<float>& gLog, const std::vector<float>& beta,
                           int acceptLen, int H, int S) {
    auto dState = ops.allocate(state0.size() * sizeof(float));
    auto dk     = ops.allocate(k.size() * sizeof(float));
    auto dv     = ops.allocate(v.size() * sizeof(float));
    auto dg     = ops.allocate(gLog.size() * sizeof(float));
    auto db     = ops.allocate(beta.size() * sizeof(float));
    ops.uploadHostBytes(dState.get(), state0.data(), state0.size() * sizeof(float));
    ops.uploadHostBytes(dk.get(), k.data(), k.size() * sizeof(float));
    ops.uploadHostBytes(dv.get(), v.data(), v.size() * sizeof(float));
    ops.uploadHostBytes(dg.get(), gLog.data(), gLog.size() * sizeof(float));
    ops.uploadHostBytes(db.get(), beta.data(), beta.size() * sizeof(float));

    kern.setPtr  (0, dk.get());
    kern.setPtr  (1, dv.get());
    kern.setPtr  (2, dg.get());
    kern.setPtr  (3, db.get());
    kern.setPtr  (4, dState.get());
    kern.setValue(5, static_cast<std::int32_t>(acceptLen));
    kern.setValue(6, static_cast<std::int32_t>(H));
    kern.setValue(7, static_cast<std::int32_t>(S));
    kern.launch(ctx.stream(),
                static_cast<std::uint32_t>(H), 1, 1,
                static_cast<std::uint32_t>(S), 1, 1);
    ops.flush();

    std::vector<float> out(state0.size());
    ops.readbackToHost(out.data(), dState.get(), out.size() * sizeof(float));
    return out;
}

// Reference via the device gated_deltanet_ar kernel (same op order, same
// hardware) — its final state must be BIT-IDENTICAL to the fold's.
std::vector<float> runAr(CudaComputeContext& ctx, GpuOps& ops, CudaKernel& kern,
                         const std::vector<float>& state0,
                         const std::vector<float>& q, const std::vector<float>& k,
                         const std::vector<float>& v, const std::vector<float>& gLog,
                         const std::vector<float>& beta, int T, int H, int S) {
    auto dState = ops.allocate(state0.size() * sizeof(float));
    auto dq     = ops.allocate(q.size() * sizeof(float));
    auto dk     = ops.allocate(k.size() * sizeof(float));
    auto dv     = ops.allocate(v.size() * sizeof(float));
    auto dg     = ops.allocate(gLog.size() * sizeof(float));
    auto db     = ops.allocate(beta.size() * sizeof(float));
    auto dOut   = ops.allocate(static_cast<std::size_t>(T) * H * S * sizeof(float));
    ops.uploadHostBytes(dState.get(), state0.data(), state0.size() * sizeof(float));
    ops.uploadHostBytes(dq.get(), q.data(), q.size() * sizeof(float));
    ops.uploadHostBytes(dk.get(), k.data(), k.size() * sizeof(float));
    ops.uploadHostBytes(dv.get(), v.data(), v.size() * sizeof(float));
    ops.uploadHostBytes(dg.get(), gLog.data(), gLog.size() * sizeof(float));
    ops.uploadHostBytes(db.get(), beta.data(), beta.size() * sizeof(float));

    kern.setPtr  (0, dq.get());
    kern.setPtr  (1, dk.get());
    kern.setPtr  (2, dv.get());
    kern.setPtr  (3, dg.get());
    kern.setPtr  (4, db.get());
    kern.setPtr  (5, dState.get());
    kern.setPtr  (6, dOut.get());
    kern.setValue(7, static_cast<std::int32_t>(T));
    kern.setValue(8, static_cast<std::int32_t>(H));
    kern.setValue(9, static_cast<std::int32_t>(S));
    kern.launch(ctx.stream(),
                static_cast<std::uint32_t>(H), 1, 1,
                static_cast<std::uint32_t>(S), 1, 1);
    ops.flush();

    std::vector<float> out(state0.size());
    ops.readbackToHost(out.data(), dState.get(), out.size() * sizeof(float));
    return out;
}

// One geometry: fold `acceptLen` of an L-token window; compare to the CPU
// recurrence (math) and the device AR kernel (bit-identity).
void checkParity(int L, int H, int S, int acceptLen, std::uint32_t seed) {
    CudaComputeContext ctx{};
    GpuOps ops{ctx};
    CudaModule foldMod = CudaModule::fromFile(ctx.cudaContext(),
                                              resolvePtx("gated_deltanet_fold"));
    CudaKernel foldK = foldMod.getFunction("gated_deltanet_fold");
    CudaModule arMod = CudaModule::fromFile(ctx.cudaContext(),
                                            resolvePtx("gated_deltanet_ar"));
    CudaKernel arK = arMod.getFunction("gated_deltanet_ar");

    const std::size_t elemsTHS = static_cast<std::size_t>(L) * H * S;
    const std::size_t elemsTH  = static_cast<std::size_t>(L) * H;
    const std::size_t elemsHSS = static_cast<std::size_t>(H) * S * S;

    const auto state0 = randVec(elemsHSS, seed ^ 0x11u);
    const auto q      = randVec(elemsTHS, seed ^ 0x22u);
    const auto k      = randVec(elemsTHS, seed ^ 0x33u);
    const auto v      = randVec(elemsTHS, seed ^ 0x44u);
    auto gLogRaw      = randVec(elemsTH, seed ^ 0x55u);
    auto betaRaw      = randVec(elemsTH, seed ^ 0x66u);
    std::vector<float> gLog(elemsTH), beta(elemsTH);
    for (std::size_t i = 0; i < elemsTH; ++i) {
        gLog[i] = -std::abs(gLogRaw[i]);          // log-decay <= 0 => g in (0,1]
        beta[i] = 0.5f * (betaRaw[i] + 1.0f);     // beta in [0,1)
    }

    // (a) CPU golden truth: recurrence over the accepted prefix (state only —
    // q/out ignored for the comparison).
    std::vector<float> stateCpu = state0;
    std::vector<float> outCpu(static_cast<std::size_t>(acceptLen) * H * S);
    ::mimirmind::compute::gatedDeltaNetRecurrent(
        q.data(), k.data(), v.data(), gLog.data(), beta.data(),
        stateCpu.data(), outCpu.data(),
        static_cast<std::size_t>(acceptLen), static_cast<std::size_t>(H),
        static_cast<std::size_t>(S));

    // (b) device AR kernel over the accepted prefix (bit-identity reference).
    const auto stateAr = runAr(ctx, ops, arK, state0, q, k, v, gLog, beta,
                               acceptLen, H, S);

    // fold under test.
    const auto stateFold = runFold(ctx, ops, foldK, state0, k, v, gLog, beta,
                                   acceptLen, H, S);

    double maxAbsCpu = 0.0, maxAbsAr = 0.0;
    for (std::size_t i = 0; i < elemsHSS; ++i) {
        maxAbsCpu = std::max(maxAbsCpu,
                             std::abs(static_cast<double>(stateFold[i]) - stateCpu[i]));
        maxAbsAr = std::max(maxAbsAr,
                            std::abs(static_cast<double>(stateFold[i]) - stateAr[i]));
    }
    std::printf("[fold] L=%d H=%d S=%d acceptLen=%d  maxAbs(fold-cpu)=%.3e "
                "maxAbs(fold-ar)=%.3e\n", L, H, S, acceptLen, maxAbsCpu, maxAbsAr);

    // fold vs device AR: identical ops on identical hardware -> bit-identical.
    EXPECT_TRUE(maxAbsAr == 0.0);
    // fold vs CPU recurrence: same op order, FMA-level differences only.
    EXPECT_TRUE(maxAbsCpu < 1e-3);
}

} // namespace

TEST(gdn_fold_full_accept_s128) {
    // Full-window fold (acceptLen == L), production head_dim S=128, block 8.
    checkParity(/*L=*/8, /*H=*/16, /*S=*/128, /*acceptLen=*/8, 0xF01Du);
}

TEST(gdn_fold_partial_accept_s128) {
    // Partial accept: window of 8, only 3 tokens accepted.
    checkParity(/*L=*/8, /*H=*/16, /*S=*/128, /*acceptLen=*/3, 0xF01Eu);
}

TEST(gdn_fold_partial_accept_s64) {
    checkParity(/*L=*/7, /*H=*/8, /*S=*/64, /*acceptLen=*/5, 0xF01Fu);
}

TEST(gdn_fold_zero_accept_noop) {
    // acceptLen == 0 must leave the checkpoint untouched.
    checkParity(/*L=*/8, /*H=*/4, /*S=*/64, /*acceptLen=*/0, 0xF020u);
}

int main() {
    return mm::test::run();
}
