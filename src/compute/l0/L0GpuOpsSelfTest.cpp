// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "compute/l0/L0GpuOpsSelfTest.hpp"

#include "compute/Attention.hpp"
#include "compute/l0/GpuOps.hpp"
#include "core/gpu/l0/UsmAllocator.hpp"
#include "runtime/KvCache.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace mimirmind::compute::l0 {

void L0GpuOpsSelfTest::run(core::l0::UsmAllocator& allocator) {
    runQuantI8(allocator);
    runQkvSplit(allocator);
    // M5i.J: verify the single-WG streaming FlashAttention prefill kernel.
    // Skipped when the rollback flag is on: there's nothing to gate.
    if (_ops.prefillFlashEnabled()) {
        runAttentionPrefillFlash(allocator);
    }
}

void L0GpuOpsSelfTest::runQuantI8(core::l0::UsmAllocator& allocator) {
    // x_quant_i8: per-row symmetric int8 quantisation. Feeds the DP4A
    // Q8_0 matmul (M8.H) — a broken quant kernel silently corrupts
    // every DP4A matmul on the target iGPU, so this runs first.
    constexpr std::size_t M    = 2;
    constexpr std::size_t K    = 64;
    constexpr std::size_t xBytes = M * K * sizeof(float);
    constexpr std::size_t yBytes = M * K * sizeof(std::int8_t);
    constexpr std::size_t sBytes = M     * sizeof(float);

    void* xUsm = allocator.allocate(xBytes);
    void* yUsm = allocator.allocate(yBytes);
    void* sUsm = allocator.allocate(sBytes);

    // Row 0: mixed positive/negative, max at k=13. Row 1: all zeros
    // to exercise the amax=0 branch.
    std::vector<float> xh(M * K, 0.0F);
    for (std::size_t k = 0; k < K; ++k) {
        xh[k] = static_cast<float>(k) * 0.125F - 3.0F;
    }
    xh[13] = 5.5F; // guaranteed row-0 amax
    std::memcpy(xUsm, xh.data(), xBytes);
    std::memset(yUsm, 0x7F, yBytes);
    const float sPoison = -1.0e6F;
    for (std::size_t m = 0; m < M; ++m) {
        std::memcpy(static_cast<char*>(sUsm) + m * sizeof(float),
                    &sPoison, sizeof(float));
    }

    _ops.xQuantI8Async(static_cast<const float*>(xUsm),
                       static_cast<std::int8_t*>(yUsm),
                       static_cast<float*>(sUsm),
                       M, K);
    _ops.flush();

    std::vector<std::int8_t> qGot(M * K);
    std::vector<float>       sGot(M);
    std::memcpy(qGot.data(), yUsm, yBytes);
    std::memcpy(sGot.data(), sUsm, sBytes);

    // Row 0: scale = 5.5/127, dequant round-trip within 0.5*scale.
    const float amax0 = 5.5F;
    const float sRef0 = amax0 / 127.0F;
    if (!(std::fabs(sGot[0] - sRef0) <= 1e-6F)) {
        std::ostringstream os;
        os << "GpuOps::selfTest[x_quant_i8]: scale[0] mismatch — got="
           << sGot[0] << " ref=" << sRef0;
        throw std::runtime_error(os.str());
    }
    const float tol0 = 0.5F * sRef0 + 1e-6F;
    for (std::size_t k = 0; k < K; ++k) {
        const float deq = static_cast<float>(qGot[k]) * sGot[0];
        const float d   = std::fabs(deq - xh[k]);
        if (!(d <= tol0)) {
            std::ostringstream os;
            os << "GpuOps::selfTest[x_quant_i8]: row 0 dequant "
               << "mismatch at k=" << k << " x=" << xh[k]
               << " q=" << static_cast<int>(qGot[k])
               << " deq=" << deq << " diff=" << d
               << " tol=" << tol0;
            throw std::runtime_error(os.str());
        }
    }

    // Row 1: all-zero input → scale=0, all quants=0.
    if (!(sGot[1] == 0.0F)) {
        std::ostringstream os;
        os << "GpuOps::selfTest[x_quant_i8]: zero-row scale[1] "
           << "expected 0 got=" << sGot[1];
        throw std::runtime_error(os.str());
    }
    for (std::size_t k = 0; k < K; ++k) {
        if (qGot[K + k] != 0) {
            std::ostringstream os;
            os << "GpuOps::selfTest[x_quant_i8]: zero-row quant "
               << "at k=" << k << " expected 0 got="
               << static_cast<int>(qGot[K + k]);
            throw std::runtime_error(os.str());
        }
    }

    allocator.deallocate(xUsm, xBytes);
    allocator.deallocate(yUsm, yBytes);
    allocator.deallocate(sUsm, sBytes);
}

void L0GpuOpsSelfTest::runQkvSplit(core::l0::UsmAllocator& allocator) {
    // qkv_split: full QKV path (hasV=true) plus alt-attention path
    // (hasV=false) on a tiny fixed shape.
    constexpr std::size_t M   = 3;
    constexpr std::size_t Nq  = 8;
    constexpr std::size_t Nkv = 4;

    auto runCase = [&](bool hasV, const char* label) {
        const std::size_t Nfused = Nq + Nkv * (hasV ? 2 : 1);

        void* fUsm = allocator.allocate(M * Nfused * sizeof(float));
        void* qUsm = allocator.allocate(M * Nq     * sizeof(float));
        void* kUsm = allocator.allocate(M * Nkv    * sizeof(float));
        void* vUsm = allocator.allocate(M * Nkv    * sizeof(float));

        std::vector<float> fused(M * Nfused);
        for (std::size_t i = 0; i < fused.size(); ++i) {
            fused[i] = static_cast<float>(i) * 0.125F;
        }
        std::memcpy(fUsm, fused.data(), fused.size() * sizeof(float));
        // Poison the outputs so under-fills show up. Sized to the
        // largest destination (`M * Nq`) so the Q memcpy stays in
        // bounds; K and V only read the first `M * Nkv` entries.
        std::vector<float> poison(M * Nq, -1.0e6F);
        std::memcpy(qUsm, poison.data(), M * Nq  * sizeof(float));
        std::memcpy(kUsm, poison.data(), M * Nkv * sizeof(float));
        std::memcpy(vUsm, poison.data(), M * Nkv * sizeof(float));

        _ops.qkvSplitAsync(static_cast<const float*>(fUsm),
                           static_cast<float*>(qUsm),
                           static_cast<float*>(kUsm),
                           static_cast<float*>(vUsm),
                           M, Nq, Nkv, hasV);
        _ops.flush();

        auto verify = [&](const void* usm, const float* ref,
                          std::size_t n, const char* which) {
            std::vector<float> got(n);
            std::memcpy(got.data(), usm, n * sizeof(float));
            float maxDiff = 0.0F;
            std::size_t badIdx = 0;
            for (std::size_t i = 0; i < n; ++i) {
                const float d = std::fabs(got[i] - ref[i]);
                if (d > maxDiff) { maxDiff = d; badIdx = i; }
            }
            if (!(maxDiff <= 1e-6F)) {
                std::ostringstream os;
                os << "GpuOps::selfTest[" << label << "/" << which
                   << "]: qkv_split output mismatch — maxDiff=" << maxDiff
                   << " at i=" << badIdx
                   << " got=" << got[badIdx] << " ref=" << ref[badIdx];
                throw std::runtime_error(os.str());
            }
        };

        std::vector<float> refQ(M * Nq);
        std::vector<float> refK(M * Nkv);
        std::vector<float> refV(M * Nkv, -1.0e6F);
        for (std::size_t m = 0; m < M; ++m) {
            for (std::size_t i = 0; i < Nq; ++i) {
                refQ[m * Nq + i] = fused[m * Nfused + i];
            }
            for (std::size_t j = 0; j < Nkv; ++j) {
                refK[m * Nkv + j] = fused[m * Nfused + Nq + j];
            }
            if (hasV) {
                for (std::size_t j = 0; j < Nkv; ++j) {
                    refV[m * Nkv + j] =
                        fused[m * Nfused + Nq + Nkv + j];
                }
            }
        }
        verify(qUsm, refQ.data(), M * Nq,  "Q");
        verify(kUsm, refK.data(), M * Nkv, "K");
        if (hasV) {
            verify(vUsm, refV.data(), M * Nkv, "V");
        }

        allocator.deallocate(fUsm, M * Nfused * sizeof(float));
        allocator.deallocate(qUsm, M * Nq     * sizeof(float));
        allocator.deallocate(kUsm, M * Nkv    * sizeof(float));
        allocator.deallocate(vUsm, M * Nkv    * sizeof(float));
    };

    runCase(/*hasV=*/true,  "full");
    runCase(/*hasV=*/false, "qk-only");
}

void L0GpuOpsSelfTest::runAttentionPrefillFlash(
        core::l0::UsmAllocator& allocator) {
    // M5i.J: verify the single-WG streaming FlashAttention prefill kernel
    // against compute::multiHeadAttention (CPU reference) on a tiny
    // T_q=8 case. Same SPV-regression-guard role as the qkv_split block
    // above — catches driver/ocloc miscompilation before the first block
    // runs and gives a targeted error rather than a mysterious model
    // divergence later.
    constexpr std::size_t T_q      = 8;
    constexpr std::size_t T_k      = 8;
    constexpr std::size_t nHeads   = 4;
    constexpr std::size_t nKvHeads = 2;
    constexpr std::size_t headDim  = 32;
    const float scale = 1.0F / std::sqrt(
        static_cast<float>(headDim));

    const std::size_t qN  = T_q * nHeads   * headDim;
    const std::size_t kvN = T_k * nKvHeads * headDim;
    const std::size_t oN  = qN;

    void* qUsm = allocator.allocate(qN  * sizeof(float));
    void* kUsm = allocator.allocate(kvN * sizeof(float));
    void* vUsm = allocator.allocate(kvN * sizeof(float));
    void* oUsm = allocator.allocate(oN  * sizeof(float));

    // Deterministic ramp inputs — same seed math wouldn't help since
    // we have no RNG in the runtime, and this keeps the test
    // reproducible without pulling <random> into a load-time gate.
    std::vector<float> qHost(qN), kHost(kvN), vHost(kvN);
    for (std::size_t i = 0; i < qN;  ++i)
        qHost[i] = static_cast<float>((i * 7  + 1) % 17) * 0.125F - 1.0F;
    for (std::size_t i = 0; i < kvN; ++i)
        kHost[i] = static_cast<float>((i * 11 + 3) % 19) * 0.125F - 1.25F;
    for (std::size_t i = 0; i < kvN; ++i)
        vHost[i] = static_cast<float>((i * 13 + 5) % 23) * 0.0625F - 0.75F;
    std::memcpy(qUsm, qHost.data(), qN  * sizeof(float));
    std::memcpy(kUsm, kHost.data(), kvN * sizeof(float));
    std::memcpy(vUsm, vHost.data(), kvN * sizeof(float));
    std::memset(oUsm, 0,            oN  * sizeof(float));

    _ops.attentionPrefillFlashAsync(
        static_cast<const float*>(qUsm),
        static_cast<const float*>(kUsm),
        static_cast<const float*>(vUsm),
        T_q, nHeads, nKvHeads, headDim,
        /*positionOffset=*/0, scale,
        static_cast<float*>(oUsm),
        /*slidingWindow=*/0,
        runtime::KvDtype::F32);
    _ops.flush();

    std::vector<float> outCpu(oN, 0.0F);
    std::vector<float> scratch(T_k);
    // compute::multiHeadAttention bakes 1/sqrt(headDim) into its Q·K
    // scale — matches what we pass here, no pre-scale needed.
    multiHeadAttention(qHost.data(), kHost.data(), vHost.data(),
                       T_q, T_k, nHeads, nKvHeads, headDim,
                       /*positionOffset=*/0,
                       scratch.data(), outCpu.data());

    std::vector<float> outGpu(oN);
    std::memcpy(outGpu.data(), oUsm, oN * sizeof(float));

    constexpr float kTol = 5e-4F;
    float       maxDiff = 0.0F;
    std::size_t badIdx  = 0;
    for (std::size_t i = 0; i < oN; ++i) {
        const float d = std::fabs(outGpu[i] - outCpu[i]);
        if (d > maxDiff) { maxDiff = d; badIdx = i; }
    }
    if (!(maxDiff <= kTol)) {
        std::ostringstream os;
        os << "GpuOps::selfTest[attention_prefill_flash]: "
           << "output mismatch — maxDiff=" << maxDiff
           << " at i=" << badIdx
           << " gpu=" << outGpu[badIdx]
           << " cpu=" << outCpu[badIdx]
           << " tol="  << kTol;
        throw std::runtime_error(os.str());
    }

    allocator.deallocate(qUsm, qN  * sizeof(float));
    allocator.deallocate(kUsm, kvN * sizeof(float));
    allocator.deallocate(vUsm, kvN * sizeof(float));
    allocator.deallocate(oUsm, oN  * sizeof(float));
}

} // namespace mimirmind::compute::l0
