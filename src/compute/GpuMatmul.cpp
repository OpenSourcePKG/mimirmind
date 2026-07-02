#include "compute/GpuMatmul.hpp"

#include "compute/GpuOps.hpp"
#include "compute/Matmul.hpp"
#include "compute/QuantType.hpp"
#include "compute/QuantTypeRegistry.hpp"
#include "runtime/L0Context.hpp"
#include "runtime/Log.hpp"
#include "runtime/UsmAllocator.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace mimirmind::compute {

namespace {

bool envSet(const char* name) noexcept {
    const char* v = std::getenv(name);
    if (v == nullptr) return false;
    const std::string_view s{v};
    return !s.empty() && s != "0" && s != "false" && s != "off";
}

// Synthesise a plausible block-quantised weight buffer for the given
// QuantType. Random bytes with the fp16 scales / mins clamped to a
// small positive value so mat-mul dot products stay finite. The bench
// only cares about wall-clock throughput, not numerical correctness,
// but avoiding NaN keeps the driver happy.
void fillSyntheticWeights(model::GgmlType type,
                          std::uint8_t*   dst,
                          std::size_t     nbytes) {
    std::mt19937 rng{0xC0FFEEU};
    std::uniform_int_distribution<int> distByte(0, 255);
    for (std::size_t i = 0; i < nbytes; ++i) {
        dst[i] = static_cast<std::uint8_t>(distByte(rng));
    }

    // fp16 tiny positive constant used as super-block / block scale.
    constexpr std::uint16_t kHalfTiny  = 0x2400U;  // ~0.02
    constexpr std::uint16_t kHalfSmall = 0x2E00U;  // ~0.09
    (void)kHalfSmall;

    if (type == model::GgmlType::Q6_K) {
        const std::size_t bs = 210;
        for (std::size_t off = 0; off + bs <= nbytes; off += bs) {
            std::memcpy(dst + off + 208, &kHalfTiny, 2);
        }
    } else if (type == model::GgmlType::Q4_K) {
        const std::size_t bs = 144;
        for (std::size_t off = 0; off + bs <= nbytes; off += bs) {
            std::memcpy(dst + off,     &kHalfTiny, 2);   // d
            std::memcpy(dst + off + 2, &kHalfTiny, 2);   // dmin
        }
    } else if (type == model::GgmlType::Q8_0) {
        const std::size_t bs = 34;
        for (std::size_t off = 0; off + bs <= nbytes; off += bs) {
            std::memcpy(dst + off, &kHalfTiny, 2);
        }
    }
}

double medianMs(std::vector<double> xs) {
    if (xs.empty()) return 0.0;
    std::sort(xs.begin(), xs.end());
    const std::size_t mid = xs.size() / 2;
    return xs.size() % 2 == 1
        ? xs[mid]
        : 0.5 * (xs[mid - 1] + xs[mid]);
}

} // namespace

GpuMatmul::GpuMatmul(runtime::L0Context&    ctx,
                     GpuOps&                ops,
                     runtime::UsmAllocator& alloc,
                     runtime::CommandQueue& queue)
    : _ctx{ctx},
      _ops{ops},
      _alloc{alloc},
      _queue{queue}
{
    const auto loadSlot = [&ctx](std::string_view moduleName) {
        const std::string nameStr{moduleName};
        auto module = std::make_unique<runtime::GpuModule>(ctx, moduleName);
        runtime::GpuKernel kernel{module->kernel(nameStr.c_str())};
        return KernelSlot{std::move(module), kernel};
    };

    std::ostringstream loaded;
    bool first = true;

    for (const QuantType* qt : allQuantTypes()) {
        const auto vecName = qt->gpuMatmulModule();
        if (vecName.empty()) {
            continue;
        }

        KernelSlot                vecSlot = loadSlot(vecName);
        std::optional<KernelSlot> gemmSlot;
        std::size_t               gemmMTile = 1;

        const auto gemmName = qt->gpuMatmulGemmModule();
        if (!gemmName.empty()) {
            gemmSlot.emplace(loadSlot(gemmName));
            gemmMTile = qt->gpuMatmulGemmMTile();
        }

        // useGemm stays false until autotune() runs. This makes the
        // pre-autotune default the safe matvec-loop path.
        _entries.emplace(qt->ggmlType(),
                         Entry{std::move(vecSlot),
                               std::move(gemmSlot),
                               gemmMTile,
                               /*useGemm=*/false});

        if (!first) {
            loaded << " + ";
        }
        loaded << qt->name();
        if (!gemmName.empty()) {
            loaded << "(vec+gemm)";
        } else {
            loaded << "(vec)";
        }
        first = false;
    }

    // M8.H.1 — DP4A Q8_0 matvec. Guarded because the extension is not
    // universally advertised on older Intel iGPUs; a failure here just
    // disables the DP4A path (callers fall back to the plain matvec).
    try {
        _q8_0Dp4aSlot.emplace(loadSlot("matmul_q8_0_vec_dp4a"));
        MM_LOG_INFO("gpummm",
                    "GpuMatmul: matmul_q8_0_vec_dp4a loaded (local={}, sg={}, "
                    "{} outputs/group)",
                    kDp4aLocalSize, kDp4aSubgroupSize, kDp4aOutputsPerGroup);
    } catch (const std::exception& e) {
        MM_LOG_WARN("gpummm",
                    "GpuMatmul: matmul_q8_0_vec_dp4a load failed ({}) — "
                    "DP4A path disabled, Q8_0 matvec stays on the FP32-dequant "
                    "kernel",
                    e.what());
        _q8_0Dp4aSlot.reset();
    }

    // M8.H.3 — persistent Xq / Xscale scratch. Only allocate when the
    // DP4A path actually loaded; there's no point paying 23 MiB of USM
    // for a kernel we can't dispatch.
    if (_q8_0Dp4aSlot.has_value()) {
        _dp4aXqBytes    = kDp4aMaxM * kDp4aMaxK;               // int8
        _dp4aScaleBytes = kDp4aMaxM * sizeof(float);
        _dp4aXqUsm      = _alloc.allocate(_dp4aXqBytes);
        _dp4aScaleUsm   = _alloc.allocate(_dp4aScaleBytes);
        MM_LOG_INFO("gpummm",
                    "GpuMatmul: DP4A scratch reserved — Xq {} bytes "
                    "(max M={}, K={}), Xscale {} bytes",
                    _dp4aXqBytes, kDp4aMaxM, kDp4aMaxK, _dp4aScaleBytes);
    }

    MM_LOG_INFO("gpummm",
                "GpuMatmul ready — {} kernels loaded, local_size={} "
                "(sg={}, {} outputs/group). Autotune pending — call "
                "autotune() to pick between matvec-loop and GEMM per type.",
                loaded.str(), kLocalSize, kSubgroupSize, kOutputsPerGroup);
}

GpuMatmul::~GpuMatmul() {
    if (_dp4aXqUsm != nullptr) {
        _alloc.deallocate(_dp4aXqUsm, _dp4aXqBytes);
        _dp4aXqUsm = nullptr;
    }
    if (_dp4aScaleUsm != nullptr) {
        _alloc.deallocate(_dp4aScaleUsm, _dp4aScaleBytes);
        _dp4aScaleUsm = nullptr;
    }
}

void GpuMatmul::autotune(runtime::UsmAllocator& allocator,
                         std::size_t            hiddenDim,
                         std::size_t            mBatch) {
    if (mBatch < 2) {
        // Autotune only meaningful for M > 1 (the matvec-vs-GEMM decision
        // only matters on the batched path).
        mBatch = 16;
    }

    const bool forceDisable       = envSet("MIMIRMIND_DISABLE_GEMM");
    const bool forceEnable        = envSet("MIMIRMIND_FORCE_GEMM");
    const bool forceDisableDp4a   = envSet("MIMIRMIND_DISABLE_DP4A");
    const bool forceEnableDp4a    = envSet("MIMIRMIND_FORCE_DP4A");

    if (forceDisable) {
        for (auto& [type, entry] : _entries) {
            (void)type;
            entry.useGemm = false;
            entry.autotuneSource = "env_disable_gemm";
        }
        MM_LOG_INFO("gpummm",
                    "autotune: MIMIRMIND_DISABLE_GEMM=1 — every type "
                    "pinned to the matvec-loop path");
        return;
    }
    if (forceEnable) {
        for (auto& [type, entry] : _entries) {
            (void)type;
            entry.useGemm = entry.gemm.has_value();
            entry.autotuneSource = "env_force_gemm";
        }
        MM_LOG_INFO("gpummm",
                    "autotune: MIMIRMIND_FORCE_GEMM=1 — every type with "
                    "a GEMM kernel pinned to the GEMM path");
        return;
    }

    // MIMIRMIND_FORCE_DP4A=1 pins Q8_0 to the DP4A path without a
    // timing bench (still parity-checked below in the loop).
    if (forceEnableDp4a && _q8_0Dp4aSlot.has_value()) {
        const auto it = _entries.find(model::GgmlType::Q8_0);
        if (it != _entries.end()) {
            MM_LOG_INFO("gpummm",
                        "autotune: MIMIRMIND_FORCE_DP4A=1 — Q8_0 will be "
                        "pinned to the DP4A path after parity gate");
        }
    }

    // Round N and K to super-block-aligned sizes so every quant type
    // sees the same shape. Q4_K/Q6_K need K % 256 == 0; Q8_0 needs
    // K % 32 == 0. Aligning to 256 covers all three.
    const std::size_t K = ((hiddenDim + 255) / 256) * 256;
    const std::size_t N = K;
    const std::size_t M = mBatch;

    // Shared X / Y / scratch USM — reused across every type.
    const std::size_t xBytes = M * K * sizeof(float);
    const std::size_t yBytes = M * N * sizeof(float);
    const std::size_t sBytes = K * sizeof(float);
    void* xUsm = allocator.allocate(xBytes);
    void* yUsm = allocator.allocate(yBytes);
    void* sUsm = allocator.allocate(sBytes);

    {
        std::vector<float> xInit(M * K);
        std::mt19937 rng{0xB00BB00BU};
        std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
        for (auto& v : xInit) v = dist(rng);
        std::memcpy(xUsm, xInit.data(), xBytes);
    }
    std::memset(yUsm, 0, yBytes);
    std::memset(sUsm, 0, sBytes);

    constexpr int nWarmup = 2;
    constexpr int nTimed  = 5;

    for (auto& [type, entry] : _entries) {
        if (!entry.gemm.has_value()) {
            // No GEMM to compare against — stay on matvec-loop.
            entry.useGemm = false;
            entry.autotuneSource = "no_gemm";
            continue;
        }

        const QuantType* qt = quantType(type);
        if (qt == nullptr) {
            entry.useGemm = false;
            entry.autotuneSource = "no_gemm";
            continue;
        }

        // Synthesise weights of size [N, K].
        const std::size_t nSuper  = K / qt->blockElements();
        const std::size_t wBytes  = N * nSuper * qt->blockBytes();
        void* wUsm = allocator.allocate(wBytes);
        fillSyntheticWeights(type,
                             static_cast<std::uint8_t*>(wUsm),
                             wBytes);

        // Warmup: run one of each so both kernels are JIT'd.
        for (int i = 0; i < nWarmup; ++i) {
            // matvec-loop
            for (std::size_t m = 0; m < M; ++m) {
                matmulAsync(type, wUsm, N, K,
                            static_cast<const float*>(xUsm) + m * K,
                            /*M=*/1,
                            static_cast<float*>(yUsm) + m * N,
                            static_cast<float*>(sUsm));
            }
            _queue.flush();

            // GEMM — force it by temporarily flipping useGemm on.
            entry.useGemm = true;
            matmulAsync(type, wUsm, N, K,
                        static_cast<const float*>(xUsm), M,
                        static_cast<float*>(yUsm),
                        static_cast<float*>(sUsm));
            _queue.flush();
            entry.useGemm = false;
        }

        // Parity gate — before we let GEMM win the timing race, verify
        // the two paths compute the same values within tolerance on
        // this iGPU. A driver bug or a broken SPV would otherwise
        // silently corrupt inference. Small tolerance (5e-2) because
        // synthetic random weights can produce large accumulator
        // magnitudes; anything above that is a genuine kernel bug.
        {
            std::vector<float> yVec(M * N);
            std::vector<float> yGem(M * N);

            entry.useGemm = false;
            for (std::size_t m = 0; m < M; ++m) {
                matmulAsync(type, wUsm, N, K,
                            static_cast<const float*>(xUsm) + m * K,
                            /*M=*/1,
                            static_cast<float*>(yUsm) + m * N,
                            static_cast<float*>(sUsm));
            }
            _queue.flush();
            std::memcpy(yVec.data(), yUsm, yBytes);

            entry.useGemm = true;
            matmulAsync(type, wUsm, N, K,
                        static_cast<const float*>(xUsm), M,
                        static_cast<float*>(yUsm),
                        static_cast<float*>(sUsm));
            _queue.flush();
            entry.useGemm = false;
            std::memcpy(yGem.data(), yUsm, yBytes);

            float maxDiff = 0.0F;
            float maxRel  = 0.0F;
            for (std::size_t i = 0; i < yVec.size(); ++i) {
                const float d = std::fabs(yVec[i] - yGem[i]);
                if (d > maxDiff) maxDiff = d;
                const float ref = std::fabs(yVec[i]);
                if (ref > 1e-6F) {
                    const float r = d / ref;
                    if (r > maxRel) maxRel = r;
                }
            }
            constexpr float kAbsTol = 5e-2F;
            constexpr float kRelTol = 5e-2F;
            if (!(maxDiff <= kAbsTol) && !(maxRel <= kRelTol)) {
                MM_LOG_WARN("gpummm",
                            "autotune parity FAIL for {} — matvec vs gemm "
                            "maxDiff={:.6g} maxRel={:.6g}. Pinning to "
                            "matvec-loop and skipping timing bench.",
                            qt->name(), maxDiff, maxRel);
                entry.useGemm = false;
                entry.autotuneSource = "parity_fail";
                allocator.deallocate(wUsm, wBytes);
                continue;
            }
            MM_LOG_INFO("gpummm",
                        "autotune parity OK for {} — maxDiff={:.6g} "
                        "maxRel={:.6g}",
                        qt->name(), maxDiff, maxRel);
        }

        using clk = std::chrono::steady_clock;

        // Timed matvec-loop: M matmulAsyncs + one flush per iteration.
        std::vector<double> vecMs;
        vecMs.reserve(nTimed);
        for (int it = 0; it < nTimed; ++it) {
            const auto t0 = clk::now();
            for (std::size_t m = 0; m < M; ++m) {
                matmulAsync(type, wUsm, N, K,
                            static_cast<const float*>(xUsm) + m * K,
                            /*M=*/1,
                            static_cast<float*>(yUsm) + m * N,
                            static_cast<float*>(sUsm));
            }
            _queue.flush();
            const auto t1 = clk::now();
            std::chrono::duration<double, std::milli> dt = t1 - t0;
            vecMs.push_back(dt.count());
        }

        // Timed GEMM.
        entry.useGemm = true;
        std::vector<double> gemmMs;
        gemmMs.reserve(nTimed);
        for (int it = 0; it < nTimed; ++it) {
            const auto t0 = clk::now();
            matmulAsync(type, wUsm, N, K,
                        static_cast<const float*>(xUsm), M,
                        static_cast<float*>(yUsm),
                        static_cast<float*>(sUsm));
            _queue.flush();
            const auto t1 = clk::now();
            std::chrono::duration<double, std::milli> dt = t1 - t0;
            gemmMs.push_back(dt.count());
        }

        const double vecMed  = medianMs(std::move(vecMs));
        const double gemmMed = medianMs(std::move(gemmMs));

        // 5 % margin — noise floor on the iGPU shifts by more than 1-2 %
        // between runs. Below that we prefer the matvec-loop as the
        // conservative default.
        const bool pickGemm = (gemmMed * 1.05 < vecMed);
        entry.useGemm        = pickGemm;
        entry.lastVecMs      = vecMed;
        entry.lastGemmMs     = gemmMed;
        entry.autotuneSource = "bench";

        MM_LOG_INFO("gpummm",
                    "autotune: {} N={} K={} M={} — matvec-loop {:.2f} ms, "
                    "gemm {:.2f} ms → picked {}",
                    qt->name(), N, K, M, vecMed, gemmMed,
                    pickGemm ? "gemm" : "matvec-loop");

        // M8.H.3 — DP4A 3rd variant for Q8_0. Only runs after the
        // matvec-vs-gemm decision above so the existing telemetry stays
        // meaningful. Overrides useGemm on win.
        if (type == model::GgmlType::Q8_0
            && _q8_0Dp4aSlot.has_value()
            && !forceDisableDp4a)
        {
            const std::size_t xqBytes = M * K * sizeof(std::int8_t);
            const std::size_t xsBytes = M * sizeof(float);
            if (xqBytes > _dp4aXqBytes || xsBytes > _dp4aScaleBytes) {
                MM_LOG_WARN("gpummm",
                            "autotune: DP4A bench for {} skipped — "
                            "bench shape (M={}, K={}) exceeds scratch "
                            "bounds. Bump kDp4aMax* together.",
                            qt->name(), M, K);
            } else {
                entry.useDp4a = false;
                entry.useGemm = false;
                std::vector<float> yVec(M * N);
                for (std::size_t m = 0; m < M; ++m) {
                    matmulAsync(type, wUsm, N, K,
                                static_cast<const float*>(xUsm) + m * K,
                                /*M=*/1,
                                static_cast<float*>(yUsm) + m * N,
                                static_cast<float*>(sUsm));
                }
                _queue.flush();
                std::memcpy(yVec.data(), yUsm, yBytes);

                entry.useDp4a = true;
                std::vector<float> yDp4a(M * N);
                for (int i = 0; i < nWarmup; ++i) {
                    matmulAsync(type, wUsm, N, K,
                                static_cast<const float*>(xUsm), M,
                                static_cast<float*>(yUsm),
                                static_cast<float*>(sUsm));
                    _queue.flush();
                }
                matmulAsync(type, wUsm, N, K,
                            static_cast<const float*>(xUsm), M,
                            static_cast<float*>(yUsm),
                            static_cast<float*>(sUsm));
                _queue.flush();
                std::memcpy(yDp4a.data(), yUsm, yBytes);

                float maxAbs   = 0.0F;
                float maxDiff  = 0.0F;
                for (std::size_t i = 0; i < yVec.size(); ++i) {
                    maxAbs  = std::max(maxAbs,  std::fabs(yVec[i]));
                    maxDiff = std::max(maxDiff,
                                       std::fabs(yVec[i] - yDp4a[i]));
                }
                // INT8-quant-of-X noise on random inputs on a synthetic
                // weight bench: 5 % of max|ref| is the empirically-safe
                // ceiling. Real-inference activations are much better-
                // conditioned so a broken kernel would still overshoot
                // this by orders of magnitude.
                const float dp4aTol = std::max(0.05F * maxAbs, 1e-3F);
                if (!(maxDiff <= dp4aTol)) {
                    MM_LOG_WARN("gpummm",
                                "autotune parity FAIL for Q8_0 DP4A — "
                                "maxDiff={:.6g} maxRef={:.6g} tol={:.6g}. "
                                "Pinning to previous choice ({}) and "
                                "skipping DP4A timing bench.",
                                maxDiff, maxAbs, dp4aTol,
                                pickGemm ? "gemm" : "matvec-loop");
                    entry.useDp4a        = false;
                    entry.useGemm        = pickGemm;
                    entry.autotuneSource = "dp4a_parity_fail";
                } else {
                    MM_LOG_INFO("gpummm",
                                "autotune parity OK for Q8_0 DP4A — "
                                "maxDiff={:.6g} maxRef={:.6g} tol={:.6g}",
                                maxDiff, maxAbs, dp4aTol);

                    std::vector<double> dp4aMs;
                    dp4aMs.reserve(nTimed);
                    for (int it2 = 0; it2 < nTimed; ++it2) {
                        const auto t0 = clk::now();
                        matmulAsync(type, wUsm, N, K,
                                    static_cast<const float*>(xUsm), M,
                                    static_cast<float*>(yUsm),
                                    static_cast<float*>(sUsm));
                        _queue.flush();
                        const auto t1 = clk::now();
                        std::chrono::duration<double, std::milli> dt = t1 - t0;
                        dp4aMs.push_back(dt.count());
                    }
                    const double dp4aMed = medianMs(std::move(dp4aMs));
                    entry.lastDp4aMs = dp4aMed;

                    const double bestNonDp4a =
                        pickGemm ? gemmMed : vecMed;
                    // MIMIRMIND_FORCE_DP4A=1 skips the timing race.
                    const bool pickDp4a =
                        forceEnableDp4a ||
                        (dp4aMed * 1.05 < bestNonDp4a);
                    entry.useDp4a = pickDp4a;
                    if (pickDp4a) {
                        entry.useGemm = false;
                    } else {
                        entry.useGemm = pickGemm;
                    }
                    entry.autotuneSource =
                        forceEnableDp4a ? "env_force_dp4a" : "bench";
                    MM_LOG_INFO("gpummm",
                                "autotune: Q8_0 DP4A {:.2f} ms vs "
                                "best-non-dp4a {:.2f} ms → picked {}",
                                dp4aMed, bestNonDp4a,
                                pickDp4a
                                    ? "dp4a"
                                    : (pickGemm ? "gemm" : "matvec-loop"));
                }
            }
        }

        allocator.deallocate(wUsm, wBytes);
    }

    allocator.deallocate(xUsm, xBytes);
    allocator.deallocate(yUsm, yBytes);
    allocator.deallocate(sUsm, sBytes);
}

bool GpuMatmul::supports(model::GgmlType type) const noexcept {
    return _entries.contains(type);
}

void GpuMatmul::matmulAsync(model::GgmlType type,
                            const void*     W,
                            std::size_t     N,
                            std::size_t     K,
                            const float*    X,
                            std::size_t     M,
                            float*          Y,
                            float*          scratch) {
    const auto it = _entries.find(type);
    if (it == _entries.end()) {
        // CPU fallback. If async GPU work is pending, sync it first so
        // ordering vs prior matmulAsync calls is preserved.
        _queue.flush();
        compute::matmul(type, W, N, K, X, M, Y, scratch);
        return;
    }

    Entry& entry = it->second;

    // M8.H.3 — DP4A hot path for Q8_0 when autotune picked it AND the
    // request fits the persistent scratch. Bounds overflow falls
    // through to vec/gemm below with a one-shot warn so the constants
    // can be bumped if it keeps happening.
    if (type == model::GgmlType::Q8_0
        && entry.useDp4a
        && _q8_0Dp4aSlot.has_value())
    {
        if (M <= kDp4aMaxM && K <= kDp4aMaxK) {
            dispatchQ8_0Dp4aFromFloat(X, W, N, K, M, Y);
            return;
        }
        if (!_dp4aScratchOverflowWarned) {
            MM_LOG_WARN("gpummm",
                        "GpuMatmul: DP4A dispatch declined for Q8_0 "
                        "(M={}, K={}) — exceeds scratch bounds "
                        "(kDp4aMaxM={}, kDp4aMaxK={}). Falling back to "
                        "vec/gemm. Bump both constants together if this "
                        "happens for every request.",
                        M, K, kDp4aMaxM, kDp4aMaxK);
            _dp4aScratchOverflowWarned = true;
        }
    }

    // M5h: 4 outputs per workgroup (4 subgroups × 16 threads).
    const std::uint32_t nGroups = static_cast<std::uint32_t>(
        (N + kOutputsPerGroup - 1) / kOutputsPerGroup);

    // Prefill hot path — batched GEMM kernel when available and M > 1.
    // Handles all M rows in a single Level-Zero dispatch and amortises
    // the W dequant work M_TILE-fold. Falls through to matvec for M=1
    // (decode) even when GEMM is available, since the matvec kernel is
    // more launch-efficient for a single row.
    //
    // `useGemm` is set by autotune() based on measured GEMM-vs-matvec
    // wall time on the current iGPU. Until autotune runs it stays
    // false, so early pre-load matmuls take the safe matvec-loop.
    if (M > 1 && entry.gemm.has_value() && entry.useGemm) {
        runtime::GpuKernel& kern = entry.gemm->kernel;
        kern.setGroupSize(kLocalSize, 1, 1);
        kern.setPtr(0, X);
        kern.setPtr(1, W);
        kern.setPtr(2, Y);
        kern.setValue<std::int32_t>(3, static_cast<std::int32_t>(K));
        kern.setValue<std::int32_t>(4, static_cast<std::int32_t>(N));
        kern.setValue<std::int32_t>(5, static_cast<std::int32_t>(M));

        const std::uint32_t mGroups = static_cast<std::uint32_t>(
            (M + entry.gemmMTile - 1) / entry.gemmMTile);
        _queue.appendLaunch(kern, nGroups, mGroups, 1);
        return;
    }

    // Matvec fallback: M=1, or a QuantType with no GEMM kernel yet.
    // One appendLaunch per row of X. Per the Level Zero spec, args are
    // captured at append time, so each loop iteration's setPtr/setValue
    // do not affect the previously-recorded commands.
    runtime::GpuKernel& kern = entry.vec.kernel;
    kern.setGroupSize(kLocalSize, 1, 1);
    for (std::size_t m = 0; m < M; ++m) {
        const float* xRow = X + m * K;
        float*       yRow = Y + m * N;

        kern.setPtr(0, xRow);
        kern.setPtr(1, W);
        kern.setPtr(2, yRow);
        kern.setValue<std::int32_t>(3, static_cast<std::int32_t>(K));
        kern.setValue<std::int32_t>(4, static_cast<std::int32_t>(N));

        _queue.appendLaunch(kern, nGroups, 1, 1);
    }
}

void GpuMatmul::matmul(model::GgmlType type,
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

void GpuMatmul::matmulQ8_0Dp4aAsync(const std::int8_t* Xq,
                                    const float*       Xscale,
                                    const void*        W,
                                    std::size_t        N,
                                    std::size_t        K,
                                    std::size_t        M,
                                    float*             Y) {
    if (!_q8_0Dp4aSlot.has_value()) {
        throw std::runtime_error(
            "GpuMatmul::matmulQ8_0Dp4aAsync: DP4A kernel not loaded on "
            "this iGPU — check dp4aAvailable() before calling");
    }
    if (M == 0 || N == 0 || K == 0) {
        return;
    }
    if (K % 32 != 0) {
        throw std::runtime_error(
            "GpuMatmul::matmulQ8_0Dp4aAsync: K=" + std::to_string(K) +
            " is not a multiple of 32 (Q8_0 block size)");
    }

    runtime::GpuKernel& kern = _q8_0Dp4aSlot->kernel;
    kern.setGroupSize(kDp4aLocalSize, 1, 1);
    kern.setValue<std::int32_t>(4, static_cast<std::int32_t>(K));
    kern.setValue<std::int32_t>(5, static_cast<std::int32_t>(N));

    const std::uint32_t nGroups = static_cast<std::uint32_t>(
        (N + kDp4aOutputsPerGroup - 1) / kDp4aOutputsPerGroup);

    // One appendLaunch per row of X. Same rationale as matmulAsync's
    // matvec loop — args are captured at append time so per-row setPtr
    // does not affect prior recorded commands.
    for (std::size_t m = 0; m < M; ++m) {
        const std::int8_t* xqRow = Xq + m * K;
        const float*       xsRow = Xscale + m;
        float*             yRow  = Y + m * N;

        kern.setPtr(0, xqRow);
        kern.setPtr(1, xsRow);
        kern.setPtr(2, W);
        kern.setPtr(3, yRow);

        _queue.appendLaunch(kern, nGroups, 1, 1);
    }
}

void GpuMatmul::dispatchQ8_0Dp4aFromFloat(const float* X,
                                          const void*  W,
                                          std::size_t  N,
                                          std::size_t  K,
                                          std::size_t  M,
                                          float*       Y) {
    // Bounds are the caller's responsibility (matmulAsync checks them
    // before routing here). This method just wires the two kernels
    // together on the shared queue.
    auto* xq = static_cast<std::int8_t*>(_dp4aXqUsm);
    auto* xs = static_cast<float*>(_dp4aScaleUsm);

    _ops.xQuantI8Async(X, xq, xs, M, K);
    matmulQ8_0Dp4aAsync(xq, xs, W, N, K, M, Y);
}

void GpuMatmul::sync() {
    _queue.flush();
}

std::vector<GpuMatmul::AutotuneReport> GpuMatmul::autotuneReport() const {
    std::vector<AutotuneReport> out;
    out.reserve(_entries.size());
    for (const auto& [type, entry] : _entries) {
        const QuantType* qt = quantType(type);
        const bool dp4aForThisType =
            type == model::GgmlType::Q8_0 && _q8_0Dp4aSlot.has_value();
        out.push_back(AutotuneReport{
            .name           = qt != nullptr ? std::string{qt->name()} : "??",
            .gemmAvailable  = entry.gemm.has_value(),
            .gemmPicked     = entry.useGemm,
            .vecMs          = entry.lastVecMs,
            .gemmMs         = entry.lastGemmMs,
            .dp4aAvailable  = dp4aForThisType,
            .dp4aPicked     = entry.useDp4a,
            .dp4aMs         = entry.lastDp4aMs,
            .source         = entry.autotuneSource.empty()
                                ? std::string{"pending"}
                                : entry.autotuneSource,
        });
    }
    return out;
}

} // namespace mimirmind::compute