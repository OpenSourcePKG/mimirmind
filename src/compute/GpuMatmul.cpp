#include "compute/GpuMatmul.hpp"

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

GpuMatmul::GpuMatmul(runtime::L0Context& ctx, runtime::CommandQueue& queue)
    : _ctx{ctx},
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

    MM_LOG_INFO("gpummm",
                "GpuMatmul ready — {} kernels loaded, local_size={} "
                "(sg={}, {} outputs/group). Autotune pending — call "
                "autotune() to pick between matvec-loop and GEMM per type.",
                loaded.str(), kLocalSize, kSubgroupSize, kOutputsPerGroup);
}

void GpuMatmul::autotune(runtime::UsmAllocator& allocator,
                         std::size_t            hiddenDim,
                         std::size_t            mBatch) {
    if (mBatch < 2) {
        // Autotune only meaningful for M > 1 (the matvec-vs-GEMM decision
        // only matters on the batched path).
        mBatch = 16;
    }

    const bool forceDisable = envSet("MIMIRMIND_DISABLE_GEMM");
    const bool forceEnable  = envSet("MIMIRMIND_FORCE_GEMM");

    if (forceDisable) {
        for (auto& [type, entry] : _entries) {
            (void)type;
            entry.useGemm = false;
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
        }
        MM_LOG_INFO("gpummm",
                    "autotune: MIMIRMIND_FORCE_GEMM=1 — every type with "
                    "a GEMM kernel pinned to the GEMM path");
        return;
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
            continue;
        }

        const QuantType* qt = quantType(type);
        if (qt == nullptr) {
            entry.useGemm = false;
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
        entry.useGemm = pickGemm;

        MM_LOG_INFO("gpummm",
                    "autotune: {} N={} K={} M={} — matvec-loop {:.2f} ms, "
                    "gemm {:.2f} ms → picked {}",
                    qt->name(), N, K, M, vecMed, gemmMed,
                    pickGemm ? "gemm" : "matvec-loop");

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

void GpuMatmul::sync() {
    _queue.flush();
}

} // namespace mimirmind::compute