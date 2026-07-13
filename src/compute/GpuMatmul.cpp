#include "compute/GpuMatmul.hpp"

#include "compute/GpuOps.hpp"
#include "compute/Matmul.hpp"
#include "compute/QuantType.hpp"
#include "compute/QuantTypeRegistry.hpp"
#include "core/config/Config.hpp"
#include "core/l0/L0Context.hpp"
#include "core/log/Log.hpp"
#include "core/l0/UsmAllocator.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace mimirmind::compute {

namespace {

// kAutotuneMBuckets + kAutotuneBucketCount moved to the .hpp so
// Entry / AutotuneReport / GpuMatmul.cpp all see the same size.
constexpr std::size_t kGemmMinMNever =
    std::numeric_limits<std::size_t>::max();

// Format a multi-bucket kernel-timing row as
//   "M=16 X.XX/M=64 X.XX/..."
// so the two-bench-per-line "autotune:" logs scale with bucket count.
std::string formatBucketRow(std::span<const double> ms) {
    std::string out;
    out.reserve(ms.size() * 16);
    for (std::size_t i = 0; i < ms.size(); ++i) {
        if (i > 0) out += " | ";
        out += "M=";
        out += std::to_string(kAutotuneMBuckets[i]);
        char buf[32];
        std::snprintf(buf, sizeof(buf), " %.2f", ms[i]);
        out += buf;
    }
    return out;
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
    } else if (type == model::GgmlType::Q5_K) {
        // 176 B super-block: fp16 d + fp16 dmin + 12 scales + 32 qh + 128 qs.
        // Without this override, random bytes at offsets 0..3 get
        // interpreted as fp16 by the dequant path (possibly ±Inf, NaN, or
        // 65504), which makes the vec-parity check "pass" (CPU and GPU
        // agree) but produces astronomical output magnitudes — the
        // parity check then can't tell an actual layout bug from
        // consistent-but-wrong behaviour.
        const std::size_t bs = 176;
        for (std::size_t off = 0; off + bs <= nbytes; off += bs) {
            std::memcpy(dst + off,     &kHalfTiny, 2);   // d
            std::memcpy(dst + off + 2, &kHalfTiny, 2);   // dmin
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

        // gemmMinM defaults to kGemmMinMNever until autotune() runs, so
        // the pre-autotune default is the safe matvec-loop path even
        // for M=Mmax requests that arrive before autotune finishes.
        _entries.emplace(qt->ggmlType(),
                         Entry{std::move(vecSlot),
                               std::move(gemmSlot),
                               gemmMTile,
                               /*gemmV2=*/std::nullopt,
                               /*gemmV2MTile=*/1,
                               /*useGemmV2=*/false,
                               /*gemmV2MsAtM=*/{},
                               kGemmMinMNever,   // gemmMinM
                               /*dp4a=*/std::nullopt,
                               false,            // useDp4a
                               {},               // vecMsAtM
                               {},               // gemmMsAtM
                               0.0,              // lastDp4aMs
                               ""});             // autotuneSource

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

    // M8.K.1 + M8.K.1b — v2 GEMM prototypes with shrunk SLM (X_TILE=256).
    // Q8_0 was the first landed (M8.K.1), Q6_K + Q4_K added in M8.K.1b.
    // All three use identical WG geometry (LOCAL=64, SG=16, 4 outputs/WG,
    // M_TILE=8) and the same 8 KiB SLM/WG budget — the only difference
    // is the block-dequant math per type. Guarded so a driver refusal
    // (unlikely at 8 KiB but possible on quirky iGPUs) just disables
    // that type's v2 path.
    const std::array<std::pair<model::GgmlType, const char*>, 3>
        kV2Kernels = {{
            {model::GgmlType::Q8_0, "matmul_q8_0_gemm_v2"},
            {model::GgmlType::Q6_K, "matmul_q6k_gemm_v2"},
            {model::GgmlType::Q4_K, "matmul_q4k_gemm_v2"},
        }};
    for (const auto& [type, moduleName] : kV2Kernels) {
        const auto it = _entries.find(type);
        if (it == _entries.end()) continue;
        try {
            it->second.gemmV2.emplace(loadSlot(moduleName));
            it->second.gemmV2MTile = kGemmV2MTile;
            MM_LOG_INFO("gpummm",
                        "GpuMatmul: {} loaded (M_TILE={}, X_TILE=256, "
                        "SLM=8 KiB/WG) — v2 path benched alongside v1",
                        moduleName, kGemmV2MTile);
        } catch (const std::exception& e) {
            MM_LOG_WARN("gpummm",
                        "GpuMatmul: {} load failed ({}) — v2 path "
                        "disabled for this type",
                        moduleName, e.what());
            it->second.gemmV2.reset();
        }
    }

    // M8.H.1 / M8.M — DP4A matvec per quant type. Guarded because the
    // integer_dot_product extension is not universally advertised on
    // older Intel iGPUs; per-type failures leave the entry without a
    // DP4A slot and callers fall back to plain matvec for that type.
    struct Dp4aKernel { model::GgmlType type; const char* module; };
    for (const auto& [dtype, module] : std::initializer_list<Dp4aKernel>{
             {model::GgmlType::Q8_0, "matmul_q8_0_vec_dp4a"},
             {model::GgmlType::Q4_K, "matmul_q4k_vec_dp4a"}})
    {
        auto it = _entries.find(dtype);
        if (it == _entries.end()) continue;   // type not registered
        Entry& entry = it->second;
        try {
            entry.dp4a.emplace(loadSlot(module));
            MM_LOG_INFO("gpummm",
                        "GpuMatmul: {} loaded (local={}, sg={}, {} outputs/group)",
                        module,
                        kDp4aLocalSize, kDp4aSubgroupSize, kDp4aOutputsPerGroup);
        } catch (const std::exception& e) {
            MM_LOG_WARN("gpummm",
                        "GpuMatmul: {} load failed ({}) — DP4A path "
                        "disabled for this type, matvec stays on the "
                        "FP32-dequant kernel",
                        module, e.what());
            entry.dp4a.reset();
        }
    }

    // Persistent Xq / Xscale scratch. Only allocate when at least one
    // DP4A path actually loaded; there's no point paying 23 MiB of USM
    // for a kernel we can't dispatch.
    const bool anyDp4a = std::any_of(
        _entries.begin(), _entries.end(),
        [](const auto& kv) { return kv.second.dp4a.has_value(); });
    if (anyDp4a) {
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

void GpuMatmul::autotune(runtime::UsmAllocator&          allocator,
                         std::size_t                     hiddenDim,
                         const runtime::FeatureSettings& features) {
    // M8.J — M-buckets replace the single `mBatch` argument; every
    // bench-time decision runs against kAutotuneMBuckets.
    const bool forceDisable     = features.gemm == runtime::TriState::Disable;
    const bool forceEnable      = features.gemm == runtime::TriState::Force;
    const bool forceDisableDp4a = features.dp4a == runtime::TriState::Disable;
    const bool forceEnableDp4a  = features.dp4a == runtime::TriState::Force;
    const std::size_t envMinM   = features.gemmMinM.value_or(std::size_t{0});

    // features.gemmMinM — pin the crossover threshold on every type
    // that has a GEMM kernel and skip the timing bench entirely.
    // Wins over features.gemm if set. Debug-only lever.
    if (envMinM > 0) {
        for (auto& [type, entry] : _entries) {
            (void)type;
            entry.gemmMinM =
                entry.gemm.has_value() ? envMinM : kGemmMinMNever;
            entry.autotuneSource = "cfg_gemm_min_m";
        }
        MM_LOG_INFO("gpummm",
                    "autotune: features.gemmMinM={} — every type with "
                    "a GEMM kernel pinned to that threshold, timing bench "
                    "skipped", envMinM);
        return;
    }

    if (forceDisable) {
        for (auto& [type, entry] : _entries) {
            (void)type;
            entry.gemmMinM = kGemmMinMNever;
            entry.autotuneSource = "cfg_disable_gemm";
        }
        MM_LOG_INFO("gpummm",
                    "autotune: features.gemm=disable — every type "
                    "pinned to the matvec-loop path");
        return;
    }
    if (forceEnable) {
        for (auto& [type, entry] : _entries) {
            (void)type;
            entry.gemmMinM =
                entry.gemm.has_value() ? std::size_t{2} : kGemmMinMNever;
            entry.autotuneSource = "cfg_force_gemm";
        }
        MM_LOG_INFO("gpummm",
                    "autotune: features.gemm=force — every type with "
                    "a GEMM kernel pinned to the GEMM path (gemmMinM=2)");
        return;
    }

    if (forceEnableDp4a && dp4aAvailable()) {
        MM_LOG_INFO("gpummm",
                    "autotune: features.dp4a=force — every type with "
                    "a DP4A kernel will be pinned to that path after "
                    "the parity gate");
    }

    // Round N and K to super-block-aligned sizes so every quant type
    // sees the same shape. Q4_K/Q6_K need K % 256 == 0; Q8_0 needs
    // K % 32 == 0. Aligning to 256 covers all three.
    const std::size_t K    = ((hiddenDim + 255) / 256) * 256;
    const std::size_t N    = K;
    const std::size_t Mmax = kAutotuneMBuckets.back();

    // Shared X / Y / scratch USM sized for the largest bucket; smaller
    // buckets slice a prefix of the same buffer.
    const std::size_t xBytes = Mmax * K * sizeof(float);
    const std::size_t yBytes = Mmax * N * sizeof(float);
    const std::size_t sBytes = K * sizeof(float);
    void* xUsm = allocator.allocate(xBytes);
    void* yUsm = allocator.allocate(yBytes);
    void* sUsm = allocator.allocate(sBytes);

    {
        std::vector<float> xInit(Mmax * K);
        std::mt19937 rng{0xB00BB00BU};
        std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
        for (auto& v : xInit) v = dist(rng);
        std::memcpy(xUsm, xInit.data(), xBytes);
    }
    std::memset(yUsm, 0, yBytes);
    std::memset(sUsm, 0, sBytes);

    constexpr int nWarmup = 2;
    constexpr int nTimed  = 5;

    using clk = std::chrono::steady_clock;

    for (auto& [type, entry] : _entries) {
        const QuantType* qt = quantType(type);
        if (qt == nullptr) {
            entry.gemmMinM = kGemmMinMNever;
            entry.autotuneSource = "no_gemm";
            continue;
        }

        // CPU vs GPU vec parity — even types without a GEMM kernel get
        // this check. Catches quantized matvec kernels whose bit-layout
        // disagrees with the CPU dequant (Q5_K bringup for E4B lived
        // here for a day). Sample at M=1 with the synthetic X row so
        // the cost is minimal.
        {
            const std::size_t nSuper = K / qt->blockElements();
            const std::size_t wBytes = N * nSuper * qt->blockBytes();
            void* wUsm = allocator.allocate(wBytes);
            fillSyntheticWeights(type,
                                 static_cast<std::uint8_t*>(wUsm),
                                 wBytes);

            std::vector<float> yGpu(N, 0.0F);
            std::vector<float> yCpu(N, 0.0F);
            std::vector<float> cpuScratch(K, 0.0F);

            entry.gemmMinM = kGemmMinMNever;   // force matvec-loop
            std::memset(yUsm, 0, N * sizeof(float));
            matmulAsync(type, wUsm, N, K,
                        static_cast<const float*>(xUsm), /*M=*/1,
                        static_cast<float*>(yUsm),
                        static_cast<float*>(sUsm));
            _queue.flush();
            std::memcpy(yGpu.data(), yUsm, N * sizeof(float));

            compute::matmul(type, wUsm, N, K,
                            static_cast<const float*>(xUsm), /*M=*/1,
                            yCpu.data(),
                            cpuScratch.data());

            float maxDiff = 0.0F;
            float maxRef  = 0.0F;
            for (std::size_t i = 0; i < N; ++i) {
                const float d = std::fabs(yGpu[i] - yCpu[i]);
                if (d > maxDiff) maxDiff = d;
                const float r = std::fabs(yCpu[i]);
                if (r > maxRef) maxRef = r;
            }
            const float relTol = maxRef * 1e-2F;
            const float absTol = 5e-2F;
            const bool ok = maxDiff <= std::max(relTol, absTol);
            if (!ok) {
                MM_LOG_ERROR("gpummm",
                             "vec parity FAIL for {} — GPU vs CPU "
                             "maxDiff={:.6g} maxRef={:.6g} (tol=max("
                             "{:.4g},{:.4g})). Matvec kernel disagrees "
                             "with reference dequant; this WILL produce "
                             "garbage output on any model using this "
                             "quant type.",
                             qt->name(), maxDiff, maxRef, relTol, absTol);
            } else {
                MM_LOG_INFO("gpummm",
                            "vec parity OK for {} — GPU vs CPU "
                            "maxDiff={:.6g} maxRef={:.6g}",
                            qt->name(), maxDiff, maxRef);
            }

            allocator.deallocate(wUsm, wBytes);
        }

        if (!entry.gemm.has_value()) {
            entry.gemmMinM = kGemmMinMNever;
            entry.autotuneSource = "no_gemm";
            continue;
        }

        // Synthesise weights of size [N, K].
        const std::size_t nSuper = K / qt->blockElements();
        const std::size_t wBytes = N * nSuper * qt->blockBytes();
        void* wUsm = allocator.allocate(wBytes);
        fillSyntheticWeights(type,
                             static_cast<std::uint8_t*>(wUsm),
                             wBytes);

        // Warmup at Mmax — biggest cost, primes both kernels' JIT and
        // the GEMM SLM staging path for the timing loop below.
        for (int i = 0; i < nWarmup; ++i) {
            entry.gemmMinM = kGemmMinMNever;   // force matvec-loop
            for (std::size_t m = 0; m < Mmax; ++m) {
                matmulAsync(type, wUsm, N, K,
                            static_cast<const float*>(xUsm) + m * K,
                            /*M=*/1,
                            static_cast<float*>(yUsm) + m * N,
                            static_cast<float*>(sUsm));
            }
            _queue.flush();

            entry.gemmMinM = 2;                // force GEMM
            matmulAsync(type, wUsm, N, K,
                        static_cast<const float*>(xUsm), Mmax,
                        static_cast<float*>(yUsm),
                        static_cast<float*>(sUsm));
            _queue.flush();
        }

        // Parity gate at Mmax — one shape, cheapest to verify at the
        // largest realistic bench-size. If matvec and GEMM disagree
        // beyond tolerance, the whole GEMM path is disabled for this
        // type (gemmMinM = MAX). Wouldn't matter WHICH bucket we
        // checked at; the kernels are shape-agnostic.
        {
            const std::size_t elts = Mmax * N;
            std::vector<float> yVec(elts);
            std::vector<float> yGem(elts);

            entry.gemmMinM = kGemmMinMNever;
            for (std::size_t m = 0; m < Mmax; ++m) {
                matmulAsync(type, wUsm, N, K,
                            static_cast<const float*>(xUsm) + m * K,
                            /*M=*/1,
                            static_cast<float*>(yUsm) + m * N,
                            static_cast<float*>(sUsm));
            }
            _queue.flush();
            std::memcpy(yVec.data(), yUsm, elts * sizeof(float));

            entry.gemmMinM = 2;
            matmulAsync(type, wUsm, N, K,
                        static_cast<const float*>(xUsm), Mmax,
                        static_cast<float*>(yUsm),
                        static_cast<float*>(sUsm));
            _queue.flush();
            std::memcpy(yGem.data(), yUsm, elts * sizeof(float));

            float maxDiff = 0.0F;
            float maxRel  = 0.0F;
            for (std::size_t i = 0; i < elts; ++i) {
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
                entry.gemmMinM       = kGemmMinMNever;
                entry.autotuneSource = "parity_fail";
                allocator.deallocate(wUsm, wBytes);
                continue;
            }
            MM_LOG_INFO("gpummm",
                        "autotune parity OK for {} — maxDiff={:.6g} "
                        "maxRel={:.6g}",
                        qt->name(), maxDiff, maxRel);
        }

        // Timing loop — bench matvec-loop and GEMM at every M bucket.
        // Buckets are held in a stack array so the per-M-medians land
        // straight into `entry.{vec,gemm}MsAtM` at the matching index.
        for (std::size_t bi = 0; bi < kAutotuneMBuckets.size(); ++bi) {
            const std::size_t Mb = kAutotuneMBuckets[bi];

            entry.gemmMinM = kGemmMinMNever;   // force matvec-loop
            std::vector<double> vecMs;
            vecMs.reserve(nTimed);
            for (int it = 0; it < nTimed; ++it) {
                const auto t0 = clk::now();
                for (std::size_t m = 0; m < Mb; ++m) {
                    matmulAsync(type, wUsm, N, K,
                                static_cast<const float*>(xUsm) + m * K,
                                /*M=*/1,
                                static_cast<float*>(yUsm) + m * N,
                                static_cast<float*>(sUsm));
                }
                _queue.flush();
                const auto t1 = clk::now();
                vecMs.push_back(
                    std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
            entry.vecMsAtM[bi] = medianMs(std::move(vecMs));

            entry.gemmMinM = 2;                // force GEMM
            std::vector<double> gemmMs;
            gemmMs.reserve(nTimed);
            for (int it = 0; it < nTimed; ++it) {
                const auto t0 = clk::now();
                matmulAsync(type, wUsm, N, K,
                            static_cast<const float*>(xUsm), Mb,
                            static_cast<float*>(yUsm),
                            static_cast<float*>(sUsm));
                _queue.flush();
                const auto t1 = clk::now();
                gemmMs.push_back(
                    std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
            entry.gemmMsAtM[bi] = medianMs(std::move(gemmMs));
        }

        // Derive gemmMinM: smallest bucket-M where gemm × 1.05 < vec.
        // 5 % margin is the noise floor between iGPU runs; below that we
        // stick with matvec-loop as the conservative default.
        entry.gemmMinM = kGemmMinMNever;
        for (std::size_t bi = 0; bi < kAutotuneMBuckets.size(); ++bi) {
            if (entry.gemmMsAtM[bi] * 1.05 < entry.vecMsAtM[bi]) {
                entry.gemmMinM = kAutotuneMBuckets[bi];
                break;
            }
        }
        entry.autotuneSource = "bench";

        MM_LOG_INFO("gpummm",
                    "autotune: {} N={} K={} — vec:[{}] | gemm:[{}] → "
                    "gemmMinM={}",
                    qt->name(), N, K,
                    formatBucketRow(entry.vecMsAtM),
                    formatBucketRow(entry.gemmMsAtM),
                    entry.gemmMinM == kGemmMinMNever
                        ? std::string{"never"}
                        : std::to_string(entry.gemmMinM));

        // M8.K.1 + M8.K.1b — v2 GEMM bench for every type that has a
        // v2 kernel loaded (Q8_0, Q6_K, Q4_K). Runs alongside v1 at
        // every M-bucket so operators can see the crossover in the
        // logs; the actual dispatch decision only flips to v2 when
        // `features.gemmV2: true` is set in config.json.
        if (entry.gemmV2.has_value()) {
            const std::size_t v2Tile = entry.gemmV2MTile;
            // Temporarily route through v2 by flipping useGemmV2 for
            // the bench, restore after.
            const bool savedUseV2 = entry.useGemmV2;
            entry.useGemmV2 = true;

            // Force GEMM dispatch (bypass matvec) for the bench by
            // setting gemmMinM=2. Restore afterwards.
            const std::size_t savedMinM = entry.gemmMinM;
            entry.gemmMinM = 2;

            // Warmup — one shot at Mmax to prime the JIT.
            matmulAsync(type, wUsm, N, K,
                        static_cast<const float*>(xUsm), Mmax,
                        static_cast<float*>(yUsm),
                        static_cast<float*>(sUsm));
            _queue.flush();

            for (std::size_t bi = 0; bi < kAutotuneMBuckets.size(); ++bi) {
                const std::size_t Mb = kAutotuneMBuckets[bi];
                std::vector<double> v2Ms;
                v2Ms.reserve(nTimed);
                for (int it = 0; it < nTimed; ++it) {
                    const auto t0 = clk::now();
                    matmulAsync(type, wUsm, N, K,
                                static_cast<const float*>(xUsm), Mb,
                                static_cast<float*>(yUsm),
                                static_cast<float*>(sUsm));
                    _queue.flush();
                    const auto t1 = clk::now();
                    v2Ms.push_back(
                        std::chrono::duration<double, std::milli>(t1 - t0)
                            .count());
                }
                entry.gemmV2MsAtM[bi] = medianMs(std::move(v2Ms));
            }
            entry.useGemmV2 = savedUseV2;
            entry.gemmMinM  = savedMinM;

            (void)v2Tile;
            MM_LOG_INFO("gpummm",
                        "autotune: {} GEMM v2 (M_TILE={}, X_TILE=256, "
                        "SLM=8 KiB/WG) — v2:[{}] | v1:[{}] | vec:[{}]",
                        qt->name(), entry.gemmV2MTile,
                        formatBucketRow(entry.gemmV2MsAtM),
                        formatBucketRow(entry.gemmMsAtM),
                        formatBucketRow(entry.vecMsAtM));

            // Config opt-in. Only fires when the v2 bench actually
            // completed for all buckets AND the operator asked for it.
            //
            // M8.K.1 follow-up: when the operator enables v2 AND v2
            // wins vs matvec at some bucket where v1 lost, re-derive
            // gemmMinM using v2's timings. Without this the dispatch
            // stays at gemmMinM=never (v1 lost) even though v2 would
            // win, forcing the operator to also set features.gemmMinM
            // manually. Winning is defined against matvec (vecMsAtM),
            // not against v1, because that's the actual dispatch
            // fallback when GEMM isn't picked.
            if (features.gemmV2) {
                entry.useGemmV2 = true;
                for (std::size_t bi = 0;
                     bi < kAutotuneMBuckets.size(); ++bi)
                {
                    if (entry.gemmV2MsAtM[bi] * 1.05
                            < entry.vecMsAtM[bi])
                    {
                        entry.gemmMinM = kAutotuneMBuckets[bi];
                        break;
                    }
                }
                MM_LOG_INFO("gpummm",
                            "features.gemmV2=true — {} GEMM will "
                            "dispatch through v2 when M >= gemmMinM={} "
                            "(re-derived from v2 vs matvec bench)",
                            qt->name(),
                            entry.gemmMinM == kGemmMinMNever
                                ? std::string{"never"}
                                : std::to_string(entry.gemmMinM));
            }
        }

        // M8.H.3 / M8.M — DP4A bench for any type that has a DP4A slot
        // (Q8_0 and Q4_K currently). Benched at M=16 only
        // (kAutotuneMBuckets[0]); DP4A is shape-agnostic so a M=16 win
        // covers all M.
        if (entry.dp4a.has_value() && !forceDisableDp4a) {
            const std::size_t Mdp = kAutotuneMBuckets[0];
            const std::size_t xqBytes = Mdp * K * sizeof(std::int8_t);
            const std::size_t xsBytes = Mdp * sizeof(float);
            if (xqBytes > _dp4aXqBytes || xsBytes > _dp4aScaleBytes) {
                MM_LOG_WARN("gpummm",
                            "autotune: DP4A bench for {} skipped — "
                            "bench shape (M={}, K={}) exceeds scratch "
                            "bounds. Bump kDp4aMax* together.",
                            qt->name(), Mdp, K);
            } else {
                entry.useDp4a  = false;
                entry.gemmMinM = kGemmMinMNever;  // force matvec-loop ref
                const std::size_t elts = Mdp * N;
                std::vector<float> yVec(elts);
                for (std::size_t m = 0; m < Mdp; ++m) {
                    matmulAsync(type, wUsm, N, K,
                                static_cast<const float*>(xUsm) + m * K,
                                /*M=*/1,
                                static_cast<float*>(yUsm) + m * N,
                                static_cast<float*>(sUsm));
                }
                _queue.flush();
                std::memcpy(yVec.data(), yUsm, elts * sizeof(float));

                entry.useDp4a = true;
                std::vector<float> yDp4a(elts);
                for (int i = 0; i < nWarmup; ++i) {
                    matmulAsync(type, wUsm, N, K,
                                static_cast<const float*>(xUsm), Mdp,
                                static_cast<float*>(yUsm),
                                static_cast<float*>(sUsm));
                    _queue.flush();
                }
                matmulAsync(type, wUsm, N, K,
                            static_cast<const float*>(xUsm), Mdp,
                            static_cast<float*>(yUsm),
                            static_cast<float*>(sUsm));
                _queue.flush();
                std::memcpy(yDp4a.data(), yUsm, elts * sizeof(float));

                float maxAbs  = 0.0F;
                float maxDiff = 0.0F;
                for (std::size_t i = 0; i < elts; ++i) {
                    maxAbs  = std::max(maxAbs,  std::fabs(yVec[i]));
                    maxDiff = std::max(maxDiff,
                                       std::fabs(yVec[i] - yDp4a[i]));
                }
                const float dp4aTol = std::max(0.05F * maxAbs, 1e-3F);
                if (!(maxDiff <= dp4aTol)) {
                    MM_LOG_WARN("gpummm",                                "autotune parity FAIL for {} DP4A — "
                                "maxDiff={:.6g} maxRef={:.6g} tol={:.6g}. "
                                "Sticking with matvec/gemm decision, "
                                "skipping DP4A timing bench.",
                                qt->name(), maxDiff, maxAbs, dp4aTol);
                    entry.useDp4a        = false;
                    entry.autotuneSource = "dp4a_parity_fail";
                } else {
                    MM_LOG_INFO("gpummm",
                                "autotune parity OK for {} DP4A — "
                                "maxDiff={:.6g} maxRef={:.6g} tol={:.6g}",
                                qt->name(), maxDiff, maxAbs, dp4aTol);

                    std::vector<double> dp4aMs;
                    dp4aMs.reserve(nTimed);
                    for (int it2 = 0; it2 < nTimed; ++it2) {
                        const auto t0 = clk::now();
                        matmulAsync(type, wUsm, N, K,
                                    static_cast<const float*>(xUsm), Mdp,
                                    static_cast<float*>(yUsm),
                                    static_cast<float*>(sUsm));
                        _queue.flush();
                        const auto t1 = clk::now();
                        dp4aMs.push_back(
                            std::chrono::duration<double, std::milli>(t1 - t0)
                                .count());
                    }
                    const double dp4aMed = medianMs(std::move(dp4aMs));
                    entry.lastDp4aMs = dp4aMed;

                    const double bestNonDp4a =
                        std::min(entry.vecMsAtM[0], entry.gemmMsAtM[0]);
                    const bool pickDp4a =
                        forceEnableDp4a ||
                        (dp4aMed * 1.05 < bestNonDp4a);
                    entry.useDp4a = pickDp4a;
                    if (pickDp4a) {
                        entry.autotuneSource =
                            forceEnableDp4a ? "env_force_dp4a" : "bench";
                    }
                    MM_LOG_INFO("gpummm",
                                "autotune: {} DP4A {:.2f} ms vs "
                                "best-non-dp4a@M=16 {:.2f} ms → picked {}",
                                qt->name(), dp4aMed, bestNonDp4a,
                                pickDp4a ? "dp4a" : "matvec-or-gemm-by-M");
                }

                // Restore the M-threshold that the timing loop derived
                // — the DP4A bench mutated it as a dispatch-control
                // temporarily. useDp4a takes priority at dispatch time
                // when true, so if DP4A won the caller still hits DP4A;
                // if it lost, the M-threshold is what applies.
                entry.gemmMinM = kGemmMinMNever;
                for (std::size_t bi = 0;
                     bi < kAutotuneMBuckets.size(); ++bi)
                {
                    if (entry.gemmMsAtM[bi] * 1.05 < entry.vecMsAtM[bi]) {
                        entry.gemmMinM = kAutotuneMBuckets[bi];
                        break;
                    }
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

    // DP4A hot path for any type whose autotune picked it AND that has
    // a DP4A kernel loaded. Bounds overflow falls through to vec/gemm
    // below with a one-shot warn.
    if (entry.useDp4a && entry.dp4a.has_value()) {
        if (M <= kDp4aMaxM && K <= kDp4aMaxK) {
            dispatchDp4aFromFloat(type, X, W, N, K, M, Y);
            return;
        }
        if (!_dp4aScratchOverflowWarned) {
            const QuantType* qtWarn = quantType(type);
            MM_LOG_WARN("gpummm",
                        "GpuMatmul: DP4A dispatch declined for {} "
                        "(M={}, K={}) — exceeds scratch bounds "
                        "(kDp4aMaxM={}, kDp4aMaxK={}). Falling back to "
                        "vec/gemm. Bump both constants together if this "
                        "happens for every request.",
                        qtWarn != nullptr
                            ? std::string{qtWarn->name()} : "type?",
                        M, K, kDp4aMaxM, kDp4aMaxK);
            _dp4aScratchOverflowWarned = true;
        }
    }

    // M5h: 4 outputs per workgroup (4 subgroups × 16 threads).
    const std::uint32_t nGroups = static_cast<std::uint32_t>(
        (N + kOutputsPerGroup - 1) / kOutputsPerGroup);

    // M8.J — batched GEMM kernel wins only past a per-type threshold
    // learned by autotune. `gemmMinM` = kGemmMinMNever means matvec-
    // loop is always faster on this shape / iGPU / driver combination.
    // M=1 (decode) never triggers GEMM because gemmMinM is at least 2
    // (autotune only benches M=16 and up; the M=1 path stays on matvec
    // for launch-efficiency reasons even when GEMM would numerically win).
    if (M >= entry.gemmMinM && entry.gemm.has_value()) {
        // M8.K.1 — if the v2 GEMM prototype is loaded AND the operator
        // opted into it, dispatch v2 instead of v1. Signature and
        // launch geometry are identical; only the M-tile differs, so
        // the per-workgroup count needs the v2 M-tile.
        const bool useV2 =
            entry.useGemmV2 && entry.gemmV2.has_value();
        runtime::GpuKernel& kern =
            useV2 ? entry.gemmV2->kernel : entry.gemm->kernel;
        const std::size_t mTile =
            useV2 ? entry.gemmV2MTile : entry.gemmMTile;

        kern.setGroupSize(kLocalSize, 1, 1);
        kern.setPtr(0, X);
        kern.setPtr(1, W);
        kern.setPtr(2, Y);
        kern.setValue<std::int32_t>(3, static_cast<std::int32_t>(K));
        kern.setValue<std::int32_t>(4, static_cast<std::int32_t>(N));
        kern.setValue<std::int32_t>(5, static_cast<std::int32_t>(M));

        const std::uint32_t mGroups = static_cast<std::uint32_t>(
            (M + mTile - 1) / mTile);
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

void GpuMatmul::matmulDp4aAsync(model::GgmlType    type,
                                const std::int8_t* Xq,
                                const float*       Xscale,
                                const void*        W,
                                std::size_t        N,
                                std::size_t        K,
                                std::size_t        M,
                                float*             Y) {
    auto it = _entries.find(type);
    if (it == _entries.end() || !it->second.dp4a.has_value()) {
        throw std::runtime_error(
            "GpuMatmul::matmulDp4aAsync: DP4A kernel not loaded for "
            "this type — check dp4aAvailable(type) before calling");
    }
    if (M == 0 || N == 0 || K == 0) {
        return;
    }
    const QuantType* qt = quantType(type);
    const std::size_t blockElts =
        qt != nullptr ? qt->blockElements() : 0;
    if (blockElts == 0 || K % blockElts != 0) {
        throw std::runtime_error(
            "GpuMatmul::matmulDp4aAsync: K=" + std::to_string(K) +
            " is not a multiple of blockElements=" +
            std::to_string(blockElts) + " for this quant type");
    }

    runtime::GpuKernel& kern = it->second.dp4a->kernel;
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

void GpuMatmul::dispatchDp4aFromFloat(model::GgmlType type,
                                      const float*    X,
                                      const void*     W,
                                      std::size_t     N,
                                      std::size_t     K,
                                      std::size_t     M,
                                      float*          Y) {
    // Bounds are the caller's responsibility (matmulAsync checks them
    // before routing here). This method just wires the two kernels
    // together on the shared queue.
    auto* xq = static_cast<std::int8_t*>(_dp4aXqUsm);
    auto* xs = static_cast<float*>(_dp4aScaleUsm);

    _ops.xQuantI8Async(X, xq, xs, M, K);
    matmulDp4aAsync(type, xq, xs, W, N, K, M, Y);
}

bool GpuMatmul::dp4aAvailable() const noexcept {
    for (const auto& [type, entry] : _entries) {
        (void)type;
        if (entry.dp4a.has_value()) return true;
    }
    return false;
}

bool GpuMatmul::dp4aAvailable(model::GgmlType type) const noexcept {
    auto it = _entries.find(type);
    return it != _entries.end() && it->second.dp4a.has_value();
}

void GpuMatmul::sync() {
    _queue.flush();
}

std::vector<GpuMatmul::AutotuneReport> GpuMatmul::autotuneReport() const {
    std::vector<AutotuneReport> out;
    out.reserve(_entries.size());
    for (const auto& [type, entry] : _entries) {
        const QuantType* qt = quantType(type);
        const bool dp4aForThisType = entry.dp4a.has_value();
        AutotuneReport r{};
        r.name          = qt != nullptr ? std::string{qt->name()} : "??";
        r.gemmAvailable = entry.gemm.has_value();
        r.gemmPicked    = entry.gemmMinM != kGemmMinMNever;
        // Legacy vec_ms / gemm_ms mirror the M=16 bucket for backward
        // compat with pre-M8.J telemetry consumers.
        r.vecMs         = entry.vecMsAtM[0];
        r.gemmMs        = entry.gemmMsAtM[0];
        for (std::size_t i = 0; i < kAutotuneMBuckets.size(); ++i) {
            r.mBuckets[i]  = kAutotuneMBuckets[i];
            r.vecMsAtM[i]  = entry.vecMsAtM[i];
            r.gemmMsAtM[i] = entry.gemmMsAtM[i];
        }
        r.gemmMinM      = entry.gemmMinM;
        r.dp4aAvailable = dp4aForThisType;
        r.dp4aPicked    = entry.useDp4a;
        r.dp4aMs        = entry.lastDp4aMs;
        r.gemmV2Available = entry.gemmV2.has_value();
        r.gemmV2Picked    = entry.useGemmV2;
        for (std::size_t i = 0; i < kAutotuneMBuckets.size(); ++i) {
            r.gemmV2MsAtM[i] = entry.gemmV2MsAtM[i];
        }
        r.source        = entry.autotuneSource.empty()
                              ? std::string{"pending"}
                              : entry.autotuneSource;
        out.push_back(std::move(r));
    }
    return out;
}

} // namespace mimirmind::compute