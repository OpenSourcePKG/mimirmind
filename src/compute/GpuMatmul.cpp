#include "compute/GpuMatmul.hpp"

#include "compute/Matmul.hpp"
#include "compute/QuantType.hpp"
#include "compute/QuantTypeRegistry.hpp"
#include "runtime/L0Context.hpp"
#include "runtime/Log.hpp"

#include <cstdint>
#include <cstdlib>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

namespace mimirmind::compute {

namespace {

// Env-var kill-switch. Set MIMIRMIND_DISABLE_GEMM=1 to force the
// per-token matvec-loop path even for M>1. The M5i GEMM design has
// only been validated for correctness on the dev-time HD Graphics 630
// (Gen9.5, 2017) — a very different µarch from L0_TARGET_HOST's
// Xe-LPG (Gen13, Meteor Lake). This switch lets us A/B on the target
// without a rebuild.
bool gemmDisabledByEnv() noexcept {
    const char* v = std::getenv("MIMIRMIND_DISABLE_GEMM");
    if (v == nullptr) return false;
    const std::string_view s{v};
    return !s.empty() && s != "0" && s != "false" && s != "off";
}

} // namespace

GpuMatmul::GpuMatmul(runtime::L0Context& ctx, runtime::CommandQueue& queue)
    : _ctx{ctx},
      _queue{queue}
{
    const bool gemmDisabled = gemmDisabledByEnv();

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
        const bool gemmEnabled = !gemmName.empty() && !gemmDisabled;
        if (gemmEnabled) {
            gemmSlot.emplace(loadSlot(gemmName));
            gemmMTile = qt->gpuMatmulGemmMTile();
        }

        _entries.emplace(qt->ggmlType(),
                         Entry{std::move(vecSlot),
                               std::move(gemmSlot),
                               gemmMTile});

        if (!first) {
            loaded << " + ";
        }
        loaded << qt->name();
        if (gemmEnabled) {
            loaded << "(vec+gemm)";
        } else if (!gemmName.empty()) {
            loaded << "(vec, gemm disabled)";
        } else {
            loaded << "(vec)";
        }
        first = false;
    }

    MM_LOG_INFO("gpummm",
                "GpuMatmul ready — {} kernels loaded, local_size={} "
                "(sg={}, {} outputs/group){}",
                loaded.str(), kLocalSize, kSubgroupSize, kOutputsPerGroup,
                gemmDisabled ? " [MIMIRMIND_DISABLE_GEMM=1]" : "");
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
    if (M > 1 && entry.gemm.has_value()) {
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