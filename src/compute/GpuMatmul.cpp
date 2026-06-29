#include "compute/GpuMatmul.hpp"

#include "compute/Matmul.hpp"
#include "compute/QuantType.hpp"
#include "compute/QuantTypeRegistry.hpp"
#include "runtime/L0Context.hpp"
#include "runtime/Log.hpp"

#include <cstdint>
#include <sstream>
#include <string>

namespace mimirmind::compute {

GpuMatmul::GpuMatmul(runtime::L0Context& ctx, runtime::CommandQueue& queue)
    : _ctx{ctx},
      _queue{queue}
{
    std::ostringstream loaded;
    bool first = true;

    for (const QuantType* qt : allQuantTypes()) {
        const auto moduleName = qt->gpuMatmulModule();
        if (moduleName.empty()) {
            continue;
        }

        // GpuModule wants a string_view; the kernel-lookup wants c_str.
        const std::string nameStr{moduleName};
        auto module = std::make_unique<runtime::GpuModule>(_ctx, moduleName);
        runtime::GpuKernel kernel{module->kernel(nameStr.c_str())};

        _entries.emplace(qt->ggmlType(),
                         Entry{std::move(module), kernel});

        if (!first) {
            loaded << " + ";
        }
        loaded << qt->name();
        first = false;
    }

    MM_LOG_INFO("gpummm",
                "GpuMatmul ready — {} kernels loaded, local_size={} "
                "(sg={}, {} outputs/group)",
                loaded.str(), kLocalSize, kSubgroupSize, kOutputsPerGroup);
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

    runtime::GpuKernel& kern = it->second.kernel;

    // M5h: 4 outputs per workgroup (4 subgroups × 16 threads).
    const std::uint32_t groups = static_cast<std::uint32_t>(
        (N + kOutputsPerGroup - 1) / kOutputsPerGroup);
    kern.setGroupSize(kLocalSize, 1, 1);

    // One appendLaunch per row of X. Per the Level Zero spec, args are
    // captured at append time, so each loop iteration's setPtr/setValue
    // do not affect the previously-recorded commands.
    for (std::size_t m = 0; m < M; ++m) {
        const float* xRow = X + m * K;
        float*       yRow = Y + m * N;

        kern.setPtr(0, xRow);
        kern.setPtr(1, W);
        kern.setPtr(2, yRow);
        kern.setValue<std::int32_t>(3, static_cast<std::int32_t>(K));
        kern.setValue<std::int32_t>(4, static_cast<std::int32_t>(N));

        _queue.appendLaunch(kern, groups, 1, 1);
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