#include "compute/GpuMatmul.hpp"

#include "compute/Matmul.hpp"
#include "runtime/L0Context.hpp"
#include "runtime/Log.hpp"

#include <cstdint>

namespace mimirmind::compute {

GpuMatmul::GpuMatmul(runtime::L0Context& ctx)
    : _ctx{ctx},
      _queue{ctx},
      _q4kModule{ctx, "matmul_q4k_vec"},
      _q4kKernel{_q4kModule.kernel("matmul_q4k_vec")},
      _q6kModule{ctx, "matmul_q6k_vec"},
      _q6kKernel{_q6kModule.kernel("matmul_q6k_vec")}
{
    MM_LOG_INFO("gpummm",
                "GpuMatmul ready — Q4_K + Q6_K kernels loaded, local_size={}",
                kLocalSize);
}

bool GpuMatmul::supports(model::GgmlType type) const noexcept {
    return type == model::GgmlType::Q4_K || type == model::GgmlType::Q6_K;
}

void GpuMatmul::matmulAsync(model::GgmlType type,
                            const void*     W,
                            std::size_t     N,
                            std::size_t     K,
                            const float*    X,
                            std::size_t     M,
                            float*          Y,
                            float*          scratch) {
    if (!supports(type)) {
        // CPU fallback. If async GPU work is pending, sync it first so
        // ordering vs prior matmulAsync calls is preserved.
        _queue.flush();
        compute::matmul(type, W, N, K, X, M, Y, scratch);
        return;
    }

    runtime::GpuKernel& kern = (type == model::GgmlType::Q4_K)
        ? _q4kKernel : _q6kKernel;

    const std::uint32_t groups =
        static_cast<std::uint32_t>((N + kLocalSize - 1) / kLocalSize);
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