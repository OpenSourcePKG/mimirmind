#include "compute/GpuOps.hpp"

#include "runtime/L0Context.hpp"
#include "runtime/Log.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace mimirmind::compute {

namespace {

std::int32_t toInt32(std::size_t v, const char* tag) {
    if (v > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::runtime_error(
            std::string{"GpuOps: "} + tag +
            " overflows int32 ("  + std::to_string(v) + ")");
    }
    return static_cast<std::int32_t>(v);
}

std::uint32_t groupsForN(std::size_t n, std::uint32_t local) {
    const std::size_t g = (n + local - 1) / local;
    if (g > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("GpuOps: workgroup count overflows uint32");
    }
    return static_cast<std::uint32_t>(g);
}

} // namespace

GpuOps::GpuOps(runtime::L0Context& ctx, runtime::CommandQueue& queue)
    : _ctx{ctx},
      _queue{queue},
      _rmsnormModule    {ctx, "rmsnorm"},
      _rmsnormKernel    {_rmsnormModule.kernel("rmsnorm")},
      _addBiasModule    {ctx, "add_bias"},
      _addBiasKernel    {_addBiasModule.kernel("add_bias")},
      _addResidualModule{ctx, "add_residual"},
      _addResidualKernel{_addResidualModule.kernel("add_residual")},
      _siluMulModule    {ctx, "silu_mul"},
      _siluMulKernel    {_siluMulModule.kernel("silu_mul")}
{
    MM_LOG_INFO("gpuops",
                "GpuOps ready — rmsnorm/add_bias/add_residual/silu_mul loaded "
                "(rms local={}, elementwise local={})",
                kRmsnormLocalSize, kElementwiseLocalSize);
}

void GpuOps::rmsNormAsync(const float* x,
                          std::size_t  M,
                          std::size_t  K,
                          const float* weight,
                          float        eps,
                          float*       y) {
    if (M == 0 || K == 0) {
        return;
    }
    const std::int32_t Ki = toInt32(K, "rmsNorm K");
    _rmsnormKernel.setPtr(0, x);
    _rmsnormKernel.setPtr(1, weight);
    _rmsnormKernel.setPtr(2, y);
    _rmsnormKernel.setValue<float>(3, eps);
    _rmsnormKernel.setValue<std::int32_t>(4, Ki);
    _rmsnormKernel.setGroupSize(kRmsnormLocalSize, 1, 1);
    // One workgroup per row.
    _queue.appendLaunch(_rmsnormKernel,
                        static_cast<std::uint32_t>(M), 1, 1);
}

void GpuOps::addBiasAsync(float*       y,
                          std::size_t  M,
                          std::size_t  K,
                          const float* bias) {
    if (M == 0 || K == 0) {
        return;
    }
    const std::int32_t Mi = toInt32(M, "addBias M");
    const std::int32_t Ki = toInt32(K, "addBias K");
    _addBiasKernel.setPtr(0, y);
    _addBiasKernel.setPtr(1, bias);
    _addBiasKernel.setValue<std::int32_t>(2, Mi);
    _addBiasKernel.setValue<std::int32_t>(3, Ki);
    _addBiasKernel.setGroupSize(kElementwiseLocalSize, 1, 1);
    _queue.appendLaunch(_addBiasKernel,
                        groupsForN(M * K, kElementwiseLocalSize), 1, 1);
}

void GpuOps::addResidualAsync(float*       y,
                              const float* x,
                              std::size_t  n) {
    if (n == 0) {
        return;
    }
    const std::int32_t ni = toInt32(n, "addResidual n");
    _addResidualKernel.setPtr(0, y);
    _addResidualKernel.setPtr(1, x);
    _addResidualKernel.setValue<std::int32_t>(2, ni);
    _addResidualKernel.setGroupSize(kElementwiseLocalSize, 1, 1);
    _queue.appendLaunch(_addResidualKernel,
                        groupsForN(n, kElementwiseLocalSize), 1, 1);
}

void GpuOps::siluMulAsync(float*       gate,
                          const float* up,
                          std::size_t  n) {
    if (n == 0) {
        return;
    }
    const std::int32_t ni = toInt32(n, "siluMul n");
    _siluMulKernel.setPtr(0, gate);
    _siluMulKernel.setPtr(1, up);
    _siluMulKernel.setValue<std::int32_t>(2, ni);
    _siluMulKernel.setGroupSize(kElementwiseLocalSize, 1, 1);
    _queue.appendLaunch(_siluMulKernel,
                        groupsForN(n, kElementwiseLocalSize), 1, 1);
}

} // namespace mimirmind::compute