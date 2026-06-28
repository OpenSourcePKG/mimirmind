#include "compute/Matmul.hpp"

#include "compute/Dequant.hpp"
#include "runtime/Log.hpp"

#include <cstdint>
#include <stdexcept>

namespace mimirmind::compute {

namespace {

std::size_t bytesPerRow(model::GgmlType type, std::size_t K) {
    const auto info = model::typeInfo(type);
    if (info.blockSize == 0 || info.typeSize == 0) {
        throw std::runtime_error("matmul: unsupported weight type");
    }
    if (K % info.blockSize != 0) {
        throw std::runtime_error(
            "matmul: K not a multiple of the type's block size");
    }
    return (K / info.blockSize) * info.typeSize;
}

} // namespace

void matmul(model::GgmlType weightType,
            const void*     W,
            std::size_t     N,
            std::size_t     K,
            const float*    X,
            std::size_t     M,
            float*          Y,
            float*          scratch) {
    if (W == nullptr || X == nullptr || Y == nullptr || scratch == nullptr) {
        throw std::runtime_error("matmul: null pointer");
    }
    if (N == 0 || K == 0) {
        return;
    }

    const std::size_t rowBytes = bytesPerRow(weightType, K);
    const auto*       base     = static_cast<const std::uint8_t*>(W);

    MM_LOG_DEBUG("matmul",
                 "start — M={} K={} N={} type={} row={} bytes",
                 M, K, N, model::typeInfo(weightType).name, rowBytes);

    constexpr std::size_t kProgressEvery = 8192;

    for (std::size_t n = 0; n < N; ++n) {
        dequantToF32(weightType, base + n * rowBytes, K, scratch);

        for (std::size_t m = 0; m < M; ++m) {
            const float* xr = X + m * K;
            double acc = 0.0;
            for (std::size_t k = 0; k < K; ++k) {
                acc += static_cast<double>(xr[k]) *
                       static_cast<double>(scratch[k]);
            }
            Y[m * N + n] = static_cast<float>(acc);
        }

        if (((n + 1) % kProgressEvery) == 0) {
            MM_LOG_DEBUG("matmul", "progress — {}/{} rows", n + 1, N);
        }
    }

    MM_LOG_DEBUG("matmul", "done — {} output rows produced", N);
}

} // namespace mimirmind::compute