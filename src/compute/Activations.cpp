#include "compute/Activations.hpp"

#include <cmath>

namespace mimirmind::compute {

void siluInPlace(float* x, std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i) {
        const float v = x[i];
        x[i] = v / (1.0F + std::exp(-v));
    }
}

void mulInPlace(float* a, const float* b, std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i) {
        a[i] *= b[i];
    }
}

} // namespace mimirmind::compute