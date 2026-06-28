#include "compute/Rope.hpp"

#include "runtime/Log.hpp"

#include <cmath>
#include <stdexcept>

namespace mimirmind::compute {

void applyRopeInPlace(float*      x,
                      std::size_t seqLen,
                      std::size_t numHeads,
                      std::size_t headDim,
                      std::size_t startPos,
                      float       base) {
    if (headDim % 2 != 0) {
        throw std::runtime_error("RoPE: headDim must be even");
    }
    const std::size_t halfDim = headDim / 2;
    const float       invDim  = 1.0F / static_cast<float>(headDim);

    for (std::size_t p = 0; p < seqLen; ++p) {
        const float pos = static_cast<float>(startPos + p);
        for (std::size_t h = 0; h < numHeads; ++h) {
            float* head = x + (p * numHeads + h) * headDim;
            for (std::size_t i = 0; i < halfDim; ++i) {
                const float freq  = std::pow(base,
                    -static_cast<float>(2 * i) * invDim);
                const float theta = pos * freq;
                const float c     = std::cos(theta);
                const float s     = std::sin(theta);

                const float a = head[i];
                const float b = head[i + halfDim];
                head[i]           = a * c - b * s;
                head[i + halfDim] = a * s + b * c;
            }
        }
    }

    MM_LOG_DEBUG("rope",
                 "applied — seqLen={} heads={} headDim={} startPos={} base={}",
                 seqLen, numHeads, headDim, startPos,
                 static_cast<double>(base));
}

} // namespace mimirmind::compute