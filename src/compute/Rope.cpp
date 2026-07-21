// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "compute/Rope.hpp"

#include "core/log/Log.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

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

void applyRopeInPlaceWithFactors(float*       x,
                                 const float* freqFactors,
                                 std::size_t  seqLen,
                                 std::size_t  numHeads,
                                 std::size_t  headDim,
                                 std::size_t  startPos,
                                 float        base) {
    if (headDim % 2 != 0) {
        throw std::runtime_error("RoPE: headDim must be even");
    }
    if (freqFactors == nullptr) {
        throw std::runtime_error(
            "RoPE-with-factors: freqFactors is null (use applyRopeInPlace "
            "for the unweighted variant)");
    }
    const std::size_t halfDim = headDim / 2;
    const float       invDim  = 1.0F / static_cast<float>(headDim);

    for (std::size_t p = 0; p < seqLen; ++p) {
        const float pos = static_cast<float>(startPos + p);
        for (std::size_t h = 0; h < numHeads; ++h) {
            float* head = x + (p * numHeads + h) * headDim;
            for (std::size_t i = 0; i < halfDim; ++i) {
                const float f = freqFactors[i];
                if (f == 0.0F) {
                    throw std::runtime_error(
                        "RoPE-with-factors: freqFactors[" +
                        std::to_string(i) + "] is zero (would NaN)");
                }
                const float freq  = std::pow(base,
                    -static_cast<float>(2 * i) * invDim) / f;
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
                 "applied-with-factors — seqLen={} heads={} headDim={} "
                 "startPos={} base={}",
                 seqLen, numHeads, headDim, startPos,
                 static_cast<double>(base));
}

void applyMropeInPlace(float*              x,
                       std::size_t         seqLen,
                       std::size_t         numHeads,
                       std::size_t         headDim,
                       std::size_t         startPos,
                       float               base,
                       const std::int32_t* sections) {
    if (headDim % 2 != 0) {
        throw std::runtime_error("IMRoPE: headDim must be even");
    }
    const std::size_t halfDim = headDim / 2;
    const float       invDim  = 1.0F / static_cast<float>(headDim);

    const std::int32_t s0 = sections ? sections[0] : 0;
    const std::int32_t s1 = sections ? sections[1] : 0;
    const std::int32_t s2 = sections ? sections[2] : 0;
    const std::int32_t s3 = sections ? sections[3] : 0;
    const std::int32_t sectDims = s0 + s1 + s2 + s3;

    for (std::size_t p = 0; p < seqLen; ++p) {
        // Text-only: all four axis positions are the sequence position.
        const float pos       = static_cast<float>(startPos + p);
        const float posAxis[4] = {pos, pos, pos, pos};
        for (std::size_t h = 0; h < numHeads; ++h) {
            float* head = x + (p * numHeads + h) * headDim;
            for (std::size_t i = 0; i < halfDim; ++i) {
                // IMRoPE sector rule (ggml is_imrope branch). sectDims==0
                // degenerates to plain RoPE (posSel = time axis).
                float posSel = posAxis[0];
                if (sectDims > 0) {
                    const std::int32_t sector =
                        static_cast<std::int32_t>(i) % sectDims;
                    if (sector % 3 == 1 && sector < 3 * s1) {
                        posSel = posAxis[1];
                    } else if (sector % 3 == 2 && sector < 3 * s2) {
                        posSel = posAxis[2];
                    } else if (sector % 3 == 0 && sector < 3 * s0) {
                        posSel = posAxis[0];
                    } else {
                        posSel = posAxis[3];
                    }
                }
                const float freq  = std::pow(base,
                    -static_cast<float>(2 * i) * invDim);
                const float theta = posSel * freq;
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
                 "applied-imrope — seqLen={} heads={} headDim={} startPos={} "
                 "base={} sections=[{},{},{},{}]",
                 seqLen, numHeads, headDim, startPos,
                 static_cast<double>(base), s0, s1, s2, s3);
}

} // namespace mimirmind::compute