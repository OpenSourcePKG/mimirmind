// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/audio/WhisperConvStem.hpp"

#include <cmath>

namespace mimirmind::runtime::audio {

namespace {

// Exact (erf) GELU, as HF Whisper uses.
inline float gelu(float x) {
    return 0.5F * x * (1.0F + std::erf(x * 0.7071067811865476F));  // 1/sqrt(2)
}

// One conv1d layer, kernel 3, given stride and pad 1.
//   in  : [inC x inT] channel-major
//   W   : [outC x inC x 3] row-major
//   B   : [outC]
//   out : [outC x outT] channel-major, outT = (inT + 2*pad - 3)/stride + 1
// GELU is applied to every output element.
std::vector<float> conv1dGelu(const float* in, std::size_t inC, std::size_t inT,
                              const float* W, const float* B, std::size_t outC,
                              std::size_t stride) {
    constexpr int pad = 1;
    constexpr int kernel = 3;
    const std::size_t outT =
        (inT + 2 * pad - kernel) / stride + 1;
    std::vector<float> out(outC * outT);
    for (std::size_t oc = 0; oc < outC; ++oc) {
        for (std::size_t ot = 0; ot < outT; ++ot) {
            float acc = B[oc];
            const long start = static_cast<long>(ot * stride) - pad;
            for (std::size_t ic = 0; ic < inC; ++ic) {
                const float* wk = &W[(oc * inC + ic) * kernel];
                const float* ir = &in[ic * inT];
                for (int k = 0; k < kernel; ++k) {
                    const long it = start + k;
                    if (it >= 0 && static_cast<std::size_t>(it) < inT) {
                        acc += wk[k] * ir[it];
                    }
                }
            }
            out[oc * outT + ot] = gelu(acc);
        }
    }
    return out;
}

} // namespace

ConvStemOutput whisperConvStem(const float* mel, std::size_t nMels,
                               std::size_t nFrames, std::size_t dModel,
                               const float* conv1W, const float* conv1B,
                               const float* conv2W, const float* conv2B) {
    // conv1: nMels -> dModel, stride 1  => [dModel x nFrames]
    const std::vector<float> c1 =
        conv1dGelu(mel, nMels, nFrames, conv1W, conv1B, dModel, 1);
    // conv2: dModel -> dModel, stride 2 => [dModel x nCtx]
    const std::size_t nCtx = (nFrames + 2 - 3) / 2 + 1;
    const std::vector<float> c2 =
        conv1dGelu(c1.data(), dModel, nFrames, conv2W, conv2B, dModel, 2);

    // Transpose [dModel x nCtx] (channel-major) -> [nCtx x dModel] (time-major).
    ConvStemOutput out;
    out.nCtx = nCtx;
    out.dModel = dModel;
    out.data.resize(nCtx * dModel);
    for (std::size_t ch = 0; ch < dModel; ++ch) {
        for (std::size_t t = 0; t < nCtx; ++t) {
            out.data[t * dModel + ch] = c2[ch * nCtx + t];
        }
    }
    return out;
}

} // namespace mimirmind::runtime::audio
