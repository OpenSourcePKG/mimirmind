// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "compute/GatedDeltaNet.hpp"

#include <cmath>
#include <vector>

namespace mimirmind::compute {

void gatedDeltaNetRecurrent(const float* q,
                            const float* k,
                            const float* v,
                            const float* gLog,
                            const float* beta,
                            float*       state,
                            float*       out,
                            std::size_t  T,
                            std::size_t  H,
                            std::size_t  S) {
    const float qScale = 1.0F / std::sqrt(static_cast<float>(S));

    // Per-token scratch reused across heads: the predicted value s^T k and
    // the gated error d, both length S.
    std::vector<float> sk(S);
    std::vector<float> d(S);
    std::vector<float> qs(S);

    for (std::size_t t = 0; t < T; ++t) {
        for (std::size_t h = 0; h < H; ++h) {
            const float* qt = q + (t * H + h) * S;
            const float* kt = k + (t * H + h) * S;
            const float* vt = v + (t * H + h) * S;
            float*       s  = state + h * S * S;   // s[i*S + j]
            float*       ot = out + (t * H + h) * S;

            const float gDecay = std::exp(gLog[t * H + h]);
            const float b       = beta[t * H + h];

            for (std::size_t i = 0; i < S; ++i) {
                qs[i] = qt[i] * qScale;
            }

            // Decay the state, then predict sk[j] = sum_i s[i,j] * k[i]
            // from the decayed state. Fuse both passes over i.
            for (std::size_t j = 0; j < S; ++j) {
                sk[j] = 0.0F;
            }
            for (std::size_t i = 0; i < S; ++i) {
                const float ki = kt[i];
                float* srow = s + i * S;
                for (std::size_t j = 0; j < S; ++j) {
                    srow[j] *= gDecay;
                    sk[j]   += srow[j] * ki;
                }
            }

            // Gated prediction error d[j] = (v[j] - sk[j]) * beta.
            for (std::size_t j = 0; j < S; ++j) {
                d[j] = (vt[j] - sk[j]) * b;
            }

            // Rank-1 update s[i,j] += k[i] * d[j], and read out from the
            // UPDATED state o[j] = sum_i s[i,j] * qs[i]. Two passes: the
            // readout must see the full updated matrix.
            for (std::size_t i = 0; i < S; ++i) {
                const float ki = kt[i];
                float* srow = s + i * S;
                for (std::size_t j = 0; j < S; ++j) {
                    srow[j] += ki * d[j];
                }
            }
            for (std::size_t j = 0; j < S; ++j) {
                ot[j] = 0.0F;
            }
            for (std::size_t i = 0; i < S; ++i) {
                const float qi = qs[i];
                const float* srow = s + i * S;
                for (std::size_t j = 0; j < S; ++j) {
                    ot[j] += srow[j] * qi;
                }
            }
        }
    }
}

namespace {
// Numerically-stable softplus: log(1 + exp(x)).
inline float softplus(float x) {
    return x > 0.0F ? x + std::log1p(std::exp(-x))
                    : std::log1p(std::exp(x));
}
} // namespace

void deltanetGate(const float* alpha,
                  const float* ssmA,
                  const float* ssmDt,
                  float*       gLog,
                  std::size_t  T,
                  std::size_t  H) {
    for (std::size_t t = 0; t < T; ++t) {
        for (std::size_t h = 0; h < H; ++h) {
            const float sp = softplus(alpha[t * H + h] + ssmDt[h]);
            gLog[t * H + h] = -std::exp(ssmA[h]) * sp;
        }
    }
}

void sigmoidInPlace(float* y, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        y[i] = 1.0F / (1.0F + std::exp(-y[i]));
    }
}

void gatherHeadsFromChannels(const float* src,
                             float*       dst,
                             std::size_t  T,
                             std::size_t  offset,
                             std::size_t  srcHeads,
                             std::size_t  dstHeads,
                             std::size_t  S,
                             std::size_t  convTotalWidth) {
    for (std::size_t t = 0; t < T; ++t) {
        const float* srcTok = src + t * convTotalWidth + offset;
        float*       dstTok = dst + t * dstHeads * S;
        for (std::size_t hd = 0; hd < dstHeads; ++hd) {
            const std::size_t srcHead = hd % srcHeads;
            const float* srcHeadPtr = srcTok + srcHead * S;
            float*       dstHeadPtr = dstTok + hd * S;
            for (std::size_t s = 0; s < S; ++s) {
                dstHeadPtr[s] = srcHeadPtr[s];
            }
        }
    }
}

void causalConv1dSilu(const float* convInput,
                      const float* kernel,
                      float*       out,
                      std::size_t  T,
                      std::size_t  channels,
                      std::size_t  kernelSize) {
    for (std::size_t t = 0; t < T; ++t) {
        for (std::size_t c = 0; c < channels; ++c) {
            float acc = 0.0F;
            for (std::size_t kk = 0; kk < kernelSize; ++kk) {
                // convInput row (t + kk) — the state-prepended input
                // (time-major, channel contiguous). kernel is channel-major
                // (GGUF ssm_conv1d.weight [d_conv, conv_dim], ne0=d_conv
                // contiguous → weight[c*K + kk], matching ggml_ssm_conv).
                acc += convInput[(t + kk) * channels + c] *
                       kernel[c * kernelSize + kk];
            }
            // SiLU: acc * sigmoid(acc).
            out[t * channels + c] = acc / (1.0F + std::exp(-acc));
        }
    }
}

void l2NormInPlace(float*      x,
                   std::size_t rows,
                   std::size_t dim,
                   float       eps) {
    for (std::size_t r = 0; r < rows; ++r) {
        float* row = x + r * dim;
        double sumSq = 0.0;
        for (std::size_t j = 0; j < dim; ++j) {
            sumSq += static_cast<double>(row[j]) * static_cast<double>(row[j]);
        }
        // ggml_l2_norm: scale = 1 / max(sqrt(sumSq), eps).
        const float scale =
            1.0F / std::fmax(std::sqrt(static_cast<float>(sumSq)), eps);
        for (std::size_t j = 0; j < dim; ++j) {
            row[j] *= scale;
        }
    }
}

} // namespace mimirmind::compute