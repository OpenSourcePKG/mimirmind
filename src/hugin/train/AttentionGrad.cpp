// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "hugin/train/AttentionGrad.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>


namespace {
/// Explicit float -> double widening (the build runs with -Wdouble-promotion).
constexpr double f64(float x) { return x; }
} // namespace

namespace mimirmind::hugin::train {

float AttentionGrad::effectiveScale(const Shape& s) {
    return s.scale > 0.0F ? s.scale : 1.0F / std::sqrt(static_cast<float>(s.headDim));
}

void AttentionGrad::keyRange(const Shape& s, std::size_t p, std::size_t& kMin, std::size_t& kMax) {
    kMax = std::min(s.positionOffset + p + 1, s.tk);
    kMin = (s.window > 0 && kMax > s.window) ? kMax - s.window : 0;
}

void AttentionGrad::forward(const float* q, const float* k, const float* v, const Shape& s,
                            float* out, std::vector<float>& probs) {
    if (s.heads == 0 || s.kvHeads == 0 || s.heads % s.kvHeads != 0) {
        throw std::runtime_error("AttentionGrad: heads must be a multiple of kvHeads");
    }
    const std::size_t D = s.headDim;
    const float       sc = effectiveScale(s);
    probs.assign(s.heads * s.tq * s.tk, 0.0F);
    for (std::size_t h = 0; h < s.heads; ++h) {
        const std::size_t hk = h * s.kvHeads / s.heads;
        for (std::size_t p = 0; p < s.tq; ++p) {
            std::size_t kMin = 0, kMax = 0;
            keyRange(s, p, kMin, kMax);
            const float* qr = q + (p * s.heads + h) * D;
            float*       pr = probs.data() + (h * s.tq + p) * s.tk;
            float mx = -INFINITY;
            for (std::size_t j = kMin; j < kMax; ++j) {
                const float* kr = k + (j * s.kvHeads + hk) * D;
                double dot = 0;
                for (std::size_t d = 0; d < D; ++d) dot += f64(qr[d]) * f64(kr[d]);
                pr[j] = static_cast<float>(dot) * sc;
                mx    = std::max(mx, pr[j]);
            }
            double sum = 0;
            for (std::size_t j = kMin; j < kMax; ++j) {
                pr[j] = std::exp(pr[j] - mx);
                sum  += f64(pr[j]);
            }
            float* o = out + (p * s.heads + h) * D;
            std::fill(o, o + D, 0.0F);
            for (std::size_t j = kMin; j < kMax; ++j) {
                pr[j] = static_cast<float>(f64(pr[j]) / sum);
                const float* vr = v + (j * s.kvHeads + hk) * D;
                for (std::size_t d = 0; d < D; ++d) o[d] += pr[j] * vr[d];
            }
        }
    }
}

void AttentionGrad::backward(const float* dOut, const float* q, const float* k, const float* v,
                             const Shape& s, const std::vector<float>& probs,
                             float* dq, float* dk, float* dv) {
    const std::size_t D  = s.headDim;
    const float       sc = effectiveScale(s);
    std::fill(dq, dq + s.tq * s.heads * D, 0.0F);
    std::fill(dk, dk + s.tk * s.kvHeads * D, 0.0F);
    std::fill(dv, dv + s.tk * s.kvHeads * D, 0.0F);
    std::vector<double> dp(s.tk);
    for (std::size_t h = 0; h < s.heads; ++h) {
        const std::size_t hk = h * s.kvHeads / s.heads;
        for (std::size_t p = 0; p < s.tq; ++p) {
            std::size_t kMin = 0, kMax = 0;
            keyRange(s, p, kMin, kMax);
            const float* pr  = probs.data() + (h * s.tq + p) * s.tk;
            const float* dor = dOut + (p * s.heads + h) * D;
            const float* qr  = q + (p * s.heads + h) * D;
            float*       dqr = dq + (p * s.heads + h) * D;
            // dV_j += P_j dO;  dP_j = dO . V_j;  dS_j = P_j (dP_j - sum_i P_i dP_i)
            double pdp = 0;
            for (std::size_t j = kMin; j < kMax; ++j) {
                const float* vr  = v + (j * s.kvHeads + hk) * D;
                float*       dvr = dv + (j * s.kvHeads + hk) * D;
                double acc = 0;
                for (std::size_t d = 0; d < D; ++d) {
                    acc    += f64(dor[d]) * f64(vr[d]);
                    dvr[d] += pr[j] * dor[d];
                }
                dp[j] = acc;
                pdp  += f64(pr[j]) * acc;
            }
            for (std::size_t j = kMin; j < kMax; ++j) {
                const float  ds  = static_cast<float>(f64(pr[j]) * (dp[j] - pdp)) * sc;
                const float* kr  = k + (j * s.kvHeads + hk) * D;
                float*       dkr = dk + (j * s.kvHeads + hk) * D;
                for (std::size_t d = 0; d < D; ++d) {
                    dqr[d] += ds * kr[d];
                    dkr[d] += ds * qr[d];
                }
            }
        }
    }
}

} // namespace mimirmind::hugin::train
