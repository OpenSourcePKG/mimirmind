// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/encoder/LinearProbeTrainer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>

namespace mimirmind::runtime::encoder {

namespace {

// Softmax of `z` in place (double), max-subtracted for stability.
void softmaxInPlace(std::vector<double>& z) {
    double m = z[0];
    for (double v : z) {
        m = std::max(m, v);
    }
    double sum = 0.0;
    for (double& v : z) {
        v = std::exp(v - m);
        sum += v;
    }
    for (double& v : z) {
        v /= sum;
    }
}

// argmax of W·x + b over classes for one example (accuracy eval).
std::size_t predict(const std::vector<double>& W, const std::vector<double>& b,
                    const float* x, std::size_t K, std::size_t D) {
    std::size_t best = 0;
    double bestZ = -std::numeric_limits<double>::infinity();
    for (std::size_t c = 0; c < K; ++c) {
        const double* w = W.data() + c * D;
        double z = b[c];
        for (std::size_t d = 0; d < D; ++d) {
            z += w[d] * static_cast<double>(x[d]);
        }
        if (z > bestZ) {
            bestZ = z;
            best  = c;
        }
    }
    return best;
}

} // namespace

LinearProbeTrainer::Result LinearProbeTrainer::train(
        std::span<const float> features, std::size_t dim,
        std::span<const std::int32_t> labels, std::size_t numClasses,
        const Config& cfg) {
    if (dim == 0) {
        throw std::invalid_argument("LinearProbeTrainer: dim must be > 0");
    }
    if (numClasses < 2) {
        throw std::invalid_argument("LinearProbeTrainer: numClasses must be >= 2");
    }
    if (features.size() % dim != 0) {
        throw std::invalid_argument("LinearProbeTrainer: features size not a multiple of dim");
    }
    const std::size_t N = features.size() / dim;
    if (N != labels.size()) {
        throw std::invalid_argument("LinearProbeTrainer: labels count != example count");
    }
    if (N < numClasses) {
        throw std::invalid_argument("LinearProbeTrainer: need at least numClasses examples");
    }
    for (std::int32_t y : labels) {
        if (y < 0 || static_cast<std::size_t>(y) >= numClasses) {
            throw std::invalid_argument("LinearProbeTrainer: label out of [0, numClasses)");
        }
    }

    const std::size_t K = numClasses;
    const std::size_t D = dim;

    // Deterministic train/val split.
    std::vector<std::size_t> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::mt19937_64 rng(cfg.seed);
    std::shuffle(idx.begin(), idx.end(), rng);
    std::size_t nVal = static_cast<std::size_t>(static_cast<double>(N) * cfg.valSplit);
    // Keep at least numClasses training examples.
    if (N - nVal < K) {
        nVal = N - K;
    }
    const std::size_t nTrain = N - nVal;

    // Parameters + Adam state (double).
    std::vector<double> W(K * D, 0.0);
    std::vector<double> b(K, 0.0);
    std::vector<double> mW(K * D, 0.0), vW(K * D, 0.0);
    std::vector<double> mb(K, 0.0), vb(K, 0.0);
    std::vector<double> gW(K * D, 0.0), gB(K, 0.0);

    const double beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
    double finalLoss = 0.0;
    std::size_t epochsRun = 0;

    for (std::size_t epoch = 0; epoch < cfg.epochs; ++epoch) {
        std::fill(gW.begin(), gW.end(), 0.0);
        std::fill(gB.begin(), gB.end(), 0.0);
        double loss = 0.0;

        for (std::size_t t = 0; t < nTrain; ++t) {
            const std::size_t i = idx[t];
            const float* x = features.data() + i * D;
            const std::size_t y = static_cast<std::size_t>(labels[i]);

            std::vector<double> z(K);
            for (std::size_t c = 0; c < K; ++c) {
                const double* w = W.data() + c * D;
                double s = b[c];
                for (std::size_t d = 0; d < D; ++d) {
                    s += w[d] * static_cast<double>(x[d]);
                }
                z[c] = s;
            }
            softmaxInPlace(z);
            loss += -std::log(std::max(z[y], 1e-12));

            // grad of cross-entropy wrt logits = p - onehot(y)
            for (std::size_t c = 0; c < K; ++c) {
                const double g = z[c] - (c == y ? 1.0 : 0.0);
                double* gw = gW.data() + c * D;
                for (std::size_t d = 0; d < D; ++d) {
                    gw[d] += g * static_cast<double>(x[d]);
                }
                gB[c] += g;
            }
        }

        const double invN = 1.0 / static_cast<double>(nTrain);
        // L2 reg term on W for the reported loss.
        double reg = 0.0;
        for (double w : W) {
            reg += w * w;
        }
        loss = loss * invN + 0.5 * cfg.l2 * reg;
        finalLoss = loss;

        // Adam update (bias-corrected).
        const std::size_t tstep = epoch + 1;
        const double bc1 = 1.0 - std::pow(beta1, static_cast<double>(tstep));
        const double bc2 = 1.0 - std::pow(beta2, static_cast<double>(tstep));
        for (std::size_t j = 0; j < K * D; ++j) {
            const double g = gW[j] * invN + cfg.l2 * W[j];
            mW[j] = beta1 * mW[j] + (1.0 - beta1) * g;
            vW[j] = beta2 * vW[j] + (1.0 - beta2) * g * g;
            const double mhat = mW[j] / bc1;
            const double vhat = vW[j] / bc2;
            W[j] -= cfg.lr * mhat / (std::sqrt(vhat) + eps);
        }
        for (std::size_t c = 0; c < K; ++c) {
            const double g = gB[c] * invN;   // no reg on bias
            mb[c] = beta1 * mb[c] + (1.0 - beta1) * g;
            vb[c] = beta2 * vb[c] + (1.0 - beta2) * g * g;
            const double mhat = mb[c] / bc1;
            const double vhat = vb[c] / bc2;
            b[c] -= cfg.lr * mhat / (std::sqrt(vhat) + eps);
        }
        epochsRun = tstep;
    }

    // Accuracy.
    std::size_t trainHits = 0;
    for (std::size_t t = 0; t < nTrain; ++t) {
        const std::size_t i = idx[t];
        if (predict(W, b, features.data() + i * D, K, D) ==
            static_cast<std::size_t>(labels[i])) {
            ++trainHits;
        }
    }
    std::size_t valHits = 0;
    for (std::size_t t = nTrain; t < N; ++t) {
        const std::size_t i = idx[t];
        if (predict(W, b, features.data() + i * D, K, D) ==
            static_cast<std::size_t>(labels[i])) {
            ++valHits;
        }
    }

    Result r{};
    r.numClasses    = K;
    r.dim           = D;
    r.nTrain        = nTrain;
    r.nVal          = nVal;
    r.trainAccuracy = nTrain > 0 ? static_cast<double>(trainHits) / static_cast<double>(nTrain) : 0.0;
    r.valAccuracy   = nVal > 0 ? static_cast<double>(valHits) / static_cast<double>(nVal) : 0.0;
    r.finalLoss     = finalLoss;
    r.epochsRun     = epochsRun;
    r.weight.resize(K * D);
    r.bias.resize(K);
    for (std::size_t j = 0; j < K * D; ++j) {
        r.weight[j] = static_cast<float>(W[j]);
    }
    for (std::size_t c = 0; c < K; ++c) {
        r.bias[c] = static_cast<float>(b[c]);
    }
    return r;
}

} // namespace mimirmind::runtime::encoder
