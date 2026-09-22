// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace mimirmind::runtime::encoder {

/**
 * In-process trainer for a "System-One" typed-decision head (8.23): a
 * multinomial logistic-regression (softmax) linear probe on frozen bge-m3
 * sentence embeddings. This is the ONLY training MimirMind does — a small convex
 * fit over already-computed features, not model training. No ML framework: pure
 * host-side Adam + L2, deterministic, sub-second for thousands of examples.
 *
 * Given a feature matrix X [N x D] (L2-normalized encoder embeddings) and integer
 * class labels y [N] over K classes, it minimises cross-entropy + (l2/2)||W||^2
 * and returns W [K x D] (row-major) + bias [K] — exactly the shape
 * DecisionHead consumes. Features are supplied by the caller (DecideEngine embeds
 * the training texts through the same path inference uses), so train == serve.
 */
class LinearProbeTrainer {
public:
    struct Config {
        std::size_t epochs{300};        // full-batch Adam steps
        double      lr{0.05};           // Adam step size
        double      l2{1e-3};           // L2 weight decay on W (not bias)
        double      valSplit{0.2};      // fraction held out for val metrics (0 = none)
        std::uint64_t seed{0x9E3779B9}; // deterministic shuffle for the split
    };

    struct Result {
        std::vector<float> weight;      // [K * D] row-major (row c = class c)
        std::vector<float> bias;        // [K]
        std::size_t        numClasses{0};
        std::size_t        dim{0};
        std::size_t        nTrain{0};
        std::size_t        nVal{0};
        double             trainAccuracy{0.0};
        double             valAccuracy{0.0};   // NaN-free; 0 when no val split
        double             finalLoss{0.0};     // train cross-entropy + reg
        std::size_t        epochsRun{0};
    };

    /// Fit W,b. `features` is N contiguous D-vectors (features[i*D .. i*D+D]);
    /// `labels[i]` in [0, numClasses). Throws std::invalid_argument on shape
    /// mismatch, an empty set, numClasses < 2, or a label out of range.
    [[nodiscard]] static Result train(std::span<const float> features,
                                      std::size_t dim,
                                      std::span<const std::int32_t> labels,
                                      std::size_t numClasses,
                                      const Config& cfg);
};

} // namespace mimirmind::runtime::encoder
