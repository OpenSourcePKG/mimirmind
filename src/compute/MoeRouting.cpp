#include "compute/MoeRouting.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace mimirmind::compute {

void moeTopKRoute(const float*  logits,
                  std::size_t   T,
                  std::size_t   nExperts,
                  std::size_t   topK,
                  std::int32_t* outIdx,
                  float*        outWeight) {
    if (logits == nullptr || outIdx == nullptr || outWeight == nullptr) {
        throw std::runtime_error("moeTopKRoute: null pointer");
    }
    if (nExperts == 0) {
        throw std::runtime_error("moeTopKRoute: nExperts must be > 0");
    }
    if (topK == 0 || topK > nExperts) {
        throw std::runtime_error(
            "moeTopKRoute: topK=" + std::to_string(topK) +
            " out of range (1.." + std::to_string(nExperts) + ")");
    }

    std::vector<float>       probs(nExperts);
    std::vector<std::size_t> idx(nExperts);

    for (std::size_t t = 0; t < T; ++t) {
        const float* row = logits + t * nExperts;

        // 1. softmax with max-subtract; double sum for numerical stability.
        float maxL = row[0];
        for (std::size_t e = 1; e < nExperts; ++e) {
            if (row[e] > maxL) {
                maxL = row[e];
            }
        }
        double sum = 0.0;
        for (std::size_t e = 0; e < nExperts; ++e) {
            probs[e] = std::exp(row[e] - maxL);
            sum += static_cast<double>(probs[e]);
        }
        const float invSum = static_cast<float>(1.0 / sum);
        for (auto& p : probs) {
            p *= invSum;
        }

        // 2. partial_sort indices by probability, descending. Ties resolve
        //    in implementation-defined order — callers should not rely on
        //    a stable order when probabilities are equal.
        std::iota(idx.begin(), idx.end(), std::size_t{0});
        std::partial_sort(
            idx.begin(),
            idx.begin() + static_cast<std::ptrdiff_t>(topK),
            idx.end(),
            [&](std::size_t a, std::size_t b) {
                return probs[a] > probs[b];
            });

        // 3. renormalise the kept K weights so they sum to ~1.
        double kept = 0.0;
        for (std::size_t k = 0; k < topK; ++k) {
            kept += static_cast<double>(probs[idx[k]]);
        }
        const float invKept = kept > 0.0
            ? static_cast<float>(1.0 / kept)
            : 1.0F;
        for (std::size_t k = 0; k < topK; ++k) {
            outIdx[t * topK + k]    = static_cast<std::int32_t>(idx[k]);
            outWeight[t * topK + k] = probs[idx[k]] * invKept;
        }
    }
}

} // namespace mimirmind::compute