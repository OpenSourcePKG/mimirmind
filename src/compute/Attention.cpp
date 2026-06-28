#include "compute/Attention.hpp"

#include "compute/Softmax.hpp"
#include "runtime/Log.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mimirmind::compute {

void multiHeadAttention(const float* q,
                        const float* k,
                        const float* v,
                        std::size_t  T_q,
                        std::size_t  T_k,
                        std::size_t  nHeads,
                        std::size_t  nKvHeads,
                        std::size_t  headDim,
                        std::size_t  positionOffset,
                        float*       scratch,
                        float*       out) {
    if (nKvHeads == 0 || nHeads == 0 || nHeads % nKvHeads != 0) {
        throw std::runtime_error(
            "attention: nHeads must be a positive multiple of nKvHeads");
    }
    const std::size_t qStride  = nHeads   * headDim;
    const std::size_t kvStride = nKvHeads * headDim;
    const float       scale    = 1.0F /
        std::sqrt(static_cast<float>(headDim));

    for (std::size_t hq = 0; hq < nHeads; ++hq) {
        const std::size_t hkv = (hq * nKvHeads) / nHeads;

        for (std::size_t p = 0; p < T_q; ++p) {
            const std::size_t absPos = positionOffset + p;
            const std::size_t kMax   = std::min(absPos + 1, T_k);

            const float* qVec = q + p * qStride + hq * headDim;

            for (std::size_t kk = 0; kk < kMax; ++kk) {
                const float* kVec = k + kk * kvStride + hkv * headDim;
                double acc = 0.0;
                for (std::size_t d = 0; d < headDim; ++d) {
                    acc += static_cast<double>(qVec[d]) *
                           static_cast<double>(kVec[d]);
                }
                scratch[kk] = static_cast<float>(acc) * scale;
            }
            for (std::size_t kk = kMax; kk < T_k; ++kk) {
                scratch[kk] = 0.0F;
            }

            const std::size_t live = kMax;
            softmaxRows(scratch, 1, T_k, &live);

            float* outVec = out + p * qStride + hq * headDim;
            for (std::size_t d = 0; d < headDim; ++d) {
                double acc = 0.0;
                for (std::size_t kk = 0; kk < kMax; ++kk) {
                    const float* vVec = v + kk * kvStride + hkv * headDim;
                    acc += static_cast<double>(scratch[kk]) *
                           static_cast<double>(vVec[d]);
                }
                outVec[d] = static_cast<float>(acc);
            }
        }
    }

    MM_LOG_DEBUG("attn",
                 "done — T_q={} T_k={} posOffset={} nHeads={} nKvHeads={} "
                 "headDim={} (GQA group size {})",
                 T_q, T_k, positionOffset, nHeads, nKvHeads, headDim,
                 nHeads / nKvHeads);
}

} // namespace mimirmind::compute