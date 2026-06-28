#include "compute/Attention.hpp"

#include "compute/Softmax.hpp"
#include "runtime/Log.hpp"

#include <cmath>
#include <stdexcept>

namespace mimirmind::compute {

void multiHeadAttentionPrefill(const float* q,
                               const float* k,
                               const float* v,
                               std::size_t  seqLen,
                               std::size_t  nHeads,
                               std::size_t  nKvHeads,
                               std::size_t  headDim,
                               float*       scratch,
                               float*       out) {
    if (nKvHeads == 0 || nHeads == 0 || nHeads % nKvHeads != 0) {
        throw std::runtime_error(
            "attention: nHeads must be a positive multiple of nKvHeads");
    }
    const std::size_t qStride  = nHeads   * headDim;   // per token in q/out
    const std::size_t kvStride = nKvHeads * headDim;   // per token in k/v
    const float       scale    = 1.0F /
        std::sqrt(static_cast<float>(headDim));

    // For each (query head, query position): compute scores against all
    // earlier key positions, softmax (causal), then weighted sum of V.
    for (std::size_t hq = 0; hq < nHeads; ++hq) {
        const std::size_t hkv = (hq * nKvHeads) / nHeads;

        for (std::size_t p = 0; p < seqLen; ++p) {
            const float* qVec = q + p * qStride + hq * headDim;

            // Scores against keys at positions 0..p (live = p + 1).
            for (std::size_t kk = 0; kk <= p; ++kk) {
                const float* kVec = k + kk * kvStride + hkv * headDim;
                double acc = 0.0;
                for (std::size_t d = 0; d < headDim; ++d) {
                    acc += static_cast<double>(qVec[d]) *
                           static_cast<double>(kVec[d]);
                }
                scratch[kk] = static_cast<float>(acc) * scale;
            }

            // Mask the rest (will be zeroed by softmaxRows but explicit
            // here for safety against later changes).
            for (std::size_t kk = p + 1; kk < seqLen; ++kk) {
                scratch[kk] = 0.0F;
            }

            const std::size_t live = p + 1;
            softmaxRows(scratch, 1, seqLen, &live);

            float* outVec = out + p * qStride + hq * headDim;
            for (std::size_t d = 0; d < headDim; ++d) {
                double acc = 0.0;
                for (std::size_t kk = 0; kk <= p; ++kk) {
                    const float* vVec = v + kk * kvStride + hkv * headDim;
                    acc += static_cast<double>(scratch[kk]) *
                           static_cast<double>(vVec[d]);
                }
                outVec[d] = static_cast<float>(acc);
            }
        }
    }

    MM_LOG_DEBUG("attn",
                 "prefill done — seqLen={} nHeads={} nKvHeads={} headDim={} "
                 "(GQA group size {})",
                 seqLen, nHeads, nKvHeads, headDim, nHeads / nKvHeads);
}

} // namespace mimirmind::compute