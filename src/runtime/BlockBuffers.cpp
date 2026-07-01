#include "runtime/BlockBuffers.hpp"

#include <algorithm>

namespace mimirmind::runtime {

BlockBuffers allocBlockBuffers(UsmAllocator&           allocator,
                               const model::LlmConfig& config,
                               std::size_t             maxT,
                               std::size_t             maxSeq,
                               std::size_t             qDimMax,
                               std::size_t             kvDimMax,
                               bool                    withFusedQkv) {
    BlockBuffers b{};
    b.maxT    = maxT;
    b.maxSeq  = maxSeq;
    b.d_model = config.embeddingLength;
    b.q_dim   = qDimMax;
    b.kv_dim  = kvDimMax;
    b.ff_dim  = config.feedForwardLength;

    const std::size_t qBytes            = maxT * b.q_dim   * sizeof(float);
    const std::size_t normBytes         = maxT * b.d_model * sizeof(float);
    const std::size_t attnOutBytes      = maxT * b.q_dim   * sizeof(float);
    const std::size_t projOutBytes      = maxT * b.d_model * sizeof(float);
    const std::size_t gateOutBytes      = maxT * b.ff_dim  * sizeof(float);
    const std::size_t upOutBytes        = maxT * b.ff_dim  * sizeof(float);
    const std::size_t scoreScratchBytes = maxSeq           * sizeof(float);
    const std::size_t matmulScratchBytes =
        std::max({b.d_model, b.q_dim, b.ff_dim}) * sizeof(float);

    b.qBuf          = UsmHandle{allocator, qBytes};
    b.normBuf       = UsmHandle{allocator, normBytes};
    b.attnOut       = UsmHandle{allocator, attnOutBytes};
    b.projOut       = UsmHandle{allocator, projOutBytes};
    b.gateOut       = UsmHandle{allocator, gateOutBytes};
    b.upOut         = UsmHandle{allocator, upOutBytes};
    b.matmulScratch = UsmHandle{allocator, matmulScratchBytes};
    b.scoreScratch  = UsmHandle{allocator, scoreScratchBytes};

    if (withFusedQkv) {
        // Q + K + V fused output. Widest layer: Q width + two KV widths.
        const std::size_t fusedBytes =
            maxT * (b.q_dim + 2 * b.kv_dim) * sizeof(float);
        b.qkvFusedScratch = UsmHandle{allocator, fusedBytes};
    }

    if (config.expertCount > 0) {
        const std::size_t moeBytes = maxT * b.d_model * sizeof(float);
        b.moeAccumBuf  = UsmHandle{allocator, moeBytes};
        b.expertOutBuf = UsmHandle{allocator, moeBytes};

        // Expert-grouping scratch (M5i.F). Per-block worst-case row
        // count = maxT * expertUsedCount (K_top). Feed-forward per
        // expert is inferred from the block layout at run time —
        // BlockBuffers is sized on ff_dim (dense path A) which is a
        // safe upper bound: for Gemma 4 26B-A4B ff_dim == ffPerExpert.
        const std::size_t nRowsMax  = maxT * config.expertUsedCount;
        const std::size_t xBytes    = nRowsMax * b.d_model   * sizeof(float);
        const std::size_t gateBytes = nRowsMax * b.ff_dim    * sizeof(float);
        b.moeXCompact    = UsmHandle{allocator, xBytes};
        b.moeGateCompact = UsmHandle{allocator, gateBytes};
        b.moeUpCompact   = UsmHandle{allocator, gateBytes};
        b.moeDownCompact = UsmHandle{allocator, xBytes};
    }
    return b;
}

} // namespace mimirmind::runtime