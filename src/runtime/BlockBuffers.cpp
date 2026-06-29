#include "runtime/BlockBuffers.hpp"

#include <algorithm>

namespace mimirmind::runtime {

BlockBuffers allocBlockBuffers(UsmAllocator&           allocator,
                               const model::LlmConfig& config,
                               std::size_t             maxT,
                               std::size_t             maxSeq) {
    BlockBuffers b{};
    b.maxT    = maxT;
    b.maxSeq  = maxSeq;
    b.d_model = config.embeddingLength;
    b.q_dim   = config.headCount * config.headDim();
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

    if (config.expertCount > 0) {
        const std::size_t moeBytes = maxT * b.d_model * sizeof(float);
        b.moeAccumBuf  = UsmHandle{allocator, moeBytes};
        b.expertOutBuf = UsmHandle{allocator, moeBytes};
    }
    return b;
}

} // namespace mimirmind::runtime