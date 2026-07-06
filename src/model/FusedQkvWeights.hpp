#pragma once

#include "model/GgufTypes.hpp"

#include <cstddef>
#include <optional>
#include <vector>

namespace mimirmind::runtime {
class UsmAllocator;
}

namespace mimirmind::model {

class WeightsMap;

/**
 * Fuses per-block attn_q / attn_k / attn_v weight tensors into a single
 * contiguous USM allocation per block, so that the QKV projections can
 * dispatch as ONE matmul instead of three. See ADR M5i for context.
 *
 * Fusion is enabled when:
 *   - Both attn_q.weight and attn_k.weight exist for the block
 *   - All present tensors share the same GgmlType
 *   - MIMIRMIND_DISABLE_FUSED_QKV is unset (or set to "0"/"false"/"off")
 *
 * When the layer's V is derived from raw K (Gemma 4 altAttention), the
 * fused block contains only [W_q | W_k] and `hasV` is false.
 *
 * Owns its USM allocations; releases them in the destructor. Not
 * movable/copyable — construct once per model, share by reference.
 */
class FusedQkvWeights {
public:
    struct Block {
        void*       usmPtr{nullptr};
        GgmlType    type{GgmlType::Unknown};
        std::size_t Nq{0};        // Q output-row count
        std::size_t Nkv{0};       // K/V output-row count (same for both)
        std::size_t K{0};         // input dim (d_model)
        bool        hasV{false};  // false → fused = [W_q | W_k] only
        std::size_t nbytes{0};    // total USM allocation size (for dealloc)
    };

    /// Iterate `numBlocks` transformer blocks in `weights` and build a
    /// fused USM allocation for every one that qualifies. Blocks that
    /// don't qualify (missing tensors, mismatched types, env var
    /// disabled) get no allocation and `find()` returns nullptr for
    /// them.
    ///
    /// `sharedKvLayers` is the trailing count of blocks that reuse K/V
    /// from an earlier layer (Gemma 4 E4B = 18). Those blocks never
    /// touch the K/V projection at inference time and therefore don't
    /// need a fused-QKV entry. Default 0 = every block owns K/V (the
    /// pre-existing Qwen / non-E4B-Gemma-4 case).
    ///
    /// When `requantMismatchToQ8_0` is true, own-KV blocks whose
    /// attn_q/k/v types don't match get rescued: all three tensors are
    /// dequantized to f32 and re-quantized to Q8_0 in a new USM
    /// allocation, and the fused block is emitted as Q8_0. Quality is
    /// neutral-to-better (Q8_0 outranks Q4_K/Q5_K/Q6_K on per-element
    /// error). Enables Q8_0 GEMM v2 dispatch on prefill even when the
    /// underlying model was quantized to K-quants.
    FusedQkvWeights(const WeightsMap&      weights,
                    runtime::UsmAllocator& allocator,
                    std::size_t            numBlocks,
                    std::size_t            sharedKvLayers        = 0,
                    bool                   requantMismatchToQ8_0 = false);

    ~FusedQkvWeights();

    FusedQkvWeights(const FusedQkvWeights&)            = delete;
    FusedQkvWeights& operator=(const FusedQkvWeights&) = delete;
    FusedQkvWeights(FusedQkvWeights&&)                 = delete;
    FusedQkvWeights& operator=(FusedQkvWeights&&)      = delete;

    /// Returns the fused block if it exists, nullptr otherwise.
    [[nodiscard]] const Block* find(std::size_t blockIdx) const noexcept;

    /// True when at least one block was fused. Used for banner logging
    /// and for deciding whether to allocate qkvFusedScratch.
    [[nodiscard]] bool anyFused() const noexcept { return _anyFused; }

    /// True when the env-var disable switch is set. Diagnostic-only.
    [[nodiscard]] bool disabledByEnv() const noexcept { return _disabledByEnv; }

    /// Block-count telemetry for /v1/system/status.
    [[nodiscard]] std::size_t fusedCount()   const noexcept { return _fusedCount; }
    [[nodiscard]] std::size_t skippedCount() const noexcept { return _skippedCount; }
    [[nodiscard]] std::size_t totalUsmBytes() const noexcept { return _totalBytes; }
    [[nodiscard]] std::size_t requantCount() const noexcept { return _requantCount; }

private:
    runtime::UsmAllocator&              _alloc;
    std::vector<std::optional<Block>>   _blocks;
    bool                                _anyFused{false};
    bool                                _disabledByEnv{false};
    std::size_t                         _fusedCount{0};
    std::size_t                         _skippedCount{0};
    std::size_t                         _totalBytes{0};
    std::size_t                         _requantCount{0};
};

} // namespace mimirmind::model