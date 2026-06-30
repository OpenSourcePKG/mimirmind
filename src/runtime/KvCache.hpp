#pragma once

#include <cstddef>
#include <vector>

namespace mimirmind::runtime {

class UsmAllocator;

/**
 * Per-layer rolling K/V cache for the autoregressive decoder.
 *
 * Each layer stores its own [maxSeq, kvDim] f32 buffer for K and V.
 * Layers can have *different* kvDim values (e.g. Gemma 4 has SWA layers
 * with 8 × 256 KV heads and full-attention layers with 2 × 512). Pass
 * a per-layer kvDim vector at construction; uniform Qwen-style models
 * just fill all entries with the same value.
 *
 * Caller flow per forward pass:
 *   for each new chunk of T tokens (T == prompt length on prefill, == 1
 *   on decode):
 *
 *     for each layer L in [0, nLayers):
 *         project K and V into writeSlotK(L) / writeSlotV(L)
 *         apply RoPE in place over those new T rows using startPos = length()
 *         run attention reading baseK(L)/baseV(L) with T_k = length() + T,
 *             positionOffset = length()
 *     cache.commit(T)
 *
 * The cache itself never auto-advances — `commit(T)` is the explicit
 * signal that a forward step is done and all layers' new K/V slots are
 * filled.
 */
class KvCache {
public:
    /// Per-layer dims. kvDimPerLayer[L] == nKvHeads(L) * headDim(L).
    KvCache(UsmAllocator&            alloc,
            std::size_t              maxSeq,
            std::vector<std::size_t> kvDimPerLayer);

    /// Uniform (Qwen/Llama) convenience: fills every layer with the same dim.
    KvCache(UsmAllocator& alloc,
            std::size_t   nLayers,
            std::size_t   maxSeq,
            std::size_t   nKvHeads,
            std::size_t   headDim);

    ~KvCache();

    KvCache(const KvCache&)            = delete;
    KvCache& operator=(const KvCache&) = delete;
    KvCache(KvCache&&)                 = delete;
    KvCache& operator=(KvCache&&)      = delete;

    /// Pointer to the next free K slot for `layer`. Caller will write T
    /// rows starting here, T <= maxSeq - length().
    [[nodiscard]] float* writeSlotK(std::size_t layer) noexcept;
    [[nodiscard]] float* writeSlotV(std::size_t layer) noexcept;

    /// Base pointer for `layer`. Attention reads from here with T_k =
    /// length() + T (the newly-written rows are physically present even
    /// before commit).
    [[nodiscard]] const float* baseK(std::size_t layer) const noexcept;
    [[nodiscard]] const float* baseV(std::size_t layer) const noexcept;

    void commit(std::size_t T) noexcept { _length += T; }
    void reset() noexcept                { _length = 0; }

    /// Shrink the logical length to `n`. The K/V data at positions
    /// `[0, n)` stays valid and re-usable. Caller must guarantee `n <=
    /// length()` — used by the prefix-cache path to walk the engine
    /// back to the longest-common-prefix length before re-prefilling
    /// the divergent suffix.
    void truncate(std::size_t n) noexcept {
        _length = (n <= _length) ? n : _length;
    }

    [[nodiscard]] std::size_t length()           const noexcept { return _length; }
    [[nodiscard]] std::size_t maxSeq()           const noexcept { return _maxSeq; }
    [[nodiscard]] std::size_t kvDim(std::size_t l) const noexcept { return _kvDim[l]; }

private:
    void allocateLayers();

    UsmAllocator&            _alloc;
    std::size_t              _maxSeq;
    std::vector<std::size_t> _kvDim;
    std::vector<std::size_t> _layerBytes;
    std::size_t              _length{0};
    std::vector<void*>       _kBuf;
    std::vector<void*>       _vBuf;
};

} // namespace mimirmind::runtime