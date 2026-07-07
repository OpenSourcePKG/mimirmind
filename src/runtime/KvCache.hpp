#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace mimirmind::runtime {

class UsmAllocator;

/**
 * Element dtype for K/V storage. Baseline F32 today; FP16 lands in
 * M10.2 Phase 0 as the 2× bandwidth/RAM win before Q8_0 (M10.2.1) is
 * layered on top. See Synaipse
 * `Memory/mimirmind/todos/m10-2-neg1-kv-cache-dtype-layer-design.md`
 * for the full migration plan.
 */
enum class KvDtype : std::uint8_t {
    F32  = 0,   // 4 B per element — baseline.
    FP16 = 1,   // 2 B per element — Phase 0 target.
    // Q8_0 kommt in M10.2.1
};

[[nodiscard]] constexpr std::size_t kvElementBytes(KvDtype d) noexcept {
    switch (d) {
        case KvDtype::F32:  return 4;
        case KvDtype::FP16: return 2;
    }
    return 0;  // unreachable; enum is closed.
}

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
    ///
    /// `kvSourceLayer` is Gemma 4 E4B's shared-KV mechanism: when
    /// `kvSourceLayer[L] != L`, layer L does NOT get its own USM buffer.
    /// `baseK/V(L)` and `writeSlotK/V(L)` alias `kvSourceLayer[L]`'s
    /// buffers. Passing an empty vector = identity (every layer owns its
    /// K/V). Entries must satisfy `kvSourceLayer[L] <= L` (dependency
    /// ordering) and both source and aliased layer must have the same
    /// kvDim (the backend guarantees this via SWA-vs-full offset math).
    KvCache(UsmAllocator&            alloc,
            std::size_t              maxSeq,
            std::vector<std::size_t> kvDimPerLayer,
            std::vector<std::size_t> kvSourceLayer = {},
            KvDtype                  dtype = KvDtype::F32);

    /// Uniform (Qwen/Llama) convenience: fills every layer with the same dim.
    KvCache(UsmAllocator& alloc,
            std::size_t   nLayers,
            std::size_t   maxSeq,
            std::size_t   nKvHeads,
            std::size_t   headDim,
            KvDtype       dtype = KvDtype::F32);

    ~KvCache();

    KvCache(const KvCache&)            = delete;
    KvCache& operator=(const KvCache&) = delete;
    KvCache(KvCache&&)                 = delete;
    KvCache& operator=(KvCache&&)      = delete;

    /// Pointer to the next free K slot for `layer`. Caller will write T
    /// rows starting here, T <= maxSeq - length(). Element count per
    /// row is `kvDim(layer)`; element size is `elementBytes()`.
    [[nodiscard]] void* writeSlotK(std::size_t layer) noexcept;
    [[nodiscard]] void* writeSlotV(std::size_t layer) noexcept;

    /// Base pointer for `layer`. Attention reads from here with T_k =
    /// length() + T (the newly-written rows are physically present even
    /// before commit).
    [[nodiscard]] const void* baseK(std::size_t layer) const noexcept;
    [[nodiscard]] const void* baseV(std::size_t layer) const noexcept;

    /// F32-typed convenience wrappers — asserts dtype() == F32 in dev
    /// builds and returns the same address as the void* getters, cast
    /// to float*. Callers that still assume F32 KV storage should use
    /// these until they gain dtype-awareness (M10.2 Commit 5).
    [[nodiscard]] float* writeSlotKf32(std::size_t layer) noexcept {
        assert(_dtype == KvDtype::F32);
        return static_cast<float*>(writeSlotK(layer));
    }
    [[nodiscard]] float* writeSlotVf32(std::size_t layer) noexcept {
        assert(_dtype == KvDtype::F32);
        return static_cast<float*>(writeSlotV(layer));
    }
    [[nodiscard]] const float* baseKf32(std::size_t layer) const noexcept {
        assert(_dtype == KvDtype::F32);
        return static_cast<const float*>(baseK(layer));
    }
    [[nodiscard]] const float* baseVf32(std::size_t layer) const noexcept {
        assert(_dtype == KvDtype::F32);
        return static_cast<const float*>(baseV(layer));
    }

    [[nodiscard]] KvDtype     dtype()        const noexcept { return _dtype; }
    [[nodiscard]] std::size_t elementBytes() const noexcept {
        return kvElementBytes(_dtype);
    }

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
    KvDtype                  _dtype{KvDtype::F32};
    /// Bytes actually owned per layer for dealloc. Aliased layers hold
    /// 0 here so the destructor skips them (their buffer belongs to the
    /// source layer).
    std::vector<std::size_t> _layerBytes;
    /// kvSourceLayer[L] == L for own-KV, < L for aliased (Gemma 4 E4B).
    /// Empty when the caller passes no override — treated as identity.
    std::vector<std::size_t> _kvSource;
    std::size_t              _length{0};
    std::vector<void*>       _kBuf;
    std::vector<void*>       _vBuf;
};

} // namespace mimirmind::runtime