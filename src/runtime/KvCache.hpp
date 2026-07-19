// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "compute/ComputeBuffer.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace mimirmind::compute { class ComputeOps; }

namespace mimirmind::runtime {

/**
 * Element dtype for K/V storage. Baseline F32 today; FP16 lands in
 * M10.2 Phase 0 as the 2× bandwidth/RAM win, Q8_0 in Phase 1a
 * (M10.2.0) as the 4× win with block-based storage. See Synaipse
 * `Memory/mimirmind/todos/m10-2-neg1-kv-cache-dtype-layer-design.md`
 * and `m10-2-0-kv-cache-q8-0-design.md` for the full migration plans.
 */
enum class KvDtype : std::uint8_t {
    F32  = 0,   // 4 B per element — baseline.
    FP16 = 1,   // 2 B per element — Phase 0 target.
    Q8_0 = 2,   // 32-element blocks × 34 B (fp16 scale + 32 int8) —
                // Phase 1a target. NOT per-element addressable; use
                // blockElements()/blockBytes()/rowBytes() and route
                // through the Q8_0-aware kernels.
};

/// Uniform-dtype element size. Only defined for F32 and FP16 — Q8_0
/// is block-based (see kvBlockBytes / kvBytesForElements). Returns 0
/// on Q8_0 so constexpr contexts stay well-defined; the KvCache
/// `elementBytes()` member method throws on Q8_0 for runtime callers.
[[nodiscard]] constexpr std::size_t kvElementBytes(KvDtype d) noexcept {
    switch (d) {
        case KvDtype::F32:  return 4;
        case KvDtype::FP16: return 2;
        case KvDtype::Q8_0: return 0;    // sentinel — use kvBlockBytes
    }
    return 0;
}

/// Elements per storage block. 1 for uniform dtypes (F32/FP16), 32
/// for Q8_0.
[[nodiscard]] constexpr std::size_t kvBlockElements(KvDtype d) noexcept {
    switch (d) {
        case KvDtype::F32:  return 1;
        case KvDtype::FP16: return 1;
        case KvDtype::Q8_0: return 32;
    }
    return 1;
}

/// Bytes per storage block. `kvBlockBytes(d) / kvBlockElements(d)` is
/// the *average* per-element footprint (1.0625 B for Q8_0, i.e. 34/32).
[[nodiscard]] constexpr std::size_t kvBlockBytes(KvDtype d) noexcept {
    switch (d) {
        case KvDtype::F32:  return 4;
        case KvDtype::FP16: return 2;
        case KvDtype::Q8_0: return 34;   // fp16 scale + 32 int8
    }
    return 0;
}

/// Storage bytes needed to hold `nElements` values encoded as `d`.
/// For F32/FP16 this is `nElements * elementBytes`. For Q8_0 it is
/// `(nElements / 32) * 34`. Caller must guarantee `nElements` is a
/// multiple of `kvBlockElements(d)` — for Q8_0 that means multiples
/// of 32. Assertion below covers the F32/FP16 trivial case too.
[[nodiscard]] constexpr std::size_t kvBytesForElements(
    KvDtype d, std::size_t nElements) noexcept
{
    const auto be = kvBlockElements(d);
    return (nElements / be) * kvBlockBytes(d);
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
    KvCache(compute::ComputeOps&     ops,
            std::size_t              maxSeq,
            std::vector<std::size_t> kvDimPerLayer,
            std::vector<std::size_t> kvSourceLayer = {},
            KvDtype                  dtype = KvDtype::F32);

    /// Uniform (Qwen/Llama) convenience: fills every layer with the same dim.
    KvCache(compute::ComputeOps& ops,
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

    /// Bytes per element. Only defined for F32 and FP16; **throws
    /// `std::logic_error` on Q8_0** because per-element addressing
    /// isn't meaningful there. Callers that need dtype-agnostic byte
    /// arithmetic should use `rowBytes(layer)` (matches the memcpy
    /// footprint of a T=1 K/V slot).
    [[nodiscard]] std::size_t elementBytes() const {
        if (_dtype == KvDtype::Q8_0) {
            throw std::logic_error(
                "KvCache::elementBytes: not defined for Q8_0 — use "
                "rowBytes(layer) or blockBytes()/blockElements() "
                "instead");
        }
        return kvElementBytes(_dtype);
    }

    /// M10.2 Phase 1a — dtype-aware block accessors. Works for all
    /// three dtypes uniformly. For F32/FP16 blockElements() == 1 and
    /// blockBytes() == elementBytes(); for Q8_0 blockElements() == 32
    /// and blockBytes() == 34.
    [[nodiscard]] std::size_t blockElements() const noexcept {
        return kvBlockElements(_dtype);
    }
    [[nodiscard]] std::size_t blockBytes() const noexcept {
        return kvBlockBytes(_dtype);
    }
    /// Byte footprint of one KV row (one token, kvDim(layer) elements)
    /// in the layer's storage dtype. Correct for all dtypes; the
    /// preferred call for any dtype-agnostic size / offset math.
    [[nodiscard]] std::size_t rowBytes(std::size_t layer) const noexcept {
        return kvBytesForElements(_dtype, _kvDim[layer]);
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
    void allocateLayers(compute::ComputeOps& ops);

    std::size_t              _maxSeq;
    std::vector<std::size_t> _kvDim;
    KvDtype                  _dtype{KvDtype::F32};
    /// Bytes owned per layer — populated for own-KV entries, 0 for
    /// aliased ones. Retained after Schicht 5.5 for the info-log's
    /// "own vs alias / saved bytes" reporting.
    std::vector<std::size_t> _layerBytes;
    /// kvSourceLayer[L] == L for own-KV, < L for aliased (Gemma 4 E4B).
    /// Empty when the caller passes no override — treated as identity.
    std::vector<std::size_t> _kvSource;
    std::size_t              _length{0};
    // Schicht 5.5 — RAII ownership of the per-layer device allocations.
    // Own-KV layers hold their ComputeBuffer here; aliased entries are
    // empty (default-constructed). `_kBuf` / `_vBuf` alias into the
    // owning layer's raw pointer for both cases so downstream reads
    // don't branch on ownership.
    std::vector<compute::ComputeBuffer> _kOwners;
    std::vector<compute::ComputeBuffer> _vOwners;
    std::vector<void*>       _kBuf;
    std::vector<void*>       _vBuf;
};

} // namespace mimirmind::runtime