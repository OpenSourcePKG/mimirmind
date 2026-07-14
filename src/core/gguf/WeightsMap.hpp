#pragma once

#include "core/gguf/GgufReader.hpp"

#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace mimirmind::core::ipc {
struct TensorManifest;
}

namespace mimirmind::core::gguf {

/**
 * O(1) name → tensor lookup over a parsed `GgufReader`. Thin wrapper for
 * now; gains layer-aware accessors as soon as the forward pass needs
 * them (attnQ(blockIdx), ffnDown(blockIdx), ...).
 *
 * Two construction paths:
 *   - `WeightsMap(reader)`  — standalone mode. Borrows tensor structs
 *     from `reader`; reader must outlive the map.
 *   - `WeightsMap::fromAttachedChunked(manifest, chunkBases)` — chunk
 *     attached mode (M-Munin.1a). Materialises the tensor list from a
 *     v2 manifest plus the pointer table returned by
 *     `IpcImporter::openChunks`. Every tensor's `usmPtr` is
 *     `chunkBases[entry.chunkIndex] + entry.chunkOffset` — no
 *     per-tensor IPC handshake happens at all.
 */
class WeightsMap {
public:
    explicit WeightsMap(const GgufReader& reader);

    /**
     * Build a WeightsMap from a v2 manifest and the imported chunk base
     * pointers. Each `manifest.tensors[i]` produces one owned GgufTensor
     * with `usmPtr = static_cast<std::byte*>(chunkBases[chunkIndex]) +
     * chunkOffset`. Throws `std::runtime_error` on out-of-range
     * chunkIndex (the manifest parser already refuses these, but we
     * guard defensively — a corrupt in-memory manifest would otherwise
     * dereference past `chunkBases`).
     *
     * `nelements` is recomputed from `dims` because the manifest ships
     * only `bytes`; downstream code that reads `t.nelements` relies on
     * it being populated. `fileOffset` is set to zero — meaningless in
     * attached mode, and setting it makes the origin obvious in dumps.
     */
    [[nodiscard]] static WeightsMap fromAttachedChunked(
        const ::mimirmind::core::ipc::TensorManifest& manifest,
        std::span<void* const>                        chunkBases);

    /// Returns nullptr if no tensor has that name.
    [[nodiscard]] const GgufTensor* find(std::string_view name) const noexcept;

    /// Throws std::runtime_error mentioning the name if not present.
    [[nodiscard]] const GgufTensor& require(std::string_view name) const;

    /// Convenience for per-layer tensors named `blk.<idx>.<suffix>`
    /// (the llama.cpp / GGUF convention). Returns nullptr if missing.
    [[nodiscard]] const GgufTensor* findBlock(std::size_t blockIdx,
                                              std::string_view suffix) const;

    [[nodiscard]] std::size_t size() const noexcept { return _byName.size(); }

    /// True when this map owns its tensor entries (attached mode).
    /// Diagnostic only — the forward-pass code doesn't care.
    [[nodiscard]] bool isAttached() const noexcept { return !_owned.empty(); }

private:
    /// Attached-mode ctor used internally by `fromAttachedChunked` to
    /// take ownership of the resolved tensor list. Not part of the
    /// public API — callers must go through the chunked factory so the
    /// manifest → usmPtr mapping stays in one place.
    explicit WeightsMap(std::vector<GgufTensor> attachedTensors);

    // Populated only in attached mode; `_byName` points into this vector
    // when set. In reader mode it stays empty and `_byName` points into
    // the reader's own `tensors()` vector.
    std::vector<GgufTensor>                            _owned;
    std::unordered_map<std::string, const GgufTensor*> _byName;
};

} // namespace mimirmind::core::gguf