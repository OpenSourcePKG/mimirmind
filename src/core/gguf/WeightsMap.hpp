#pragma once

#include "core/gguf/GgufReader.hpp"

#include <cstddef>
#include <string>
#include <string_view>
#include <unordered_map>

namespace mimirmind::core::gguf {

/**
 * O(1) name → tensor lookup over a parsed `GgufReader`. Thin wrapper for
 * now; gains layer-aware accessors as soon as the forward pass needs
 * them (attnQ(blockIdx), ffnDown(blockIdx), ...).
 *
 * Two construction paths:
 *   - `WeightsMap(reader)`  — standalone mode. Borrows tensor structs
 *     from `reader`; reader must outlive the map.
 *   - `WeightsMap(std::vector<GgufTensor>&&)` — attached mode. Owns the
 *     tensor list itself (the entries carry `usmPtr` values that came
 *     from IpcImporter). Used by the mimirmind worker after attaching
 *     to a running Munin daemon — no local GgufReader::loadTensors on
 *     this path; the pointers point into USM owned by Munin.
 */
class WeightsMap {
public:
    explicit WeightsMap(const GgufReader& reader);

    /// Attached-mode ctor. Takes ownership of the tensor list; the map
    /// will find/require by name against these entries directly.
    explicit WeightsMap(std::vector<GgufTensor> attachedTensors);

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
    // Populated only in attached mode; `_byName` points into this vector
    // when set. In reader mode it stays empty and `_byName` points into
    // the reader's own `tensors()` vector.
    std::vector<GgufTensor>                            _owned;
    std::unordered_map<std::string, const GgufTensor*> _byName;
};

} // namespace mimirmind::core::gguf