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
 * Lifetime: borrows from `reader`. Reader must outlive the map.
 */
class WeightsMap {
public:
    explicit WeightsMap(const GgufReader& reader);

    /// Returns nullptr if no tensor has that name.
    [[nodiscard]] const GgufTensor* find(std::string_view name) const noexcept;

    /// Throws std::runtime_error mentioning the name if not present.
    [[nodiscard]] const GgufTensor& require(std::string_view name) const;

    /// Convenience for per-layer tensors named `blk.<idx>.<suffix>`
    /// (the llama.cpp / GGUF convention). Returns nullptr if missing.
    [[nodiscard]] const GgufTensor* findBlock(std::size_t blockIdx,
                                              std::string_view suffix) const;

    [[nodiscard]] std::size_t size() const noexcept { return _byName.size(); }

private:
    std::unordered_map<std::string, const GgufTensor*> _byName;
};

} // namespace mimirmind::core::gguf