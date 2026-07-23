// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/safetensors/SafetensorsHeader.hpp"
#include "core/safetensors/SafetensorsReader.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace mimirmind::core::safetensors {

/**
 * A whole ModelOpt / HuggingFace checkpoint, possibly sharded across
 * several `*.safetensors` files with a `model.safetensors.index.json`
 * weight-map. Opens every referenced shard and presents one flat,
 * unified tensor namespace on top of the per-shard `SafetensorsReader`s.
 *
 * Resolution rules for `open(path)`:
 *   - a directory containing `model.safetensors.index.json` → sharded load;
 *     every tensor named in the weight-map must resolve in its shard.
 *   - a directory containing a single `model.safetensors` (no index) →
 *     single-shard load.
 *   - a path to one `*.safetensors` file → single-shard load.
 *
 * Move-only; owns the shard readers (and thus their mmaps). Like the
 * per-shard reader it interprets NO quantisation scheme — that is
 * `core::modelopt` — and uploads nothing to device.
 */
class SafetensorsModel {
public:
    SafetensorsModel() = default;
    ~SafetensorsModel() = default;

    SafetensorsModel(const SafetensorsModel&)            = delete;
    SafetensorsModel& operator=(const SafetensorsModel&) = delete;
    SafetensorsModel(SafetensorsModel&&) noexcept            = default;
    SafetensorsModel& operator=(SafetensorsModel&&) noexcept = default;

    /// Resolve, open, and index every shard. Throws std::runtime_error on a
    /// missing/ambiguous checkpoint, an unreadable/malformed shard or index,
    /// a weight-map tensor absent from its shard, or the same tensor name
    /// appearing in more than one shard.
    void open(std::string_view path);

    /// Release all shard mmaps and reset. Idempotent.
    void close() noexcept;

    [[nodiscard]] bool        isOpen()      const noexcept { return !_shards.empty(); }
    [[nodiscard]] std::size_t shardCount()  const noexcept { return _shards.size(); }
    [[nodiscard]] std::size_t tensorCount() const noexcept { return _flat.size(); }

    /// `metadata.total_size` from the index (tensor-data bytes only), or 0
    /// for a single-file checkpoint with no index.
    [[nodiscard]] std::uint64_t declaredTotalSize() const noexcept { return _totalSize; }

    /// All tensors across all shards, as pointers into the owning readers.
    /// Stable for the model's lifetime.
    [[nodiscard]] const std::vector<const SafetensorsTensor*>& tensors() const noexcept {
        return _flat;
    }

    /// Lookup by exact name across all shards, or nullptr. O(log n).
    [[nodiscard]] const SafetensorsTensor* find(std::string_view name) const noexcept;

    /// Zero-copy view of a tensor's bytes in its shard's mmap, valid for the
    /// model's lifetime. Empty span if the name is unknown.
    [[nodiscard]] std::span<const std::uint8_t>
    tensorBytes(std::string_view name) const noexcept;

private:
    void openSingle(std::string_view file);
    void openSharded(std::string_view dir, std::string_view indexFile);
    void reindex();

    std::vector<SafetensorsReader>     _shards;
    std::map<std::string, std::size_t> _tensorToShard;  ///< name -> shard idx
    std::vector<const SafetensorsTensor*> _flat;
    std::uint64_t                      _totalSize{0};
};

} // namespace mimirmind::core::safetensors