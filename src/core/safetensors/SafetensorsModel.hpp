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

    /// One in-memory shard for openFromShards(). `bytes` is NON-OWNING and
    /// must outlive the model (M-Munin shm attach: mmap'd memfd chunk).
    struct ShardImage {
        std::string_view              name;   ///< shard filename, for diagnostics
        std::span<const std::uint8_t> bytes;  ///< the whole *.safetensors image
    };

    /// Build the model from already-in-memory shards instead of files — the
    /// M-Munin shm attach path, where each shard lives in an mmap'd memfd
    /// chunk the worker holds. Parses each via SafetensorsReader::openBytes,
    /// then indexes exactly like a file-based load (duplicate tensor names
    /// across shards are rejected). `declaredTotalSize` carries the index's
    /// `metadata.total_size` (0 = undeclared, as for a single-file load).
    /// Throws std::runtime_error on an empty shard list or any malformation.
    void openFromShards(std::span<const ShardImage> shards,
                        std::uint64_t               declaredTotalSize = 0);

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
    /// model's lifetime. Empty span if the name is unknown. A byte override
    /// (see `overrideTensorBytes`) wins over the mmap view.
    [[nodiscard]] std::span<const std::uint8_t>
    tensorBytes(std::string_view name) const noexcept;

    /// Rename every tensor across all shards through `fn` (identity keeps a
    /// name) and rebuild the model-level index. Load-time normalisation seam
    /// for checkpoints in a different naming dialect (compressed-tensors
    /// `weight_packed`, missing `language_model.` prefix, ...).
    void normalizeNames(const std::function<std::string(const std::string&)>& fn);

    /// Replace the bytes served for `name` by an owned buffer (the mmap is
    /// read-only). Used for tiny value fix-ups, e.g. inverting
    /// compressed-tensors' RECIPROCAL `weight_global_scale` into the direct
    /// ModelOpt `weight_scale_2` convention.
    void overrideTensorBytes(std::string_view name,
                             std::vector<std::uint8_t> bytes);

    /// Fused-projection splits (see SafetensorsReader::addDerivedRowSlice).
    /// Mutation-batch discipline: run all derive/duplicate calls first, then
    /// removes, then ONE rebuildIndexes() — find() results and tensors()
    /// pointers are unreliable between the first mutation and the rebuild.
    bool deriveRowSlice(std::string_view src, std::string newName,
                        std::uint64_t rowBegin, std::uint64_t rowCount);
    bool duplicateTensorAs(std::string_view src, std::string newName);
    bool removeTensor(std::string_view name);
    void rebuildIndexes();

private:
    void openSingle(std::string_view file);
    void openSharded(std::string_view dir, std::string_view indexFile);
    void reindex();

    std::vector<SafetensorsReader>     _shards;
    std::map<std::string, std::size_t> _tensorToShard;  ///< name -> shard idx
    std::vector<const SafetensorsTensor*> _flat;
    std::uint64_t                      _totalSize{0};
    std::map<std::string, std::vector<std::uint8_t>, std::less<>> _byteOverrides;
};

} // namespace mimirmind::core::safetensors