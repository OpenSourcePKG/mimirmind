// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace mimirmind::core::safetensors {

/// Parsed `model.safetensors.index.json` — the shard map for a checkpoint
/// split across several `*.safetensors` files.
struct SafetensorsIndex {
    /// tensor name -> shard filename (e.g. "model-00001-of-00003.safetensors").
    std::map<std::string, std::string> weightMap;

    /// `metadata.total_size` if present, else 0. NOTE: this counts only
    /// tensor-data bytes across all shards, NOT the per-shard JSON headers,
    /// so it is smaller than the summed file sizes — expose it, do not use
    /// it as a byte-exact integrity check.
    std::uint64_t totalSize{0};

    /// Distinct shard filenames referenced by `weightMap`, ascending order.
    [[nodiscard]] std::vector<std::string> shardFiles() const;
};

/**
 * Parse the JSON text of a `model.safetensors.index.json`. Pure — takes the
 * text, does no I/O — so it is unit-testable without a real checkpoint.
 *
 * Throws std::runtime_error on: malformed JSON, a missing/non-object
 * `weight_map`, a non-string shard filename, an empty `weight_map`, or a
 * `metadata.total_size` that is present but not an unsigned integer.
 */
[[nodiscard]] SafetensorsIndex parseSafetensorsIndex(std::string_view jsonText);

} // namespace mimirmind::core::safetensors