// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace mimirmind::core::safetensors {

/// One shard of a safetensors checkpoint: file name and its size in bytes.
struct ShardSize {
    std::string   name;
    std::uint64_t bytes{0};
};

/**
 * Identity of a safetensors (NVFP4 / FP8 / BF16) checkpoint, built from its
 * shard {name, size} set plus the index's declared `total_size`.
 *
 * Format: `nvfp4:<16 hex fnv1a64>`. This is the exact value the Munin shm
 * daemon advertises as `modelFingerprint` for an NVFP4 model, so an attached
 * worker, a standalone engine, and Hugin all name the same checkpoint the
 * same way (Runes are bound to it). Header-level identity, not a content
 * hash — Runes add a per-tensor content hash on top.
 */
[[nodiscard]] std::string shardSetFingerprint(std::span<const ShardSize> shards,
                                              std::uint64_t              declaredTotalSize);

/**
 * Resolve the shard set of the checkpoint at `path` (a directory with
 * `model.safetensors.index.json` or `model.safetensors`, or a single
 * `*.safetensors` file), stat each shard, and return `shardSetFingerprint`.
 * Throws std::runtime_error when the checkpoint cannot be resolved.
 */
[[nodiscard]] std::string checkpointFingerprint(std::string_view path);

} // namespace mimirmind::core::safetensors
