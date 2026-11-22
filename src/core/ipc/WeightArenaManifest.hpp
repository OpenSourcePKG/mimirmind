// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstdint>
#include <expected>
#include <string>
#include <vector>

namespace mimirmind::core::ipc {

/// One buffer of a materialized tensor inside the shared weight arena.
struct WeightArenaBuffer {
    std::string   role;      ///< "data", "tcNibble", "tcSfb", "tcGlobals"
    std::uint64_t offset{0}; ///< byte offset from the arena base
    std::uint64_t bytes{0};
};

/// One materialized tensor as the engine holds it after load (format flags
/// name the on-device layout, e.g. "fp8", "nvfp4blk", "nvfp4tc", "f32").
struct WeightArenaTensor {
    std::string                    name;
    std::vector<std::uint64_t>     dims;
    std::uint64_t                  elems{0};
    std::vector<std::string>       flags;
    std::vector<WeightArenaBuffer> buffers;
};

/**
 * 7.2.3 — what an engine publishes about its memfd weight arena so an
 * attacher (Hugin) can rebuild the engine's weight map on the same bytes.
 * Serialized as compact JSON in the export handshake frame.
 */
struct WeightArenaManifest {
    static constexpr int kVersion = 1;

    int                            version{kVersion};
    std::string                    modelId;
    std::string                    architecture;
    std::string                    checkpointDir;
    std::string                    fingerprint;   ///< checkpointFingerprint(dir)
    std::uint64_t                  mapBytes{0};   ///< extent the attacher maps
    std::vector<std::string>       runes;         ///< runes the engine applied
    bool                           baseModified{false}; ///< a rune rewrote base bytes
    std::vector<WeightArenaTensor> tensors;

    [[nodiscard]] std::string toJson() const;
    [[nodiscard]] static std::expected<WeightArenaManifest, std::string>
    fromJson(const std::string& text);
};

} // namespace mimirmind::core::ipc
