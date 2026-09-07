// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/safetensors/SafetensorsModel.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace mimirmind::runtime::nvfp4 {

/**
 * qwen4_exp PLE n-gram embedding table (5.27 I-4). The ~51 GiB FP8 table
 * (128 shards × [2500012, 160] F8_E4M3, one logical [padded_vocab, 160] +
 * a single per-table F32 `weight_scale`) is FAR too big to keep resident on
 * the 128 GiB unified pool alongside the weights, so it stays OFF-VRAM:
 * this class owns a `SafetensorsModel` whose shard mmaps hold the table on
 * disk/page-cache, and only the ~16 rows/token the n-gram gather touches ever
 * fault in (Flash-DGX tiering; the plan's "host-gather, off-VRAM PLE").
 *
 * `gatherDequantBf16` reads the requested rows straight from the mmap and
 * dequantises F8_E4M3 × scale → BF16 on the host, ready for a tiny H2D + the
 * device PLE projections. Move-only (owns the SafetensorsModel mmaps).
 */
class PleNgramTable {
public:
    PleNgramTable() = default;

    PleNgramTable(const PleNgramTable&)            = delete;
    PleNgramTable& operator=(const PleNgramTable&) = delete;
    PleNgramTable(PleNgramTable&&) noexcept            = default;
    PleNgramTable& operator=(PleNgramTable&&) noexcept = default;

    /// Open the checkpoint at `checkpointDir` and bind the n-gram shards for
    /// the given HF layer (e.g. "model.language_model.layers.1"). Enumerates
    /// `<hfLayerPrefix>.ple.ple_embedding.ngram_embedding.shard_{0,1,...}.weight`
    /// until absent, validates each is 2-D F8_E4M3 with `cols` columns, and
    /// reads `<...>.ngram_embedding.weight_scale` (F32 scalar). Throws on any
    /// missing/malformed tensor.
    void open(const std::string& checkpointDir, const std::string& hfLayerPrefix,
              std::size_t cols);

    [[nodiscard]] bool          isOpen()   const noexcept { return !_shardData.empty(); }
    [[nodiscard]] std::int64_t  rowsTotal() const noexcept { return _rowsTotal; }
    [[nodiscard]] std::size_t   cols()      const noexcept { return _cols; }
    [[nodiscard]] float         scale()     const noexcept { return _scale; }

    /// The checkpoint's stored `ple_embedding.layer_multipliers` (int64
    /// [ngram_size]) — read verbatim so the host n-gram hash matches the model
    /// bit-for-bit without re-deriving them from vocab_size/seed.
    [[nodiscard]] const std::vector<std::int64_t>& layerMultipliers() const noexcept {
        return _mult;
    }

    /// Gather + dequantise `ids.size()` rows into `outF32` (row-major
    /// [ids.size(), cols] F32 — the layout the BF16-weight matmul consumes as
    /// its activation input). Each id must be in [0, rowsTotal). Reads FP8 bytes
    /// from the mmap (cold rows fault from disk); out = e4m3(byte) * scale.
    void gatherDequantF32(std::span<const std::int64_t> ids, float* outF32) const;

private:
    core::safetensors::SafetensorsModel _sm;
    std::vector<const std::uint8_t*>    _shardData;    ///< host ptr per shard (into mmap)
    std::vector<std::int64_t>           _shardRowStart; ///< cumulative first-row per shard (+ total at end)
    std::int64_t                        _rowsTotal{0};
    std::size_t                         _cols{0};
    float                               _scale{1.0F};
    std::vector<std::int64_t>           _mult;   ///< stored layer_multipliers
};

} // namespace mimirmind::runtime::nvfp4
