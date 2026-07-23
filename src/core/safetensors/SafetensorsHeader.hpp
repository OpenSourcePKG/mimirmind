// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/safetensors/SafetensorsDtype.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <vector>

namespace mimirmind::core::safetensors {

/// One tensor entry parsed from a safetensors header. Byte offsets are
/// ABSOLUTE within the file buffer (the 8-byte length prefix and JSON
/// header are already added in), so a consumer can slice the buffer
/// directly at `[dataBegin, dataEnd)`.
struct SafetensorsTensor {
    std::string                name;
    SafetensorsDtype           dtype{SafetensorsDtype::Unknown};
    std::vector<std::uint64_t> shape;          ///< empty for a scalar (nelements == 1)
    std::uint64_t              nelements{0};    ///< product of shape (1 for scalar)
    std::size_t                nbytes{0};       ///< dataEnd - dataBegin
    std::size_t                dataBegin{0};    ///< absolute offset of first byte
    std::size_t                dataEnd{0};      ///< absolute offset one past last
};

/// Result of parsing a safetensors header off a whole-file buffer.
struct ParsedSafetensorsHeader {
    std::vector<SafetensorsTensor>     tensors;     ///< ascending name order
    std::map<std::string, std::string> metadata;    ///< `__metadata__`, empty if absent
    std::size_t                        dataOffset{0}; ///< 8 + headerLen
};

/**
 * Parse the header of a safetensors file given the WHOLE file as a byte
 * span (8-byte little-endian header length, then that many bytes of JSON,
 * then the tensor-data region). Pure and allocation-only — no mmap, no I/O
 * — so the parse logic is unit-testable against hand-crafted buffers
 * without a real file or a modern-libstdc++ toolchain.
 *
 * Throws std::runtime_error on: a buffer shorter than the declared header,
 * malformed JSON, a per-tensor entry missing/mistyping `dtype` / `shape` /
 * `data_offsets`, an unsupported/unknown dtype, a duplicate tensor name, a
 * `data_offsets` range outside the data region or with begin > end, or a
 * byte length inconsistent with dtype width × element count.
 *
 * `data_offsets` in the header are relative to the start of the data
 * region; the returned `dataBegin` / `dataEnd` are absolute within `file`.
 */
[[nodiscard]] ParsedSafetensorsHeader
parseSafetensorsHeader(std::span<const std::uint8_t> file);

} // namespace mimirmind::core::safetensors