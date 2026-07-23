// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace mimirmind::core::safetensors {

/**
 * safetensors tensor element dtype, as spelled in the container header's
 * per-tensor `dtype` field (see github.com/huggingface/safetensors — the
 * format is a JSON header followed by a raw tensor-data region).
 *
 * This is a pure container-level concept: it says how one stored element is
 * encoded, nothing about quantisation scheme. The ModelOpt scheme layer
 * (`core::modelopt`) sits ABOVE this and interprets which sidecar tensors of
 * which dtype form an NVFP4 / FP8 weight. Keeping the dtype here means the
 * container reader never has to depend on the interpretation layer.
 *
 * Only the subset the loader encounters is listed; anything else parses to
 * `Unknown` and must stop the load.
 */
enum class SafetensorsDtype : std::uint8_t {
    F32,      ///< "F32"     — global / weight / input scales, biases, router
    F16,      ///< "F16"
    BF16,     ///< "BF16"    — norms, embeddings, MTP, vision tower
    F8_E4M3,  ///< "F8_E4M3" — FP8 weights and NVFP4 block scales
    F8_E5M2,  ///< "F8_E5M2"
    U8,       ///< "U8"      — packed NVFP4 (2x E2M1 per byte)
    I8,       ///< "I8"
    I16,      ///< "I16"
    I32,      ///< "I32"
    I64,      ///< "I64"
    Bool,     ///< "BOOL"
    Unknown,
};

/// Parse a safetensors header `dtype` string. Returns `Unknown` for any
/// dtype outside the recognised subset. Case-sensitive (the format spells
/// dtypes in upper case).
[[nodiscard]] SafetensorsDtype dtypeFromString(std::string_view s) noexcept;

/// Bytes per stored element for a dtype. Returns 0 for `Unknown`. NOTE:
/// this is bytes per STORED element — for U8-packed NVFP4 that is 1 (the
/// byte holds two fp4 values); the 2x packing is a scheme property, not a
/// dtype property, and lives in `core::modelopt`.
[[nodiscard]] std::size_t dtypeWidth(SafetensorsDtype d) noexcept;

/// Human-readable name, matching the header spelling ("F32", "U8", ...).
[[nodiscard]] std::string_view dtypeName(SafetensorsDtype d) noexcept;

} // namespace mimirmind::core::safetensors