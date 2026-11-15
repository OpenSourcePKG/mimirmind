// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace mimirmind::core::rune {

/**
 * Rune — the on-disk unit of a weight change (roadmap 7.1, ADR
 * "Hugin adaptation companion + Runes").
 *
 * Invariant the format exists for: the base model file is never modified.
 * Every change is a Rune that is hash-bound to one base model, applied after
 * load, and fully reversible by not loading it.
 *
 * Container: a Rune IS a valid safetensors file. Payload blobs are ordinary
 * F32 tensors; the manifest (this header's `RuneManifest`) is a JSON string
 * stored under the safetensors `__metadata__` key `kRuneMetadataKey`. Any
 * safetensors tool can therefore list a Rune's payloads.
 *
 * Tensor naming follows the engine's GGUF convention (`blk.<i>.<suffix>`,
 * `token_embd.weight`, ...), i.e. the names a `WeightsMap` resolves.
 *
 * Shape convention for the ops: a target tensor with GGUF dims
 * `[ne0, ne1, ...]` is viewed as a row-major matrix of
 * `rows = nelements / ne0` rows by `cols = ne0` columns (ne0 is innermost).
 */

/// `__metadata__` key holding the manifest JSON.
inline constexpr std::string_view kRuneMetadataKey = "mimirmind.rune";

/// Manifest `format` field value.
inline constexpr std::string_view kRuneFormatName = "mimirmind-rune";

/// Highest manifest version this build understands.
inline constexpr std::uint32_t kRuneFormatVersion = 1;

/// What one op does to its target tensor.
enum class RuneOpKind : std::uint8_t {
    WriteSlice,        ///< overwrite rows [rowBegin,rowEnd) with payload values
    Scale,             ///< multiply each row (or column) by a payload factor
    Mask,              ///< load-time removal of experts/heads/layers (not yet applicable)
    LowRank,           ///< overlay y = Wx + s*B(Ax) (not yet applicable)
    Steer,             ///< residual-stream activation offset (not yet applicable)
    WriteQuantBlocks,  ///< overwrite pre-quantised NVFP4 blocks (not yet applicable)
};

/// Axis of a `Scale` op.
enum class RuneScaleAxis : std::uint8_t {
    Rows,   ///< payload length == rows; row r *= payload[r]
    Cols,   ///< payload length == cols; column c *= payload[c]
};

/// One manifest op. Fields not used by `kind` stay at their defaults.
struct RuneOp {
    RuneOpKind    kind{RuneOpKind::WriteSlice};
    std::string   target;        ///< tensor name (GGUF convention)
    std::string   tensorHash;    ///< `fnv1a64:<16 hex>` of the target bytes BEFORE this op
    std::string   payload;       ///< name of the F32 payload tensor in the container
    std::uint64_t rowBegin{0};   ///< WriteSlice only
    std::uint64_t rowEnd{0};     ///< WriteSlice only (exclusive)
    RuneScaleAxis axis{RuneScaleAxis::Rows};   ///< Scale only
};

/// Parsed manifest.
struct RuneManifest {
    std::uint32_t       version{kRuneFormatVersion};
    std::string         name;              ///< human-readable rune name
    std::string         baseFingerprint;   ///< identity of the base model it was built against
    std::string         provenanceJson{"{}"};  ///< free-form JSON object (who/what/eval), kept verbatim
    std::vector<RuneOp> ops;               ///< applied in order
};

/// Manifest spelling of an op kind ("write_slice", "scale", ...).
[[nodiscard]] std::string_view opKindName(RuneOpKind k) noexcept;

/// Inverse of `opKindName`; nullopt for unknown spellings.
[[nodiscard]] std::optional<RuneOpKind> opKindFromName(std::string_view s) noexcept;

/// True for op kinds the host applier can execute in this build.
[[nodiscard]] bool opKindApplicable(RuneOpKind k) noexcept;

/// FNV-1a 64 over `bytes`, formatted as `fnv1a64:<16 lowercase hex>`. Identity
/// check for a target tensor's contents, not a security boundary.
[[nodiscard]] std::string tensorContentHash(const void* bytes, std::size_t n) noexcept;

} // namespace mimirmind::core::rune
