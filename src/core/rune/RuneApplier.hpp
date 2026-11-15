// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/rune/RuneFile.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace mimirmind::core::rune {

/// Element encoding of a target tensor as seen by the applier.
enum class RuneElemType : std::uint8_t {
    F32,
    F16,
    BF16,
    Unsupported,   ///< quantised / packed layouts — refused, never silently skipped
};

/**
 * Host-accessible view of one target tensor. `bytes` must be writable host
 * memory (a staging copy for device-resident weights — the engine-side Futhark
 * integration downloads, applies, uploads).
 *
 * `hasDerivedCopies` must be set when the engine keeps a derived copy of the
 * tensor that kernels read instead of (or in addition to) these bytes — e.g.
 * fused-QKV buffers, FP4 tensor-core side banks, repacked layouts. Patching
 * only the primary bytes would then change nothing the model actually reads,
 * so the applier refuses such targets.
 */
struct RuneTensorView {
    std::string                 name;
    RuneElemType                type{RuneElemType::Unsupported};
    std::vector<std::uint64_t>  dims;      ///< GGUF order, dims[0] innermost
    std::span<std::byte>        bytes;
    bool                        hasDerivedCopies{false};
};

/// Resolve a target tensor name to a view; nullopt if the model has no such tensor.
using RuneTensorResolver = std::function<std::optional<RuneTensorView>(std::string_view)>;

/// One tensor the applier changed (in first-touch order).
struct RuneTouchedTensor {
    std::string   name;
    std::string   hashBefore;   ///< content hash before the rune touched it
    std::string   hashAfter;    ///< content hash after all of the rune's ops
};

/// Result of a successful apply.
struct RuneApplyReport {
    std::string                     runeName;
    std::size_t                     opsApplied{0};
    std::vector<RuneTouchedTensor>  touched;
};

/**
 * Host-side Rune applier — the backend-neutral core of Futhark (the in-engine Rune applier).
 *
 * Contract:
 *  - Rejects the whole rune (nothing modified) when the base fingerprint does
 *    not match, an op kind is not applicable in this build, or a target is
 *    missing / of an unsupported type / has derived copies.
 *  - Applies ops in manifest order. Before each op it checks the op's
 *    `tensor_hash` against the target's CURRENT bytes, so stacked runes and
 *    multiple ops on one tensor chain naturally (each op is bound to the
 *    state it was authored against).
 *  - Any failure mid-way restores every byte this call modified and returns
 *    the error: apply is all-or-nothing.
 *  - A successful apply keeps byte backups; `rollback()` restores them
 *    (latest apply first) for a runtime undo without reloading the model.
 *
 * Math is done in F32 and rounded back to the target encoding with
 * round-to-nearest-even.
 */
class RuneApplier {
public:
    [[nodiscard]] std::expected<RuneApplyReport, std::string>
    apply(const RuneFile& rune, std::string_view baseFingerprint,
          const RuneTensorResolver& resolve);

    /// Undo every successful apply, newest first. Returns the number of
    /// tensors restored. Backups are released afterwards.
    std::size_t rollback() noexcept;

    /// Number of applies that can still be rolled back.
    [[nodiscard]] std::size_t appliedCount() const noexcept { return _history.size(); }

private:
    struct Backup {
        std::span<std::byte>    where;
        std::vector<std::byte>  original;
    };
    using BackupSet = std::vector<Backup>;

    static void restore(BackupSet& set) noexcept;

    std::vector<BackupSet> _history;
};

/// IEEE binary16 <-> binary32 and bfloat16 <-> binary32 (round-to-nearest-even).
[[nodiscard]] float         halfBitsToFloat(std::uint16_t h) noexcept;
[[nodiscard]] std::uint16_t floatToHalfBits(float f) noexcept;
[[nodiscard]] float         bf16BitsToFloat(std::uint16_t b) noexcept;
[[nodiscard]] std::uint16_t floatToBf16Bits(float f) noexcept;

} // namespace mimirmind::core::rune
