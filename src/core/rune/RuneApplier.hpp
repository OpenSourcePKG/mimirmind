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
    Q8_0,          ///< GGML Q8_0 (f16 scale + 32 int8 per block along ne0): exact Scale
                   ///< only (see FP8_E4M3_B32)
    FP8_E4M3_B32,  ///< blocked FP8 (f16 scale + 32 E4M3 per block along ne0, the loader's
                   ///< attention format): exact Scale only — factor 0 zeroes a block, any
                   ///< other factor multiplies its f16 scale and must stay exactly
                   ///< representable (powers of two), e.g. head ablation / amplification
    Unsupported,   ///< other quantised / packed layouts — refused, never silently skipped
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

/// A Steer op resolved for the engine: add `values` (payload x alpha,
/// d_model floats) to the residual stream after `block`, every row.
struct RuneSteer {
    std::uint32_t       block{0};
    std::vector<float>  values;
};

/// A SteerProject / SteerIf op resolved for the engine (7.11). On the rows it
/// gates on, after `block`:
///   Add:     x += values                      (values = payload x alpha)
///   Project: x -= alpha * (x . values) values (values = unit direction)
/// The gate is 1 for every row without a condition; with one it is
/// [ cos(x_conditionBlock, condition) > threshold ] per row, read on the
/// residual after `conditionBlock` (condition = unit direction).
struct RuneCondSteer {
    std::uint32_t       block{0};
    RuneSteerMode       mode{RuneSteerMode::Add};
    std::vector<float>  values;
    float               alpha{1.0F};   ///< Project only (Add folds alpha into values)
    bool                hasCondition{false};
    std::uint32_t       conditionBlock{0};
    std::vector<float>  condition;
    float               threshold{0.0F};
};

/// A Mask op resolved for the engine: experts with keep[e] == 0 are never
/// routed to in `block` (router logit pushed to -inf before top-K).
struct RuneExpertMask {
    std::uint32_t              block{0};
    std::vector<std::uint8_t>  keep;   ///< [n_experts], 1 = kept, 0 = disabled
};

/// A LowRank op resolved for the engine (7.4.3): on the projection `target`
/// (weight [out, in]) the engine adds y += B (A x) after the base matmul;
/// alpha is folded into b.
struct RuneLowRank {
    std::string         target;
    std::uint32_t       rank{0};
    std::uint64_t       in{0};
    std::uint64_t       out{0};
    std::vector<float>  a;   ///< [rank, in]
    std::vector<float>  b;   ///< [out, rank], already x alpha
};

/// Result of a successful apply.
struct RuneApplyReport {
    std::string                     runeName;
    std::size_t                     opsApplied{0};
    std::vector<RuneTouchedTensor>  touched;
    std::vector<RuneSteer>          steers;   ///< activation overlays; no weight touched
    std::vector<RuneCondSteer>      condSteers;   ///< 7.11 projection / conditional overlays
    std::vector<RuneExpertMask>     expertMasks;   ///< router overlays; no weight touched
    std::vector<RuneLowRank>        lowRanks;      ///< projection overlays; no weight touched
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
 * round-to-nearest-even. Block-scaled targets (Q8_0, FP8_E4M3_B32) accept only
 * Scale with factors uniform per 32-column block (axis=cols): 0 zeroes the
 * block, 1 leaves it, any other factor multiplies the block's f16 scale and
 * is refused unless the product is exact in f16 (powers of two) — so a block
 * Scale never introduces rounding.
 *
 * Steer ops touch no weight: they are validated (finite payload) and handed
 * back in `RuneApplyReport::steers` for the engine's residual overlay.
 * Mask ops (expert disable) touch no weight either: target = a block's router,
 * payload = one 0/1 factor per expert (= router rows), at least one expert
 * disabled and one kept; handed back in `RuneApplyReport::expertMasks`.
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
