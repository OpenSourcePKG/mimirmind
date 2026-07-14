// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstddef>

namespace mimirmind::compute {

/**
 * Numerically-stable row-wise softmax over `[M, K]`, in place.
 *
 * If `causalKeepPerRow` is non-zero, treat each row `m` as having only
 * the first `causalKeepPerRow[m]` columns "live"; columns >=
 * causalKeepPerRow[m] are set to 0.0 and not included in the normaliser.
 * Pass nullptr for unmasked softmax. (Used for attention's causal mask
 * where row p only attends to positions <= p.)
 */
void softmaxRows(float*               data,
                 std::size_t          M,
                 std::size_t          K,
                 const std::size_t*   causalKeepPerRow = nullptr) noexcept;

} // namespace mimirmind::compute