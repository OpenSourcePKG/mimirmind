// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstddef>

namespace mimirmind::compute {

/// Row-wise residual-stream ops for the 7.11 Rune projection / conditional
/// steer overlays (Futhark `steer_project` / `steer_if`). Segregated out of the
/// backend-neutral `ComputeOps` (8.30.6 ISP-split): a backend that implements
/// them returns a non-null `IRuneSteerOps*` from `ComputeOps::runeSteerOps()`;
/// the engine refuses such a rune loudly on a backend that returns nullptr.
///
/// Both ops are pure device work on the compute stream (no host sync), so they
/// run inside captured CUDA graphs. Rows are independent: no op reads a row
/// other than the one it writes, which is what makes the gate per row (and so
/// per token) valid on every batched / ragged serving path.
class IRuneSteerOps {
public:
    virtual ~IRuneSteerOps() = default;

    /// out[r] = <x[r,:], w>, divided by ||x[r,:]|| when `normalize` (0 for a
    /// zero row). x is [rows, d] row-major, w is [d], out is [rows].
    virtual void rowDotAsync(const float* x, std::size_t rows, std::size_t d,
                             const float* w, float* out, bool normalize) = 0;

    /// x[r,:] += alpha * g[r] * c[r] * v, with
    ///   g[r] = gate == nullptr ? 1 : (gate[r] > threshold ? 1 : 0)
    ///   c[r] = coef == nullptr ? 1 : coef[r]
    /// x is [rows, d], v is [d], gate / coef are [rows].
    virtual void steerRowsAsync(float* x, std::size_t rows, std::size_t d,
                                const float* v, float alpha,
                                const float* gate, float threshold,
                                const float* coef) = 0;
};

} // namespace mimirmind::compute
