// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

namespace mimirmind::core::backend {

/**
 * How `BackendPool::select` picks an entry when the caller has no
 * explicit token.
 *
 * `Auto` is the only mode wired for the first cut — it walks the pool
 * entries in `BackendKind` enum order (LevelZero, Hip, Cuda, Cpu) and
 * returns the first available. Two follow-up modes are named up front
 * so the surface stays stable when they land:
 *
 * - `BestFor(model)` — perf-heuristic pick (VRAM fit, historical
 *   throughput from perf-regression ledger). Not implemented in this
 *   cut; the enum is reserved so downstream code compiles when it
 *   lands.
 * - `Parallel` — multi-entry return for tensor / model parallelism.
 *   Needs a compute-layer story for cross-device copies first; deferred.
 */
enum class SelectionMode {
    Auto,
};

} // namespace mimirmind::core::backend
