// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/config/Config.hpp"
#include "mimirmind/CliArgs.hpp"

namespace mimirmind::cli {

/**
 * 5.19 Increment C — lossy-tier quality gate.
 *
 * Loads the configured model once and runs a small DE goldset through a
 * greedy (temperature 0) A/B: the same prompt is generated with the candidate
 * lossy prefill flag OFF (the exact-F32 / BF16 reference path) and then ON,
 * and the two decoded token streams are compared. Because prefill is the only
 * phase the lossy flags touch, a divergence in the greedy decode trajectory is
 * exactly the flag's quality effect.
 *
 * A flag that produces TOKEN-EXACT output across the goldset is provably safe
 * to promote to `apply:true` in configs/hw/<fp>/probe-result.json (it cannot
 * change a greedy answer); a flag that diverges prints ref-vs-candidate text
 * for human semantic review rather than pretending to auto-judge meaning.
 *
 * CUDA-only in effect: on other backends the prefill-flag setters are no-ops,
 * so every case is trivially identical and the gate reports "nothing to gate".
 *
 * Exit code: 0 if every case is TOKEN-EXACT, 2 if any case diverges (review
 * required), 1 on a load/generate error.
 */
int runQualityGate(const CliArgs&                        args,
                   const ::mimirmind::core::config::Config& cfg);

} // namespace mimirmind::cli
