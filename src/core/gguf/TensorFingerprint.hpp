// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <string>

namespace mimirmind::core::gguf {

class GgufReader;

/**
 * Cheap identity check for a parsed GGUF. Deterministic: identical byte
 * inputs produce identical outputs across processes and machines.
 *
 * The Munin daemon computes this at model load; an attached mimirmind
 * worker computes the same value from its own local GGUF header (the
 * worker opens the file for parsing but skips `loadTensors`, so no USM
 * is spent) and refuses the attach if the two disagree. That is the
 * mechanism that catches "Munin has Q6 loaded, worker was configured
 * for Q8" — the M-Munin ADR's central guarantee.
 *
 * Format: `<tensorCount>.<totalTensorBytes>.<hex64 fnv1a>`. The two
 * leading fields are decorative but useful in logs (they're the first
 * things an operator glances at when debugging a fingerprint mismatch).
 * The FNV-1a hash covers header version, alignment, tensor-data offset,
 * total tensor bytes, metadata count, and each tensor's name / type /
 * dims / bytes / file-offset.
 *
 * Explicitly *not* a cryptographic hash. The fingerprint is an identity
 * check, not a security boundary. A cheap header-only hash is what the
 * ADR asks for; a full-file SHA over 25 GiB is a Tier-3 follow-up.
 */
[[nodiscard]] std::string tensorFingerprint(const GgufReader& reader);

} // namespace mimirmind::core::gguf