// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/gguf/WeightsMap.hpp"
#include "runtime/nvfp4/NvFp4Materializer.hpp"

#include <vector>

namespace mimirmind::runtime::nvfp4 {

/**
 * Wrap the materialised BF16 tensors in a GGUF-convention `WeightsMap` the
 * existing arch backend addresses (`findBlock(N, "attn_q.weight")` etc.).
 * Each `GgufTensor` is BF16 with `usmPtr` pointing at the corresponding
 * `MaterializedTensor`'s device buffer — so `mats` (which owns those buffers)
 * MUST outlive the returned map. Takes a non-const ref because `usmPtr` is a
 * mutable `void*` in the GGUF tensor model.
 */
[[nodiscard]] core::gguf::WeightsMap
buildBf16WeightsMap(std::vector<MaterializedTensor>& mats);

} // namespace mimirmind::runtime::nvfp4