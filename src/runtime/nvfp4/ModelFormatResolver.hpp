// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/config/Config.hpp"

#include <string_view>

namespace mimirmind::runtime::nvfp4 {

/**
 * Resolve a model entry's declared `format` to a concrete one by probing the
 * filesystem when it is `Auto`. Kept out of `Config` (which stays
 * filesystem-free) — this is the load-path helper the note refers to.
 *
 * For `Auto`:
 *   - a directory containing `model.safetensors.index.json` or
 *     `model.safetensors`            -> Nvfp4
 *   - a path ending in `.safetensors`-> Nvfp4
 *   - anything else (e.g. a `.gguf`) -> Gguf
 * A non-`Auto` declared format is returned unchanged (no probing), so an
 * explicit `format` always wins and a missing/renamed path fails later in
 * the concrete loader with a clearer message.
 */
[[nodiscard]] core::config::ModelFormat
resolveModelFormat(core::config::ModelFormat declared, std::string_view path);

} // namespace mimirmind::runtime::nvfp4