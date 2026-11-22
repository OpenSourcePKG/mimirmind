// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/ipc/WeightArenaManifest.hpp"
#include "runtime/nvfp4/NvFp4Materializer.hpp"

#include <cstddef>
#include <functional>
#include <vector>

namespace mimirmind::core::ipc {
class SharedWeightArena;
}

namespace mimirmind::runtime::nvfp4 {

/**
 * 7.2.3 — bridge between the engine's materialized tensors and the shared
 * weight arena manifest.
 */
struct SharedArenaWeights {
    /// Copies `bytes` from `src` to `dst` (device-visible pointers).
    using CopyFn = std::function<void(void* dst, const void* src, std::size_t bytes)>;

    /// Move every buffer that is not already inside `arena` into it (allocate
    /// from the arena, copy, swap the buffer; the old buffer is freed). The
    /// allocation goes through `arena` directly, so the moved buffer's deleter
    /// returns the range to the arena. Returns the number of buffers moved —
    /// 0 when the loader allocated everything inside the arena scope.
    static std::size_t migrateInto(std::vector<MaterializedTensor>& mats,
                                   core::ipc::SharedWeightArena&    arena,
                                   const CopyFn&                    copy);

    /// Describe `mats` (all buffers must lie inside `arena`; throws otherwise).
    /// Header fields (model id, fingerprint, ...) are left for the caller.
    static core::ipc::WeightArenaManifest
    describe(const std::vector<MaterializedTensor>& mats,
             const core::ipc::SharedWeightArena&    arena);

    /// Rebuild the tensors on an attached mapping. The buffers are NON-owning
    /// views into `base`; the mapping must outlive them.
    static std::vector<MaterializedTensor>
    bind(const core::ipc::WeightArenaManifest& manifest, void* base);
};

} // namespace mimirmind::runtime::nvfp4
