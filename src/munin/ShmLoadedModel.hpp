// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/gguf/GgufReader.hpp"
#include "core/ipc/TensorManifest.hpp"
#include "munin/ShmChunkAllocator.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mimirmind::munin {

/**
 * One model held resident by a CUDA/GB10 Munin in POSIX-shm — the shm
 * analogue of LoadedModel (ADR 2026-08-14, step 2). The GgufReader parses
 * the file and `loadTensorsIntoShmChunks` copies each tensor's raw payload
 * into the memfd-backed ShmChunkAllocator; dropping the ShmLoadedModel
 * releases every chunk (munmap + close), which invalidates every attached
 * worker's mapping — the same shutdown semantics the L0 Munin promises.
 *
 * MVP materialization is raw GGUF bytes (no dequant on the Munin side — the
 * attached worker dequantises at compute time), so there is no WeightsMap
 * here yet; that lands with the worker-attach path (step 3).
 *
 * Pure host code — builds in mimirmind_core_common, no GPU SDK.
 * `chunks` is declared before `reader` so it outlives the reader's tensor
 * views (destruction is reverse declaration order).
 */
struct ShmLoadedModel {
    std::string                        id{};
    std::string                        path{};
    std::string                        fingerprint{};
    std::uint64_t                      totalBytes{0};
    std::unique_ptr<ShmChunkAllocator> chunks{};

    // ---- GGUF mode -------------------------------------------------------
    // Populated for a GGUF model (format ""/"gguf"): the reader holds the
    // per-tensor {chunkIndex, chunkOffset}. Null in NVFP4 mode.
    std::unique_ptr<::mimirmind::core::gguf::GgufReader> reader{};

    // ---- NVFP4 mode ------------------------------------------------------
    // Populated for an NVFP4 (compressed-tensors safetensors) checkpoint:
    // the chunks hold each shard's raw *.safetensors image and `shards`
    // describes them. The worker reconstructs a SafetensorsModel via
    // SafetensorsModel::openFromShards and runs the NVFP4 materialization.
    bool                                    isNvfp4{false};
    std::vector<::mimirmind::core::ipc::ShardDesc> shards{};
    std::uint64_t                           declaredTotalSize{0};

    /**
     * Build the wire manifest. GGUF mode (v2): one ChunkDesc per shm chunk
     * plus every tensor's {name, type, dims, bytes, chunkIndex, chunkOffset}.
     * NVFP4 mode: format="nvfp4", the same ChunkDescs, plus the shard list +
     * declaredTotalSize (no per-tensor entries). The worker resolves against
     * the imported chunk bases either way. Mirrors LoadedModel::buildManifest.
     */
    [[nodiscard]] ::mimirmind::core::ipc::TensorManifest buildManifest() const;
};

} // namespace mimirmind::munin
