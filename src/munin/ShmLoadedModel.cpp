// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "munin/ShmLoadedModel.hpp"

#include <cstdint>

namespace mimirmind::munin {

::mimirmind::core::ipc::TensorManifest ShmLoadedModel::buildManifest() const {
    using ::mimirmind::core::ipc::ChunkDesc;
    using ::mimirmind::core::ipc::ManifestEntry;
    using ::mimirmind::core::ipc::TensorManifest;

    TensorManifest m{};
    m.modelId          = id;
    m.modelFingerprint = fingerprint;

    // Chunks first — the worker walks this list to size its per-chunk import
    // loop. `bytes` is the used footprint, not the raw chunk size.
    const std::uint32_t nChunks = chunks ? chunks->chunkCount() : 0;
    m.chunks.reserve(nChunks);
    for (std::uint32_t i = 0; i < nChunks; ++i) {
        ChunkDesc cd{};
        cd.chunkIndex = i;
        cd.bytes      = chunks->chunkUsedBytes(i);
        m.chunks.push_back(cd);
    }

    if (isNvfp4) {
        // NVFP4: the chunks hold raw *.safetensors images; ship the shard
        // list so the worker can wrap each span and openFromShards.
        m.format            = "nvfp4";
        m.declaredTotalSize = declaredTotalSize;
        m.shards            = shards;
        return m;
    }

    // GGUF: per-tensor {chunkIndex, chunkOffset}.
    const auto& ts = reader->tensors();
    m.tensors.reserve(ts.size());
    for (const auto& t : ts) {
        ManifestEntry e{};
        e.name        = t.name;
        e.type        = t.type;
        e.dims        = t.dimensions;
        e.bytes       = static_cast<std::uint64_t>(t.nbytes);
        e.chunkIndex  = t.chunkIndex;
        e.chunkOffset = t.chunkOffset;
        m.tensors.push_back(std::move(e));
    }
    return m;
}

} // namespace mimirmind::munin
