// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

// POSIX-shm load path for `GgufReader` (M-Munin.CUDA, ADR 2026-08-14 step 2).
// Mirrors `GgufReaderChunks.cpp` byte-for-byte in structure, but targets a
// `ShmChunkAllocator` (memfd + mmap host RAM) instead of the L0 USM
// `ChunkAllocator`. Kept a separate translation unit so it stays in
// `mimirmind_core_common` — no Level Zero, no CUDA — while the L0 variant
// lives in `mimirmind_core_l0`. The two loops are intentional duplicates:
// unifying them into one allocator-templated method touches the L0 TU, which
// is done later on an L0 box. Until then this path carries no L0 build risk.

#include "core/gguf/GgufReader.hpp"

#include "core/log/Log.hpp"
#include "munin/ShmChunkAllocator.hpp"

#include <cstring>
#include <stdexcept>

namespace mimirmind::core::gguf {

namespace {

double bytesToMiB(std::size_t b) noexcept {
    return static_cast<double>(b) / (1024.0 * 1024.0);
}

} // namespace

void GgufReader::loadTensorsIntoShmChunks(
    ::mimirmind::munin::ShmChunkAllocator& chunks) {
    if (!_file.isOpen()) {
        throw std::runtime_error(
            "GgufReader::loadTensorsIntoShmChunks: not open");
    }
    if (!_tensorBuffers.empty()) {
        throw std::runtime_error(
            "GgufReader::loadTensorsIntoShmChunks: reader already loaded via "
            "loadTensors — mixing ownership regimes is not supported");
    }
    if (_chunkLoaded) {
        return; // idempotent
    }

    MM_LOG_INFO("gguf",
                "loading {} tensor(s) into chunk-shm ({:.2f} MiB total)",
                _tensors.size(), bytesToMiB(_totalTensorBytes));

    std::size_t loadedBytes = 0;
    std::size_t loadedCount = 0;

    for (auto& t : _tensors) {
        if (t.usmPtr != nullptr) {
            continue;
        }

        const std::size_t absOffset = _tensorDataOffset +
                                      static_cast<std::size_t>(t.fileOffset);
        if (absOffset + t.nbytes > _file.size()) {
            MM_LOG_ERROR("gguf",
                         "tensor '{}' walks off EOF: abs_offset={} bytes={} "
                         "file_size={}",
                         t.name, absOffset, t.nbytes, _file.size());
            throw std::runtime_error(
                "GgufReader::loadTensorsIntoShmChunks: tensor data out of "
                "bounds for " + t.name);
        }

        const auto a  = chunks.allocate(t.nbytes);
        t.usmPtr      = a.ptr;
        t.chunkIndex  = a.chunkIndex;
        t.chunkOffset = a.chunkOffset;
        std::memcpy(t.usmPtr, _file.data() + absOffset, t.nbytes);

        loadedBytes += t.nbytes;
        ++loadedCount;

        if (loadedCount % 50 == 0) {
            MM_LOG_DEBUG("gguf",
                         "chunk-shm-progress: {}/{} tensors, {:.2f} MiB, "
                         "chunks={}",
                         loadedCount, _tensors.size(), bytesToMiB(loadedBytes),
                         chunks.chunkCount());
        }
    }

    _chunkLoaded = true;

    MM_LOG_INFO("gguf",
                "chunk-shm-load complete: {} tensor(s), {:.2f} MiB packed into "
                "{} chunk(s) ({:.2f} MiB / chunk max)",
                loadedCount, bytesToMiB(loadedBytes),
                chunks.chunkCount(), bytesToMiB(chunks.chunkBytes()));

    if (_file.isOpen()) {
        const std::size_t freed = _file.size();
        _file.close();
        MM_LOG_INFO("gguf",
                    "dropped mmap of source GGUF, freed ~{:.2f} MiB of page cache",
                    static_cast<double>(freed) / (1024.0 * 1024.0));
    }
}

} // namespace mimirmind::core::gguf
