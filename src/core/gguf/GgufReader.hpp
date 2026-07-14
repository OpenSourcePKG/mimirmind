// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/gguf/GgufTypes.hpp"
#include "core/os/MappedFile.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace mimirmind::core::l0 {
class UsmAllocator;
}

namespace mimirmind::munin {
class ChunkAllocator;
}

namespace mimirmind::core::gguf {

/// Array stored in metadata. For primitive element types `raw` holds the
/// packed little-endian element data and `strings` is empty. For string
/// arrays it's the reverse. Nested arrays are not supported (GGUF v3
/// disallows them).
struct GgufArray {
    GgufValueType            elementType{GgufValueType::UInt8};
    std::uint64_t            count{0};
    std::vector<std::uint8_t> raw{};
    std::vector<std::string>  strings{};
};

using MetadataValue = std::variant<
    bool,
    std::uint8_t,  std::int8_t,
    std::uint16_t, std::int16_t,
    std::uint32_t, std::int32_t,
    std::uint64_t, std::int64_t,
    float,         double,
    std::string,
    GgufArray>;

struct GgufTensor {
    std::string                 name;
    GgmlType                    type{GgmlType::Unknown};
    std::vector<std::uint64_t>  dimensions;
    std::uint64_t               nelements{0};
    std::size_t                 nbytes{0};
    std::uint64_t               fileOffset{0};   // within the tensor-data region
    void*                       usmPtr{nullptr}; // set by loadTensors[IntoChunks]()

    // Chunk placement, populated by loadTensorsIntoChunks(). Meaningless
    // (both zero) when the tensor was loaded via the per-tensor
    // UsmAllocator path — those callers do not need chunk coordinates
    // because their pointer is the whole allocation.
    std::uint32_t               chunkIndex{0};
    std::uint64_t               chunkOffset{0};
};

/**
 * GGUF v3 reader. Parses header + metadata + tensor index from a mmap'd
 * file, and (optionally) copies each tensor's payload into per-tensor USM
 * allocations.
 *
 * Lifetime: the reader owns the MappedFile and any USM it has allocated.
 * Destroying it releases everything. Move-only.
 */
class GgufReader {
public:
    GgufReader() = default;
    ~GgufReader();

    GgufReader(const GgufReader&)            = delete;
    GgufReader& operator=(const GgufReader&) = delete;
    GgufReader(GgufReader&&) noexcept            = default;
    GgufReader& operator=(GgufReader&&) noexcept = default;

    /// Parse header + metadata + tensor index. Throws on bad magic /
    /// unsupported version / truncated file / unknown tensor type.
    void open(std::string_view path);

    /// Allocate per-tensor USM via `allocator` and memcpy the payload from
    /// the mmap. Idempotent. Throws if a tensor's offset/size walks off
    /// the end of the file.
    void loadTensors(core::l0::UsmAllocator& allocator);

    /// Munin-side load path. Packs every tensor into `chunks` via
    /// bump-allocation and memcpy'ies the payload from the mmap. Each
    /// tensor's `usmPtr`, `chunkIndex`, and `chunkOffset` are populated
    /// so the manifest builder can transport chunk coordinates over IPC
    /// (see M-Munin.1a ADR). The mmap is closed after the last copy
    /// exactly as in `loadTensors`.
    ///
    /// Idempotent. Cannot be mixed with `loadTensors` for the same
    /// reader — the two ownership regimes (per-tensor deallocate vs
    /// per-chunk drop) are mutually exclusive.
    void loadTensorsIntoChunks(::mimirmind::munin::ChunkAllocator& chunks);

    /// Release USM, drop the mmap, reset state. Idempotent.
    void close() noexcept;

    [[nodiscard]] std::uint32_t version()          const noexcept { return _version; }
    [[nodiscard]] std::size_t   alignment()        const noexcept { return _alignment; }
    [[nodiscard]] std::size_t   tensorDataOffset() const noexcept { return _tensorDataOffset; }
    [[nodiscard]] std::size_t   totalTensorBytes() const noexcept { return _totalTensorBytes; }
    [[nodiscard]] std::size_t   metadataCount()    const noexcept { return _metadata.size(); }
    [[nodiscard]] std::size_t   tensorCount()      const noexcept { return _tensors.size(); }

    [[nodiscard]] const std::vector<GgufTensor>& tensors() const noexcept { return _tensors; }
    [[nodiscard]] const std::map<std::string, MetadataValue>& metadata() const noexcept { return _metadata; }

    [[nodiscard]] const GgufTensor*    findTensor(std::string_view name) const noexcept;
    [[nodiscard]] const MetadataValue* findMetadata(std::string_view key) const noexcept;

private:
    core::l0::MappedFile                  _file{};
    std::uint32_t                        _version{0};
    std::size_t                          _alignment{32};
    std::size_t                          _tensorDataOffset{0};
    std::size_t                          _totalTensorBytes{0};
    std::map<std::string, MetadataValue> _metadata{};
    std::vector<GgufTensor>              _tensors{};
    core::l0::UsmAllocator*               _allocator{nullptr};
    // True once loadTensorsIntoChunks() ran successfully. Blocks close()
    // from calling per-tensor deallocate() — chunk memory is owned by
    // the ChunkAllocator and freed as a whole when it is destroyed.
    bool                                  _chunkLoaded{false};
};

} // namespace mimirmind::core::gguf