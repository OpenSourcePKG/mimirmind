// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/os/MappedFile.hpp"
#include "core/safetensors/SafetensorsHeader.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace mimirmind::core::safetensors {

/**
 * Reader for a single `*.safetensors` file.
 *
 * Owns a read-only mmap of the file and the parsed tensor index. The heavy
 * lifting — header validation — lives in the pure `parseSafetensorsHeader`;
 * this class is the thin mmap + lookup wrapper on top. It does NOT
 * interpret quantisation schemes (that is `core::modelopt`) and does NOT
 * upload to device (a later loader step). Mirrors `GgufReader`'s lifetime
 * model: move-only, `open()` throws on any malformation.
 *
 * A ModelOpt checkpoint is usually sharded across several files with a
 * `model.safetensors.index.json` weight-map; this reader handles one shard.
 * The multi-shard orchestration layers on top.
 */
class SafetensorsReader {
public:
    SafetensorsReader() = default;
    ~SafetensorsReader() = default;

    SafetensorsReader(const SafetensorsReader&)            = delete;
    SafetensorsReader& operator=(const SafetensorsReader&) = delete;
    SafetensorsReader(SafetensorsReader&&) noexcept            = default;
    SafetensorsReader& operator=(SafetensorsReader&&) noexcept = default;

    /// mmap `path` and parse its header + tensor index. Throws
    /// std::runtime_error on an unopenable/empty file or any malformation
    /// reported by `parseSafetensorsHeader`.
    void open(std::string_view path);

    /// Release the mmap and reset state. Idempotent.
    void close() noexcept;

    [[nodiscard]] bool             isOpen()      const noexcept { return _file.isOpen(); }
    [[nodiscard]] std::string_view path()        const noexcept { return _file.path(); }
    [[nodiscard]] std::size_t      tensorCount() const noexcept { return _tensors.size(); }

    /// Parsed tensors, in ascending name order.
    [[nodiscard]] const std::vector<SafetensorsTensor>& tensors() const noexcept {
        return _tensors;
    }

    /// Lookup by exact name, or nullptr. O(log n).
    [[nodiscard]] const SafetensorsTensor* find(std::string_view name) const noexcept;

    /// `__metadata__` key/value pairs (both strings). Empty if absent.
    [[nodiscard]] const std::map<std::string, std::string>& metadata() const noexcept {
        return _metadata;
    }

    /// Zero-copy view of a tensor's bytes in the mmap, valid for the
    /// reader's lifetime. Empty span if `t` does not belong to this reader
    /// (offsets outside the mapping).
    [[nodiscard]] std::span<const std::uint8_t>
    tensorBytes(const SafetensorsTensor& t) const noexcept;

    /// Absolute offset where the tensor-data region begins (= 8 + headerLen).
    [[nodiscard]] std::size_t dataOffset() const noexcept { return _dataOffset; }

private:
    l0::MappedFile                     _file;
    std::vector<SafetensorsTensor>     _tensors;
    std::map<std::string, std::size_t> _index;      ///< name -> _tensors index
    std::map<std::string, std::string> _metadata;
    std::size_t                        _dataOffset{0};
};

} // namespace mimirmind::core::safetensors