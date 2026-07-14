// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace mimirmind::core::l0 {

/**
 * POSIX mmap RAII wrapper. PROT_READ + MAP_PRIVATE — we never write to
 * the mapping (tensor data is memcpy'd out into USM). madvise(MADV_RANDOM)
 * because the GGUF reader jumps between the header, metadata, tensor
 * index, and far-apart tensor payload offsets.
 */
class MappedFile {
public:
    MappedFile() = default;
    explicit MappedFile(std::string_view path);
    ~MappedFile();

    MappedFile(const MappedFile&)            = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    MappedFile(MappedFile&&) noexcept;
    MappedFile& operator=(MappedFile&&) noexcept;

    [[nodiscard]] const std::uint8_t* data() const noexcept { return _data; }
    [[nodiscard]] std::size_t         size() const noexcept { return _size; }
    [[nodiscard]] std::string_view    path() const noexcept { return _path; }
    [[nodiscard]] bool                isOpen() const noexcept { return _data != nullptr; }
    [[nodiscard]] std::span<const std::uint8_t> bytes() const noexcept {
        return {_data, _size};
    }

    void close() noexcept;

private:
    std::uint8_t* _data{nullptr};
    std::size_t   _size{0};
    int           _fd{-1};
    std::string   _path{};
};

} // namespace mimirmind::core::l0