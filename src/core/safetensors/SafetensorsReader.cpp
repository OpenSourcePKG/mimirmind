// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/safetensors/SafetensorsReader.hpp"

#include <utility>

namespace mimirmind::core::safetensors {

void SafetensorsReader::open(std::string_view path) {
    close();

    // MappedFile's constructor throws on open/fstat/empty/mmap failure.
    _file = l0::MappedFile(path);

    ParsedSafetensorsHeader parsed = parseSafetensorsHeader(_file.bytes());

    _tensors    = std::move(parsed.tensors);
    _metadata   = std::move(parsed.metadata);
    _dataOffset = parsed.dataOffset;

    _index.clear();
    for (std::size_t i = 0; i < _tensors.size(); ++i) {
        _index.emplace(_tensors[i].name, i);
    }
}

void SafetensorsReader::close() noexcept {
    _file.close();
    _tensors.clear();
    _index.clear();
    _metadata.clear();
    _dataOffset = 0;
}

const SafetensorsTensor* SafetensorsReader::find(std::string_view name) const noexcept {
    // std::map has no heterogeneous find before C++14's transparent
    // comparator; a std::string key lookup is fine here (load-time path).
    const auto it = _index.find(std::string(name));
    if (it == _index.end()) {
        return nullptr;
    }
    return &_tensors[it->second];
}

std::span<const std::uint8_t>
SafetensorsReader::tensorBytes(const SafetensorsTensor& t) const noexcept {
    const auto whole = _file.bytes();
    if (t.dataEnd > whole.size() || t.dataBegin > t.dataEnd) {
        return {};
    }
    return whole.subspan(t.dataBegin, t.nbytes);
}

} // namespace mimirmind::core::safetensors