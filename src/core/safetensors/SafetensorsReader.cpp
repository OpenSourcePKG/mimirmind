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

void SafetensorsReader::openBytes(std::span<const std::uint8_t> bytes,
                                  std::string_view               sourceName) {
    close();

    // Parse BEFORE committing any state so a malformed image leaves the
    // reader closed (isOpen() stays false), not half-open with a set span.
    ParsedSafetensorsHeader parsed = parseSafetensorsHeader(bytes);

    // Non-owning: the caller guarantees `bytes` outlives this reader.
    _external     = bytes;
    _externalName = std::string{sourceName};
    _tensors      = std::move(parsed.tensors);
    _metadata     = std::move(parsed.metadata);
    _dataOffset   = parsed.dataOffset;

    _index.clear();
    for (std::size_t i = 0; i < _tensors.size(); ++i) {
        _index.emplace(_tensors[i].name, i);
    }
}

void SafetensorsReader::close() noexcept {
    _file.close();
    _external = {};
    _externalName.clear();
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
    const auto whole = allBytes();
    if (t.dataEnd > whole.size() || t.dataBegin > t.dataEnd) {
        return {};
    }
    return whole.subspan(t.dataBegin, t.nbytes);
}

void SafetensorsReader::renameTensors(
    const std::function<std::string(const std::string&)>& fn) {
    _index.clear();
    for (std::size_t i = 0; i < _tensors.size(); ++i) {
        _tensors[i].name = fn(_tensors[i].name);
        _index.emplace(_tensors[i].name, i);
    }
}

bool SafetensorsReader::addDerivedRowSlice(std::string_view src,
                                           std::string      newName,
                                           std::uint64_t    rowBegin,
                                           std::uint64_t    rowCount) {
    const SafetensorsTensor* s = find(src);
    if (s == nullptr || s->shape.empty() || s->shape[0] == 0) {
        return false;
    }
    const std::uint64_t rows = s->shape[0];
    if (rowBegin + rowCount > rows || rowCount == 0
        || s->nbytes % rows != 0 || s->nelements % rows != 0) {
        return false;
    }
    const std::size_t bytesPerRow = s->nbytes / rows;
    const std::uint64_t elemsPerRow = s->nelements / rows;
    SafetensorsTensor d;
    d.name      = std::move(newName);
    d.dtype     = s->dtype;
    d.shape     = s->shape;
    d.shape[0]  = rowCount;
    d.nelements = elemsPerRow * rowCount;
    d.nbytes    = bytesPerRow * rowCount;
    d.dataBegin = s->dataBegin + bytesPerRow * rowBegin;
    d.dataEnd   = d.dataBegin + d.nbytes;
    _tensors.push_back(std::move(d));   // pointers invalidated; see header
    return true;
}

bool SafetensorsReader::duplicateTensorAs(std::string_view src,
                                          std::string      newName) {
    const SafetensorsTensor* s = find(src);
    if (s == nullptr) {
        return false;
    }
    SafetensorsTensor d = *s;
    d.name = std::move(newName);
    _tensors.push_back(std::move(d));
    return true;
}

bool SafetensorsReader::removeTensor(std::string_view name) {
    for (auto it = _tensors.begin(); it != _tensors.end(); ++it) {
        if (it->name == name) {
            _tensors.erase(it);
            return true;
        }
    }
    return false;
}

void SafetensorsReader::rebuildIndex() {
    _index.clear();
    for (std::size_t i = 0; i < _tensors.size(); ++i) {
        _index.emplace(_tensors[i].name, i);
    }
}

} // namespace mimirmind::core::safetensors