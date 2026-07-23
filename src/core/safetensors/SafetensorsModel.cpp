// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/safetensors/SafetensorsModel.hpp"

#include "core/os/MappedFile.hpp"
#include "core/safetensors/SafetensorsIndex.hpp"

#include <filesystem>
#include <stdexcept>
#include <utility>

namespace mimirmind::core::safetensors {

namespace fs = std::filesystem;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("safetensors model: " + msg);
}

/// Read a small text file (the index JSON) into a string via mmap.
std::string readTextFile(const fs::path& p) {
    l0::MappedFile m(p.string()); // throws on unopenable / empty
    const auto b = m.bytes();
    return std::string(reinterpret_cast<const char*>(b.data()), b.size());
}

} // namespace

void SafetensorsModel::open(std::string_view path) {
    close();

    const fs::path p{std::string(path)};
    std::error_code ec;

    if (fs::is_directory(p, ec)) {
        const fs::path indexPath  = p / "model.safetensors.index.json";
        const fs::path singlePath = p / "model.safetensors";
        if (fs::is_regular_file(indexPath, ec)) {
            openSharded(p.string(), indexPath.string());
        } else if (fs::is_regular_file(singlePath, ec)) {
            openSingle(singlePath.string());
        } else {
            fail("directory '" + p.string()
                 + "' has neither model.safetensors.index.json nor model.safetensors");
        }
    } else if (fs::is_regular_file(p, ec)) {
        openSingle(p.string());
    } else {
        fail("path '" + p.string() + "' is not a file or directory");
    }

    reindex();
}

void SafetensorsModel::openSingle(std::string_view file) {
    _shards.reserve(1);
    _shards.emplace_back();
    _shards.back().open(file);
    _totalSize = 0; // no index → no declared total
}

void SafetensorsModel::openSharded(std::string_view dir, std::string_view indexFile) {
    const SafetensorsIndex index = parseSafetensorsIndex(readTextFile(fs::path{std::string(indexFile)}));
    const std::vector<std::string> shardFiles = index.shardFiles();

    const fs::path base{std::string(dir)};
    _shards.reserve(shardFiles.size());

    // shard filename -> index into _shards, so we can resolve the weight-map.
    std::map<std::string, std::size_t> shardSlot;
    for (const std::string& shardName : shardFiles) {
        const fs::path shardPath = base / shardName;
        const std::size_t slot = _shards.size();
        _shards.emplace_back();
        _shards.back().open(shardPath.string());
        shardSlot.emplace(shardName, slot);
    }

    // Every tensor promised by the weight-map must actually live in the
    // shard it points at — catch a truncated or mismatched checkpoint now,
    // not on first access.
    for (const auto& [name, shardName] : index.weightMap) {
        const auto slotIt = shardSlot.find(shardName);
        if (slotIt == shardSlot.end()) {
            fail("weight-map references unknown shard '" + shardName + "'");
        }
        if (_shards[slotIt->second].find(name) == nullptr) {
            fail("tensor '" + name + "' promised in shard '" + shardName
                 + "' but not found there");
        }
    }

    _totalSize = index.totalSize;
}

void SafetensorsModel::reindex() {
    _tensorToShard.clear();
    _flat.clear();

    for (std::size_t s = 0; s < _shards.size(); ++s) {
        for (const SafetensorsTensor& t : _shards[s].tensors()) {
            const auto [it, inserted] = _tensorToShard.emplace(t.name, s);
            if (!inserted) {
                fail("tensor '" + t.name + "' appears in more than one shard");
            }
            _flat.push_back(&t);
        }
    }
}

void SafetensorsModel::close() noexcept {
    _shards.clear();
    _tensorToShard.clear();
    _flat.clear();
    _totalSize = 0;
}

const SafetensorsTensor* SafetensorsModel::find(std::string_view name) const noexcept {
    const auto it = _tensorToShard.find(std::string(name));
    if (it == _tensorToShard.end()) {
        return nullptr;
    }
    return _shards[it->second].find(name);
}

std::span<const std::uint8_t>
SafetensorsModel::tensorBytes(std::string_view name) const noexcept {
    const auto it = _tensorToShard.find(std::string(name));
    if (it == _tensorToShard.end()) {
        return {};
    }
    const SafetensorsReader& shard = _shards[it->second];
    const SafetensorsTensor* t = shard.find(name);
    if (t == nullptr) {
        return {};
    }
    return shard.tensorBytes(*t);
}

} // namespace mimirmind::core::safetensors