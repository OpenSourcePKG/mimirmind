// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/safetensors/CheckpointFingerprint.hpp"

#include "core/safetensors/SafetensorsIndex.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <system_error>
#include <vector>

namespace mimirmind::core::safetensors {

namespace {

namespace fs = std::filesystem;

std::string fnv1aHex(std::string_view s) {
    std::uint64_t h = 1469598103934665603ULL;
    for (const char ch : s) {
        const auto c = static_cast<unsigned char>(ch);
        h ^= c;
        h *= 1099511628211ULL;
    }
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(h));
    return std::string{buf};
}

std::uint64_t sizeOf(const fs::path& p) {
    std::error_code ec;
    const auto sz = fs::file_size(p, ec);
    if (ec) {
        throw std::runtime_error("checkpointFingerprint: cannot stat shard '" +
                                 p.string() + "': " + ec.message());
    }
    return sz;
}

} // namespace

std::string shardSetFingerprint(std::span<const ShardSize> shards,
                                std::uint64_t              declaredTotalSize) {
    std::string acc;
    for (const auto& s : shards) {
        acc += s.name;
        acc += ':';
        acc += std::to_string(s.bytes);
        acc += ';';
    }
    acc += "total:";
    acc += std::to_string(declaredTotalSize);
    return "nvfp4:" + fnv1aHex(acc);
}

std::string checkpointFingerprint(std::string_view path) {
    const fs::path p{std::string{path}};
    std::error_code ec;
    std::vector<ShardSize> shards;
    std::uint64_t          total = 0;

    if (fs::is_directory(p, ec)) {
        const fs::path idx    = p / "model.safetensors.index.json";
        const fs::path single = p / "model.safetensors";
        if (fs::is_regular_file(idx, ec)) {
            std::ifstream in(idx, std::ios::binary);
            if (!in) {
                throw std::runtime_error("checkpointFingerprint: cannot read '" +
                                         idx.string() + "'");
            }
            const std::string text((std::istreambuf_iterator<char>(in)),
                                   std::istreambuf_iterator<char>());
            const SafetensorsIndex index = parseSafetensorsIndex(text);
            total = index.totalSize;
            for (const auto& name : index.shardFiles()) {
                shards.push_back({name, sizeOf(p / name)});
            }
        } else if (fs::is_regular_file(single, ec)) {
            shards.push_back({"model.safetensors", sizeOf(single)});
        } else {
            throw std::runtime_error("checkpointFingerprint: directory '" + p.string() +
                                     "' has neither model.safetensors.index.json "
                                     "nor model.safetensors");
        }
    } else if (fs::is_regular_file(p, ec)) {
        shards.push_back({p.filename().string(), sizeOf(p)});
    } else {
        throw std::runtime_error("checkpointFingerprint: '" + p.string() +
                                 "' is not a file or directory");
    }
    if (shards.empty()) {
        throw std::runtime_error("checkpointFingerprint: '" + p.string() +
                                 "' resolved to no shards");
    }
    return shardSetFingerprint(shards, total);
}

} // namespace mimirmind::core::safetensors
