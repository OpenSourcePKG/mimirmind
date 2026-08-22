// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "munin/ShmModelStore.hpp"

#include "core/gguf/GgufReader.hpp"
#include "core/gguf/TensorFingerprint.hpp"
#include "core/log/Log.hpp"
#include "core/os/MappedFile.hpp"
#include "core/safetensors/SafetensorsIndex.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace mimirmind::munin {

namespace fs = std::filesystem;

namespace {

using ::mimirmind::core::config::ModelEntry;
using ::mimirmind::core::config::ModelFormat;

/// FNV-1a of `s`, as 16 lowercase hex chars. Used to fingerprint an NVFP4
/// checkpoint from its shard {name, size} set + declared total — cheap and
/// reproducible by the attaching worker from the same local files.
std::string fnv1aHex(std::string_view s) {
    std::uint64_t h = 1469598103934665603ULL;
    for (const unsigned char c : s) {
        h ^= c;
        h *= 1099511628211ULL;
    }
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(h));
    return std::string{buf};
}

/// Decide whether an entry is an NVFP4 checkpoint. Explicit format wins;
/// Auto probes the filesystem (a directory is a safetensors checkpoint, a
/// file is GGUF) — the same rule Config documents for ModelFormat::Auto.
bool wantNvfp4(const ModelEntry& m) {
    if (m.format == ModelFormat::Nvfp4) return true;
    if (m.format == ModelFormat::Gguf)  return false;
    std::error_code ec;
    return fs::is_directory(fs::path{m.path}, ec);
}

struct ShardRef {
    std::string name;
    fs::path    full;
};
struct NvfpLayout {
    std::vector<ShardRef> shards;
    std::uint64_t         declaredTotalSize{0};
};

/// Resolve the shard file set for an NVFP4 checkpoint (dir with index.json /
/// model.safetensors, or a single *.safetensors file).
NvfpLayout resolveNvfp4(const std::string& path) {
    NvfpLayout out;
    const fs::path p{path};
    std::error_code ec;

    if (fs::is_directory(p, ec)) {
        const fs::path idx    = p / "model.safetensors.index.json";
        const fs::path single = p / "model.safetensors";
        if (fs::is_regular_file(idx, ec)) {
            ::mimirmind::core::l0::MappedFile mf(idx.string());  // throws if unreadable
            const auto b = mf.bytes();
            const auto index = ::mimirmind::core::safetensors::parseSafetensorsIndex(
                std::string_view{reinterpret_cast<const char*>(b.data()), b.size()});
            out.declaredTotalSize = index.totalSize;
            for (const auto& name : index.shardFiles()) {
                out.shards.push_back({name, p / name});
            }
        } else if (fs::is_regular_file(single, ec)) {
            out.shards.push_back({"model.safetensors", single});
        } else {
            throw std::runtime_error{
                "ShmModelStore: NVFP4 directory '" + path +
                "' has neither model.safetensors.index.json nor model.safetensors"};
        }
    } else if (fs::is_regular_file(p, ec)) {
        out.shards.push_back({p.filename().string(), p});
    } else {
        throw std::runtime_error{
            "ShmModelStore: NVFP4 path '" + path + "' is not a file or directory"};
    }

    if (out.shards.empty()) {
        throw std::runtime_error{
            "ShmModelStore: NVFP4 checkpoint '" + path + "' resolved to no shards"};
    }
    return out;
}

std::uint64_t alignUp4k(std::uint64_t v) noexcept {
    constexpr std::uint64_t a = 4096;
    return (v + a - 1) & ~(a - 1);
}

/// Load an NVFP4 checkpoint: copy each shard's raw *.safetensors image into
/// a memfd chunk and record its {chunkIndex, chunkOffset, bytes}. Chunk size
/// is sized to the largest shard so every shard fits (bump never spans).
void loadNvfp4Into(ShmLoadedModel& lm, const std::string& path) {
    const NvfpLayout layout = resolveNvfp4(path);

    std::uint64_t maxShard = 4096;
    for (const auto& s : layout.shards) {
        std::error_code ec;
        const auto sz = fs::file_size(s.full, ec);
        if (ec) {
            throw std::runtime_error{
                "ShmModelStore: cannot stat shard '" + s.full.string() +
                "': " + ec.message()};
        }
        if (sz > maxShard) maxShard = sz;
    }

    const std::size_t chunkBytes = static_cast<std::size_t>(alignUp4k(maxShard));
    lm.chunks = std::make_unique<ShmChunkAllocator>(chunkBytes);

    std::string fpAcc;
    for (const auto& s : layout.shards) {
        ::mimirmind::core::l0::MappedFile mf(s.full.string());  // throws if unreadable
        const auto bytes = mf.bytes();
        const auto a     = lm.chunks->allocate(bytes.size());
        std::memcpy(a.ptr, bytes.data(), bytes.size());

        ::mimirmind::core::ipc::ShardDesc sd{};
        sd.name        = s.name;
        sd.chunkIndex  = a.chunkIndex;
        sd.chunkOffset = a.chunkOffset;
        sd.bytes       = bytes.size();
        lm.shards.push_back(std::move(sd));

        lm.totalBytes += bytes.size();
        fpAcc += s.name;
        fpAcc += ':';
        fpAcc += std::to_string(bytes.size());
        fpAcc += ';';
    }

    lm.isNvfp4           = true;
    lm.declaredTotalSize = layout.declaredTotalSize;
    fpAcc += "total:";
    fpAcc += std::to_string(layout.declaredTotalSize);
    lm.fingerprint = "nvfp4:" + fnv1aHex(fpAcc);
}

/// Load a GGUF model: raw tensor bytes into memfd chunks + tensor fingerprint.
void loadGgufInto(ShmLoadedModel& lm, const std::string& path, std::size_t chunkBytes) {
    lm.chunks = std::make_unique<ShmChunkAllocator>(chunkBytes);
    lm.reader = std::make_unique<::mimirmind::core::gguf::GgufReader>();
    lm.reader->open(path);
    lm.reader->loadTensorsIntoShmChunks(*lm.chunks);
    lm.totalBytes  = lm.reader->totalTensorBytes();
    lm.fingerprint = ::mimirmind::core::gguf::tensorFingerprint(*lm.reader);
}

} // namespace

ShmModelStore::ShmModelStore(const ::mimirmind::core::config::Config& cfg,
                             std::size_t                              chunkBytes) {
    std::size_t loaded = 0;
    for (const auto& m : cfg.models) {
        if (!m.loadOnStart) {
            continue;
        }
        if (m.id.empty() || m.path.empty()) {
            throw std::runtime_error{
                "ShmModelStore: model entry with loadOnStart:true has empty id "
                "or path (id='" + m.id + "', path='" + m.path + "')"};
        }
        if (_byId.contains(m.id)) {
            throw std::runtime_error{
                "ShmModelStore: duplicate model id '" + m.id +
                "' — every loadable entry must have a unique id"};
        }

        const bool nvfp4 = wantNvfp4(m);
        MM_LOG_INFO("munin",
                    "ShmModelStore: loading {} model id='{}' path='{}'",
                    nvfp4 ? "NVFP4" : "GGUF", m.id, m.path);

        auto lm  = std::make_unique<ShmLoadedModel>();
        lm->id   = m.id;
        lm->path = m.path;
        if (nvfp4) {
            loadNvfp4Into(*lm, m.path);
        } else {
            loadGgufInto(*lm, m.path, chunkBytes);
        }

        MM_LOG_INFO("munin",
                    "ShmModelStore: loaded id='{}' bytes={} chunks={} "
                    "{}={} fingerprint='{}'",
                    lm->id, lm->totalBytes, lm->chunks->chunkCount(),
                    nvfp4 ? "shards" : "tensors",
                    nvfp4 ? lm->shards.size()
                          : (lm->reader ? lm->reader->tensorCount() : 0),
                    lm->fingerprint);

        _byId.emplace(m.id, std::move(lm));
        ++loaded;
    }

    if (loaded == 0) {
        throw std::runtime_error{
            "ShmModelStore: no models with loadOnStart:true in config — Munin "
            "has nothing to hold. Set at least one model to loadOnStart:true."};
    }
    MM_LOG_INFO("munin", "ShmModelStore: {} model(s) resident in shm", loaded);
}

ShmModelStore::~ShmModelStore() = default;

const ShmLoadedModel*
ShmModelStore::find(std::string_view modelId) const noexcept {
    const std::string key{modelId};
    const auto it = _byId.find(key);
    if (it == _byId.end()) {
        return nullptr;
    }
    return it->second.get();
}

std::vector<ShmModelStore::ModelSummary> ShmModelStore::summaries() const {
    std::vector<ModelSummary> out;
    out.reserve(_byId.size());
    for (const auto& [id, m] : _byId) {
        ModelSummary s{};
        s.id          = id;
        s.fingerprint = m->fingerprint;
        s.totalBytes  = m->totalBytes;
        s.tensorCount = m->isNvfp4
            ? static_cast<std::uint32_t>(m->shards.size())
            : static_cast<std::uint32_t>(m->reader ? m->reader->tensorCount() : 0);
        out.push_back(std::move(s));
    }
    return out;
}

} // namespace mimirmind::munin
