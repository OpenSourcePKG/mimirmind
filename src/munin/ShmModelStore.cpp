// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "munin/ShmModelStore.hpp"

#include "core/gguf/GgufReader.hpp"
#include "core/gguf/TensorFingerprint.hpp"
#include "core/log/Log.hpp"

#include <cstdint>
#include <stdexcept>

namespace mimirmind::munin {

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
        MM_LOG_INFO("munin",
                    "ShmModelStore: loading model id='{}' path='{}'",
                    m.id, m.path);

        auto lm    = std::make_unique<ShmLoadedModel>();
        lm->id     = m.id;
        lm->path   = m.path;
        lm->chunks = std::make_unique<ShmChunkAllocator>(chunkBytes);
        lm->reader = std::make_unique<::mimirmind::core::gguf::GgufReader>();
        lm->reader->open(m.path);
        lm->reader->loadTensorsIntoShmChunks(*lm->chunks);
        lm->totalBytes  = lm->reader->totalTensorBytes();
        lm->fingerprint = ::mimirmind::core::gguf::tensorFingerprint(*lm->reader);

        MM_LOG_INFO("munin",
                    "ShmModelStore: loaded id='{}' tensors={} bytes={} "
                    "chunks={} fingerprint='{}'",
                    lm->id, lm->reader->tensorCount(), lm->totalBytes,
                    lm->chunks->chunkCount(), lm->fingerprint);

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
        s.tensorCount = static_cast<std::uint32_t>(m->reader->tensorCount());
        out.push_back(std::move(s));
    }
    return out;
}

} // namespace mimirmind::munin
