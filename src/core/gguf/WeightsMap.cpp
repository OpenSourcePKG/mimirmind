// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/gguf/WeightsMap.hpp"

#include "core/ipc/TensorManifest.hpp"
#include "core/log/Log.hpp"

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>

namespace mimirmind::core::gguf {

WeightsMap::WeightsMap(const GgufReader& reader) {
    const auto& tensors = reader.tensors();
    _byName.reserve(tensors.size() * 2);
    for (const auto& t : tensors) {
        _byName.emplace(t.name, &t);
    }
    MM_LOG_INFO("weights", "indexed {} tensors for O(1) lookup", _byName.size());
}

WeightsMap::WeightsMap(std::vector<GgufTensor> attachedTensors)
    : _owned{std::move(attachedTensors)} {
    _byName.reserve(_owned.size() * 2);
    for (const auto& t : _owned) {
        _byName.emplace(t.name, &t);
    }
    MM_LOG_INFO("weights",
                "indexed {} attached tensors for O(1) lookup "
                "(attached-mode; usmPtrs point at Munin-owned USM)",
                _byName.size());
}

WeightsMap WeightsMap::fromAttachedChunked(
    const ::mimirmind::core::ipc::TensorManifest& manifest,
    std::span<void* const>                        chunkBases) {
    std::vector<GgufTensor> owned;
    owned.reserve(manifest.tensors.size());

    for (std::size_t i = 0; i < manifest.tensors.size(); ++i) {
        const auto& me = manifest.tensors[i];
        if (me.chunkIndex >= chunkBases.size()) {
            std::ostringstream os;
            os << "WeightsMap::fromAttachedChunked: tensor[" << i << "] '"
               << me.name << "' references chunkIndex=" << me.chunkIndex
               << " but only " << chunkBases.size()
               << " chunk base(s) available";
            throw std::runtime_error(os.str());
        }
        void* base = chunkBases[me.chunkIndex];
        if (base == nullptr) {
            std::ostringstream os;
            os << "WeightsMap::fromAttachedChunked: tensor[" << i << "] '"
               << me.name << "' references chunkIndex=" << me.chunkIndex
               << " whose base pointer is null";
            throw std::runtime_error(os.str());
        }

        GgufTensor t{};
        t.name        = me.name;
        t.type        = me.type;
        t.dimensions  = me.dims;
        // Manifest ships bytes; nelements is derived downstream data
        // some codepaths still read. Product-of-dims is the invariant
        // used everywhere in the standalone path too.
        std::uint64_t nel = 1;
        for (auto d : t.dimensions) nel *= d;
        t.nelements   = nel;
        t.nbytes      = static_cast<std::size_t>(me.bytes);
        t.fileOffset  = 0;
        t.chunkIndex  = me.chunkIndex;
        t.chunkOffset = me.chunkOffset;
        t.usmPtr      = static_cast<std::byte*>(base) + me.chunkOffset;
        owned.push_back(std::move(t));
    }

    MM_LOG_INFO("weights",
                "materialised {} tensor(s) from {} chunk base(s) — "
                "attached-chunked mode (M-Munin.1a)",
                owned.size(), chunkBases.size());

    return WeightsMap{std::move(owned)};
}

const GgufTensor* WeightsMap::find(std::string_view name) const noexcept {
    const auto it = _byName.find(std::string{name});
    return it == _byName.end() ? nullptr : it->second;
}

const GgufTensor& WeightsMap::require(std::string_view name) const {
    if (const auto* t = find(name)) {
        return *t;
    }
    MM_LOG_ERROR("weights", "required tensor '{}' missing", name);
    throw std::runtime_error("WeightsMap::require: tensor '" +
                             std::string{name} + "' not in model");
}

const GgufTensor* WeightsMap::findBlock(std::size_t blockIdx,
                                        std::string_view suffix) const {
    std::string key = "blk.";
    key += std::to_string(blockIdx);
    key += '.';
    key.append(suffix);
    return find(key);
}

} // namespace mimirmind::core::gguf