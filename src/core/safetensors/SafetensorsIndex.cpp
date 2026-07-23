// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/safetensors/SafetensorsIndex.hpp"

#include <nlohmann/json.hpp>

#include <set>
#include <stdexcept>

namespace mimirmind::core::safetensors {

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("safetensors index: " + msg);
}

} // namespace

std::vector<std::string> SafetensorsIndex::shardFiles() const {
    std::set<std::string> uniq;
    for (const auto& [name, shard] : weightMap) {
        uniq.insert(shard);
    }
    return {uniq.begin(), uniq.end()};
}

SafetensorsIndex parseSafetensorsIndex(std::string_view jsonText) {
    nlohmann::json j =
        nlohmann::json::parse(jsonText.begin(), jsonText.end(), nullptr,
                              /*allow_exceptions=*/false);
    if (j.is_discarded()) {
        fail("not valid JSON");
    }
    if (!j.is_object()) {
        fail("top level is not an object");
    }

    if (!j.contains("weight_map")) {
        fail("missing 'weight_map'");
    }
    const auto& wm = j.at("weight_map");
    if (!wm.is_object()) {
        fail("'weight_map' is not an object");
    }

    SafetensorsIndex out;
    for (const auto& [name, shard] : wm.items()) {
        if (!shard.is_string()) {
            fail("weight_map['" + name + "'] is not a string");
        }
        out.weightMap.emplace(name, shard.get<std::string>());
    }
    if (out.weightMap.empty()) {
        fail("'weight_map' is empty");
    }

    if (j.contains("metadata") && j.at("metadata").is_object()) {
        const auto& meta = j.at("metadata");
        if (meta.contains("total_size")) {
            const auto& ts = meta.at("total_size");
            if (!ts.is_number_unsigned()) {
                fail("metadata.total_size is not an unsigned integer");
            }
            out.totalSize = ts.get<std::uint64_t>();
        }
    }

    return out;
}

} // namespace mimirmind::core::safetensors