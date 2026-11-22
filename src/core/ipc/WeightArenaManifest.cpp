// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/ipc/WeightArenaManifest.hpp"

#include <nlohmann/json.hpp>

namespace mimirmind::core::ipc {

std::string WeightArenaManifest::toJson() const {
    nlohmann::json j;
    j["version"]       = version;
    j["modelId"]       = modelId;
    j["architecture"]  = architecture;
    j["checkpointDir"] = checkpointDir;
    j["fingerprint"]   = fingerprint;
    j["mapBytes"]      = mapBytes;
    j["runes"]         = runes;
    j["baseModified"]  = baseModified;
    nlohmann::json ts  = nlohmann::json::array();
    for (const auto& t : tensors) {
        nlohmann::json bj = nlohmann::json::array();
        for (const auto& b : t.buffers) {
            bj.push_back({{"role", b.role}, {"offset", b.offset}, {"bytes", b.bytes}});
        }
        ts.push_back({{"name", t.name},
                      {"dims", t.dims},
                      {"elems", t.elems},
                      {"flags", t.flags},
                      {"buffers", std::move(bj)}});
    }
    j["tensors"] = std::move(ts);
    return j.dump();
}

std::expected<WeightArenaManifest, std::string>
WeightArenaManifest::fromJson(const std::string& text) {
    const nlohmann::json j = nlohmann::json::parse(text, nullptr, false);
    if (!j.is_object()) {
        return std::unexpected("weight arena manifest: not a JSON object");
    }
    try {
        WeightArenaManifest m;
        m.version = j.at("version").get<int>();
        if (m.version != kVersion) {
            return std::unexpected("weight arena manifest: version " +
                                   std::to_string(m.version) + ", expected " +
                                   std::to_string(kVersion));
        }
        m.modelId       = j.at("modelId").get<std::string>();
        m.architecture  = j.at("architecture").get<std::string>();
        m.checkpointDir = j.at("checkpointDir").get<std::string>();
        m.fingerprint   = j.at("fingerprint").get<std::string>();
        m.mapBytes      = j.at("mapBytes").get<std::uint64_t>();
        m.runes         = j.at("runes").get<std::vector<std::string>>();
        m.baseModified  = j.at("baseModified").get<bool>();
        for (const auto& tj : j.at("tensors")) {
            WeightArenaTensor t;
            t.name  = tj.at("name").get<std::string>();
            t.dims  = tj.at("dims").get<std::vector<std::uint64_t>>();
            t.elems = tj.at("elems").get<std::uint64_t>();
            t.flags = tj.at("flags").get<std::vector<std::string>>();
            for (const auto& bj : tj.at("buffers")) {
                WeightArenaBuffer b;
                b.role   = bj.at("role").get<std::string>();
                b.offset = bj.at("offset").get<std::uint64_t>();
                b.bytes  = bj.at("bytes").get<std::uint64_t>();
                if (b.offset + b.bytes < b.offset || b.offset + b.bytes > m.mapBytes) {
                    return std::unexpected("weight arena manifest: tensor '" + t.name +
                                           "' buffer '" + b.role +
                                           "' lies outside the mapped extent");
                }
                t.buffers.push_back(std::move(b));
            }
            m.tensors.push_back(std::move(t));
        }
        return m;
    } catch (const nlohmann::json::exception& e) {
        return std::unexpected(std::string{"weight arena manifest: "} + e.what());
    }
}

} // namespace mimirmind::core::ipc
