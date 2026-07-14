// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/ipc/TensorManifest.hpp"

#include <nlohmann/json.hpp>

#include <sstream>

namespace mimirmind::core::ipc {

using ::mimirmind::core::gguf::GgmlType;

std::string TensorManifest::toJson() const {
    nlohmann::json j;
    j["protocol_version"]  = protocolVersion;
    j["model_id"]          = modelId;
    j["model_fingerprint"] = modelFingerprint;

    auto chunkArr = nlohmann::json::array();
    chunkArr.get_ref<nlohmann::json::array_t&>().reserve(chunks.size());
    for (const auto& c : chunks) {
        nlohmann::json e;
        e["chunk_index"] = c.chunkIndex;
        e["bytes"]       = c.bytes;
        chunkArr.push_back(std::move(e));
    }
    j["chunks"] = std::move(chunkArr);

    auto arr = nlohmann::json::array();
    arr.get_ref<nlohmann::json::array_t&>().reserve(tensors.size());
    for (const auto& t : tensors) {
        nlohmann::json e;
        e["name"]         = t.name;
        e["type_id"]      = static_cast<std::uint32_t>(t.type);
        e["dims"]         = t.dims;
        e["bytes"]        = t.bytes;
        e["chunk_index"]  = t.chunkIndex;
        e["chunk_offset"] = t.chunkOffset;
        arr.push_back(std::move(e));
    }
    j["tensors"] = std::move(arr);

    // Compact serialisation — no indentation, saves ~50% wire bytes on
    // a 720-tensor manifest. Deterministic key order isn't required
    // (the receiver walks the same keys explicitly).
    return j.dump();
}

std::expected<TensorManifest, std::string>
TensorManifest::fromJson(std::string_view s) {
    nlohmann::json j;
    try {
        j = nlohmann::json::parse(s);
    } catch (const nlohmann::json::parse_error& e) {
        std::ostringstream os;
        os << "TensorManifest: JSON parse error: " << e.what();
        return std::unexpected(os.str());
    }
    if (!j.is_object()) {
        return std::unexpected(std::string{"TensorManifest: root is not an object"});
    }

    auto readField = [&](const char* key) -> std::expected<const nlohmann::json*, std::string> {
        if (!j.contains(key)) {
            std::ostringstream os;
            os << "TensorManifest: missing required field '" << key << "'";
            return std::unexpected(os.str());
        }
        return &j[key];
    };

    TensorManifest m{};

    // protocol_version — reject drift up front.
    {
        auto v = readField("protocol_version");
        if (!v) return std::unexpected(v.error());
        if (!(*v)->is_number_unsigned()) {
            return std::unexpected(std::string{
                "TensorManifest: protocol_version must be an unsigned integer"});
        }
        m.protocolVersion = (*v)->get<std::uint32_t>();
        if (m.protocolVersion != kCurrentProtocolVersion) {
            std::ostringstream os;
            os << "TensorManifest: protocol_version=" << m.protocolVersion
               << " does not match receiver's " << kCurrentProtocolVersion
               << " — refusing to attach (rebuild Munin + mimirmind together)";
            return std::unexpected(os.str());
        }
    }
    {
        auto v = readField("model_id");
        if (!v) return std::unexpected(v.error());
        if (!(*v)->is_string()) {
            return std::unexpected(std::string{"TensorManifest: model_id must be a string"});
        }
        m.modelId = (*v)->get<std::string>();
    }
    {
        auto v = readField("model_fingerprint");
        if (!v) return std::unexpected(v.error());
        if (!(*v)->is_string()) {
            return std::unexpected(std::string{
                "TensorManifest: model_fingerprint must be a string"});
        }
        m.modelFingerprint = (*v)->get<std::string>();
    }
    {
        auto v = readField("chunks");
        if (!v) return std::unexpected(v.error());
        if (!(*v)->is_array()) {
            return std::unexpected(std::string{"TensorManifest: chunks must be an array"});
        }
        const auto& arr = **v;
        m.chunks.reserve(arr.size());
        for (std::size_t i = 0; i < arr.size(); ++i) {
            const auto& e = arr[i];
            if (!e.is_object()) {
                std::ostringstream os;
                os << "TensorManifest: chunks[" << i << "] is not an object";
                return std::unexpected(os.str());
            }
            ChunkDesc cd{};
            try {
                cd.chunkIndex = e.at("chunk_index").get<std::uint32_t>();
                cd.bytes      = e.at("bytes").get<std::uint64_t>();
            } catch (const nlohmann::json::exception& x) {
                std::ostringstream os;
                os << "TensorManifest: chunks[" << i
                   << "]: field extraction failed: " << x.what();
                return std::unexpected(os.str());
            }
            m.chunks.push_back(cd);
        }
    }
    {
        auto v = readField("tensors");
        if (!v) return std::unexpected(v.error());
        if (!(*v)->is_array()) {
            return std::unexpected(std::string{"TensorManifest: tensors must be an array"});
        }
        const auto& arr = **v;
        m.tensors.reserve(arr.size());
        for (std::size_t i = 0; i < arr.size(); ++i) {
            const auto& e = arr[i];
            if (!e.is_object()) {
                std::ostringstream os;
                os << "TensorManifest: tensors[" << i << "] is not an object";
                return std::unexpected(os.str());
            }
            ManifestEntry me{};
            try {
                me.name        = e.at("name").get<std::string>();
                me.type        = static_cast<GgmlType>(e.at("type_id").get<std::uint32_t>());
                me.dims        = e.at("dims").get<std::vector<std::uint64_t>>();
                me.bytes       = e.at("bytes").get<std::uint64_t>();
                me.chunkIndex  = e.at("chunk_index").get<std::uint32_t>();
                me.chunkOffset = e.at("chunk_offset").get<std::uint64_t>();
            } catch (const nlohmann::json::exception& x) {
                std::ostringstream os;
                os << "TensorManifest: tensors[" << i
                   << "]: field extraction failed: " << x.what();
                return std::unexpected(os.str());
            }
            // Sanity: referenced chunk must exist. Cheap consistency check
            // at parse-time saves the worker from a later out-of-range
            // index-into-empty-chunks-vector crash.
            if (me.chunkIndex >= m.chunks.size()) {
                std::ostringstream os;
                os << "TensorManifest: tensors[" << i << "] '" << me.name
                   << "' references chunk_index=" << me.chunkIndex
                   << " but manifest only declares " << m.chunks.size()
                   << " chunk(s)";
                return std::unexpected(os.str());
            }
            m.tensors.push_back(std::move(me));
        }
    }
    return m;
}

} // namespace mimirmind::core::ipc