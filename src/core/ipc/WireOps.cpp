#include "core/ipc/WireOps.hpp"

#include <nlohmann/json.hpp>

#include <sstream>

namespace mimirmind::core::ipc {

std::expected<RequestEnvelope, std::string>
RequestEnvelope::fromJson(std::string_view j) {
    nlohmann::json parsed;
    try {
        parsed = nlohmann::json::parse(j);
    } catch (const nlohmann::json::parse_error& e) {
        std::ostringstream os;
        os << "RequestEnvelope: JSON parse error: " << e.what();
        return std::unexpected(os.str());
    }
    if (!parsed.is_object()) {
        return std::unexpected(std::string{
            "RequestEnvelope: root is not an object"});
    }
    if (!parsed.contains("op") || !parsed["op"].is_string()) {
        return std::unexpected(std::string{
            "RequestEnvelope: missing required string field 'op'"});
    }

    RequestEnvelope r{};
    r.op = parsed["op"].get<std::string>();

    // modelId is required for attach, optional for healthz. The dispatcher
    // enforces per-op semantics; here we just extract if present so the
    // caller does not need to re-parse.
    if (parsed.contains("modelId")) {
        if (!parsed["modelId"].is_string()) {
            return std::unexpected(std::string{
                "RequestEnvelope: 'modelId' must be a string when present"});
        }
        r.modelId = parsed["modelId"].get<std::string>();
    }
    return r;
}

std::string HealthzResponse::toJson() const {
    nlohmann::json j;
    j["protocol_version"] = protocolVersion;
    j["status"]           = status;
    j["governor_owner"]   = governorOwner;
    j["pid"]              = pid;

    auto arr = nlohmann::json::array();
    arr.get_ref<nlohmann::json::array_t&>().reserve(models.size());
    for (const auto& m : models) {
        nlohmann::json e;
        e["id"]           = m.id;
        e["fingerprint"]  = m.fingerprint;
        e["total_bytes"]  = m.totalBytes;
        e["tensor_count"] = m.tensorCount;
        arr.push_back(std::move(e));
    }
    j["models"] = std::move(arr);
    return j.dump();
}

std::expected<HealthzResponse, std::string>
HealthzResponse::fromJson(std::string_view j) {
    nlohmann::json parsed;
    try {
        parsed = nlohmann::json::parse(j);
    } catch (const nlohmann::json::parse_error& e) {
        std::ostringstream os;
        os << "HealthzResponse: JSON parse error: " << e.what();
        return std::unexpected(os.str());
    }
    if (!parsed.is_object()) {
        return std::unexpected(std::string{
            "HealthzResponse: root is not an object"});
    }

    HealthzResponse r{};
    try {
        r.protocolVersion = parsed.at("protocol_version").get<std::uint32_t>();
        r.status          = parsed.at("status").get<std::string>();
        r.governorOwner   = parsed.at("governor_owner").get<std::string>();
        r.pid             = parsed.at("pid").get<std::uint32_t>();

        const auto& arr = parsed.at("models");
        if (!arr.is_array()) {
            return std::unexpected(std::string{
                "HealthzResponse: 'models' must be an array"});
        }
        r.models.reserve(arr.size());
        for (const auto& e : arr) {
            ModelSummaryWire s{};
            s.id          = e.at("id").get<std::string>();
            s.fingerprint = e.at("fingerprint").get<std::string>();
            s.totalBytes  = e.at("total_bytes").get<std::uint64_t>();
            s.tensorCount = e.at("tensor_count").get<std::uint32_t>();
            r.models.push_back(std::move(s));
        }
    } catch (const nlohmann::json::exception& x) {
        std::ostringstream os;
        os << "HealthzResponse: field extraction failed: " << x.what();
        return std::unexpected(os.str());
    }
    return r;
}

std::string makeErrorJson(std::string_view message) {
    nlohmann::json j;
    j["error"] = std::string{message};
    return j.dump();
}

std::expected<std::string, std::string>
parseErrorJson(std::string_view j) {
    nlohmann::json parsed;
    try {
        parsed = nlohmann::json::parse(j);
    } catch (const nlohmann::json::parse_error& e) {
        std::ostringstream os;
        os << "parseErrorJson: JSON parse error: " << e.what();
        return std::unexpected(os.str());
    }
    if (!parsed.is_object() || !parsed.contains("error")
        || !parsed["error"].is_string()) {
        return std::unexpected(std::string{
            "parseErrorJson: no string 'error' field in envelope"});
    }
    return parsed["error"].get<std::string>();
}

} // namespace mimirmind::core::ipc