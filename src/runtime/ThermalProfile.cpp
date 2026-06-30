#include "runtime/ThermalProfile.hpp"

#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>

namespace mimirmind::runtime {

namespace {

[[noreturn]] void fail(std::string_view path, std::string_view msg) {
    std::ostringstream os;
    os << "ThermalProfile '" << path << "': " << msg;
    throw std::runtime_error(os.str());
}

template <typename T>
std::optional<T> readOptional(const nlohmann::json& j, const char* key) {
    if (!j.contains(key) || j[key].is_null()) {
        return std::nullopt;
    }
    return j[key].get<T>();
}

void validatePair(std::string_view             path,
                  std::string_view             prefix,
                  bool                         soft_present,
                  bool                         hard_present) {
    if (soft_present != hard_present) {
        std::ostringstream os;
        os << prefix << ": both soft and hard thresholds must be set together "
           << "(found only one — either remove both to disable monitoring "
           << "or add the missing field)";
        fail(path, os.str());
    }
}

} // namespace

ThermalProfile loadThermalProfile(std::string_view path) {
    std::ifstream in{std::string{path}};
    if (!in.is_open()) {
        fail(path, "cannot open file");
    }

    nlohmann::json j;
    try {
        in >> j;
    } catch (const std::exception& e) {
        fail(path, std::string{"invalid JSON: "} + e.what());
    }
    if (!j.is_object()) {
        fail(path, "top-level JSON must be an object");
    }

    ThermalProfile p{};

    if (!j.contains("name") || !j["name"].is_string()) {
        fail(path, "missing required field 'name' (string)");
    }
    p.name = j["name"].get<std::string>();

    if (j.contains("description") && j["description"].is_string()) {
        p.description = j["description"].get<std::string>();
    }

    // ---- Package temperature ------------------------------------------------
    p.package_temp_soft_c = readOptional<float>(j, "package_temp_soft_c");
    p.package_temp_hard_c = readOptional<float>(j, "package_temp_hard_c");
    validatePair(path, "package_temp",
                 p.package_temp_soft_c.has_value(),
                 p.package_temp_hard_c.has_value());

    if (p.hasPackageLimits() && !(*p.package_temp_soft_c < *p.package_temp_hard_c)) {
        std::ostringstream os;
        os << "package_temp_soft_c (" << *p.package_temp_soft_c
           << ") must be strictly less than package_temp_hard_c ("
           << *p.package_temp_hard_c << ")";
        fail(path, os.str());
    }

    if (j.contains("package_throttle_max_ms") &&
        j["package_throttle_max_ms"].is_number_integer()) {
        const auto v = j["package_throttle_max_ms"].get<int>();
        if (v < 0) {
            fail(path, "package_throttle_max_ms must be >= 0");
        }
        p.package_throttle_max_ms = v;
    }

    p.gpu_target_temp_c = readOptional<float>(j, "gpu_target_temp_c");
    if (p.gpu_target_temp_c.has_value() && p.hasPackageLimits()) {
        // Sanity: target should sit below soft so the per-token pause
        // is only a safety net behind the governor.
        if (*p.gpu_target_temp_c >= *p.package_temp_soft_c) {
            std::ostringstream os;
            os << "gpu_target_temp_c (" << *p.gpu_target_temp_c
               << ") must be strictly less than package_temp_soft_c ("
               << *p.package_temp_soft_c
               << ") — the governor is meant to keep the chip below "
                  "the soft threshold so per-token pacing only kicks in "
                  "if the governor cannot keep up";
            fail(path, os.str());
        }
    }

    return p;
}

} // namespace mimirmind::runtime