#include "runtime/SystemMonitor.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <stdexcept>
#include <string>
#include <sys/types.h>

namespace mimirmind::runtime {

namespace {

constexpr std::string_view kThermalDir = "/sys/class/thermal";
constexpr std::string_view kHwmonDir   = "/sys/class/hwmon";
constexpr std::string_view kMeminfo    = "/proc/meminfo";

/// Read the first line of a file, trimming the trailing newline. Returns
/// empty string on any I/O error — callers should treat empty as "not
/// available" rather than throwing, since transient sysfs hiccups would
/// otherwise abort generation.
std::string readFirstLine(const std::string& path) {
    std::ifstream in{path};
    if (!in.is_open()) {
        return {};
    }
    std::string line;
    if (!std::getline(in, line)) {
        return {};
    }
    while (!line.empty() &&
           (line.back() == '\n' || line.back() == '\r' || line.back() == ' ')) {
        line.pop_back();
    }
    return line;
}

/// Enumerate child entries of `dir` whose name starts with `prefix`.
/// Returns full paths (dir + "/" + entry). Empty result on any error.
std::vector<std::string>
listDirWithPrefix(std::string_view dir, std::string_view prefix) {
    std::vector<std::string> out;
    DIR* d = ::opendir(std::string{dir}.c_str());
    if (d == nullptr) {
        return out;
    }
    while (auto* e = ::readdir(d)) {
        const std::string_view name{e->d_name};
        if (name.size() < prefix.size()) {
            continue;
        }
        if (name.substr(0, prefix.size()) != prefix) {
            continue;
        }
        std::string full{dir};
        full.push_back('/');
        full.append(name);
        out.push_back(std::move(full));
    }
    ::closedir(d);
    return out;
}

/// True iff `s` starts with `prefix`. std::string_view::starts_with is
/// C++20 but feature-gated on libstdc++; this avoids the version check.
bool startsWith(std::string_view s, std::string_view prefix) noexcept {
    return s.size() >= prefix.size() &&
           s.substr(0, prefix.size()) == prefix;
}

} // namespace

SystemMonitor::SystemMonitor(bool                      requirePackageTemp,
                             bool                      requireRam,
                             std::chrono::milliseconds minRefreshInterval)
    : _minRefresh{minRefreshInterval} {
    probePackageTempSource();
    if (requirePackageTemp && _packageTempPath.empty()) {
        throw std::runtime_error(
            "SystemMonitor: package temperature sensor required but no "
            "x86_pkg_temp thermal zone or coretemp Package-id hwmon entry "
            "was found in /sys");
    }
    if (requireRam) {
        std::ifstream in{std::string{kMeminfo}};
        if (!in.is_open()) {
            throw std::runtime_error(
                "SystemMonitor: /proc/meminfo not readable — RAM monitoring "
                "required by profile but the container has no access");
        }
    }
}

void SystemMonitor::probePackageTempSource() {
    // Tier 1: thermal_zone with type == "x86_pkg_temp" or "pkg_temp".
    for (const auto& zone : listDirWithPrefix(kThermalDir, "thermal_zone")) {
        const std::string typePath = zone + "/type";
        const std::string type     = readFirstLine(typePath);
        if (type == "x86_pkg_temp" || type == "pkg_temp") {
            _packageTempPath   = zone + "/temp";
            _packageTempSource = "thermal_zone:" + type;
            return;
        }
    }

    // Tier 2: hwmon entry whose name is "coretemp" with a "Package id N" label.
    for (const auto& hwmon : listDirWithPrefix(kHwmonDir, "hwmon")) {
        const std::string name = readFirstLine(hwmon + "/name");
        if (name != "coretemp") {
            continue;
        }
        for (const auto& label : listDirWithPrefix(hwmon, "temp")) {
            // Pick out the "tempN_label" files.
            if (!startsWith(label.substr(hwmon.size() + 1), "temp")) {
                continue;
            }
            if (label.size() < 6 ||
                label.substr(label.size() - 6) != "_label") {
                continue;
            }
            const std::string labelValue = readFirstLine(label);
            if (!startsWith(labelValue, "Package id")) {
                continue;
            }
            // /sys/.../hwmonN/tempK_label  ->  /sys/.../hwmonN/tempK_input
            _packageTempPath = label.substr(0, label.size() - 6) + "_input";
            _packageTempSource = "hwmon:coretemp:" + labelValue;
            return;
        }
    }

    _packageTempPath.clear();
    _packageTempSource = "(none detected)";
}

std::optional<float> SystemMonitor::readPackageTempC() const {
    if (_packageTempPath.empty()) {
        return std::nullopt;
    }
    const std::string raw = readFirstLine(_packageTempPath);
    if (raw.empty()) {
        return std::nullopt;
    }
    char* end = nullptr;
    const long milliC = std::strtol(raw.c_str(), &end, 10);
    if (end == raw.c_str()) {
        return std::nullopt;
    }
    return static_cast<float>(milliC) / 1000.0F;
}

std::pair<std::optional<std::size_t>, std::optional<std::size_t>>
SystemMonitor::readRam() const {
    std::ifstream in{std::string{kMeminfo}};
    if (!in.is_open()) {
        return {std::nullopt, std::nullopt};
    }
    std::optional<std::size_t> total;
    std::optional<std::size_t> available;
    std::string                line;
    while (std::getline(in, line)) {
        const auto colon = line.find(':');
        if (colon == std::string::npos) {
            continue;
        }
        const std::string key = line.substr(0, colon);
        if (key != "MemTotal" && key != "MemAvailable") {
            continue;
        }
        // The remainder is "    <kB number> kB".
        const char* p   = line.c_str() + colon + 1;
        char*       end = nullptr;
        const long  kib = std::strtol(p, &end, 10);
        if (end == p || kib < 0) {
            continue;
        }
        const auto mib = static_cast<std::size_t>(kib) / 1024U;
        if (key == "MemTotal") {
            total = mib;
        } else {
            available = mib;
        }
        if (total.has_value() && available.has_value()) {
            break;
        }
    }
    return {total, available};
}

SystemReading SystemMonitor::read() {
    const auto now = std::chrono::steady_clock::now();
    if (_lastReadAt.time_since_epoch().count() != 0 &&
        (now - _lastReadAt) < _minRefresh) {
        return _cached;
    }

    SystemReading r;
    r.taken_at       = now;
    r.package_temp_c = readPackageTempC();

    auto [total, available] = readRam();
    r.ram_total_mib     = total;
    r.ram_available_mib = available;

    _cached     = r;
    _lastReadAt = now;
    return r;
}

} // namespace mimirmind::runtime