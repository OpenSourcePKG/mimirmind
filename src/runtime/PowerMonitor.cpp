#include "runtime/PowerMonitor.hpp"

#include <algorithm>
#include <cstdlib>
#include <dirent.h>
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>

namespace mimirmind::runtime {

namespace {

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

bool tryParseU64(const std::string& s, std::uint64_t& out) {
    if (s.empty()) {
        return false;
    }
    char* end = nullptr;
    const auto v = std::strtoull(s.c_str(), &end, 10);
    if (end == s.c_str()) {
        return false;
    }
    out = static_cast<std::uint64_t>(v);
    return true;
}

bool startsWith(std::string_view s, std::string_view prefix) noexcept {
    return s.size() >= prefix.size() &&
           s.substr(0, prefix.size()) == prefix;
}

/// All direct children of `dir` whose basename starts with `prefix`.
/// Returned paths include the parent dir + "/" + basename.
std::vector<std::string>
listDirWithPrefix(std::string_view dir, std::string_view prefix) {
    std::vector<std::string> out;
    DIR* d = ::opendir(std::string{dir}.c_str());
    if (d == nullptr) {
        return out;
    }
    while (auto* e = ::readdir(d)) {
        const std::string_view name{e->d_name};
        if (!startsWith(name, prefix)) {
            continue;
        }
        std::string full{dir};
        full.push_back('/');
        full.append(name);
        out.push_back(std::move(full));
    }
    ::closedir(d);
    std::sort(out.begin(), out.end());
    return out;
}

/// True iff `path` resolves to a real file/dir we can stat. Catches the
/// Docker case where /sys/class/powercap/intel-rapl:0 is a dangling
/// symlink into masked space.
bool pathStatable(const std::string& path) {
    struct stat st{};
    return ::stat(path.c_str(), &st) == 0;
}

} // namespace

PowerMonitor::PowerMonitor(std::string_view sysfsRoot) {
    probe(sysfsRoot);
}

void PowerMonitor::probe(std::string_view sysfsRoot) {
    // intel-rapl:0 is the package socket. Children intel-rapl:0:N are
    // sub-domains (core, uncore, dram). intel-rapl:1 — when present —
    // is psys (whole-platform). Sysfs flat-view exposes both top-level
    // and sub-domain entries as siblings, so we discriminate by basename
    // colon-count to avoid double-counting.
    const auto entries = listDirWithPrefix(sysfsRoot, "intel-rapl");
    if (entries.empty()) {
        _unavailableReason =
            "no intel-rapl directories found under sysfs root — RAPL not "
            "supported on this CPU";
        return;
    }

    auto add = [&](const std::string& dir, std::string fallbackName) {
        const std::string energyPath = dir + "/energy_uj";
        const std::string namePath   = dir + "/name";
        const std::string maxPath    = dir + "/max_energy_range_uj";

        // Reading the energy file is the load-bearing operation; if that
        // fails (Docker masking, permission, parse) we treat the whole
        // domain as absent.
        const std::string energyStr = readFirstLine(energyPath);
        std::uint64_t     energy    = 0;
        if (energyStr.empty() || !tryParseU64(energyStr, energy)) {
            return;
        }
        std::uint64_t maxRange = 0;
        (void)tryParseU64(readFirstLine(maxPath), maxRange);

        Domain d;
        d.name         = readFirstLine(namePath);
        if (d.name.empty()) {
            d.name = std::move(fallbackName);
        }
        d.energyUjPath = energyPath;
        d.maxRangeUj   = maxRange;
        _domains.push_back(std::move(d));
    };

    auto basenameColons = [](std::string_view dir) {
        const auto slash = dir.rfind('/');
        const auto base  = slash == std::string_view::npos
                              ? dir
                              : dir.substr(slash + 1);
        return std::count(base.begin(), base.end(), ':');
    };

    // Tier 1: top-level domains (basename == "intel-rapl:N" → exactly 1 colon).
    for (const auto& dir : entries) {
        if (!pathStatable(dir)) {
            continue;
        }
        if (basenameColons(dir) != 1) {
            continue;
        }
        add(dir, "package");
    }

    // Tier 2: sub-domains (basename == "intel-rapl:N:M" → exactly 2 colons).
    for (const auto& dir : entries) {
        if (!pathStatable(dir)) {
            continue;
        }
        if (basenameColons(dir) != 2) {
            continue;
        }
        add(dir, "sub");
    }

    if (_domains.empty()) {
        _unavailableReason =
            "intel-rapl directories found but energy_uj is not readable — "
            "Docker likely masks /sys/devices/virtual/powercap; set "
            "security_opt: systempaths=unconfined on the runtime service";
        return;
    }

    _available = true;
    _domainNames.reserve(_domains.size());
    for (const auto& d : _domains) {
        _domainNames.push_back(d.name);
    }
}

PowerMonitor::Snapshot PowerMonitor::snapshot() const {
    Snapshot s;
    s.taken_at = std::chrono::steady_clock::now();
    if (!_available) {
        return s;
    }
    s.raw_energy_uj.reserve(_domains.size());
    for (const auto& d : _domains) {
        std::uint64_t e = 0;
        (void)tryParseU64(readFirstLine(d.energyUjPath), e);
        s.raw_energy_uj.push_back(e);
    }
    return s;
}

std::vector<double>
PowerMonitor::energyBetween(const Snapshot& start, const Snapshot& end) const {
    if (!_available) {
        return {};
    }
    if (start.raw_energy_uj.size() != _domains.size() ||
        end.raw_energy_uj.size()   != _domains.size()) {
        return {};
    }
    std::vector<double> out;
    out.reserve(_domains.size());
    for (std::size_t i = 0; i < _domains.size(); ++i) {
        const auto a = start.raw_energy_uj[i];
        const auto b = end.raw_energy_uj[i];
        std::uint64_t delta_uj = 0;
        if (b >= a) {
            delta_uj = b - a;
        } else if (_domains[i].maxRangeUj > 0) {
            // Counter wrapped once.
            delta_uj = (_domains[i].maxRangeUj - a) + b;
        }
        out.push_back(static_cast<double>(delta_uj) / 1'000'000.0);
    }
    return out;
}

std::vector<double>
PowerMonitor::averageWattsBetween(const Snapshot& start,
                                  const Snapshot& end) const {
    const auto joules = energyBetween(start, end);
    if (joules.empty()) {
        return {};
    }
    const double dt = std::chrono::duration<double>(
                          end.taken_at - start.taken_at).count();
    if (dt <= 0.0) {
        return std::vector<double>(joules.size(), 0.0);
    }
    std::vector<double> out;
    out.reserve(joules.size());
    for (double j : joules) {
        out.push_back(j / dt);
    }
    return out;
}

} // namespace mimirmind::runtime