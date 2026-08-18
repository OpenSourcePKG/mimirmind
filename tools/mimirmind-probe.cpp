// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// mimirmind-probe — offline probe-driven config generation (M-Probe.1).
//
// This is increment 1: the HW-fingerprint generator + the config-artefact
// contract. It enumerates the active compute backend + host, derives a
// stable hardware fingerprint, and writes
//
//     configs/hw/{fingerprint}/probe-result.json
//
// The runtime reads that artefact *lookup-only* (match on fingerprint, fall
// back to defaults on a miss) — no runtime autotune (that is a CLR landmine;
// see the M-Probe.1 design note). The backend x kernel-variant x shape sweep,
// the CPU-gold correctness gate, and the median/IQR margin guards
// (M-Probe.1.B-F) land in later increments; until then the artefact carries
// an empty `ops` table and `probe_status: "fingerprint-only"`, which the
// runtime loader treats as "use defaults".
//
// Runs standalone — no HTTP server, no model load. When the compute backend
// cannot be constructed (e.g. no /dev/dri), it degrades to a host-only
// fingerprint rather than failing, so the artefact contract still holds.
//
// Build:  cmake --build build --target mimirmind_probe
// Run:    ./build/mimirmind_probe [--print] [--out <dir>] [--model <id>]

#include "core/backend/BackendRegistry.hpp"
#include "core/backend/ComputeBackend.hpp"
#include "core/backend/ComputeContext.hpp"

#include <nlohmann/json.hpp>

#include <sys/utsname.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

using json = nlohmann::json;
namespace be = mimirmind::core::backend;

// FNV-1a 64-bit — the fingerprint is a stable cache key, not a security
// digest, so a fast dependency-free hash is the right tool (no OpenSSL link).
std::uint64_t fnv1a64(std::string_view s) {
    std::uint64_t h = 1469598103934665603ULL;
    for (char ch : s) {
        h ^= static_cast<unsigned char>(ch);
        h *= 1099511628211ULL;
    }
    return h;
}

std::string toHex16(std::uint64_t v) {
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(v));
    return std::string{buf};
}

// First "model name" line from /proc/cpuinfo (Linux). Empty on failure.
std::string cpuModel() {
    std::ifstream f{"/proc/cpuinfo"};
    std::string line;
    while (std::getline(f, line)) {
        const auto colon = line.find(':');
        if (colon != std::string::npos &&
            line.compare(0, 10, "model name") == 0) {
            std::string v = line.substr(colon + 1);
            const auto b = v.find_first_not_of(" \t");
            const auto e = v.find_last_not_of(" \t");
            return (b == std::string::npos) ? std::string{} : v.substr(b, e - b + 1);
        }
    }
    return {};
}

unsigned cpuThreads() {
    const unsigned hc = std::thread::hardware_concurrency();
    return hc ? hc : 0U;
}

// MemTotal from /proc/meminfo, rounded to whole GiB (a bucket — small RAM
// jitter across boots must not change the fingerprint).
std::uint64_t ramGiB() {
    std::ifstream f{"/proc/meminfo"};
    std::string key;
    std::uint64_t kb = 0;
    if (f >> key >> kb) {
        if (key == "MemTotal:") {
            return (kb + (512ULL * 1024ULL)) / (1024ULL * 1024ULL);  // kB -> GiB, rounded
        }
    }
    return 0;
}

std::string kernelRelease() {
    struct utsname u{};
    if (::uname(&u) == 0) {
        return std::string{u.release};
    }
    return {};
}

std::string iso8601UtcNow() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    struct tm tmv{};
    ::gmtime_r(&t, &tmv);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tmv);
    return std::string{buf};
}

const char* deviceKindName(be::DeviceKind k) {
    switch (k) {
        case be::DeviceKind::GpuIntegrated: return "GpuIntegrated";
        case be::DeviceKind::GpuDiscrete:   return "GpuDiscrete";
        case be::DeviceKind::Npu:           return "Npu";
        case be::DeviceKind::Cpu:           return "Cpu";
        case be::DeviceKind::Unknown:       return "Unknown";
    }
    return "Unknown";
}

// The five neutral feature flags the fingerprint + artefact care about.
struct FeatureBit { be::BackendFeature f; const char* name; };
const FeatureBit kFeatures[] = {
    {be::BackendFeature::MutableCommandLists, "MutableCommandLists"},
    {be::BackendFeature::IntegerDotProduct,   "IntegerDotProduct"},
    {be::BackendFeature::IpcHandleExport,     "IpcHandleExport"},
    {be::BackendFeature::UnifiedMemoryHost,   "UnifiedMemoryHost"},
    {be::BackendFeature::MatrixEngine,        "MatrixEngine"},
};

struct Options {
    bool        printOnly = false;
    std::string outRoot   = "configs";
    std::string modelId   = "unspecified";
};

void printUsage() {
    std::puts(
        "mimirmind-probe — HW fingerprint + probe config artefact (M-Probe.1)\n"
        "\n"
        "Usage: mimirmind_probe [--print] [--out <dir>] [--model <id>]\n"
        "\n"
        "  --print, -p     write the artefact to stdout instead of a file\n"
        "  --out <dir>     config root (default: configs) ->\n"
        "                  <dir>/hw/{fingerprint}/probe-result.json\n"
        "  --model <id>    tag the artefact with this model id\n"
        "  --help, -h      this help\n");
}

bool parseArgs(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        const std::string_view a{argv[i]};
        if (a == "--help" || a == "-h") { printUsage(); return false; }
        else if (a == "--print" || a == "-p") { o.printOnly = true; }
        else if (a == "--out" && i + 1 < argc) { o.outRoot = argv[++i]; }
        else if (a == "--model" && i + 1 < argc) { o.modelId = argv[++i]; }
        else {
            std::fprintf(stderr, "unknown arg: %.*s\n",
                         static_cast<int>(a.size()), a.data());
            printUsage();
            return false;
        }
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    Options opt;
    if (!parseArgs(argc, argv, opt)) {
        // --help returns success; a bad arg is caught above with usage. Both
        // paths here are "stop, not an error worth a non-zero code".
        return 0;
    }

    // --- Host block (always available) -------------------------------------
    const std::string hostCpu     = cpuModel();
    const unsigned    hostThreads = cpuThreads();
    const std::uint64_t hostRamGiB = ramGiB();
    const std::string hostKernel  = kernelRelease();

    json hw;
    hw["host"] = {
        {"cpu_model", hostCpu},
        {"cpu_threads", hostThreads},
        {"ram_gib", hostRamGiB},
        {"kernel", hostKernel},
    };

    // Canonical fingerprint string — only STABLE identity fields. Timing,
    // timestamps and mutable counters never enter it.
    std::ostringstream fp;
    fp << "host|" << hostCpu << '|' << hostThreads << '|' << hostRamGiB
       << '|' << hostKernel;

    // --- Available-backends probe (no construction) ------------------------
    json avail = json::array();
    for (const auto& p : be::BackendRegistry::probeAll()) {
        avail.push_back({
            {"kind", be::BackendRegistry::name(p.kind)},
            {"compiled", p.compiledIn},
            {"available", p.available},
            {"detail", p.detail},
        });
    }
    hw["available_backends"] = std::move(avail);

    // --- Active backend block (construct + observe, degrade on failure) -----
    const be::BackendKind kind = be::BackendRegistry::autoSelect();
    json backendJson;
    try {
        std::unique_ptr<be::ComputeContext> ctx =
            be::BackendRegistry::createContext(kind);
        const be::ComputeBackend&    b    = ctx->backend();
        const be::BackendDeviceInfo& info = b.deviceInfo();

        char pci[16];
        std::snprintf(pci, sizeof(pci), "0x%04x:0x%04x", info.vendorId, info.deviceId);

        json features;
        for (const auto& fb : kFeatures) {
            features[fb.name] = b.hasFeature(fb.f);
        }

        backendJson = {
            {"kind", be::BackendRegistry::name(b.kind())},
            {"device_name", info.name},
            {"uuid", info.uuid},
            {"pci", pci},
            {"device_kind", deviceKindName(info.kind)},
            {"compute_units", info.numComputeUnits},
            {"total_local_mem_gib",
             static_cast<double>(info.totalLocalMem) / (1024.0 * 1024.0 * 1024.0)},
            {"max_alloc_gib",
             static_cast<double>(info.maxMemAllocSize) / (1024.0 * 1024.0 * 1024.0)},
            {"core_clock_mhz", info.coreClockRate},
            {"bandwidth_gbps", ctx->bandwidthGBps()},
            {"features", features},
        };

        // Fingerprint from stable device identity — NOT clock/mem which can
        // jitter with governor state, and NOT bandwidth (a heuristic).
        fp << "|dev|" << be::BackendRegistry::name(b.kind()) << '|' << info.name
           << '|' << info.uuid << '|' << pci << '|' << info.numComputeUnits;
        for (const auto& fb : kFeatures) {
            fp << (b.hasFeature(fb.f) ? '1' : '0');
        }
    } catch (const std::exception& e) {
        // No device (e.g. no /dev/dri): host-only fingerprint, backend noted.
        backendJson = {
            {"kind", "none"},
            {"error", e.what()},
        };
        fp << "|dev|none";
        std::fprintf(stderr,
                     "[warn] backend '%s' could not be constructed (%s) — "
                     "emitting a host-only fingerprint\n",
                     be::BackendRegistry::name(kind), e.what());
    }
    hw["backend"] = std::move(backendJson);

    const std::string fingerprint = toHex16(fnv1a64(fp.str()));

    // --- Artefact ----------------------------------------------------------
    json artefact = {
        {"schema", "mimirmind-probe-result/v1"},
        {"generated_at", iso8601UtcNow()},
        {"fingerprint", fingerprint},
        {"model_id", opt.modelId},
        // fingerprint-only until the sweep increments (M-Probe.1.B-F) fill
        // `ops`; the runtime loader treats this status as "use defaults".
        {"probe_status", "fingerprint-only"},
        {"hardware", std::move(hw)},
        {"ops", json::array()},
    };

    const std::string dump = artefact.dump(2);

    if (opt.printOnly) {
        std::puts(dump.c_str());
        return 0;
    }

    namespace fs = std::filesystem;
    const fs::path dir = fs::path{opt.outRoot} / "hw" / fingerprint;
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) {
        std::fprintf(stderr, "[error] cannot create %s: %s\n",
                     dir.c_str(), ec.message().c_str());
        return 1;
    }
    const fs::path out = dir / "probe-result.json";
    std::ofstream os{out, std::ios::trunc};
    if (!os) {
        std::fprintf(stderr, "[error] cannot write %s\n", out.c_str());
        return 1;
    }
    os << dump << '\n';
    os.close();

    std::printf("mimirmind-probe: fingerprint=%s  backend=%s  ->  %s\n",
                fingerprint.c_str(),
                be::BackendRegistry::name(kind),
                out.c_str());
    return 0;
}
