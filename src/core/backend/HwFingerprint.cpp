// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/backend/HwFingerprint.hpp"

#include "core/backend/BackendRegistry.hpp"
#include "core/backend/ComputeBackend.hpp"

#include <sys/utsname.h>

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string_view>
#include <thread>

namespace mimirmind::core::backend {

namespace {

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

std::string cpuModel() {
    std::ifstream f{"/proc/cpuinfo"};
    std::string line;
    while (std::getline(f, line)) {
        const auto colon = line.find(':');
        if (colon != std::string::npos && line.compare(0, 10, "model name") == 0) {
            std::string v = line.substr(colon + 1);
            const auto b = v.find_first_not_of(" \t");
            const auto e = v.find_last_not_of(" \t");
            return (b == std::string::npos) ? std::string{} : v.substr(b, e - b + 1);
        }
    }
    return {};
}

std::uint64_t ramGiB() {
    std::ifstream f{"/proc/meminfo"};
    std::string key;
    std::uint64_t kb = 0;
    if (f >> key >> kb && key == "MemTotal:") {
        return (kb + (512ULL * 1024ULL)) / (1024ULL * 1024ULL);  // kB -> GiB rounded
    }
    return 0;
}

std::string kernelRelease() {
    struct utsname u{};
    return (::uname(&u) == 0) ? std::string{u.release} : std::string{};
}

} // namespace

HwProbeIdentity identityFromBackend(const ComputeBackend& backend) {
    const BackendDeviceInfo& info = backend.deviceInfo();
    HwProbeIdentity id;
    id.haveBackend     = true;
    id.backendKind     = BackendRegistry::name(backend.kind());
    id.deviceName      = info.name;
    id.uuid            = info.uuid;
    char pci[16];
    std::snprintf(pci, sizeof(pci), "0x%04x:0x%04x", info.vendorId, info.deviceId);
    id.pci             = pci;
    id.numComputeUnits = info.numComputeUnits;
    const BackendFeature feats[5] = {
        BackendFeature::MutableCommandLists, BackendFeature::IntegerDotProduct,
        BackendFeature::IpcHandleExport, BackendFeature::UnifiedMemoryHost,
        BackendFeature::MatrixEngine};
    for (std::size_t i = 0; i < 5; ++i) id.features[i] = backend.hasFeature(feats[i]);
    return id;
}

HwHostInfo gatherHostInfo() {
    HwHostInfo h;
    h.cpuModel = cpuModel();
    const unsigned hc = std::thread::hardware_concurrency();
    h.threads = hc ? hc : 0U;
    h.ramGiB = ramGiB();
    h.kernel = kernelRelease();
    return h;
}

std::string computeHwFingerprint(const HwHostInfo& host, const HwProbeIdentity& id) {
    // Canonical string — MUST stay byte-identical between the probe tool and
    // the runtime, so any change here changes every fingerprint. Order and
    // separators are load-bearing.
    std::ostringstream fp;
    fp << "host|" << host.cpuModel << '|' << host.threads << '|' << host.ramGiB
       << '|' << host.kernel;
    if (id.haveBackend) {
        fp << "|dev|" << id.backendKind << '|' << id.deviceName << '|' << id.uuid
           << '|' << id.pci << '|' << id.numComputeUnits;
        for (bool f : id.features) fp << (f ? '1' : '0');
    } else {
        fp << "|dev|none";
    }
    return toHex16(fnv1a64(fp.str()));
}

} // namespace mimirmind::core::backend
