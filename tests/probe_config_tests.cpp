// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Unit tests for the per-HW profile keys and lookup (roadmap 5.19.1).
//
// The GB10 HW fingerprint drifted twice on unchanged hardware (CUDA 13.3
// update, then a kernel update) and prod silently lost its Layer-2 flags.
// These tests pin the contract that fixes it: the hw-class key ignores the
// kernel release and the device UUID, and loadProbePicks() accepts a profile
// directory keyed by either the exact fingerprint or the hw-class key.

#include "TestFramework.hpp"

#include "core/backend/HwFingerprint.hpp"
#include "runtime/ProbeConfig.hpp"

#include <filesystem>
#include <fstream>
#include <string>

#include <unistd.h>

namespace {

namespace be = ::mimirmind::core::backend;
namespace fs = std::filesystem;
using ::mimirmind::runtime::findProfilesForDevice;
using ::mimirmind::runtime::loadModelOverlay;
using ::mimirmind::runtime::loadProbePicks;

be::HwHostInfo gb10Host() {
    be::HwHostInfo h;
    h.cpuModel = "";
    h.threads  = 20;
    h.ramGiB   = 122;
    h.kernel   = "7.0.0-1019-nvidia";
    return h;
}

be::HwProbeIdentity gb10Device() {
    be::HwProbeIdentity id;
    id.haveBackend     = true;
    id.backendKind     = "Cuda";
    id.deviceName      = "NVIDIA GB10";
    id.uuid            = "GPU-0000-aaaa";
    id.pci             = "0x10de:0x2e12";
    id.numComputeUnits = 48;
    id.features        = {false, true, true, true, true};
    return id;
}

fs::path makeConfigRoot(const char* tag) {
    const fs::path root = fs::temp_directory_path() /
                          (std::string{"mm_probe_cfg_"} + tag + "_" +
                           std::to_string(::getpid()));
    fs::remove_all(root);
    fs::create_directories(root / "hw");
    return root;
}

void writeProfile(const fs::path& root, const std::string& dirKey,
                  const std::string& fingerprint, const std::string& hwClass) {
    const fs::path d = root / "hw" / dirKey;
    fs::create_directories(d / "model-overlays");
    std::ofstream{d / "probe-result.json"}
        << R"({"fingerprint":")" << fingerprint << R"(","hw_class":")" << hwClass
        << R"(","model_id":"gb10","probe_status":"curated-partial",)"
        << R"("device":{"device_name":"NVIDIA GB10"},)"
        << R"("provenance":{"kernel":"7.0.0-1019-nvidia"},)"
        << R"("flags":{"MIMIRMIND_MMQ":{"value":1,"apply":true}}})";
    std::ofstream{d / "model-overlays" / "kolibri1.json"}
        << R"({"flags":{"MIMIRMIND_GROUPED_MOE":{"value":3,"apply":true}}})";
}

} // namespace

TEST(class_key_ignores_kernel_and_uuid) {
    auto host = gb10Host();
    auto dev  = gb10Device();
    const std::string fp0  = be::computeHwFingerprint(host, dev);
    const std::string cls0 = be::computeHwClassKey(host, dev);

    host.kernel = "7.0.0-1025-nvidia";
    dev.uuid    = "GPU-1111-bbbb";
    EXPECT_TRUE(be::computeHwFingerprint(host, dev) != fp0);   // the old drift
    EXPECT_EQ(be::computeHwClassKey(host, dev), cls0);         // the fix
}

TEST(class_key_tolerates_ram_jitter_but_tracks_device) {
    auto host = gb10Host();
    auto dev  = gb10Device();
    const std::string cls0 = be::computeHwClassKey(host, dev);

    host.ramGiB = 121;   // kernel-reserved memory moved MemTotal
    EXPECT_EQ(be::computeHwClassKey(host, dev), cls0);

    host = gb10Host();
    dev.deviceName = "NVIDIA GH200";
    EXPECT_TRUE(be::computeHwClassKey(host, dev) != cls0);
}

TEST(class_key_disjoint_from_fingerprint) {
    EXPECT_TRUE(be::computeHwClassKey(gb10Host(), gb10Device()) !=
                be::computeHwFingerprint(gb10Host(), gb10Device()));
}

TEST(lookup_by_hw_class_dir) {
    const fs::path root = makeConfigRoot("cls");
    writeProfile(root, "c1a55c1a55c1a55c", "a68b756bff8519e8", "c1a55c1a55c1a55c");

    const auto picks = loadProbePicks(root.string(), "c1a55c1a55c1a55c");
    EXPECT_TRUE(picks.has_value());
    EXPECT_EQ(picks->fingerprint, std::string{"a68b756bff8519e8"});
    EXPECT_EQ(picks->probedKernel, std::string{"7.0.0-1019-nvidia"});
    EXPECT_TRUE(picks->applyMmq.has_value() && *picks->applyMmq);

    const auto over = loadModelOverlay(root.string(), "c1a55c1a55c1a55c", "kolibri1");
    EXPECT_TRUE(over.has_value() && over->applyGroupedMoe.value_or(0) == 3);
    fs::remove_all(root);
}

TEST(lookup_rejects_key_mismatch) {
    const fs::path root = makeConfigRoot("mis");
    // Directory name matches neither field inside the artefact -> refused.
    writeProfile(root, "deadbeefdeadbeef", "a68b756bff8519e8", "c1a55c1a55c1a55c");
    EXPECT_TRUE(!loadProbePicks(root.string(), "deadbeefdeadbeef").has_value());
    fs::remove_all(root);
}

TEST(find_profiles_for_device) {
    const fs::path root = makeConfigRoot("dev");
    writeProfile(root, "a68b756bff8519e8", "a68b756bff8519e8", "");
    writeProfile(root, "78f59405170233c2", "78f59405170233c2", "");
    const auto hits = findProfilesForDevice(root.string(), "NVIDIA GB10");
    EXPECT_EQ(hits.size(), std::size_t{2});
    EXPECT_EQ(hits.front(), std::string{"78f59405170233c2"});
    EXPECT_TRUE(findProfilesForDevice(root.string(), "NVIDIA GH200").empty());
    EXPECT_TRUE(findProfilesForDevice((root / "nope").string(), "NVIDIA GB10").empty());
    fs::remove_all(root);
}

int main() { return mm::test::run(); }
