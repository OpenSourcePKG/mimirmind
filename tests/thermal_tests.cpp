// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Unit tests for the GpuClockGovernor adaptive up-gain (M9.6.6.2).
//
// The gain math and the overshoot watchdog are validated WITHOUT real
// hardware: the pure ramp function is tested directly, and adjustForTemp
// is driven against a synthetic /sys/class/drm tree (plain files the
// governor reads/writes exactly like the i915 sysfs nodes). This is the
// only pre-deploy validation for the one milestone with real thermal-
// shutdown risk (see Synaipse research/roadmap-governor-adaptive-gains).

#include "TestFramework.hpp"

#include "runtime/thermal/GpuClockGovernor.hpp"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

namespace {

using ::mimirmind::runtime::GpuClockGovernor;

// RP0 (max) and RPn (min) for the synthetic card. Match a typical Xe-LPG
// (Meteor Lake) so the clamp bounds are realistic.
constexpr std::uint32_t kRp0 = 2350;
constexpr std::uint32_t kRpn = 800;

void writeFile(const std::filesystem::path& p, std::uint32_t v) {
    std::ofstream f{p};
    f << v << "\n";
}

// Build a fresh synthetic /sys/class/drm root with a single writable
// card0 iGPU node. Returns the root to pass as sysfsRoot. Unique per call
// so tests don't alias each other's cap files.
std::string makeSysfsTree(std::uint32_t initialMax, unsigned tag) {
    const auto root = std::filesystem::temp_directory_path()
                    / ("mm_gov_" + std::to_string(tag));
    const auto card = root / "card0";
    std::filesystem::create_directories(card);
    writeFile(card / "gt_RP0_freq_mhz", kRp0);
    writeFile(card / "gt_RPn_freq_mhz", kRpn);
    writeFile(card / "gt_max_freq_mhz", initialMax);
    return root.string();
}

} // namespace

// ---------------------------------------------------------------------------
// Pure ramp — adaptiveUpGainMhzPerC(headroom) = clamp(headroom*5, 10, 100)
// ---------------------------------------------------------------------------

TEST(gov_adaptive_gain_floor_at_baseline) {
    // At/near target the gain never drops below the paranoid baseline (10).
    EXPECT_NEAR(GpuClockGovernor::adaptiveUpGainMhzPerC(0.0F),  10.0F, 1e-4F);
    EXPECT_NEAR(GpuClockGovernor::adaptiveUpGainMhzPerC(1.0F),  10.0F, 1e-4F); // 5 -> floor
    EXPECT_NEAR(GpuClockGovernor::adaptiveUpGainMhzPerC(-5.0F), 10.0F, 1e-4F); // neg -> 0 -> floor
}

TEST(gov_adaptive_gain_scales_with_headroom) {
    EXPECT_NEAR(GpuClockGovernor::adaptiveUpGainMhzPerC(4.0F),  20.0F, 1e-4F);
    EXPECT_NEAR(GpuClockGovernor::adaptiveUpGainMhzPerC(10.0F), 50.0F, 1e-4F);
    EXPECT_NEAR(GpuClockGovernor::adaptiveUpGainMhzPerC(18.0F), 90.0F, 1e-4F);
}

TEST(gov_adaptive_gain_saturates_at_down_gain) {
    // Saturates at 100 (== |down-gain|) by ~20 C headroom and stays there.
    EXPECT_NEAR(GpuClockGovernor::adaptiveUpGainMhzPerC(20.0F), 100.0F, 1e-4F);
    EXPECT_NEAR(GpuClockGovernor::adaptiveUpGainMhzPerC(40.0F), 100.0F, 1e-4F);
}

// ---------------------------------------------------------------------------
// adjustForTemp — adaptive ramp vs flat baseline (synthetic sysfs)
// ---------------------------------------------------------------------------

TEST(gov_cold_chip_ramps_faster_than_flat) {
    const auto root = makeSysfsTree(/*initialMax=*/kRpn, /*tag=*/1);
    GpuClockGovernor gov{root};
    EXPECT_TRUE(gov.available());
    gov.setTargetTempC(72.0F);
    gov.setMaxFreqMhz(kRpn);                 // start at RPn (800)

    // 4 C below target -> headroom 4 -> adaptive gain 20 -> +80 MHz.
    // Flat baseline would be gain 10 -> +40. Assert the adaptive value.
    const std::uint32_t cap = gov.adjustForTemp(68.0F);
    EXPECT_EQ(cap, std::uint32_t{880});
    EXPECT_TRUE(gov.adaptiveUp());
}

TEST(gov_deep_headroom_reaches_rp0_in_one_tick) {
    const auto root = makeSysfsTree(kRpn, 2);
    GpuClockGovernor gov{root};
    gov.setTargetTempC(72.0F);
    gov.setMaxFreqMhz(kRpn);

    // 24 C headroom -> gain saturates at 100 -> +2400 target, clamped RP0.
    const std::uint32_t cap = gov.adjustForTemp(48.0F);
    EXPECT_EQ(cap, kRp0);
}

TEST(gov_kill_switch_forces_flat_gain) {
    ::setenv("MIMIRMIND_GOVERNOR_ADAPTIVE_UP", "off", /*overwrite=*/1);
    const auto root = makeSysfsTree(kRpn, 3);
    GpuClockGovernor gov{root};             // reads env at construction
    ::unsetenv("MIMIRMIND_GOVERNOR_ADAPTIVE_UP");

    gov.setTargetTempC(72.0F);
    gov.setMaxFreqMhz(kRpn);
    EXPECT_TRUE(!gov.adaptiveUp());

    // 4 C headroom with adaptive OFF -> flat baseline gain 10 -> +40.
    const std::uint32_t cap = gov.adjustForTemp(68.0F);
    EXPECT_EQ(cap, std::uint32_t{840});
}

TEST(gov_down_gain_is_always_hard) {
    const auto root = makeSysfsTree(/*initialMax=*/1500, 4);
    GpuClockGovernor gov{root};
    gov.setTargetTempC(72.0F);
    gov.setMaxFreqMhz(1500);

    // 2 C over target -> hard down-gain 100 -> -200 MHz, regardless of adaptive.
    const std::uint32_t cap = gov.adjustForTemp(74.0F);
    EXPECT_EQ(cap, std::uint32_t{1300});
}

// ---------------------------------------------------------------------------
// Overshoot watchdog — latch on overshoot, suppress adaptive until cooled
// ---------------------------------------------------------------------------

TEST(gov_overshoot_latches_and_suppresses_adaptive) {
    const auto root = makeSysfsTree(1500, 5);
    GpuClockGovernor gov{root};
    gov.setTargetTempC(72.0F);
    gov.setMaxFreqMhz(1500);

    // Overshoot 2 C (> latch 1 C) -> latch + hard drop to 1300.
    std::uint32_t cap = gov.adjustForTemp(74.0F);
    EXPECT_EQ(cap, std::uint32_t{1300});
    EXPECT_TRUE(gov.overshootLatched());

    // Cool to 2.5 C headroom: NOT past release (3 C) -> still latched ->
    // up-gain is flat baseline 10 (+25), NOT adaptive 12.5 (+31).
    cap = gov.adjustForTemp(69.5F);
    EXPECT_TRUE(gov.overshootLatched());
    EXPECT_EQ(cap, std::uint32_t{1325});
}

TEST(gov_overshoot_releases_below_hysteresis) {
    const auto root = makeSysfsTree(1300, 6);
    GpuClockGovernor gov{root};
    gov.setTargetTempC(72.0F);
    gov.setMaxFreqMhz(1300);

    gov.adjustForTemp(74.0F);                 // latch
    EXPECT_TRUE(gov.overshootLatched());

    // Cool to 4 C headroom (> release 3 C) -> release + adaptive gain 20
    // -> +80 from the post-drop cap (1300 -> 1100 on the latch tick? no:
    // start cap 1300, latch tick drops 200 -> 1100; then +80 -> 1180).
    const std::uint32_t cap = gov.adjustForTemp(68.0F);
    EXPECT_TRUE(!gov.overshootLatched());
    EXPECT_EQ(cap, std::uint32_t{1180});
}

int main() {
    return mm::test::run();
}
