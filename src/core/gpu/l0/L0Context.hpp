// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/backend/ComputeBackend.hpp"

#include <level_zero/ze_api.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace mimirmind::core::l0 {

/**
 * Level-Zero-native device snapshot. Superset of
 * `gpu::BackendDeviceInfo` — carries L0-typed fields
 * (`ze_device_type_t`, sub-device counts, max-hardware-contexts) that
 * only the L0 backend cares about. Consumers that only need the
 * portable subset should use `ComputeBackend::deviceInfo()`.
 */
struct DeviceInfo {
    std::string  name;
    std::string  uuid;
    std::uint32_t deviceId{0};
    std::uint32_t vendorId{0};
    ze_device_type_t type{ZE_DEVICE_TYPE_GPU};
    std::uint32_t numSubDevices{0};
    std::uint32_t numComputeUnits{0};
    std::size_t  totalLocalMem{0};
    std::size_t  maxMemAllocSize{0};
    std::uint32_t maxHardwareContexts{0};
    std::uint32_t coreClockRate{0};
    bool         isIntegrated{false};
};

class L0Error : public std::runtime_error {
public:
    L0Error(const std::string& call, ze_result_t code);
    [[nodiscard]] ze_result_t code() const noexcept { return _code; }

private:
    ze_result_t _code;
};

/**
 * RAII wrapper around a Level Zero driver + context, scoped to the first
 * available GPU device. Multi-driver / multi-GPU support is deliberately
 * out of scope for M1 — Meteor Lake exposes a single iGPU and that is what
 * MimirMind targets first.
 *
 * Implements `core::gpu::ComputeBackend` — the backend-neutral interface.
 * Consumers that only need device-info + feature-flags should take
 * `ComputeBackend&`; consumers that touch raw L0 handles (CommandQueue,
 * GpuModule, UsmAllocator, GpuOps, GpuMatmul) stay on `L0Context&`
 * because those classes ARE the L0 backend. See
 * [[MimirMind — HW-Abstraktions-Strategie für Multi-Backend-Support]]
 * for the Schicht-1..6 plan.
 */
class L0Context : public ::mimirmind::core::backend::ComputeBackend {
public:
    /// `spvDirOverride`, if non-empty, is where GpuModule looks for `.spv`
    /// files before falling back to the install / build-tree defaults. Comes
    /// from `runtime.spvDir` in config.json.
    explicit L0Context(std::string spvDirOverride = {});
    ~L0Context() override;

    L0Context(const L0Context&)            = delete;
    L0Context& operator=(const L0Context&) = delete;
    L0Context(L0Context&&)                 = delete;
    L0Context& operator=(L0Context&&)      = delete;

    // ---- L0-native accessors (backend-specific consumers only) ---------

    [[nodiscard]] ze_driver_handle_t  driver()    const noexcept { return _driver; }
    [[nodiscard]] ze_device_handle_t  device()    const noexcept { return _device; }
    [[nodiscard]] ze_context_handle_t context()   const noexcept { return _context; }

    [[nodiscard]] const DeviceInfo&         info()      const noexcept { return _info; }
    [[nodiscard]] const std::vector<DeviceInfo>& allDevices() const noexcept { return _allDevices; }

    [[nodiscard]] std::string_view spvDirOverride() const noexcept { return _spvDirOverride; }

    /// True when the driver advertises `ZE_experimental_mutable_command_list`.
    /// Preflight signal for the Command-List-Replay milestone (M-CLR).
    /// Equivalent to `hasFeature(BackendFeature::MutableCommandLists)`;
    /// kept as a named accessor for readability at existing callsites.
    [[nodiscard]] bool hasMutableCommandLists() const noexcept {
        return _hasMutableCmdLists;
    }

    // ---- ComputeBackend interface (backend-agnostic consumers) -------------

    [[nodiscard]] ::mimirmind::core::backend::BackendKind
        kind() const noexcept override {
        return ::mimirmind::core::backend::BackendKind::LevelZero;
    }

    [[nodiscard]] const ::mimirmind::core::backend::BackendDeviceInfo&
        deviceInfo() const noexcept override { return _neutralInfo; }

    [[nodiscard]] bool hasFeature(
        ::mimirmind::core::backend::BackendFeature f) const noexcept override;

    [[nodiscard]] static std::string typeToString(ze_device_type_t t);
    [[nodiscard]] static std::string resultToString(ze_result_t r);

private:
    void _enumerate();
    void _createContext();
    void _probeDriverExtensions();

    ze_driver_handle_t  _driver{nullptr};
    ze_device_handle_t  _device{nullptr};
    ze_context_handle_t _context{nullptr};

    DeviceInfo              _info{};
    std::vector<DeviceInfo> _allDevices{};

    // Backend-neutral view populated at ctor time from _info. Kept as a
    // member (rather than derived on-demand) so `deviceInfo()` returning
    // a const reference is safe without lifetime shenanigans.
    ::mimirmind::core::backend::BackendDeviceInfo _neutralInfo{};

    std::string _spvDirOverride{};
    bool        _hasMutableCmdLists{false};
};

} // namespace mimirmind::core::l0