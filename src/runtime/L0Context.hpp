#pragma once

#include <level_zero/ze_api.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace mimirmind::runtime {

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
 * Mimirmind targets first.
 */
class L0Context {
public:
    L0Context();
    ~L0Context();

    L0Context(const L0Context&)            = delete;
    L0Context& operator=(const L0Context&) = delete;
    L0Context(L0Context&&)                 = delete;
    L0Context& operator=(L0Context&&)      = delete;

    [[nodiscard]] ze_driver_handle_t  driver()    const noexcept { return _driver; }
    [[nodiscard]] ze_device_handle_t  device()    const noexcept { return _device; }
    [[nodiscard]] ze_context_handle_t context()   const noexcept { return _context; }

    [[nodiscard]] const DeviceInfo&         info()      const noexcept { return _info; }
    [[nodiscard]] const std::vector<DeviceInfo>& allDevices() const noexcept { return _allDevices; }

    /// True when the driver advertises `ZE_experimental_mutable_command_list`.
    /// Preflight signal for the Command-List-Replay milestone (M-CLR).
    [[nodiscard]] bool hasMutableCommandLists() const noexcept {
        return _hasMutableCmdLists;
    }

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

    bool _hasMutableCmdLists{false};
};

} // namespace mimirmind::runtime