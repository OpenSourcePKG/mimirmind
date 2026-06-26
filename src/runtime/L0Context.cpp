#include "runtime/L0Context.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <sstream>

namespace mimirmind::runtime {

namespace {

#define ZE_CHECK(call)                                                  \
    do {                                                                \
        const ze_result_t _r = (call);                                  \
        if (_r != ZE_RESULT_SUCCESS) {                                  \
            throw L0Error(#call, _r);                                   \
        }                                                               \
    } while (false)

std::string formatUuid(const ze_device_uuid_t& uuid) {
    char buf[64];
    std::snprintf(buf, sizeof(buf),
        "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
        uuid.id[0],  uuid.id[1],  uuid.id[2],  uuid.id[3],
        uuid.id[4],  uuid.id[5],
        uuid.id[6],  uuid.id[7],
        uuid.id[8],  uuid.id[9],
        uuid.id[10], uuid.id[11], uuid.id[12], uuid.id[13], uuid.id[14], uuid.id[15]);
    return std::string{buf};
}

DeviceInfo describeDevice(ze_device_handle_t dev) {
    DeviceInfo info{};

    ze_device_properties_t props{};
    props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    ZE_CHECK(zeDeviceGetProperties(dev, &props));

    info.name                = std::string{props.name};
    info.uuid                = formatUuid(props.uuid);
    info.deviceId            = props.deviceId;
    info.vendorId            = props.vendorId;
    info.type                = props.type;
    info.coreClockRate       = props.coreClockRate;
    info.maxHardwareContexts = props.maxHardwareContexts;
    info.maxMemAllocSize     = static_cast<std::size_t>(props.maxMemAllocSize);

    // Total compute lanes — slices × subslices × EUs × threads/EU.
    const std::uint64_t lanes
        = static_cast<std::uint64_t>(props.numSlices)
        * static_cast<std::uint64_t>(props.numSubslicesPerSlice)
        * static_cast<std::uint64_t>(props.numEUsPerSubslice)
        * static_cast<std::uint64_t>(props.numThreadsPerEU);
    info.numComputeUnits = static_cast<std::uint32_t>(lanes);

    // Sub-devices live in a separate API call.
    std::uint32_t subCount = 0;
    if (zeDeviceGetSubDevices(dev, &subCount, nullptr) == ZE_RESULT_SUCCESS) {
        info.numSubDevices = subCount;
    }

    // Memory properties — sum across all memory pools.
    std::uint32_t memCount = 0;
    ZE_CHECK(zeDeviceGetMemoryProperties(dev, &memCount, nullptr));
    if (memCount > 0) {
        std::vector<ze_device_memory_properties_t> mems(memCount);
        for (auto& m : mems) {
            m.stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
        }
        ZE_CHECK(zeDeviceGetMemoryProperties(dev, &memCount, mems.data()));
        for (const auto& m : mems) {
            info.totalLocalMem += static_cast<std::size_t>(m.totalSize);
        }
    }

    // NOTE: props.maxMemAllocSize is the loader's declared per-allocation
    // cap. The real, observed cap may differ — M2 will probe it
    // empirically and store the result on the allocator.

    return info;
}

} // namespace

// --- L0Error -----------------------------------------------------------------

L0Error::L0Error(const std::string& call, ze_result_t code)
    : std::runtime_error{call + " failed: " + L0Context::resultToString(code)
                         + " (0x" + std::to_string(static_cast<unsigned>(code)) + ")"},
      _code{code}
{}

// --- L0Context ---------------------------------------------------------------

L0Context::L0Context() {
    ze_init_flags_t flags = ZE_INIT_FLAG_GPU_ONLY;
    ZE_CHECK(zeInit(flags));

    _enumerate();
    _createContext();
}

L0Context::~L0Context() {
    if (_context != nullptr) {
        zeContextDestroy(_context);
        _context = nullptr;
    }
    // Drivers and devices are global handles, no destroy.
}

void L0Context::_enumerate() {
    std::uint32_t driverCount = 0;
    ZE_CHECK(zeDriverGet(&driverCount, nullptr));
    if (driverCount == 0) {
        throw std::runtime_error{
            "Level Zero reported 0 drivers. Check that intel-level-zero-gpu "
            "is installed and /dev/dri/renderD128 is accessible."};
    }

    std::vector<ze_driver_handle_t> drivers(driverCount);
    ZE_CHECK(zeDriverGet(&driverCount, drivers.data()));

    for (auto* drv : drivers) {
        std::uint32_t deviceCount = 0;
        ZE_CHECK(zeDeviceGet(drv, &deviceCount, nullptr));
        if (deviceCount == 0) {
            continue;
        }
        std::vector<ze_device_handle_t> devs(deviceCount);
        ZE_CHECK(zeDeviceGet(drv, &deviceCount, devs.data()));

        for (auto* dev : devs) {
            DeviceInfo info = describeDevice(dev);
            _allDevices.push_back(info);

            if (_device == nullptr && info.type == ZE_DEVICE_TYPE_GPU) {
                _driver = drv;
                _device = dev;
                _info   = info;
            }
        }
    }

    if (_device == nullptr) {
        throw std::runtime_error{
            "No GPU device found by Level Zero. On Meteor Lake, verify:\n"
            "  - intel-level-zero-gpu package installed\n"
            "  - /dev/dri/renderD128 exists and is readable by current user\n"
            "  - current user is in the 'render' group"};
    }
}

void L0Context::_createContext() {
    ze_context_desc_t desc{};
    desc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
    desc.flags = 0;
    desc.pNext = nullptr;
    ZE_CHECK(zeContextCreate(_driver, &desc, &_context));
}

std::string L0Context::typeToString(ze_device_type_t t) {
    switch (t) {
        case ZE_DEVICE_TYPE_GPU:  return "GPU";
        case ZE_DEVICE_TYPE_CPU:  return "CPU";
        case ZE_DEVICE_TYPE_FPGA: return "FPGA";
        case ZE_DEVICE_TYPE_MCA:  return "MCA";
        case ZE_DEVICE_TYPE_VPU:  return "VPU";
        default:                  return "Unknown";
    }
}

std::string L0Context::resultToString(ze_result_t r) {
    switch (r) {
        case ZE_RESULT_SUCCESS:                        return "ZE_RESULT_SUCCESS";
        case ZE_RESULT_NOT_READY:                      return "ZE_RESULT_NOT_READY";
        case ZE_RESULT_ERROR_UNINITIALIZED:            return "ZE_RESULT_ERROR_UNINITIALIZED";
        case ZE_RESULT_ERROR_DEVICE_LOST:              return "ZE_RESULT_ERROR_DEVICE_LOST";
        case ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY:       return "ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY";
        case ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY:     return "ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY";
        case ZE_RESULT_ERROR_MODULE_BUILD_FAILURE:     return "ZE_RESULT_ERROR_MODULE_BUILD_FAILURE";
        case ZE_RESULT_ERROR_MODULE_LINK_FAILURE:      return "ZE_RESULT_ERROR_MODULE_LINK_FAILURE";
        case ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS: return "ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS";
        case ZE_RESULT_ERROR_NOT_AVAILABLE:            return "ZE_RESULT_ERROR_NOT_AVAILABLE";
        case ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE:   return "ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE";
        case ZE_RESULT_ERROR_UNSUPPORTED_VERSION:      return "ZE_RESULT_ERROR_UNSUPPORTED_VERSION";
        case ZE_RESULT_ERROR_UNSUPPORTED_FEATURE:      return "ZE_RESULT_ERROR_UNSUPPORTED_FEATURE";
        case ZE_RESULT_ERROR_INVALID_ARGUMENT:         return "ZE_RESULT_ERROR_INVALID_ARGUMENT";
        case ZE_RESULT_ERROR_INVALID_NULL_HANDLE:      return "ZE_RESULT_ERROR_INVALID_NULL_HANDLE";
        case ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE:     return "ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE";
        case ZE_RESULT_ERROR_INVALID_NULL_POINTER:     return "ZE_RESULT_ERROR_INVALID_NULL_POINTER";
        case ZE_RESULT_ERROR_INVALID_SIZE:             return "ZE_RESULT_ERROR_INVALID_SIZE";
        case ZE_RESULT_ERROR_INVALID_ENUMERATION:      return "ZE_RESULT_ERROR_INVALID_ENUMERATION";
        case ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION:  return "ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION";
        case ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT: return "ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT";
        case ZE_RESULT_ERROR_INVALID_NATIVE_BINARY:    return "ZE_RESULT_ERROR_INVALID_NATIVE_BINARY";
        case ZE_RESULT_ERROR_INVALID_GLOBAL_NAME:      return "ZE_RESULT_ERROR_INVALID_GLOBAL_NAME";
        case ZE_RESULT_ERROR_INVALID_KERNEL_NAME:      return "ZE_RESULT_ERROR_INVALID_KERNEL_NAME";
        case ZE_RESULT_ERROR_INVALID_FUNCTION_NAME:    return "ZE_RESULT_ERROR_INVALID_FUNCTION_NAME";
        case ZE_RESULT_ERROR_OVERLAPPING_REGIONS:      return "ZE_RESULT_ERROR_OVERLAPPING_REGIONS";
        case ZE_RESULT_ERROR_UNKNOWN:                  return "ZE_RESULT_ERROR_UNKNOWN";
        default:                                       return "UNRECOGNIZED";
    }
}

} // namespace mimirmind::runtime