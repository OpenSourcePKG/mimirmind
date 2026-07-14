// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mimirmind::core::gpu {

/**
 * Backend identifier — which underlying API is doing the compute.
 *
 * `LevelZero` is today's only production impl (Intel Xe-LPG via
 * oneAPI). The other entries are placeholders for the multi-backend
 * roadmap ([[Mimirmind — HW-Abstraktions-Strategie für Multi-Backend-Support]]):
 *
 *  - `VulkanCompute` — cross-vendor (AMD via RADV, Intel, NVIDIA)
 *                     with SPIR-V kernels; the pragmatic
 *                     home-exploration + AMD path.
 *  - `Cuda`          — NVIDIA-native; would need CUDA-C or PTX
 *                     kernels alongside our SPIR-V set.
 *  - `Hip`           — AMD ROCm; parallel kernel-set. Not on the
 *                     recommended path today (Vulkan Compute is the
 *                     lower-friction AMD route).
 *  - `Cpu`           — reference / test fallback.
 *
 * Each concrete backend is implemented in its own translation unit
 * under `src/core/gpu/<backend>/`. Consumers that only need
 * backend-agnostic device-info should take `GpuBackend&`; consumers
 * that touch backend-specific kernel APIs (CommandQueue, GpuModule,
 * UsmAllocator today) stay on the concrete `L0Context` / future
 * concrete-backend types.
 */
enum class BackendKind : std::uint8_t {
    LevelZero,
    VulkanCompute,
    Cuda,
    Hip,
    Cpu,
    Unknown,
};

/**
 * Backend-neutral device classification. Mirrors the coarse device
 * families the runtime cares about — the exact enum value should be
 * derivable from the backend's native device-type without extra HW
 * queries.
 */
enum class DeviceKind : std::uint8_t {
    GpuIntegrated,   // e.g. Meteor Lake Xe-LPG iGPU, Apple M-series
    GpuDiscrete,     // e.g. Arc B70, RTX 4090, Radeon RX 9070
    Npu,             // e.g. Meteor Lake VPU 3720 (via L0)
    Cpu,             // reference / fallback
    Unknown,
};

/**
 * Coarse capability probe. Backends set these based on their driver
 * probing at ctor time; consumers can gate optimisation paths on the
 * result without knowing which backend is live.
 *
 * Entries are additive — new features get their own enumerator, no
 * value is ever renumbered. Keep sorted by first-supported-backend.
 */
enum class BackendFeature : std::uint8_t {
    /// Level Zero: `ZE_experimental_mutable_command_list`. Also known
    /// as CLR-prerequisite. Vulkan has a rough analogue via secondary
    /// command buffers but the semantics differ.
    MutableCommandLists,

    /// Integer dot-product intrinsics (INT8×INT8 accumulate) at
    /// hardware speed. Level Zero: `cl_khr_integer_dot_product`.
    /// AMD RDNA2+, NVIDIA post-Turing, Intel Xe-LPG all have
    /// vendor-specific variants; the enum just says "yes there is
    /// a fast path", not which one.
    IntegerDotProduct,

    /// Cross-process USM handle export (Level Zero `zeMemGetIpcHandle`).
    /// Munin depends on this for the persistent-memory daemon.
    /// Backends without IPC-safe host allocations must return false.
    IpcHandleExport,

    /// Backend supports allocating memory that's both device-readable
    /// AND host-writable without explicit copy (UMA-iGPU semantics).
    /// Level Zero on Meteor Lake: yes via `zeMemAllocHost`. dGPU
    /// backends typically no.
    UnifiedMemoryHost,
};

/**
 * Backend-neutral snapshot of the currently selected compute device.
 * Populated once at backend ctor time; consumers should treat it as
 * immutable for the lifetime of the backend.
 *
 * Deliberately smaller than the L0-native `l0::DeviceInfo` — this is
 * the "portable subset" that has meaning across backends. When the
 * L0-specific `ze_device_type_t` matters, use the concrete
 * `L0Context` instead.
 */
struct BackendDeviceInfo {
    std::string   name;              // vendor-supplied device name
    std::string   uuid;              // stable device identifier (may be empty on some backends)
    std::uint32_t vendorId{0};       // PCI vendor id (0x8086, 0x10DE, 0x1002, ...)
    std::uint32_t deviceId{0};       // PCI device id
    DeviceKind    kind{DeviceKind::Unknown};
    std::uint32_t numComputeUnits{0};// EUs / SMs / CUs — comparable order-of-magnitude only
    std::size_t   totalLocalMem{0};  // total device-addressable memory in bytes (UMA-total on iGPU)
    std::size_t   maxMemAllocSize{0};// largest single allocation the backend will honour
    std::uint32_t coreClockRate{0};  // MHz, best-effort
};

/**
 * Pure-virtual multi-backend interface. Today's only impl is
 * `core::l0::L0Context`; a second (`VulkanBackend`, `CudaBackend`,
 * …) would be added under `src/core/gpu/<backend>/`.
 *
 * The interface is deliberately narrow — it exposes only what's
 * useful without knowing the concrete backend: device info + feature
 * flags + backend identity. Anything that needs to construct a
 * queue, load a module, or allocate USM still goes through the
 * concrete backend type. See
 * [[Mimirmind — HW-Abstraktions-Strategie für Multi-Backend-Support]]
 * for the Schicht-1..6 progression this fits into (this is
 * Schicht 1).
 *
 * Not thread-safe by contract — concrete backends may be, but no
 * consumer should assume so.
 */
class GpuBackend {
public:
    virtual ~GpuBackend() = default;

    GpuBackend(const GpuBackend&)            = delete;
    GpuBackend& operator=(const GpuBackend&) = delete;
    GpuBackend(GpuBackend&&)                 = delete;
    GpuBackend& operator=(GpuBackend&&)      = delete;

    [[nodiscard]] virtual BackendKind kind() const noexcept = 0;

    /// The selected device the runtime is bound to. For multi-device
    /// hosts the backend picks one at ctor time. Backend-specific
    /// multi-device enumeration (if any) stays on the concrete
    /// backend type — the neutral interface only exposes the ONE
    /// device compute actually happens on.
    [[nodiscard]] virtual const BackendDeviceInfo& deviceInfo() const noexcept = 0;

    /// Cheap runtime probe — did the backend detect a fast path for
    /// this feature on the selected device? Never throws; unknown
    /// features return false.
    [[nodiscard]] virtual bool hasFeature(BackendFeature f) const noexcept = 0;

protected:
    GpuBackend() = default;
};

} // namespace mimirmind::core::gpu