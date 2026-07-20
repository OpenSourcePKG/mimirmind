// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/backend/BackendRegistry.hpp"
#include "core/backend/ComputeBackend.hpp"
#include "core/backend/ComputeContext.hpp"

#include <memory>

namespace mimirmind::core::cpu {

/**
 * CPU implementation of `ComputeBackend`. Stateless — the CPU is always
 * available on any Linux host that runs the binary, so `hasFeature` is
 * a fixed truth-table and `deviceInfo` returns a minimal descriptor.
 *
 * The only feature we advertise is `UnifiedMemoryHost` (device pointer
 * IS a host pointer on this backend). No matrix engine, no DP4A fast
 * path, no IPC — reference/fallback semantics only.
 */
class CpuBackend : public ::mimirmind::core::backend::ComputeBackend {
public:
    CpuBackend();
    ~CpuBackend() override = default;

    [[nodiscard]] ::mimirmind::core::backend::BackendKind
        kind() const noexcept override;

    [[nodiscard]] const ::mimirmind::core::backend::BackendDeviceInfo&
        deviceInfo() const noexcept override;

    [[nodiscard]] bool
        hasFeature(::mimirmind::core::backend::BackendFeature f)
        const noexcept override;

private:
    ::mimirmind::core::backend::BackendDeviceInfo _info;
};

/**
 * CPU-backed `ComputeContext`. No handles to own — every op runs
 * inline on the calling thread through the reference implementations
 * in `src/compute/`. `compute::cpu::GpuOps` (Schicht M-CPU.0) and
 * `compute::cpu::GpuMatmul` (Schicht M-CPU.1) both take this by
 * reference for symmetry with the L0 / HIP contexts, but neither
 * actually reads any state from it today.
 *
 * Constructed via `mimirmind::core::cpu::createComputeContext()` from
 * `BackendRegistry::createContext(BackendKind::Cpu)`. Owns nothing on
 * the heap; move-only would be trivially fine but the interface
 * enforces the non-copy / non-move contract.
 */
class CpuContext : public ::mimirmind::core::backend::ComputeContext {
public:
    CpuContext() = default;
    ~CpuContext() override = default;

    [[nodiscard]] ::mimirmind::core::backend::ComputeBackend&
        backend() noexcept override { return _backend; }
    [[nodiscard]] const ::mimirmind::core::backend::ComputeBackend&
        backend() const noexcept override { return _backend; }

    /// Sustained CPU memory bandwidth. 50 GB/s covers a typical
    /// dual-channel DDR5-5600 desktop (pegenaut-skynet class);
    /// higher-end 4-channel or DDR5-8000+ systems will see more but
    /// the probe consumer only uses this to gate serving-class
    /// batching, which is CUDA-only in Bragi-v1 — so the CPU value
    /// is mostly informational at `/v1/system/info`.
    [[nodiscard]] std::size_t bandwidthGBps() const noexcept override {
        return 50;
    }

private:
    CpuBackend _backend;
};

/// Backend probe hook used by `BackendRegistry::probeAll()`. Cpu is
/// always compiled in and always available, so the returned probe is
/// `{Cpu, compiledIn=true, available=true, ...}` unconditionally.
[[nodiscard]] ::mimirmind::core::backend::BackendProbe
    probeBackend() noexcept;

/// Backend factory used by `BackendRegistry::createContext(Cpu)`.
[[nodiscard]] std::unique_ptr<::mimirmind::core::backend::ComputeContext>
    createComputeContext();

} // namespace mimirmind::core::cpu