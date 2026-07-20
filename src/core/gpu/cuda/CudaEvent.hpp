// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/gpu/cuda/CudaContext.hpp"
#include "core/gpu/cuda/CudaStream.hpp"

#include <cuda_runtime.h>

namespace mimirmind::core::cuda {

/**
 * How the event is used decides which flag we pass to
 * `cudaEventCreateWithFlags`.
 *
 *   Timing   — default: `cudaEventDefault`. Enables the driver's
 *              timestamp so `elapsedMs` returns real numbers.
 *   SyncOnly — `cudaEventDisableTiming`. Cheaper — meant for pure
 *              stream-ordering / cross-stream synchronisation where
 *              elapsed time is not needed. Calling `elapsedMs` on a
 *              SyncOnly event is a driver error.
 */
enum class CudaEventKind {
    Timing,
    SyncOnly,
};

/**
 * RAII wrapper around a `cudaEvent_t`. Mirrors HipEvent — construct,
 * record on a stream, synchronise, query, measure elapsed time between
 * two events.
 *
 * Every `cuda*` failure throws `CudaError`. Move-only, not thread-safe.
 */
class CudaEvent {
public:
    explicit CudaEvent(CudaContext&  ctx,
                       CudaEventKind kind = CudaEventKind::Timing);
    ~CudaEvent() noexcept;

    CudaEvent(const CudaEvent&)            = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    CudaEvent(CudaEvent&& other) noexcept;
    CudaEvent& operator=(CudaEvent&& other) noexcept;

    /// Insert this event into the given stream. Non-blocking.
    void record(CudaStream& stream);

    /// Block until the recorded op completes on the device.
    void synchronize();

    /// Non-blocking check. Returns true if the recorded op has finished
    /// (or if the event was never recorded — vacuously done). Returns
    /// false on driver errors.
    [[nodiscard]] bool query() const noexcept;

    /// Elapsed milliseconds from `start` to `*this`. Both events must
    /// be `CudaEventKind::Timing` and recorded. Throws `CudaError`
    /// otherwise (typically `cudaErrorInvalidResourceHandle`).
    [[nodiscard]] float elapsedMs(const CudaEvent& start) const;

    [[nodiscard]] cudaEvent_t handle() const noexcept { return _event; }

    [[nodiscard]] CudaEventKind kind() const noexcept { return _kind; }

private:
    CudaContext*  _ctx{nullptr};
    cudaEvent_t   _event{nullptr};
    CudaEventKind _kind{CudaEventKind::Timing};

    void destroy() noexcept;
};

} // namespace mimirmind::core::cuda