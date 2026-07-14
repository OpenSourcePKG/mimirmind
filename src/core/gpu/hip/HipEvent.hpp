// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/gpu/hip/HipContext.hpp"
#include "core/gpu/hip/HipStream.hpp"

#include <hip/hip_runtime.h>

namespace mimirmind::core::hip {

/**
 * How the event is used decides which flag we pass to
 * `hipEventCreateWithFlags`.
 *
 *   Timing   — default: `hipEventDefault`. Enables the driver's
 *              timestamp so `elapsedMs` returns real numbers. Slightly
 *              more expensive than sync-only events.
 *   SyncOnly — `hipEventDisableTiming`. Cheaper — meant for pure
 *              stream-ordering / cross-stream synchronisation where
 *              elapsed time is not needed. Calling `elapsedMs` on a
 *              SyncOnly event is a driver error.
 */
enum class HipEventKind {
    Timing,
    SyncOnly,
};

/**
 * RAII wrapper around a `hipEvent_t`. Small — five entry points:
 * construct, record on a stream, synchronise, query, and measure
 * elapsed time between two events. Same "concrete-backend-type,
 * no interface" style as HipStream.
 *
 * Every `hip*` failure throws `HipError`. Move-only, not thread-safe
 * by contract.
 */
class HipEvent {
public:
    explicit HipEvent(HipContext&  ctx,
                      HipEventKind kind = HipEventKind::Timing);
    ~HipEvent() noexcept;

    HipEvent(const HipEvent&)            = delete;
    HipEvent& operator=(const HipEvent&) = delete;
    HipEvent(HipEvent&& other) noexcept;
    HipEvent& operator=(HipEvent&& other) noexcept;

    /// Insert this event into the given stream. Non-blocking — returns
    /// as soon as the record op is queued. Multiple record()s on the
    /// same event overwrite: only the most recent record decides what
    /// `synchronize()` and `elapsedMs` see.
    void record(HipStream& stream);

    /// Block until the recorded op completes on the device.
    void synchronize();

    /// Non-blocking check. Returns true if the recorded op has finished
    /// (or if the event was never recorded — an unrecorded event is
    /// vacuously "done"). Returns false on driver errors — for the
    /// caller's error path use `synchronize()` instead.
    [[nodiscard]] bool query() const noexcept;

    /// Elapsed milliseconds from `start` to `*this`. Both events must
    /// have been constructed with `HipEventKind::Timing` and recorded
    /// on some stream. Throws `HipError` otherwise (typically
    /// `hipErrorInvalidResourceHandle` for SyncOnly or unrecorded).
    [[nodiscard]] float elapsedMs(const HipEvent& start) const;

    /// Raw handle. Same rationale as HipStream::handle().
    [[nodiscard]] hipEvent_t handle() const noexcept { return _event; }

    [[nodiscard]] HipEventKind kind() const noexcept { return _kind; }

private:
    HipContext*  _ctx{nullptr};
    hipEvent_t   _event{nullptr};
    HipEventKind _kind{HipEventKind::Timing};

    void destroy() noexcept;
};

} // namespace mimirmind::core::hip