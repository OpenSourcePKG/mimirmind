// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/gpu/hip/HipContext.hpp"

#include <hip/hip_runtime.h>

namespace mimirmind::core::hip {

/**
 * Selects how the constructed stream ordering relates to the legacy
 * default stream on the device.
 *
 *   BlockingDefault    — hipStreamCreate. Implicit sync w/ default
 *                        stream; simplest semantics for the bringup.
 *   NonBlocking        — hipStreamCreateWithFlags(hipStreamNonBlocking).
 *                        Independent from the default stream — required
 *                        once we have real overlap between compute and
 *                        H↔D copies in the same forward pass.
 */
enum class HipStreamKind {
    BlockingDefault,
    NonBlocking,
};

/**
 * RAII wrapper around a `hipStream_t`. The skeleton parallel to
 * `runtime::CommandQueue` for Level Zero — same "one stream per user"
 * approach, no pooling. Move-only, single-threaded by contract.
 *
 * The parent `HipContext` selects the device; the stream is created
 * on that device and inherits its context. Multiple `HipStream`s per
 * `HipContext` are allowed and independent.
 *
 * Every `hip*` failure throws `HipError`.
 */
class HipStream {
public:
    explicit HipStream(HipContext&   ctx,
                       HipStreamKind kind = HipStreamKind::BlockingDefault);
    ~HipStream() noexcept;

    HipStream(const HipStream&)            = delete;
    HipStream& operator=(const HipStream&) = delete;
    HipStream(HipStream&& other) noexcept;
    HipStream& operator=(HipStream&& other) noexcept;

    /// Block the calling thread until every op recorded on this stream
    /// completes. Cheap when the stream is idle.
    void synchronize();

    /// True if all recorded work has finished, false if any op is still
    /// in-flight. Non-blocking. Returns false on driver errors — for the
    /// caller's error path use `synchronize()` instead.
    [[nodiscard]] bool query() const noexcept;

    /// Raw handle for direct API calls (kernel launches, hipMemcpyAsync,
    /// hipEventRecord). Keep the concrete type visible — a neutral
    /// interface would obscure the perf story with no gain (see the
    /// CommandQueue backend-surface note).
    [[nodiscard]] hipStream_t handle() const noexcept { return _stream; }

    [[nodiscard]] HipStreamKind kind() const noexcept { return _kind; }

private:
    HipContext*   _ctx{nullptr};
    hipStream_t   _stream{nullptr};
    HipStreamKind _kind{HipStreamKind::BlockingDefault};

    void destroy() noexcept;
};

} // namespace mimirmind::core::hip