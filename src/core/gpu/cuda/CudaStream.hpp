// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/gpu/cuda/CudaContext.hpp"

#include <cuda_runtime.h>

namespace mimirmind::core::cuda {

/**
 * Selects how the constructed stream ordering relates to the legacy
 * default stream on the device.
 *
 *   BlockingDefault    — cudaStreamCreate. Implicit sync with the
 *                        legacy default stream; simplest semantics.
 *   NonBlocking        — cudaStreamCreateWithFlags(cudaStreamNonBlocking).
 *                        Independent from the default stream — required
 *                        once we have real overlap between compute and
 *                        H↔D copies in the same forward pass.
 */
enum class CudaStreamKind {
    BlockingDefault,
    NonBlocking,
};

/**
 * RAII wrapper around a `cudaStream_t`. Move-only, single-threaded by
 * contract — same "one stream per user" approach as `HipStream` on the
 * HIP side.
 *
 * The parent `CudaContext` selects the device; the stream is created
 * on that device. Multiple `CudaStream`s per `CudaContext` are allowed
 * and independent.
 *
 * Every `cuda*` failure throws `CudaError`.
 */
class CudaStream {
public:
    explicit CudaStream(CudaContext&   ctx,
                        CudaStreamKind kind = CudaStreamKind::BlockingDefault);
    ~CudaStream() noexcept;

    CudaStream(const CudaStream&)            = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    CudaStream(CudaStream&& other) noexcept;
    CudaStream& operator=(CudaStream&& other) noexcept;

    /// Block the calling thread until every op recorded on this stream
    /// completes. Cheap when the stream is idle.
    void synchronize();

    /// True if all recorded work has finished, false if any op is still
    /// in-flight. Non-blocking. Returns false on driver errors.
    [[nodiscard]] bool query() const noexcept;

    /// Raw handle for direct API calls (kernel launches via
    /// cuLaunchKernel, cudaMemcpyAsync, cudaEventRecord).
    [[nodiscard]] cudaStream_t handle() const noexcept { return _stream; }

    [[nodiscard]] CudaStreamKind kind() const noexcept { return _kind; }

private:
    CudaContext*   _ctx{nullptr};
    cudaStream_t   _stream{nullptr};
    CudaStreamKind _kind{CudaStreamKind::BlockingDefault};

    void destroy() noexcept;
};

} // namespace mimirmind::core::cuda