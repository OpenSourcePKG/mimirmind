// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/gpu/cuda/CudaContext.hpp"
#include "core/gpu/cuda/CudaStream.hpp"

#include <cuda_runtime.h>

#include <functional>

namespace mimirmind::core::cuda {

/**
 * RAII wrapper around a captured + instantiated CUDA graph
 * (`cudaGraph_t` + `cudaGraphExec_t`). Capture the stream work enqueued by
 * a callback once, instantiate, then `launch()` the exact same DAG per
 * token — replacing per-token kernel re-dispatch in the decode loop. The
 * CUDA peer of the L0 `CommandQueue` record/replay (M-Q3N.5 Kernaufgabe 4).
 *
 * Contract on the recorded callback: only stream-ordered, capturable ops
 * (kernel launches, device<->device async copies). No host syncs, no
 * allocations, no host-blocking work during capture. Any per-token-varying
 * scalar (KV length / write offset) must be read by the kernels from a
 * persistent device slot that is updated OUTSIDE the graph between launches
 * — never baked into a captured host->device copy, which would replay the
 * record-time value.
 *
 * Move-only; single-threaded by contract. Every `cuda*` failure throws
 * `CudaError`.
 */
class CudaGraph {
public:
    CudaGraph() = default;
    ~CudaGraph() noexcept;

    CudaGraph(const CudaGraph&)            = delete;
    CudaGraph& operator=(const CudaGraph&) = delete;
    CudaGraph(CudaGraph&& other) noexcept;
    CudaGraph& operator=(CudaGraph&& other) noexcept;

    /// Capture the ops `record` enqueues on `stream` and instantiate an
    /// executable graph, replacing any previously captured one. Uses
    /// `cudaStreamCaptureModeThreadLocal`. Throws on capture / instantiate
    /// failure (and leaves *this empty).
    void capture(CudaStream& stream, const std::function<void()>& record);

    /// Launch the instantiated graph on `stream` (`cudaGraphLaunch`). Does
    /// not synchronize. No-op when nothing is captured.
    void launch(CudaStream& stream);

    [[nodiscard]] bool valid() const noexcept { return _exec != nullptr; }

private:
    void destroy() noexcept;

    cudaGraph_t     _graph{nullptr};
    cudaGraphExec_t _exec{nullptr};
};

} // namespace mimirmind::core::cuda