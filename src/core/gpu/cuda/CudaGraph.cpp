// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/gpu/cuda/CudaGraph.hpp"

namespace mimirmind::core::cuda {

CudaGraph::~CudaGraph() noexcept {
    destroy();
}

CudaGraph::CudaGraph(CudaGraph&& other) noexcept
    : _graph{other._graph}, _exec{other._exec} {
    other._graph = nullptr;
    other._exec  = nullptr;
}

CudaGraph& CudaGraph::operator=(CudaGraph&& other) noexcept {
    if (this != &other) {
        destroy();
        _graph       = other._graph;
        _exec        = other._exec;
        other._graph = nullptr;
        other._exec  = nullptr;
    }
    return *this;
}

void CudaGraph::capture(CudaStream& stream, const std::function<void()>& record) {
    destroy();

    cudaError_t rc = cudaStreamBeginCapture(stream.handle(),
                                            cudaStreamCaptureModeThreadLocal);
    if (rc != cudaSuccess) {
        throw CudaError("cudaStreamBeginCapture", rc);
    }

    // If record() throws (e.g. a residual host sync mid-capture), the stream
    // is still in capture mode — end the capture (discarding the partial
    // graph) so the stream returns to a usable state, clear the sticky error,
    // then rethrow so the caller can fall back to immediate mode.
    try {
        record();
    } catch (...) {
        cudaGraph_t partial = nullptr;
        cudaStreamEndCapture(stream.handle(), &partial);
        if (partial != nullptr) {
            cudaGraphDestroy(partial);
        }
        cudaGetLastError();  // clear sticky runtime error
        throw;
    }

    rc = cudaStreamEndCapture(stream.handle(), &_graph);
    if (rc != cudaSuccess) {
        _graph = nullptr;
        throw CudaError("cudaStreamEndCapture", rc);
    }

    // CUDA 12+/13 signature: (exec*, graph, flags). flags 0 = default.
    rc = cudaGraphInstantiate(&_exec, _graph, 0);
    if (rc != cudaSuccess) {
        destroy();
        throw CudaError("cudaGraphInstantiate", rc);
    }
}

void CudaGraph::launch(CudaStream& stream) {
    if (_exec == nullptr) {
        return;
    }
    const cudaError_t rc = cudaGraphLaunch(_exec, stream.handle());
    if (rc != cudaSuccess) {
        throw CudaError("cudaGraphLaunch", rc);
    }
}

void CudaGraph::destroy() noexcept {
    if (_exec != nullptr) {
        cudaGraphExecDestroy(_exec);
        _exec = nullptr;
    }
    if (_graph != nullptr) {
        cudaGraphDestroy(_graph);
        _graph = nullptr;
    }
}

} // namespace mimirmind::core::cuda