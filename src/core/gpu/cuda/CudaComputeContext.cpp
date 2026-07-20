// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/gpu/cuda/CudaComputeContext.hpp"

namespace mimirmind::core::cuda {

CudaComputeContext::CudaComputeContext(Options opts)
    : _ctx{opts.deviceIndex}
    , _alloc{_ctx}
    , _stream{_ctx, opts.streamKind}
{}

CudaComputeContext::~CudaComputeContext() = default;

} // namespace mimirmind::core::cuda