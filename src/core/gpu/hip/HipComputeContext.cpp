// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/gpu/hip/HipComputeContext.hpp"

#include <memory>

namespace mimirmind::core::hip {

HipComputeContext::HipComputeContext(Options opts)
    : _ctx{opts.deviceIndex}
    , _alloc{_ctx}
    , _stream{_ctx, opts.streamKind}
{}

HipComputeContext::~HipComputeContext() = default;

// Factory hook consumed by BackendRegistry::createContext(). Lives in
// this translation unit so the common BackendRegistry TU never pulls
// in <hip/hip_runtime.h>. Uses default Options: auto-select device,
// BlockingDefault stream.
std::unique_ptr<::mimirmind::core::backend::ComputeContext>
createComputeContext() {
    return std::make_unique<HipComputeContext>();
}

} // namespace mimirmind::core::hip