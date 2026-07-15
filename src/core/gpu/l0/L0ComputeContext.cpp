// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/gpu/l0/L0ComputeContext.hpp"

#include <memory>

namespace mimirmind::core::l0 {

L0ComputeContext::L0ComputeContext(Options opts)
    : _ctx{std::move(opts.spvDirOverride)}
    , _alloc{_ctx, opts.usmProbeTotalGiB, opts.usmKind}
    , _queue{_ctx}
{}

L0ComputeContext::~L0ComputeContext() = default;

// Factory hook consumed by BackendRegistry::createContext(). Lives in
// this translation unit so the common BackendRegistry TU never pulls
// in level_zero/ze_api.h. Options are the defaults — the entry point
// deliberately does not expose them because runtime backend-selection
// happens before config.json is fully parsed; the tuned Options struct
// gets passed by callers that build an L0ComputeContext directly.
std::unique_ptr<::mimirmind::core::backend::ComputeContext>
createComputeContext() {
    return std::make_unique<L0ComputeContext>();
}

} // namespace mimirmind::core::l0