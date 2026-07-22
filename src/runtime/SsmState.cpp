// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/SsmState.hpp"

#include "compute/ComputeOps.hpp"

namespace mimirmind::runtime {

SsmState::SsmState(compute::ComputeOps& ops,
                   std::size_t          blockCount,
                   std::size_t          stateElemsPerLayer,
                   std::size_t          convStateElemsPerLayer)
    : _blockCount{blockCount},
      _stateElems{stateElemsPerLayer},
      _convStateElems{convStateElemsPerLayer},
      _state{ops.allocate(blockCount * stateElemsPerLayer * sizeof(float))},
      _convState{ops.allocate(blockCount * convStateElemsPerLayer * sizeof(float))} {
    reset(ops);
}

void SsmState::reset(compute::ComputeOps& ops) {
    ops.mulScalarAsync(_state.as<float>(), 0.0F, _blockCount * _stateElems);
    ops.mulScalarAsync(_convState.as<float>(), 0.0F, _blockCount * _convStateElems);
}

} // namespace mimirmind::runtime