// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/SsmState.hpp"

#include "compute/ComputeOps.hpp"

namespace mimirmind::runtime {

SsmState::SsmState(compute::ComputeOps& ops,
                   std::size_t          blockCount,
                   std::size_t          stateElemsPerLayer,
                   std::size_t          convStateElemsPerLayer,
                   std::size_t          nSeq)
    : _blockCount{blockCount},
      _stateElems{stateElemsPerLayer},
      _convStateElems{convStateElemsPerLayer},
      _nSeq{nSeq},
      _state{ops.allocate(blockCount * nSeq * stateElemsPerLayer * sizeof(float))},
      _convState{ops.allocate(blockCount * nSeq * convStateElemsPerLayer * sizeof(float))} {
    reset(ops);
}

void SsmState::reset(compute::ComputeOps& ops) {
    ops.mulScalarAsync(_state.as<float>(), 0.0F,
                       _blockCount * _nSeq * _stateElems);
    ops.mulScalarAsync(_convState.as<float>(), 0.0F,
                       _blockCount * _nSeq * _convStateElems);
}

} // namespace mimirmind::runtime