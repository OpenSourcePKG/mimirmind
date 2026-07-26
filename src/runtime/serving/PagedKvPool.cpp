// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/serving/PagedKvPool.hpp"

#include "compute/ComputeOps.hpp"
#include "core/log/Log.hpp"

#include <stdexcept>
#include <string>

namespace mimirmind::runtime::serving {

PagedKvPool::PagedKvPool(compute::ComputeOps& ops,
                         std::size_t          numLayers,
                         std::size_t          numBlocks,
                         std::size_t          blockSize,
                         std::size_t          numKvHeads,
                         std::size_t          headDim)
    : _numLayers{numLayers},
      _numBlocks{numBlocks},
      _blockSize{blockSize},
      _numKvHeads{numKvHeads},
      _headDim{headDim} {
    if (numLayers == 0 || numBlocks == 0 || blockSize == 0 ||
        numKvHeads == 0 || headDim == 0) {
        throw std::invalid_argument(
            "PagedKvPool: numLayers/numBlocks/blockSize/numKvHeads/headDim "
            "must all be > 0");
    }

    const std::size_t poolElems = _numBlocks * blockElems();
    const std::size_t poolBytes = poolElems * sizeof(float);

    _kOwners.reserve(_numLayers);
    _vOwners.reserve(_numLayers);
    _kPool.reserve(_numLayers);
    _vPool.reserve(_numLayers);
    for (std::size_t l = 0; l < _numLayers; ++l) {
        _kOwners.emplace_back(ops.allocate(poolBytes));
        _vOwners.emplace_back(ops.allocate(poolBytes));
        _kPool.push_back(_kOwners.back().as<float>());
        _vPool.push_back(_vOwners.back().as<float>());
    }

    MM_LOG_INFO("paged-kv-pool",
                "allocated {} full-attn layers × 2 × {} blocks × {} tok/block "
                "× {} kv_dim (fp32) = {:.1f} MiB total",
                _numLayers, _numBlocks, _blockSize, slotElems(),
                static_cast<double>(2 * _numLayers * poolBytes) /
                    (1024.0 * 1024.0));
}

float* PagedKvPool::keyPool(std::size_t layer) noexcept {
    return _kPool[layer];
}

float* PagedKvPool::valuePool(std::size_t layer) noexcept {
    return _vPool[layer];
}

const float* PagedKvPool::keyPool(std::size_t layer) const noexcept {
    return _kPool[layer];
}

const float* PagedKvPool::valuePool(std::size_t layer) const noexcept {
    return _vPool[layer];
}

void PagedKvPool::writeToken(compute::ComputeOps& ops,
                             std::size_t          layer,
                             std::uint32_t        blockId,
                             std::size_t          slot,
                             const float*         kRow,
                             const float*         vRow) {
    const std::size_t width = slotElems();               // numKvHeads*headDim
    const std::size_t off =
        (static_cast<std::size_t>(blockId) * _blockSize + slot) * width;
    const std::size_t bytes = width * sizeof(float);
    ops.appendMemoryCopy(_kPool[layer] + off, kRow, bytes);
    ops.appendMemoryCopy(_vPool[layer] + off, vRow, bytes);
}

} // namespace mimirmind::runtime::serving
