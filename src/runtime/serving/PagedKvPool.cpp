// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/serving/PagedKvPool.hpp"

#include "compute/ComputeOps.hpp"
#include "core/log/Log.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>

namespace mimirmind::runtime::serving {

PagedKvPool::PagedKvPool(compute::ComputeOps& ops,
                         std::size_t          numLayers,
                         std::size_t          numBlocks,
                         std::size_t          blockSize,
                         std::size_t          numKvHeads,
                         std::size_t          headDim,
                         KvDtype              dtype)
    : _numLayers{numLayers},
      _numBlocks{numBlocks},
      _blockSize{blockSize},
      _numKvHeads{numKvHeads},
      _headDim{headDim},
      _dtype{dtype} {
    if (numLayers == 0 || numBlocks == 0 || blockSize == 0 ||
        numKvHeads == 0 || headDim == 0) {
        throw std::invalid_argument(
            "PagedKvPool: numLayers/numBlocks/blockSize/numKvHeads/headDim "
            "must all be > 0");
    }
    // Only per-element dtypes are addressable by the paged pool's flat
    // element layout; Q8_0 (block-encoded) would need a block-aware pool.
    if (_dtype != KvDtype::F32 && _dtype != KvDtype::FP16) {
        throw std::invalid_argument(
            "PagedKvPool: only F32 and FP16 KV dtypes are supported");
    }

    const std::size_t poolElems = _numBlocks * blockElems();
    const std::size_t poolBytes = poolElems * elemBytes();

    _kOwners.reserve(_numLayers);
    _vOwners.reserve(_numLayers);
    _kPool.reserve(_numLayers);
    _vPool.reserve(_numLayers);
    for (std::size_t l = 0; l < _numLayers; ++l) {
        _kOwners.emplace_back(ops.allocate(poolBytes));
        _vOwners.emplace_back(ops.allocate(poolBytes));
        _kPool.push_back(static_cast<void*>(_kOwners.back().as<std::byte>()));
        _vPool.push_back(static_cast<void*>(_vOwners.back().as<std::byte>()));
    }

    MM_LOG_INFO("paged-kv-pool",
                "allocated {} full-attn layers × 2 × {} blocks × {} tok/block "
                "× {} kv_dim ({}) = {:.1f} MiB total",
                _numLayers, _numBlocks, _blockSize, slotElems(),
                _dtype == KvDtype::FP16 ? "fp16" : "fp32",
                static_cast<double>(2 * _numLayers * poolBytes) /
                    (1024.0 * 1024.0));
}

void* PagedKvPool::keyPool(std::size_t layer) noexcept {
    return _kPool[layer];
}

void* PagedKvPool::valuePool(std::size_t layer) noexcept {
    return _vPool[layer];
}

const void* PagedKvPool::keyPool(std::size_t layer) const noexcept {
    return _kPool[layer];
}

const void* PagedKvPool::valuePool(std::size_t layer) const noexcept {
    return _vPool[layer];
}

void PagedKvPool::writeToken(compute::ComputeOps& ops,
                             std::size_t          layer,
                             std::uint32_t        blockId,
                             std::size_t          slot,
                             const float*         kRow,
                             const float*         vRow) {
    const std::size_t width  = slotElems();              // numKvHeads*headDim
    const std::size_t rowOff =
        static_cast<std::size_t>(blockId) * _blockSize + slot;   // row position
    if (_dtype == KvDtype::FP16) {
        // Cast-on-scatter (F32 kRow/vRow -> __half) via the shared fp16 commit,
        // one row at the (blockId*blockSize+slot) row offset — mirrors the
        // batched writeTokensBatched fp16 path for the host-loop fallback.
        ops.kvCommitFp16Async(kRow, _kPool[layer], /*T=*/1, /*kvDim=*/width,
                              /*writeOffset=*/rowOff);
        ops.kvCommitFp16Async(vRow, _vPool[layer], /*T=*/1, /*kvDim=*/width,
                              /*writeOffset=*/rowOff);
        return;
    }
    const std::size_t off   = rowOff * width;
    const std::size_t bytes = width * sizeof(float);
    auto* kBase = static_cast<float*>(_kPool[layer]);
    auto* vBase = static_cast<float*>(_vPool[layer]);
    ops.appendMemoryCopy(kBase + off, kRow, bytes);
    ops.appendMemoryCopy(vBase + off, vRow, bytes);
}

void PagedKvPool::writeTokensBatched(compute::ComputeOps& ops,
                                     std::size_t          layer,
                                     const float*         kProj,
                                     const float*         vProj,
                                     const std::uint32_t* writeBlockIdDev,
                                     const std::int32_t*  writeSlotDev,
                                     std::size_t          nSeq) {
    // Scatter the compact per-seq F32 projections into the paged slots. When
    // the pool is FP16 the seam casts F32->__half on write (5.14 I1); F32 is
    // a straight scatter. dtype selects the kernel variant inside GpuOps.
    ops.writeKvTokensBatchedAsync(kProj, vProj, writeBlockIdDev, writeSlotDev,
                                  _kPool[layer], _vPool[layer], nSeq, _blockSize,
                                  slotElems(), _dtype);
}

} // namespace mimirmind::runtime::serving
