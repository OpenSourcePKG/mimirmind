// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/serving/PagedKvBlockAllocator.hpp"

#include <cassert>
#include <stdexcept>

namespace mimirmind::runtime::serving {

PagedKvBlockAllocator::PagedKvBlockAllocator(std::size_t numBlocks,
                                             std::size_t blockSize)
    : _blockSize(blockSize)
    , _blocks(numBlocks)
{
    if (numBlocks == 0) {
        throw std::invalid_argument{
            "PagedKvBlockAllocator: numBlocks must be > 0"};
    }
    if (blockSize == 0) {
        throw std::invalid_argument{
            "PagedKvBlockAllocator: blockSize must be > 0"};
    }
    _freeList.reserve(numBlocks);
    // Fill free list in reverse so allocate() hands out id 0 first
    // (LIFO pop returns the last pushed id).
    for (std::size_t i = numBlocks; i > 0; --i) {
        _freeList.push_back(static_cast<std::uint32_t>(i - 1));
    }
}

std::uint32_t PagedKvBlockAllocator::allocate() noexcept {
    if (_freeList.empty()) {
        return kInvalidBlock;
    }
    const std::uint32_t id = _freeList.back();
    _freeList.pop_back();
    assert(id < _blocks.size());
    _blocks[id].refcount = 1;
    _blocks[id].hash     = 0;
    return id;
}

void PagedKvBlockAllocator::addRef(std::uint32_t blockId) noexcept {
    if (blockId >= _blocks.size()) return;
    // Do not addRef a free block — that would inflate refcount from 0
    // and never release. Silent no-op in release; caller bug.
    if (_blocks[blockId].refcount == 0) return;
    _blocks[blockId].refcount += 1;
}

void PagedKvBlockAllocator::release(std::uint32_t blockId) noexcept {
    if (blockId >= _blocks.size()) return;
    if (_blocks[blockId].refcount == 0) return;
    _blocks[blockId].refcount -= 1;
    if (_blocks[blockId].refcount == 0) {
        _blocks[blockId].hash = 0;
        _freeList.push_back(blockId);
    }
}

void PagedKvBlockAllocator::setHash(std::uint32_t blockId,
                                    std::uint64_t hash) noexcept {
    if (blockId >= _blocks.size()) return;
    if (_blocks[blockId].refcount == 0) return;
    _blocks[blockId].hash = hash;
}

std::uint64_t PagedKvBlockAllocator::hashOf(std::uint32_t blockId) const noexcept {
    if (blockId >= _blocks.size()) return 0;
    return _blocks[blockId].hash;
}

std::uint32_t PagedKvBlockAllocator::refcountOf(std::uint32_t blockId) const noexcept {
    if (blockId >= _blocks.size()) return 0;
    return _blocks[blockId].refcount;
}

} // namespace mimirmind::runtime::serving
