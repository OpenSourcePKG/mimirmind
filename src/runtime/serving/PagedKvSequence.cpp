// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/serving/PagedKvSequence.hpp"

#include <cassert>

namespace mimirmind::runtime::serving {

PagedKvSequence::PagedKvSequence(PagedKvBlockAllocator& allocator) noexcept
    : _allocator(&allocator)
{}

PagedKvSequence::~PagedKvSequence() {
    releaseAllBlocks();
}

PagedKvSequence::PagedKvSequence(PagedKvSequence&& other) noexcept
    : _allocator(other._allocator)
    , _blockTable(std::move(other._blockTable))
    , _tokens(std::move(other._tokens))
{
    other._allocator = nullptr;
    other._blockTable.clear();
    other._tokens.clear();
}

PagedKvSequence& PagedKvSequence::operator=(PagedKvSequence&& other) noexcept {
    if (this != &other) {
        releaseAllBlocks();
        _allocator  = other._allocator;
        _blockTable = std::move(other._blockTable);
        _tokens     = std::move(other._tokens);
        other._allocator = nullptr;
        other._blockTable.clear();
        other._tokens.clear();
    }
    return *this;
}

bool PagedKvSequence::appendToken(std::int32_t tokenId) noexcept {
    if (_allocator == nullptr) return false;

    const std::size_t blkSz    = _allocator->blockSize();
    const std::size_t used     = _tokens.size();
    const bool        atBoundary = (used == _blockTable.size() * blkSz);

    if (atBoundary) {
        // Need a new block before we can accept the token. Try to
        // allocate — if the pool is exhausted, leave the sequence
        // untouched so the caller can trigger preemption and retry.
        const std::uint32_t id = _allocator->allocate();
        if (id == PagedKvBlockAllocator::kInvalidBlock) {
            return false;
        }
        _blockTable.push_back(id);
    }

    _tokens.push_back(tokenId);

    // Did this append just fill the current block?
    if (_tokens.size() % blkSz == 0) {
        assert(!_blockTable.empty());
        const std::size_t blockIdx = _blockTable.size() - 1;
        const std::uint64_t parentHash = (blockIdx == 0)
            ? kFnvBasis
            : _allocator->hashOf(_blockTable[blockIdx - 1]);
        const std::int32_t* firstTok = _tokens.data() + blockIdx * blkSz;
        const std::uint64_t hash = fnv1aChain(parentHash, firstTok, blkSz);
        _allocator->setHash(_blockTable[blockIdx], hash);
    }

    return true;
}

void PagedKvSequence::reset() noexcept {
    releaseAllBlocks();
    _blockTable.clear();
    _tokens.clear();
}

void PagedKvSequence::releaseAllBlocks() noexcept {
    if (_allocator == nullptr) return;
    for (const std::uint32_t id : _blockTable) {
        _allocator->release(id);
    }
}

std::uint64_t PagedKvSequence::fnv1aChain(std::uint64_t       seed,
                                          const std::int32_t* data,
                                          std::size_t         count) noexcept {
    constexpr std::uint64_t prime = 0x100000001b3ULL;
    std::uint64_t hash = seed;
    for (std::size_t i = 0; i < count; ++i) {
        const auto v = static_cast<std::uint32_t>(data[i]);
        for (int b = 0; b < 4; ++b) {
            const auto byte = static_cast<std::uint8_t>((v >> (b * 8)) & 0xFFU);
            hash ^= byte;
            hash *= prime;
        }
    }
    return hash;
}

} // namespace mimirmind::runtime::serving