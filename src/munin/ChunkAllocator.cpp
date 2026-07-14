#include "munin/ChunkAllocator.hpp"

#include "core/log/Log.hpp"

#include <level_zero/ze_api.h>

#include <stdexcept>
#include <string>

namespace mimirmind::munin {

namespace {

constexpr std::size_t kMinChunkBytes = 4096;

[[nodiscard]] std::uint64_t alignUp(std::uint64_t value, std::uint64_t align) noexcept {
    if (align <= 1) {
        return value;
    }
    return (value + align - 1) & ~(align - 1);
}

} // namespace

ChunkAllocator::ChunkAllocator(::mimirmind::core::l0::L0Context& ctx,
                               std::size_t                       chunkBytes)
    : _ctx(ctx), _chunkBytes(chunkBytes) {
    if (_chunkBytes < kMinChunkBytes) {
        throw std::runtime_error{
            "ChunkAllocator: chunkBytes must be at least " +
            std::to_string(kMinChunkBytes) + " (got " +
            std::to_string(_chunkBytes) + ")"};
    }
    if ((_chunkBytes % kMinChunkBytes) != 0) {
        throw std::runtime_error{
            "ChunkAllocator: chunkBytes must be a multiple of " +
            std::to_string(kMinChunkBytes) + " (got " +
            std::to_string(_chunkBytes) + ")"};
    }
}

ChunkAllocator::~ChunkAllocator() {
    for (const auto& c : _chunks) {
        if (c.base == nullptr) {
            continue;
        }
        const ze_result_t r = ::zeMemFree(_ctx.context(), c.base);
        if (r != ZE_RESULT_SUCCESS) {
            MM_LOG_WARN("munin",
                        "ChunkAllocator: zeMemFree({}) -> {} (0x{:x})",
                        c.base,
                        ::mimirmind::core::l0::L0Context::resultToString(r),
                        static_cast<unsigned>(r));
        }
    }
}

void ChunkAllocator::openChunk() {
    ze_host_mem_alloc_desc_t hostDesc{};
    hostDesc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
    hostDesc.flags = 0;

    void*             base = nullptr;
    const ze_result_t r    = ::zeMemAllocHost(
        _ctx.context(), &hostDesc, _chunkBytes, kDefaultAlignment, &base);

    if (r != ZE_RESULT_SUCCESS || base == nullptr) {
        throw ::mimirmind::core::l0::L0Error{
            "ChunkAllocator: zeMemAllocHost(chunkBytes=" +
                std::to_string(_chunkBytes) + ")",
            r};
    }

    _chunks.push_back(Chunk{base, 0});
    MM_LOG_INFO("munin",
                "ChunkAllocator: opened chunk #{} base={} size={} bytes",
                _chunks.size() - 1, base, _chunkBytes);
}

ChunkAllocator::Layout
ChunkAllocator::layoutInsideChunk(std::uint64_t currentUsed,
                                  std::size_t   bytes,
                                  std::size_t   align,
                                  std::size_t   chunkBytes) noexcept {
    const std::uint64_t effectiveAlign =
        (align < 1) ? std::uint64_t{1} : static_cast<std::uint64_t>(align);
    const std::uint64_t offset = alignUp(currentUsed, effectiveAlign);
    if (offset + static_cast<std::uint64_t>(bytes) > chunkBytes) {
        return Layout{true, 0};
    }
    return Layout{false, offset};
}

ChunkAllocator::Allocation
ChunkAllocator::allocate(std::size_t bytes, std::size_t align) {
    if (bytes == 0) {
        throw std::runtime_error{"ChunkAllocator::allocate: bytes == 0"};
    }
    if (bytes > _chunkBytes) {
        throw std::runtime_error{
            "ChunkAllocator::allocate: request " + std::to_string(bytes) +
            " exceeds chunkBytes " + std::to_string(_chunkBytes) +
            " — raise runtime.muninChunkBytes or split the tensor"};
    }

    const std::size_t effectiveAlign = align < 1 ? kDefaultAlignment : align;

    if (_chunks.empty()) {
        openChunk();
    }

    Chunk* current = &_chunks.back();
    Layout plan    = layoutInsideChunk(current->used, bytes, effectiveAlign,
                                       _chunkBytes);
    if (plan.needsNewChunk) {
        openChunk();
        current = &_chunks.back();
        plan    = layoutInsideChunk(current->used, bytes, effectiveAlign,
                                    _chunkBytes);
        if (plan.needsNewChunk) {
            // Fresh chunk cannot fit — only reachable if bytes > chunkBytes,
            // which we already refused. Defensive assertion in error form.
            throw std::runtime_error{
                "ChunkAllocator: fresh chunk cannot fit request " +
                std::to_string(bytes) + " after align " +
                std::to_string(effectiveAlign) + " — invariant violated"};
        }
    }

    Allocation out{};
    out.chunkIndex  = static_cast<std::uint32_t>(_chunks.size() - 1);
    out.chunkOffset = plan.offset;
    out.ptr         = static_cast<std::byte*>(current->base) + plan.offset;

    current->used = plan.offset + bytes;
    _bytesUsed += bytes;

    return out;
}

} // namespace mimirmind::munin