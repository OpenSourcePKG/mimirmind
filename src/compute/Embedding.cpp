#include "compute/Embedding.hpp"

#include "compute/Dequant.hpp"
#include "core/log/Log.hpp"

#include <cstring>
#include <stdexcept>

namespace mimirmind::compute {

namespace {

std::size_t bytesPerRow(model::GgmlType type, std::size_t d_model) {
    const auto info = model::typeInfo(type);
    if (info.blockSize == 0 || info.typeSize == 0) {
        throw std::runtime_error("embeddingLookup: unknown weight type");
    }
    if (d_model % info.blockSize != 0) {
        throw std::runtime_error(
            "embeddingLookup: d_model not a multiple of the type's block size");
    }
    return (d_model / info.blockSize) * info.typeSize;
}

} // namespace

void embeddingLookup(model::GgmlType                weightType,
                     const void*                    weightData,
                     std::size_t                    d_model,
                     std::size_t                    vocab_size,
                     std::span<const std::int32_t>  tokenIds,
                     float*                         dst) {
    if (weightData == nullptr) {
        throw std::runtime_error("embeddingLookup: null weight pointer");
    }
    if (d_model == 0) {
        throw std::runtime_error("embeddingLookup: d_model is 0");
    }

    const std::size_t rowBytes = bytesPerRow(weightType, d_model);
    const auto* base = static_cast<const std::uint8_t*>(weightData);

    std::size_t oob = 0;
    for (std::size_t i = 0; i < tokenIds.size(); ++i) {
        const std::int32_t id = tokenIds[i];
        float* row = dst + i * d_model;

        if (id < 0 || static_cast<std::size_t>(id) >= vocab_size) {
            std::memset(row, 0, d_model * sizeof(float));
            ++oob;
            continue;
        }

        const std::uint8_t* src = base +
            static_cast<std::size_t>(id) * rowBytes;
        dequantToF32(weightType, src, d_model, row);
    }

    if (oob > 0) {
        MM_LOG_WARN("embed",
                    "{} of {} token id(s) were out of range [0, {}), "
                    "their rows were zero-filled",
                    oob, tokenIds.size(), vocab_size);
    }

    MM_LOG_DEBUG("embed",
                 "lookup done: {} token(s) x {} dims (type={}, row={} bytes)",
                 tokenIds.size(), d_model,
                 model::typeInfo(weightType).name, rowBytes);
}

} // namespace mimirmind::compute