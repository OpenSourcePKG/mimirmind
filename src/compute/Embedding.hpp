#pragma once

#include "core/gguf/GgufTypes.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace mimirmind::compute {

/**
 * Look up `tokenIds.size()` rows from the token-embedding table.
 *
 * The table is stored in GGUF's natural layout: dim[0] = embedding length
 * (d_model), dim[1] = vocab size. So token T's embedding starts at
 * element offset T * d_model in the source. For quantised types this
 * requires d_model to be a multiple of the type's block size (true for
 * all production Llama/Qwen/Gemma vocabs where d_model is a multiple of
 * 256).
 *
 * `dst` must be a contiguous F32 buffer of `tokenIds.size() * d_model`
 * elements. Negative token IDs and IDs ≥ vocab_size are filled with
 * zeroes — the alternative is throwing mid-sequence, which is worse for
 * a forward pass.
 */
void embeddingLookup(model::GgmlType                weightType,
                     const void*                    weightData,
                     std::size_t                    d_model,
                     std::size_t                    vocab_size,
                     std::span<const std::int32_t>  tokenIds,
                     float*                         dst);

} // namespace mimirmind::compute