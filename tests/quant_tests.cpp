// Pure-CPU unit tests for compute::QuantType implementations.
//
// Built as a standalone `quant_tests` binary. No GPU, no Level Zero,
// no model file required — runs the dequant logic against hand-crafted
// byte blocks with known expected outputs.

#include "TestFramework.hpp"

#include "compute/Dequant.hpp"
#include "compute/QuantType.hpp"
#include "compute/QuantTypeRegistry.hpp"
#include "compute/quant/Bfloat16.hpp"
#include "compute/quant/Float16.hpp"
#include "compute/quant/Float32.hpp"
#include "compute/quant/Q4K.hpp"
#include "compute/quant/Q5K.hpp"
#include "compute/quant/Q6K.hpp"
#include "compute/quant/Q8_0.hpp"
#include "model/GgufTypes.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace {

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

// Little-endian write of a fp16 bit pattern into 2 bytes.
void writeHalfBits(std::uint8_t* dst, std::uint16_t bits) {
    std::memcpy(dst, &bits, sizeof(bits));
}

// fp16 bit patterns we'll reuse across tests.
constexpr std::uint16_t kHalfPosZero    = 0x0000U;
constexpr std::uint16_t kHalfNegZero    = 0x8000U;
constexpr std::uint16_t kHalfOne        = 0x3C00U;  // 1.0
constexpr std::uint16_t kHalfTwo        = 0x4000U;  // 2.0
constexpr std::uint16_t kHalfMinusFive  = 0xC500U;  // -5.0
constexpr std::uint16_t kHalfHalf       = 0x3800U;  // 0.5
constexpr std::uint16_t kHalfPosInf     = 0x7C00U;
constexpr std::uint16_t kHalfNegInf     = 0xFC00U;
constexpr std::uint16_t kHalfQNaN       = 0x7E00U;

} // namespace

// =======================================================================
// Block layout constants
// =======================================================================

TEST(blockLayout_Float32) {
    const auto& qt = mimirmind::compute::quant::Float32::instance();
    EXPECT_EQ(qt.ggmlType(),      mimirmind::model::GgmlType::F32);
    EXPECT_EQ(qt.blockElements(), 1U);
    EXPECT_EQ(qt.blockBytes(),    4U);
    EXPECT_EQ(qt.name(),          std::string_view{"F32"});
    EXPECT_EQ(qt.gpuMatmulModule(), std::string_view{});
}

TEST(blockLayout_Float16) {
    const auto& qt = mimirmind::compute::quant::Float16::instance();
    EXPECT_EQ(qt.ggmlType(),      mimirmind::model::GgmlType::F16);
    EXPECT_EQ(qt.blockElements(), 1U);
    EXPECT_EQ(qt.blockBytes(),    2U);
}

TEST(blockLayout_Bfloat16) {
    const auto& qt = mimirmind::compute::quant::Bfloat16::instance();
    EXPECT_EQ(qt.ggmlType(),      mimirmind::model::GgmlType::BF16);
    EXPECT_EQ(qt.blockElements(), 1U);
    EXPECT_EQ(qt.blockBytes(),    2U);
}

TEST(blockLayout_Q4K) {
    const auto& qt = mimirmind::compute::quant::Q4K::instance();
    EXPECT_EQ(qt.ggmlType(),        mimirmind::model::GgmlType::Q4_K);
    EXPECT_EQ(qt.blockElements(),   256U);
    EXPECT_EQ(qt.blockBytes(),      144U);
    EXPECT_EQ(qt.gpuMatmulModule(), std::string_view{"matmul_q4k_vec"});
}

TEST(blockLayout_Q5K) {
    const auto& qt = mimirmind::compute::quant::Q5K::instance();
    EXPECT_EQ(qt.ggmlType(),        mimirmind::model::GgmlType::Q5_K);
    EXPECT_EQ(qt.blockElements(),   256U);
    EXPECT_EQ(qt.blockBytes(),      176U);
    EXPECT_EQ(qt.gpuMatmulModule(), std::string_view{"matmul_q5k_vec"});
}

TEST(blockLayout_Q6K) {
    const auto& qt = mimirmind::compute::quant::Q6K::instance();
    EXPECT_EQ(qt.ggmlType(),        mimirmind::model::GgmlType::Q6_K);
    EXPECT_EQ(qt.blockElements(),   256U);
    EXPECT_EQ(qt.blockBytes(),      210U);
    EXPECT_EQ(qt.gpuMatmulModule(), std::string_view{"matmul_q6k_vec"});
}

TEST(blockLayout_Q8_0) {
    const auto& qt = mimirmind::compute::quant::Q8_0::instance();
    EXPECT_EQ(qt.ggmlType(),        mimirmind::model::GgmlType::Q8_0);
    EXPECT_EQ(qt.blockElements(),   32U);
    EXPECT_EQ(qt.blockBytes(),      34U);
    EXPECT_EQ(qt.gpuMatmulModule(), std::string_view{"matmul_q8_0_vec"});
}

// =======================================================================
// Registry lookup
// =======================================================================

TEST(registry_supportedTypes) {
    using namespace mimirmind::model;
    using namespace mimirmind::compute;
    EXPECT_TRUE(quantType(GgmlType::F32)  != nullptr);
    EXPECT_TRUE(quantType(GgmlType::F16)  != nullptr);
    EXPECT_TRUE(quantType(GgmlType::BF16) != nullptr);
    EXPECT_TRUE(quantType(GgmlType::Q4_K) != nullptr);
    EXPECT_TRUE(quantType(GgmlType::Q5_K) != nullptr);
    EXPECT_TRUE(quantType(GgmlType::Q6_K) != nullptr);
    EXPECT_TRUE(quantType(GgmlType::Q8_0) != nullptr);
}

TEST(registry_unsupportedTypes) {
    using namespace mimirmind::model;
    using namespace mimirmind::compute;
    EXPECT_TRUE(quantType(GgmlType::Q2_K)    == nullptr);
    EXPECT_TRUE(quantType(GgmlType::Q3_K)    == nullptr);
    EXPECT_TRUE(quantType(GgmlType::Q4_0)    == nullptr);
    EXPECT_TRUE(quantType(GgmlType::Unknown) == nullptr);
}

TEST(registry_allQuantTypes_size) {
    const auto all = mimirmind::compute::allQuantTypes();
    EXPECT_EQ(all.size(), 7U);
    for (const auto* qt : all) {
        EXPECT_TRUE(qt != nullptr);
    }
}

// =======================================================================
// halfToFloat / bf16ToFloat
// =======================================================================

TEST(halfToFloat_specialValues) {
    using mimirmind::compute::halfToFloat;
    EXPECT_NEAR(halfToFloat(kHalfPosZero),   0.0F, 0.0F);
    EXPECT_NEAR(halfToFloat(kHalfOne),       1.0F, 0.0F);
    EXPECT_NEAR(halfToFloat(kHalfTwo),       2.0F, 0.0F);
    EXPECT_NEAR(halfToFloat(kHalfMinusFive), -5.0F, 0.0F);
    EXPECT_NEAR(halfToFloat(kHalfHalf),      0.5F, 0.0F);

    EXPECT_TRUE(std::signbit(halfToFloat(kHalfNegZero)));
    EXPECT_TRUE(std::isinf(halfToFloat(kHalfPosInf))
                && halfToFloat(kHalfPosInf) > 0.0F);
    EXPECT_TRUE(std::isinf(halfToFloat(kHalfNegInf))
                && halfToFloat(kHalfNegInf) < 0.0F);
    EXPECT_TRUE(std::isnan(halfToFloat(kHalfQNaN)));
}

TEST(halfToFloat_subnormal) {
    using mimirmind::compute::halfToFloat;
    // Smallest positive subnormal half: 0x0001 = 2^-24 ≈ 5.96e-8
    const float v = halfToFloat(0x0001U);
    EXPECT_NEAR(v, std::ldexp(1.0F, -24), 1e-12F);
}

TEST(bf16ToFloat_specialValues) {
    using mimirmind::compute::bf16ToFloat;
    EXPECT_NEAR(bf16ToFloat(0x0000U), 0.0F, 0.0F);
    EXPECT_NEAR(bf16ToFloat(0x3F80U), 1.0F, 0.0F);   // 1.0
    EXPECT_NEAR(bf16ToFloat(0x4000U), 2.0F, 0.0F);   // 2.0
    EXPECT_NEAR(bf16ToFloat(0xC000U), -2.0F, 0.0F);  // -2.0
    EXPECT_NEAR(bf16ToFloat(0x4040U), 3.0F, 0.0F);   // 3.0
}

// =======================================================================
// Float32 dequant — passthrough
// =======================================================================

TEST(float32_passthrough) {
    const std::array<float, 5> src{1.0F, -2.5F, 0.0F, 3.14F, -1e7F};
    std::array<float, 5>       dst{};

    mimirmind::compute::quant::Float32::instance()
        .dequantToF32(src.data(), src.size(), dst.data());

    for (std::size_t i = 0; i < src.size(); ++i) {
        EXPECT_NEAR(dst[i], src[i], 0.0F);
    }
}

// =======================================================================
// Float16 dequant — array of known halfs
// =======================================================================

TEST(float16_array) {
    const std::array<std::uint16_t, 5> halfs{
        kHalfPosZero, kHalfOne, kHalfTwo, kHalfMinusFive, kHalfHalf};
    const std::array<float, 5> expected{0.0F, 1.0F, 2.0F, -5.0F, 0.5F};
    std::array<float, 5> dst{};

    mimirmind::compute::quant::Float16::instance()
        .dequantToF32(halfs.data(), halfs.size(), dst.data());

    for (std::size_t i = 0; i < halfs.size(); ++i) {
        EXPECT_NEAR(dst[i], expected[i], 0.0F);
    }
}

// =======================================================================
// Bfloat16 dequant — array of known bf16s
// =======================================================================

TEST(bfloat16_array) {
    const std::array<std::uint16_t, 4> bf16s{0x0000U, 0x3F80U, 0xC000U, 0x4040U};
    const std::array<float, 4>         expected{0.0F, 1.0F, -2.0F, 3.0F};
    std::array<float, 4> dst{};

    mimirmind::compute::quant::Bfloat16::instance()
        .dequantToF32(bf16s.data(), bf16s.size(), dst.data());

    for (std::size_t i = 0; i < bf16s.size(); ++i) {
        EXPECT_NEAR(dst[i], expected[i], 0.0F);
    }
}

// =======================================================================
// Q8_0 — hand-crafted block
// =======================================================================

// Block layout (34 bytes): fp16 d + 32 signed int8 quants.
// Single block of 32 elements, value[i] = d * qs[i].
TEST(q8_0_singleBlock) {
    std::array<std::uint8_t, 34> block{};
    writeHalfBits(block.data(), kHalfHalf);  // d = 0.5

    // qs[i] = 0, 1, -1, 2, -2, ..., 16, -16
    const std::array<std::int8_t, 32> qs{
         0,  1, -1,  2, -2,  3, -3,  4,
        -4,  5, -5,  6, -6,  7, -7,  8,
        -8,  9, -9, 10,-10, 11,-11, 12,
       -12, 13,-13, 14,-14, 15,-15, 16,
    };
    std::memcpy(block.data() + 2, qs.data(), 32);

    std::array<float, 32> dst{};
    mimirmind::compute::quant::Q8_0::instance()
        .dequantToF32(block.data(), 32, dst.data());

    for (std::size_t i = 0; i < 32; ++i) {
        const float expected = 0.5F * static_cast<float>(qs[i]);
        EXPECT_NEAR(dst[i], expected, 0.0F);
    }
}

// Two-block test verifies block stride (34 bytes) is correctly walked.
TEST(q8_0_twoBlocks) {
    std::array<std::uint8_t, 68> blocks{};
    // Block 0: d=1.0, qs[0]=10, rest=0  → value[0]=10, rest=0
    writeHalfBits(blocks.data(), kHalfOne);
    blocks[2] = static_cast<std::uint8_t>(10);  // signed 10
    // Block 1: d=2.0, qs[0]=5, rest=0 → value[32]=10, rest=0
    writeHalfBits(blocks.data() + 34, kHalfTwo);
    blocks[34 + 2] = static_cast<std::uint8_t>(5);

    std::array<float, 64> dst{};
    mimirmind::compute::quant::Q8_0::instance()
        .dequantToF32(blocks.data(), 64, dst.data());

    EXPECT_NEAR(dst[0],  10.0F, 0.0F);
    EXPECT_NEAR(dst[1],  0.0F,  0.0F);
    EXPECT_NEAR(dst[32], 10.0F, 0.0F);
    EXPECT_NEAR(dst[33], 0.0F,  0.0F);
}

TEST(q8_0_misalignedNelementsThrows) {
    std::array<std::uint8_t, 34> block{};
    std::array<float, 17>        dst{};
    try {
        mimirmind::compute::quant::Q8_0::instance()
            .dequantToF32(block.data(), 17, dst.data());
        EXPECT_TRUE(false && "expected throw");
    } catch (const std::runtime_error&) {
        // expected
    }
}

// =======================================================================
// Q8_0 reordered row layout — M8.K foundation for coalesced Xe loads.
// See kernels/matmul_q8_0_vec.cl and Q8_0.hpp for the layout contract.
// =======================================================================

// Explicit-layout single-block check: after reorder, byte 0-1 hold the
// scale and bytes 2-33 hold the quants (which happens to match the
// native single-block layout — the test verifies the reorder is a
// no-op copy for nBlocks=1 and that the accessor offsets are right).
TEST(q8_0_reorder_singleBlock_layout) {
    std::array<std::uint8_t, 34> native{};
    writeHalfBits(native.data(), kHalfHalf);  // d = 0.5
    for (std::size_t l = 0; l < 32; ++l) {
        native[2 + l] = static_cast<std::uint8_t>(l);
    }

    std::array<std::uint8_t, 34> reordered{};
    mimirmind::compute::quant::Q8_0::reorderRow(
        native.data(), 32, reordered.data());

    // Scales region: bytes [0..2).
    EXPECT_EQ(reordered[0], native[0]);
    EXPECT_EQ(reordered[1], native[1]);
    // Quants region: bytes [2..34).
    for (std::size_t l = 0; l < 32; ++l) {
        EXPECT_EQ(reordered[2 + l], native[2 + l]);
    }
}

// Two-block check exercises the actual reorder split: the second block's
// scale must land right after the first block's scale (offset 2), and
// the quants for both blocks must live in a single contiguous 64-byte
// run starting at offset 4.
TEST(q8_0_reorder_twoBlocks_layout) {
    std::array<std::uint8_t, 68> native{};
    // Block 0: d = 1.0, qs[l] = l
    writeHalfBits(native.data(), kHalfOne);
    for (std::size_t l = 0; l < 32; ++l) {
        native[2 + l] = static_cast<std::uint8_t>(l);
    }
    // Block 1: d = 2.0, qs[l] = 32 + l
    writeHalfBits(native.data() + 34, kHalfTwo);
    for (std::size_t l = 0; l < 32; ++l) {
        native[34 + 2 + l] = static_cast<std::uint8_t>(32 + l);
    }

    std::array<std::uint8_t, 68> reordered{};
    mimirmind::compute::quant::Q8_0::reorderRow(
        native.data(), 64, reordered.data());

    // Reordered layout: [d_0 (2)][d_1 (2)][qs_0 (32)][qs_1 (32)]
    EXPECT_EQ(reordered[0], native[0]);        // d_0 low
    EXPECT_EQ(reordered[1], native[1]);        // d_0 high
    EXPECT_EQ(reordered[2], native[34]);       // d_1 low (was at 34)
    EXPECT_EQ(reordered[3], native[35]);       // d_1 high
    // qs_0 lives at reordered[4..36)
    for (std::size_t l = 0; l < 32; ++l) {
        EXPECT_EQ(reordered[4 + l], native[2 + l]);
    }
    // qs_1 lives at reordered[36..68)
    for (std::size_t l = 0; l < 32; ++l) {
        EXPECT_EQ(reordered[36 + l], native[34 + 2 + l]);
    }
}

// Round-trip parity across a realistic multi-block row (K=2816 matches
// the E4B FusedQkvWeights hot-path row length).
TEST(q8_0_reorder_roundtrip_K2816) {
    constexpr std::size_t K       = 2816;
    constexpr std::size_t nblocks = K / 32;
    constexpr std::size_t bytes   = nblocks * 34;

    // Deterministic pseudo-native data: fp16 scale in low bits varies
    // per block, quants cover the full signed int8 range.
    std::vector<std::uint8_t> native(bytes);
    for (std::size_t b = 0; b < nblocks; ++b) {
        std::uint8_t* block = native.data() + b * 34;
        const std::uint16_t half =
            static_cast<std::uint16_t>(0x3800U + (b * 17U));  // ~0.5..
        writeHalfBits(block, half);
        for (std::size_t l = 0; l < 32; ++l) {
            block[2 + l] = static_cast<std::uint8_t>(
                static_cast<std::int8_t>((b * 7 + l * 11) % 251 - 125));
        }
    }

    std::vector<std::uint8_t> reordered(bytes);
    std::vector<std::uint8_t> restored(bytes);
    mimirmind::compute::quant::Q8_0::reorderRow(
        native.data(), K, reordered.data());
    mimirmind::compute::quant::Q8_0::unreorderRow(
        reordered.data(), K, restored.data());

    for (std::size_t i = 0; i < bytes; ++i) {
        if (native[i] != restored[i]) {
            EXPECT_TRUE(false && "reorder round-trip mismatch");
            return;
        }
    }
}

// Semantic parity: dequantising a reordered row must produce the same
// f32 values as dequantising the corresponding native row.
TEST(q8_0_reorder_dequantParity_K2816) {
    constexpr std::size_t K       = 2816;
    constexpr std::size_t nblocks = K / 32;
    constexpr std::size_t bytes   = nblocks * 34;

    std::vector<std::uint8_t> native(bytes);
    for (std::size_t b = 0; b < nblocks; ++b) {
        std::uint8_t* block = native.data() + b * 34;
        const std::uint16_t half =
            static_cast<std::uint16_t>(0x3800U + (b * 17U));
        writeHalfBits(block, half);
        for (std::size_t l = 0; l < 32; ++l) {
            block[2 + l] = static_cast<std::uint8_t>(
                static_cast<std::int8_t>((b * 7 + l * 11) % 251 - 125));
        }
    }

    std::vector<std::uint8_t> reordered(bytes);
    mimirmind::compute::quant::Q8_0::reorderRow(
        native.data(), K, reordered.data());

    std::vector<float> viaNative(K);
    std::vector<float> viaReordered(K);
    mimirmind::compute::quant::Q8_0::instance()
        .dequantToF32(native.data(), K, viaNative.data());
    mimirmind::compute::quant::Q8_0::dequantRowFromReorderedToF32(
        reordered.data(), K, viaReordered.data());

    for (std::size_t i = 0; i < K; ++i) {
        // Bit-exact: reorder is a pure permutation of bytes, dequant is
        // deterministic, so the two paths must agree exactly.
        EXPECT_NEAR(viaReordered[i], viaNative[i], 0.0F);
    }
}

// End-to-end: quantize a random f32 row, reorder it, dequant from the
// reordered layout, and check that the round-trip stays within the
// per-block Q8_0 quantisation error bound (1 LSB * max abs scale).
TEST(q8_0_reorder_quantRoundtrip_K1024) {
    constexpr std::size_t K       = 1024;
    constexpr std::size_t nblocks = K / 32;
    constexpr std::size_t bytes   = nblocks * 34;

    std::vector<float> src(K);
    for (std::size_t i = 0; i < K; ++i) {
        // Deterministic pseudo-normal signal, magnitude ~[-1, 1].
        src[i] = std::sin(static_cast<float>(i) * 0.13F)
               + 0.25F * std::cos(static_cast<float>(i) * 0.7F);
    }

    std::vector<std::uint8_t> native(bytes);
    mimirmind::compute::quant::Q8_0::quantizeRow(src.data(), K, native.data());

    std::vector<std::uint8_t> reordered(bytes);
    mimirmind::compute::quant::Q8_0::reorderRow(
        native.data(), K, reordered.data());

    std::vector<float> dequantized(K);
    mimirmind::compute::quant::Q8_0::dequantRowFromReorderedToF32(
        reordered.data(), K, dequantized.data());

    // Per-block error bound: |round-off| <= 0.5 * scale, and scale =
    // max|src|/127 per block. A comfortable 1e-2 relative tolerance is
    // enough for this deterministic input.
    for (std::size_t i = 0; i < K; ++i) {
        EXPECT_NEAR(dequantized[i], src[i], 1e-2F);
    }
}

// =======================================================================
// Q6_K — hand-crafted blocks
// =======================================================================

// All zeros except d=1.0, scales=1: every quant in [-32..31] is 0-32=-32,
// every scale is 1, so all 256 outputs = -32.
TEST(q6k_allZerosBits_d1_sc1) {
    std::array<std::uint8_t, 210> block{};
    // ql[0..127] = 0, qh[128..191] = 0
    for (std::size_t i = 192; i < 208; ++i) {
        block[i] = 1;                          // scales[i-192] = 1
    }
    writeHalfBits(block.data() + 208, kHalfOne);   // d = 1.0

    std::array<float, 256> dst{};
    mimirmind::compute::quant::Q6K::instance()
        .dequantToF32(block.data(), 256, dst.data());

    for (std::size_t i = 0; i < 256; ++i) {
        EXPECT_NEAR(dst[i], -32.0F, 0.0F);
    }
}

// All ones bits in ql + qh: every quant is 0x3F-32 = 31. d=1, sc=1 → all 31.
TEST(q6k_allOnesBits_d1_sc1) {
    std::array<std::uint8_t, 210> block{};
    for (std::size_t i = 0; i < 128; ++i) block[i] = 0xFFU;          // ql
    for (std::size_t i = 128; i < 192; ++i) block[i] = 0xFFU;        // qh
    for (std::size_t i = 192; i < 208; ++i) block[i] = 1;            // sc
    writeHalfBits(block.data() + 208, kHalfOne);                      // d=1

    std::array<float, 256> dst{};
    mimirmind::compute::quant::Q6K::instance()
        .dequantToF32(block.data(), 256, dst.data());

    for (std::size_t i = 0; i < 256; ++i) {
        EXPECT_NEAR(dst[i], 31.0F, 0.0F);
    }
}

// All zeros bits + d=1.0, but scales vary: sc[0]=1, sc[2]=2, sc[4]=4,
// sc[6]=8 (used for l=0..15 in the first half via is=0). Verifies that
// the s0/s2/s4/s6 mapping in the dequant routes to the right output
// positions y[l + {0,32,64,96}] within the first 128-element half.
TEST(q6k_scaleRouting_firstHalf_is0) {
    std::array<std::uint8_t, 210> block{};
    block[192 + 0] = 1;
    block[192 + 2] = 2;
    block[192 + 4] = 4;
    block[192 + 6] = 8;
    writeHalfBits(block.data() + 208, kHalfOne);

    std::array<float, 256> dst{};
    mimirmind::compute::quant::Q6K::instance()
        .dequantToF32(block.data(), 256, dst.data());

    // q = 0 - 32 = -32 for every position.
    // For l in 0..15 (is=0, first half), the four quant slots use
    // scale[is + {0, 2, 4, 6}] = sc[{0, 2, 4, 6}] = {1, 2, 4, 8}.
    for (std::size_t l = 0; l < 16; ++l) {
        EXPECT_NEAR(dst[l],       -32.0F * 1.0F, 0.0F);  // y[l + 0]
        EXPECT_NEAR(dst[l + 32],  -32.0F * 2.0F, 0.0F);  // y[l + 32]
        EXPECT_NEAR(dst[l + 64],  -32.0F * 4.0F, 0.0F);  // y[l + 64]
        EXPECT_NEAR(dst[l + 96],  -32.0F * 8.0F, 0.0F);  // y[l + 96]
    }
}

// Same routing test but on the second 128-element half: scp = sc + 8.
TEST(q6k_scaleRouting_secondHalf_is0) {
    std::array<std::uint8_t, 210> block{};
    block[192 + 8 + 0] = 1;
    block[192 + 8 + 2] = 2;
    block[192 + 8 + 4] = 4;
    block[192 + 8 + 6] = 8;
    writeHalfBits(block.data() + 208, kHalfOne);

    std::array<float, 256> dst{};
    mimirmind::compute::quant::Q6K::instance()
        .dequantToF32(block.data(), 256, dst.data());

    // Second half starts at output index 128.
    for (std::size_t l = 0; l < 16; ++l) {
        EXPECT_NEAR(dst[128 + l],       -32.0F * 1.0F, 0.0F);
        EXPECT_NEAR(dst[128 + l + 32],  -32.0F * 2.0F, 0.0F);
        EXPECT_NEAR(dst[128 + l + 64],  -32.0F * 4.0F, 0.0F);
        EXPECT_NEAR(dst[128 + l + 96],  -32.0F * 8.0F, 0.0F);
    }
}

TEST(q6k_misalignedNelementsThrows) {
    std::array<std::uint8_t, 210> block{};
    std::array<float, 128>        dst{};
    try {
        mimirmind::compute::quant::Q6K::instance()
            .dequantToF32(block.data(), 128, dst.data());
        EXPECT_TRUE(false && "expected throw");
    } catch (const std::runtime_error&) {
        // expected
    }
}

// =======================================================================
// Q4_K — hand-crafted blocks
// =======================================================================

// Q4_K block (144 bytes): fp16 d (0..1) + fp16 dmin (2..3) + 12-byte
// packed scales (4..15) + 128-byte qs (16..143).
//
// We zero everything and set d=1, dmin=0, scales[0..11]=0. Then every
// sub-block scale/min unpacks to 0 → d1/d2/m1/m2 = 0 → every output 0.
TEST(q4k_allZeros) {
    std::array<std::uint8_t, 144> block{};
    writeHalfBits(block.data() + 0, kHalfOne);   // d = 1.0
    writeHalfBits(block.data() + 2, kHalfPosZero); // dmin = 0.0
    // scales[0..11] = 0, qs[0..127] = 0 already.

    std::array<float, 256> dst{};
    mimirmind::compute::quant::Q4K::instance()
        .dequantToF32(block.data(), 256, dst.data());

    for (std::size_t i = 0; i < 256; ++i) {
        EXPECT_NEAR(dst[i], 0.0F, 0.0F);
    }
}

// Verify scale lower-4 / min lower-4 path (j<4): set scales[0..3] so
// sLo=1, mLo=0; scales[4..7] so mLo*=0; high-nibble (sHi) also 0.
// With d=1, dmin=0 → first 32 outputs use d1=1*1=1, m1=0; values are
// q[l] & 0xF. Set qs[0..31] = 0x0A (low nibble = 10) → values 10.
TEST(q4k_lowerHalfPath) {
    std::array<std::uint8_t, 144> block{};
    writeHalfBits(block.data() + 0, kHalfOne);    // d = 1.0
    writeHalfBits(block.data() + 2, kHalfPosZero);// dmin = 0.0
    block[4]  = 0x01U;  // scales[0] = sLo for j=0 (low 6 bits)
    block[8]  = 0x00U;  // scales[4] = mLo for j=0
    // scales[1..3] / [5..7] left zero → sub-blocks 1..3 still all zero
    for (std::size_t i = 16; i < 16 + 32; ++i) {
        block[i] = 0x0AU;  // qs lower nibble = 10
    }

    std::array<float, 256> dst{};
    mimirmind::compute::quant::Q4K::instance()
        .dequantToF32(block.data(), 256, dst.data());

    // First 32 outputs (lower-nibble of qs[0..31]) → d1 * 10 - m1 = 1*10 - 0 = 10
    for (std::size_t i = 0; i < 32; ++i) {
        EXPECT_NEAR(dst[i], 10.0F, 0.0F);
    }
    // Second 32 outputs use d2 (j=1, scales[1]=0) → all 0.
    for (std::size_t i = 32; i < 64; ++i) {
        EXPECT_NEAR(dst[i], 0.0F, 0.0F);
    }
}

TEST(q4k_misalignedNelementsThrows) {
    std::array<std::uint8_t, 144> block{};
    std::array<float, 100>        dst{};
    try {
        mimirmind::compute::quant::Q4K::instance()
            .dequantToF32(block.data(), 100, dst.data());
        EXPECT_TRUE(false && "expected throw");
    } catch (const std::runtime_error&) {
        // expected
    }
}

// =======================================================================
// Q5_K — hand-crafted blocks
// =======================================================================
//
// Q5_K block (176 bytes): fp16 d (0..1) + fp16 dmin (2..3) + 12-byte
// packed scales (4..15, same layout as Q4_K) + 32-byte qh (16..47) +
// 128-byte qs (48..175).
//
// Per element the dequant unpacks a 5-bit value (lo-nibble in qs plus
// a high bit from qh via a per-sub-super-block mask that shifts by 2
// bits: 0x01/0x02, 0x04/0x08, 0x10/0x20, 0x40/0x80), then applies
//   value = d * scale * q - dmin * min
// per 32-element sub-block.

TEST(q5k_allZeros_yieldsZero) {
    std::array<std::uint8_t, 176> block{};
    std::array<float, 256>        dst{};
    mimirmind::compute::quant::Q5K::instance()
        .dequantToF32(block.data(), 256, dst.data());
    for (std::size_t i = 0; i < 256; ++i) {
        EXPECT_NEAR(dst[i], 0.0F, 0.0F);
    }
}

// d=1, dmin=0, scales all zero → every value = 1 * 0 * q - 0 = 0, no
// matter what qs / qh contain.
TEST(q5k_zeroScales_swallowsQuants) {
    std::array<std::uint8_t, 176> block{};
    writeHalfBits(block.data() + 0, kHalfOne);      // d
    writeHalfBits(block.data() + 2, kHalfPosZero);  // dmin
    for (std::size_t i = 16; i < 48;  ++i) block[i] = 0xFFU;   // qh
    for (std::size_t i = 48; i < 176; ++i) block[i] = 0xFFU;   // qs

    std::array<float, 256> dst{};
    mimirmind::compute::quant::Q5K::instance()
        .dequantToF32(block.data(), 256, dst.data());
    for (std::size_t i = 0; i < 256; ++i) {
        EXPECT_NEAR(dst[i], 0.0F, 0.0F);
    }
}

// d=1, dmin=0, scale[0]=1 (via q[0]=1), qs[0]=1 (low nibble), qh[0]=0
// → dst[0] = 1 * 1 * (1 + 0) - 0 = 1. Every other output is 0 because
// its sub-block scale is still 0.
TEST(q5k_singleLowNibbleContribution) {
    std::array<std::uint8_t, 176> block{};
    writeHalfBits(block.data() + 0, kHalfOne);      // d = 1
    writeHalfBits(block.data() + 2, kHalfPosZero);  // dmin = 0
    block[4]  = 0x01U;      // scales[0] low 6 bits = 1 → scale[0]=1, min[0]=0
    block[48] = 0x01U;      // qs[0] low nibble = 1

    std::array<float, 256> dst{};
    mimirmind::compute::quant::Q5K::instance()
        .dequantToF32(block.data(), 256, dst.data());

    EXPECT_NEAR(dst[0], 1.0F, 0.0F);
    for (std::size_t i = 1; i < 256; ++i) {
        EXPECT_NEAR(dst[i], 0.0F, 0.0F);
    }
}

// qh bit routing: set only bit 0 of qh[0] (u1 mask for j=0 lower half),
// keep everything else zero. With scale[0]=1 and qs[0]=0, expect
// dst[0] = 1 * 1 * (0 + 16) - 0 = 16. Adjacent dst[1..31] stay 0
// because qh[1..31]=0 (the bit is per-quant), and dst[32..63] are the
// UPPER half of the j=0 pair which reads u2=0x02 from qh[0]=0x01,
// no contribution.
TEST(q5k_qhBitRouting_j0_lowerHalf) {
    std::array<std::uint8_t, 176> block{};
    writeHalfBits(block.data() + 0, kHalfOne);      // d = 1
    writeHalfBits(block.data() + 2, kHalfPosZero);  // dmin = 0
    block[4]  = 0x01U;      // scale[0]=1, min[0]=0
    block[16] = 0x01U;      // qh[0] bit 0 set — u1 mask for j=0 lower

    std::array<float, 256> dst{};
    mimirmind::compute::quant::Q5K::instance()
        .dequantToF32(block.data(), 256, dst.data());

    EXPECT_NEAR(dst[0], 16.0F, 0.0F);
    for (std::size_t i = 1; i < 256; ++i) {
        EXPECT_NEAR(dst[i], 0.0F, 0.0F);
    }
}

// All quants max: qs = 0xFF (lo-nibble 0xF, hi-nibble 0xF) + qh = 0xFF
// (every bit for every sub-super-block set). Scales all = 1, d = 1,
// dmin = 0. Expected q for every element = (0xF + 16) = 31 → dst = 31
// everywhere.
//
// Scales-all-1 packing (12 bytes):
//   q[0..3] = 0x01  → scale[0..3] = 1 (low 6 bits), min[0..3] = 0
//   q[4..7] = 0x00  → mins for j=0..3 stay 0
//   q[8..11]= 0x01  → scale[4..7] low 4 bits = 1, spillover bits from
//                     q[0..3] top 2 bits are 0, so scale[4..7] = 1;
//                     mins[4..7] = 0
TEST(q5k_allFives_allOnesScales) {
    std::array<std::uint8_t, 176> block{};
    writeHalfBits(block.data() + 0, kHalfOne);      // d = 1
    writeHalfBits(block.data() + 2, kHalfPosZero);  // dmin = 0
    for (std::size_t j = 0; j < 4; ++j) block[4 + j]     = 0x01U;
    for (std::size_t j = 0; j < 4; ++j) block[4 + 8 + j] = 0x01U;
    for (std::size_t i = 16; i < 48;  ++i) block[i] = 0xFFU;   // qh
    for (std::size_t i = 48; i < 176; ++i) block[i] = 0xFFU;   // qs

    std::array<float, 256> dst{};
    mimirmind::compute::quant::Q5K::instance()
        .dequantToF32(block.data(), 256, dst.data());
    for (std::size_t i = 0; i < 256; ++i) {
        EXPECT_NEAR(dst[i], 31.0F, 0.0F);
    }
}

TEST(q5k_misalignedNelementsThrows) {
    std::array<std::uint8_t, 176> block{};
    std::array<float, 100>        dst{};
    try {
        mimirmind::compute::quant::Q5K::instance()
            .dequantToF32(block.data(), 100, dst.data());
        EXPECT_TRUE(false && "expected throw");
    } catch (const std::runtime_error&) {
        // expected
    }
}

// =======================================================================
// Top-level dequantToF32 dispatch
// =======================================================================

TEST(dispatch_routesToCorrectType) {
    // Smoke-check the free dequantToF32 free function still routes via
    // the registry. Same input that q8_0_singleBlock used; expect same
    // result.
    std::array<std::uint8_t, 34> block{};
    writeHalfBits(block.data(), kHalfHalf);
    block[2] = static_cast<std::uint8_t>(10);

    std::array<float, 32> dst{};
    mimirmind::compute::dequantToF32(mimirmind::model::GgmlType::Q8_0,
                                     block.data(), 32, dst.data());
    EXPECT_NEAR(dst[0], 5.0F, 0.0F);  // d=0.5, qs[0]=10
}

TEST(dispatch_unsupportedTypeThrows) {
    std::array<float, 1> dst{};
    try {
        mimirmind::compute::dequantToF32(mimirmind::model::GgmlType::Q3_K,
                                         dst.data(), 1, dst.data());
        EXPECT_TRUE(false && "expected throw");
    } catch (const std::runtime_error&) {
        // expected
    }
}

int main() {
    return mm::test::run();
}