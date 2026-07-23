// Pure-CPU unit tests for core::modelopt::ModelOptQuant.
//
// Built as a standalone `modelopt_tests` binary. No GPU, no CUDA, no
// model file required — locks the ModelOpt/NVFP4/FP8 scheme descriptors
// and the hf_quant_config string mapping against the shapes verified on
// the on-disk safetensors index of nvidia/Qwen3.6-35B-A3B-NVFP4.

#include "TestFramework.hpp"

#include "core/modelopt/BlockScaleSwizzle.hpp"
#include "core/modelopt/NvFp4Reference.hpp"
#include "core/modelopt/HfQuantConfig.hpp"
#include "core/modelopt/ModelOptQuant.hpp"
#include "core/modelopt/ModelOptWeightLayout.hpp"
#include "core/safetensors/SafetensorsHeader.hpp"

#include <cmath>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace mo = mimirmind::core::modelopt;
namespace st = mimirmind::core::safetensors;

namespace {

// True iff calling `fn` throws std::runtime_error.
bool threw(const std::function<void()>& fn) {
    try {
        fn();
    } catch (const std::runtime_error&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

// Hand-build a SafetensorsTensor with a self-consistent byte size.
st::SafetensorsTensor mkTensor(const std::string&                name,
                               st::SafetensorsDtype              dtype,
                               const std::vector<std::uint64_t>& shape) {
    st::SafetensorsTensor t;
    t.name      = name;
    t.dtype     = dtype;
    t.shape     = shape;
    t.nelements = 1;
    for (const auto d : shape) t.nelements *= d;
    t.nbytes    = static_cast<std::size_t>(t.nelements) * st::dtypeWidth(dtype);
    t.dataBegin = 0;
    t.dataEnd   = t.nbytes;
    return t;
}

} // namespace

// =======================================================================
// hf_quant_config quant_algo string mapping
// =======================================================================

TEST(schemeFromQuantAlgo_recognised) {
    EXPECT_TRUE(mo::schemeFromQuantAlgo("W4A16_NVFP4").has_value());
    EXPECT_EQ(mo::schemeFromQuantAlgo("W4A16_NVFP4").value(),
              mo::ModelOptQuantScheme::NVFP4_E2M1_BLK16);

    EXPECT_TRUE(mo::schemeFromQuantAlgo("FP8").has_value());
    EXPECT_EQ(mo::schemeFromQuantAlgo("FP8").value(),
              mo::ModelOptQuantScheme::FP8_E4M3);
}

TEST(schemeFromQuantAlgo_rejected) {
    // MIXED_PRECISION is the top-level marker and never a leaf-module algo.
    EXPECT_TRUE(!mo::schemeFromQuantAlgo("MIXED_PRECISION").has_value());
    EXPECT_TRUE(!mo::schemeFromQuantAlgo("W8A8").has_value());
    EXPECT_TRUE(!mo::schemeFromQuantAlgo("").has_value());
    EXPECT_TRUE(!mo::schemeFromQuantAlgo("nvfp4").has_value()); // case-sensitive
}

// =======================================================================
// Scheme descriptors
// =======================================================================

TEST(schemeInfo_nvfp4) {
    const auto& i = mo::schemeInfo(mo::ModelOptQuantScheme::NVFP4_E2M1_BLK16);
    EXPECT_EQ(i.name,                std::string_view{"W4A16_NVFP4"});
    EXPECT_EQ(i.weightDtype,         mo::SafetensorsDtype::U8);
    EXPECT_EQ(i.weightPackFactor,    static_cast<std::uint8_t>(2));
    EXPECT_EQ(i.blockScaleGroupSize, static_cast<std::uint16_t>(16));
    EXPECT_EQ(i.blockScaleDtype,     mo::SafetensorsDtype::F8_E4M3);
    EXPECT_TRUE(i.hasBlockScale);
    EXPECT_TRUE(!i.hasTensorWeightScale);
    EXPECT_TRUE(i.hasGlobalScale);   // weight_scale_2
    EXPECT_TRUE(!i.hasInputScale);   // W4A16: bf16 activations, no act quant
}

TEST(schemeInfo_fp8) {
    const auto& i = mo::schemeInfo(mo::ModelOptQuantScheme::FP8_E4M3);
    EXPECT_EQ(i.name,                std::string_view{"FP8"});
    EXPECT_EQ(i.weightDtype,         mo::SafetensorsDtype::F8_E4M3);
    EXPECT_EQ(i.weightPackFactor,    static_cast<std::uint8_t>(1));
    EXPECT_EQ(i.blockScaleGroupSize, static_cast<std::uint16_t>(0));
    EXPECT_TRUE(!i.hasBlockScale);
    EXPECT_TRUE(i.hasTensorWeightScale);
    EXPECT_TRUE(!i.hasGlobalScale);
    EXPECT_TRUE(i.hasInputScale);    // W8A8: per-tensor activation scale
}

// =======================================================================
// Byte-size math against real on-disk shapes
// =======================================================================

TEST(packedRowBytes_nvfp4_real_shapes) {
    constexpr auto NVFP4 = mo::ModelOptQuantScheme::NVFP4_E2M1_BLK16;
    // gate_proj / up_proj: in = 2048  → weight U8 [512, 1024]
    EXPECT_EQ(mo::packedRowBytes(NVFP4, 2048U), std::size_t{1024});
    EXPECT_EQ(mo::blockScaleCols(NVFP4, 2048U), std::size_t{128});
    // down_proj: in = 512  → weight U8 [2048, 256], scale [2048, 32]
    EXPECT_EQ(mo::packedRowBytes(NVFP4, 512U), std::size_t{256});
    EXPECT_EQ(mo::blockScaleCols(NVFP4, 512U), std::size_t{32});
    // lm_head: in = 2048 → weight U8 [248320, 1024], scale [248320, 128]
    EXPECT_EQ(mo::packedRowBytes(NVFP4, 2048U), std::size_t{1024});
}

TEST(packedRowBytes_fp8_real_shapes) {
    constexpr auto FP8 = mo::ModelOptQuantScheme::FP8_E4M3;
    // q_proj: in = 2048 → weight F8_E4M3 [8192, 2048], unpacked, no block scale
    EXPECT_EQ(mo::packedRowBytes(FP8, 2048U), std::size_t{2048});
    EXPECT_EQ(mo::blockScaleCols(FP8, 2048U), std::size_t{0});
    // out_proj: in = 4096 → [2048, 4096]
    EXPECT_EQ(mo::packedRowBytes(FP8, 4096U), std::size_t{4096});
}

TEST(packedRowBytes_bad_divisor) {
    constexpr auto NVFP4 = mo::ModelOptQuantScheme::NVFP4_E2M1_BLK16;
    // in not a multiple of the group size (16) → 0 (invalid).
    EXPECT_EQ(mo::packedRowBytes(NVFP4, 17U), std::size_t{0});
    EXPECT_EQ(mo::blockScaleCols(NVFP4, 24U), std::size_t{0}); // 24 % 16 != 0
    // odd in-features can't pack two fp4 per byte.
    EXPECT_EQ(mo::packedRowBytes(NVFP4, 15U), std::size_t{0});
}

// =======================================================================
// HfQuantConfig — per-module scheme resolution (mixed checkpoint)
// =======================================================================

namespace {

// Mirrors nvidia/Qwen3.6-35B-A3B-NVFP4's hf_quant_config.json in miniature.
const char* kMixedConfig = R"({
  "producer": {"name": "modelopt", "version": "0.44.0"},
  "quantization": {
    "quant_algo": "MIXED_PRECISION",
    "kv_cache_quant_algo": "FP8",
    "exclude_modules": ["mtp*", "mtp.layers.0*"],
    "quantized_layers": {
      "model.language_model.layers.0.mlp.experts": {"quant_algo": "W4A16_NVFP4", "group_size": 16},
      "model.language_model.layers.0.mlp.shared_expert.gate_proj": {"quant_algo": "W4A16_NVFP4", "group_size": 16},
      "model.language_model.layers.3.self_attn.q_proj": {"quant_algo": "FP8"},
      "model.language_model.layers.0.linear_attn.out_proj": {"quant_algo": "FP8"},
      "lm_head": {"quant_algo": "W4A16_NVFP4", "group_size": 16}
    }
  }
})";

} // namespace

TEST(hfquant_top_level_fields) {
    const auto c = mo::HfQuantConfig::parse(kMixedConfig);
    EXPECT_TRUE(c.isMixed());
    EXPECT_EQ(c.topLevelAlgo(), std::string_view{"MIXED_PRECISION"});
    EXPECT_EQ(c.kvCacheAlgo(),  std::string_view{"FP8"});
    EXPECT_EQ(c.quantizedModules().size(), std::size_t{5});
}

TEST(hfquant_resolves_expert_by_prefix) {
    const auto c = mo::HfQuantConfig::parse(kMixedConfig);
    // A leaf expert weight resolves via the "...mlp.experts" module prefix.
    const auto w = c.resolve("model.language_model.layers.0.mlp.experts.5.gate_proj.weight");
    EXPECT_TRUE(w.has_value());
    EXPECT_EQ(w->scheme,    mo::ModelOptQuantScheme::NVFP4_E2M1_BLK16);
    EXPECT_EQ(w->groupSize, static_cast<std::uint16_t>(16));

    // Its scale sidecars share the module's scheme.
    EXPECT_EQ(c.schemeForTensor(
                  "model.language_model.layers.0.mlp.experts.5.gate_proj.weight_scale").value(),
              mo::ModelOptQuantScheme::NVFP4_E2M1_BLK16);
    EXPECT_EQ(c.schemeForTensor(
                  "model.language_model.layers.0.mlp.experts.200.down_proj.weight_scale_2").value(),
              mo::ModelOptQuantScheme::NVFP4_E2M1_BLK16);
}

TEST(hfquant_resolves_fp8_attention) {
    const auto c = mo::HfQuantConfig::parse(kMixedConfig);
    EXPECT_EQ(c.schemeForTensor(
                  "model.language_model.layers.3.self_attn.q_proj.weight").value(),
              mo::ModelOptQuantScheme::FP8_E4M3);
    EXPECT_EQ(c.schemeForTensor(
                  "model.language_model.layers.0.linear_attn.out_proj.input_scale").value(),
              mo::ModelOptQuantScheme::FP8_E4M3);
    // shared_expert.gate_proj is NVFP4 and must NOT collide with "...experts".
    EXPECT_EQ(c.schemeForTensor(
                  "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight").value(),
              mo::ModelOptQuantScheme::NVFP4_E2M1_BLK16);
}

TEST(hfquant_unquantised_and_excluded) {
    const auto c = mo::HfQuantConfig::parse(kMixedConfig);
    // Router gate is not in quantized_layers -> unquantised (bf16).
    EXPECT_TRUE(!c.resolve("model.language_model.layers.0.mlp.gate.weight").has_value());
    // MTP is excluded by the "mtp*" glob.
    EXPECT_TRUE(c.isExcluded("mtp.fc.weight"));
    EXPECT_TRUE(!c.resolve("mtp.fc.weight").has_value());
    EXPECT_TRUE(c.isExcluded("mtp.layers.0.self_attn.q_proj.weight"));
    // A prefix that only shares a leading substring must not match "...experts".
    EXPECT_TRUE(!c.resolve("model.language_model.layers.1.mlp.experts_typo.weight").has_value());
    // lm_head (a top-level module key) resolves.
    EXPECT_EQ(c.schemeForTensor("lm_head.weight").value(),
              mo::ModelOptQuantScheme::NVFP4_E2M1_BLK16);
}

TEST(hfquant_uniform_fallback) {
    // No quantized_layers, a real top-level scheme -> every non-excluded
    // module uses it.
    const char* uniform = R"({
      "quantization": {
        "quant_algo": "W4A16_NVFP4",
        "group_size": 16,
        "exclude_modules": ["lm_head*"]
      }
    })";
    const auto c = mo::HfQuantConfig::parse(uniform);
    EXPECT_TRUE(!c.isMixed());
    EXPECT_EQ(c.resolve("model.layers.0.mlp.foo.weight").value().scheme,
              mo::ModelOptQuantScheme::NVFP4_E2M1_BLK16);
    EXPECT_EQ(c.resolve("model.layers.0.mlp.foo.weight").value().groupSize,
              static_cast<std::uint16_t>(16));
    EXPECT_TRUE(!c.resolve("lm_head.weight").has_value()); // excluded
}

TEST(hfquant_rejects_malformed) {
    EXPECT_TRUE(threw([] { (void)mo::HfQuantConfig::parse("not json"); }));
    EXPECT_TRUE(threw([] { (void)mo::HfQuantConfig::parse(R"({})"); })); // no quantization
    // unsupported per-module algo
    EXPECT_TRUE(threw([] {
        (void)mo::HfQuantConfig::parse(
            R"({"quantization":{"quant_algo":"MIXED_PRECISION",)"
            R"("quantized_layers":{"m":{"quant_algo":"W8A8"}}}})");
    }));
    // quantized_layers entry without quant_algo
    EXPECT_TRUE(threw([] {
        (void)mo::HfQuantConfig::parse(
            R"({"quantization":{"quantized_layers":{"m":{"group_size":16}}}})");
    }));
}

// =======================================================================
// validateWeightLayout — pure ModelOpt weight assembly / validation
// =======================================================================

TEST(assemble_nvfp4_layout) {
    // Real gate_proj sidecars: weight U8 [512,1024] (in=2048), block scale
    // F8_E4M3 [512,128], global scale F32 scalar.
    const auto w  = mkTensor("w.weight",         st::SafetensorsDtype::U8,      {512, 1024});
    const auto bs = mkTensor("w.weight_scale",   st::SafetensorsDtype::F8_E4M3, {512, 128});
    const auto gs = mkTensor("w.weight_scale_2", st::SafetensorsDtype::F32,     {});

    const auto layout = mo::validateWeightLayout(
        mo::ModelOptQuantScheme::NVFP4_E2M1_BLK16, 16, w, &bs, &gs, nullptr, nullptr);

    EXPECT_EQ(layout.scheme,      mo::ModelOptQuantScheme::NVFP4_E2M1_BLK16);
    EXPECT_EQ(layout.outFeatures, std::uint64_t{512});
    EXPECT_EQ(layout.inFeatures,  std::uint64_t{2048}); // 1024 packed * 2
    EXPECT_EQ(layout.groupSize,   static_cast<std::uint16_t>(16));
}

TEST(assemble_fp8_layout) {
    // q_proj: weight F8_E4M3 [8192,2048] unpacked, F32 weight + input scales.
    const auto w  = mkTensor("q.weight",       st::SafetensorsDtype::F8_E4M3, {8192, 2048});
    const auto ws = mkTensor("q.weight_scale", st::SafetensorsDtype::F32,     {});
    const auto is = mkTensor("q.input_scale",  st::SafetensorsDtype::F32,     {});

    const auto layout = mo::validateWeightLayout(
        mo::ModelOptQuantScheme::FP8_E4M3, 0, w, nullptr, nullptr, &ws, &is);

    EXPECT_EQ(layout.outFeatures, std::uint64_t{8192});
    EXPECT_EQ(layout.inFeatures,  std::uint64_t{2048}); // packFactor 1
    EXPECT_EQ(layout.groupSize,   static_cast<std::uint16_t>(0));
}

TEST(assemble_rejects_inconsistencies) {
    const auto wU8  = mkTensor("w.weight",         st::SafetensorsDtype::U8,      {512, 1024});
    const auto bs   = mkTensor("w.weight_scale",   st::SafetensorsDtype::F8_E4M3, {512, 128});
    const auto gs   = mkTensor("w.weight_scale_2", st::SafetensorsDtype::F32,     {});
    constexpr auto NVFP4 = mo::ModelOptQuantScheme::NVFP4_E2M1_BLK16;
    constexpr auto FP8   = mo::ModelOptQuantScheme::FP8_E4M3;

    // weight dtype mismatch (F32 where U8 expected)
    const auto wF32 = mkTensor("w.weight", st::SafetensorsDtype::F32, {512, 1024});
    EXPECT_TRUE(threw([&] { (void)mo::validateWeightLayout(NVFP4, 16, wF32, &bs, &gs, nullptr, nullptr); }));

    // weight not 2-D
    const auto w1d = mkTensor("w.weight", st::SafetensorsDtype::U8, {512});
    EXPECT_TRUE(threw([&] { (void)mo::validateWeightLayout(NVFP4, 16, w1d, &bs, &gs, nullptr, nullptr); }));

    // missing block scale
    EXPECT_TRUE(threw([&] { (void)mo::validateWeightLayout(NVFP4, 16, wU8, nullptr, &gs, nullptr, nullptr); }));

    // block scale wrong column count (should be 128)
    const auto bsBad = mkTensor("w.weight_scale", st::SafetensorsDtype::F8_E4M3, {512, 64});
    EXPECT_TRUE(threw([&] { (void)mo::validateWeightLayout(NVFP4, 16, wU8, &bsBad, &gs, nullptr, nullptr); }));

    // missing global scale
    EXPECT_TRUE(threw([&] { (void)mo::validateWeightLayout(NVFP4, 16, wU8, &bs, nullptr, nullptr, nullptr); }));

    // global scale not scalar
    const auto gsVec = mkTensor("w.weight_scale_2", st::SafetensorsDtype::F32, {2});
    EXPECT_TRUE(threw([&] { (void)mo::validateWeightLayout(NVFP4, 16, wU8, &bs, &gsVec, nullptr, nullptr); }));

    // config group_size disagrees with the scheme (16)
    EXPECT_TRUE(threw([&] { (void)mo::validateWeightLayout(NVFP4, 32, wU8, &bs, &gs, nullptr, nullptr); }));

    // FP8: missing input scale
    const auto wf  = mkTensor("q.weight",       st::SafetensorsDtype::F8_E4M3, {8192, 2048});
    const auto ws  = mkTensor("q.weight_scale", st::SafetensorsDtype::F32,     {});
    EXPECT_TRUE(threw([&] { (void)mo::validateWeightLayout(FP8, 0, wf, nullptr, nullptr, &ws, nullptr); }));
}

// =======================================================================
// BlockScaleSwizzle — CUTLASS SF layout (verified bit-exact vs cute on GB10)
// =======================================================================

TEST(swizzle_sizes) {
    EXPECT_EQ(mo::swizzledBlockScaleBytes(128, 4),      std::size_t{512});
    EXPECT_EQ(mo::swizzledBlockScaleBytes(256, 8),      std::size_t{2048});
    EXPECT_EQ(mo::swizzledBlockScaleBytes(512, 16),     std::size_t{8192});
    // real lm_head SF: rows=248320, ksf=2048/16=128
    EXPECT_EQ(mo::swizzledBlockScaleBytes(248320, 128), std::size_t{31784960});
    // padding: 100 rows -> 1 tile, 4 ksf -> 1 tile
    EXPECT_EQ(mo::swizzledBlockScaleBytes(100, 4),      std::size_t{512});
}

TEST(swizzle_known_offsets) {
    // Values below are the exact offsets cute's tile_atom_to_shape_SFA
    // returned on GB10 (rows=256, ksf=8 -> mTiles=2, ksfTiles=2).
    const std::uint64_t mT = mo::sfRowTiles(256), kT = mo::sfKTiles(8);
    EXPECT_EQ(mo::swizzledScaleOffset(0,   0, mT, kT), std::size_t{0});
    EXPECT_EQ(mo::swizzledScaleOffset(0,   4, mT, kT), std::size_t{512});  // 2nd K-tile
    EXPECT_EQ(mo::swizzledScaleOffset(1,   1, mT, kT), std::size_t{17});   // m0=1 -> 16, a_s=1
    EXPECT_EQ(mo::swizzledScaleOffset(32,  0, mT, kT), std::size_t{4});    // m1=1 -> stride 4
    EXPECT_EQ(mo::swizzledScaleOffset(128, 0, mT, kT), std::size_t{1024}); // 2nd M-tile
}

TEST(swizzle_roundtrip_and_injective) {
    const std::uint64_t rows = 8, ksf = 8; // 64 unique byte values (< 256)
    std::vector<std::uint8_t> src(rows * ksf);
    for (std::size_t i = 0; i < src.size(); ++i) src[i] = static_cast<std::uint8_t>(i + 1);

    std::vector<std::uint8_t> dst(mo::swizzledBlockScaleBytes(rows, ksf), 0xEE);
    mo::swizzleBlockScale(src.data(), rows, ksf, dst.data());

    const std::uint64_t mT = mo::sfRowTiles(rows), kT = mo::sfKTiles(ksf);
    // every source scale lands at its offset, in bounds, no collisions
    std::vector<int> hitCount(dst.size(), 0);
    for (std::uint64_t m = 0; m < rows; ++m) {
        for (std::uint64_t s = 0; s < ksf; ++s) {
            const std::size_t off = mo::swizzledScaleOffset(m, s, mT, kT);
            EXPECT_TRUE(off < dst.size());
            EXPECT_EQ(dst[off], src[m * ksf + s]);
            ++hitCount[off];
        }
    }
    // injectivity: no offset written twice
    int maxHits = 0;
    for (int h : hitCount) if (h > maxHits) maxHits = h;
    EXPECT_EQ(maxHits, 1);
}

TEST(swizzle_pads_with_zero) {
    // rows=100 padded to 128: the 28 padding rows write nothing, and no
    // source scale collides into their storage, so every un-hit byte is 0.
    const std::uint64_t rows = 100, ksf = 4;
    std::vector<std::uint8_t> src(rows * ksf);
    for (std::size_t i = 0; i < src.size(); ++i) src[i] = static_cast<std::uint8_t>((i % 255) + 1);
    std::vector<std::uint8_t> dst(mo::swizzledBlockScaleBytes(rows, ksf), 0x7);
    mo::swizzleBlockScale(src.data(), rows, ksf, dst.data());

    const std::uint64_t mT = mo::sfRowTiles(rows), kT = mo::sfKTiles(ksf);
    std::vector<bool> written(dst.size(), false);
    for (std::uint64_t m = 0; m < rows; ++m)
        for (std::uint64_t s = 0; s < ksf; ++s)
            written[mo::swizzledScaleOffset(m, s, mT, kT)] = true;
    for (std::size_t i = 0; i < dst.size(); ++i)
        if (!written[i]) EXPECT_EQ(dst[i], std::uint8_t{0});
}

// =======================================================================
// NvFp4Reference — CPU dequant oracle (scalars verified vs CUTLASS on GB10)
// =======================================================================

TEST(e2m1_full_table) {
    // All 16 codes: magnitudes {0,.5,1,1.5,2,3,4,6}, bit 3 = sign.
    const float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    for (std::uint8_t n = 0; n < 16; ++n) {
        const float expect = (n & 0x8u) ? -mag[n & 0x7u] : mag[n & 0x7u];
        EXPECT_EQ(mo::e2m1ToFloat(n), expect);
    }
}

TEST(e4m3_known_values) {
    EXPECT_EQ(mo::e4m3ToFloat(0x00), 0.0f);
    EXPECT_EQ(mo::e4m3ToFloat(0x38), 1.0f);   // e=7,m=0 -> 2^0
    EXPECT_EQ(mo::e4m3ToFloat(0x40), 2.0f);   // e=8,m=0 -> 2^1
    EXPECT_EQ(mo::e4m3ToFloat(0x3C), 1.5f);   // e=7,m=4 -> 1.5
    EXPECT_EQ(mo::e4m3ToFloat(0x7E), 448.0f); // e=15,m=6 -> 2^8*1.75 (max normal)
    EXPECT_EQ(mo::e4m3ToFloat(0xB8), -1.0f);  // sign + 1.0
    EXPECT_EQ(mo::e4m3ToFloat(0x04), 0.0078125f); // subnormal m=4: 2^-6 * 4/8
    EXPECT_TRUE(std::isnan(mo::e4m3ToFloat(0x7F))); // NaN code
}

TEST(dequant_nvfp4_block) {
    // One row, 16 elements (one block). global=0.125, block_scale=2.0 (0x40),
    // nibbles: [+1.0(0x2), -1.5(0xB), +0.5(0x1), +6.0(0x7), rest 0].
    // packed: element 2j low nibble, 2j+1 high nibble.
    std::uint8_t packed[8] = {
        static_cast<std::uint8_t>(0x2 | (0xB << 4)),  // e0=+1.0, e1=-1.5
        static_cast<std::uint8_t>(0x1 | (0x7 << 4)),  // e2=+0.5, e3=+6.0
        0, 0, 0, 0, 0, 0,
    };
    std::uint8_t blockScale[1] = {0x40}; // 2.0
    float out[16];
    mo::dequantNvfp4(packed, blockScale, 0.125f, 1, 16, out);

    // value = 0.125 * 2.0 * e2m1
    EXPECT_EQ(out[0], 0.125f * 2.0f * 1.0f);
    EXPECT_EQ(out[1], 0.125f * 2.0f * -1.5f);
    EXPECT_EQ(out[2], 0.125f * 2.0f * 0.5f);
    EXPECT_EQ(out[3], 0.125f * 2.0f * 6.0f);
    EXPECT_EQ(out[4], 0.0f);
    EXPECT_EQ(out[15], 0.0f);
}

TEST(dequant_fp8_weight) {
    // Unpacked E4M3 weights x per-tensor scale.
    const std::uint8_t w[4] = {0x38, 0x40, 0xB8, 0x00}; // 1.0, 2.0, -1.0, 0
    float out[4];
    mo::dequantFp8(w, 0.5f, 4, out);
    EXPECT_EQ(out[0], 0.5f);   // 0.5 * 1.0
    EXPECT_EQ(out[1], 1.0f);   // 0.5 * 2.0
    EXPECT_EQ(out[2], -0.5f);  // 0.5 * -1.0
    EXPECT_EQ(out[3], 0.0f);
}

int main() {
    return mm::test::run();
}