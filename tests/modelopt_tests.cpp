// Pure-CPU unit tests for core::modelopt::ModelOptQuant.
//
// Built as a standalone `modelopt_tests` binary. No GPU, no CUDA, no
// model file required — locks the ModelOpt/NVFP4/FP8 scheme descriptors
// and the hf_quant_config string mapping against the shapes verified on
// the on-disk safetensors index of nvidia/Qwen3.6-35B-A3B-NVFP4.

#include "TestFramework.hpp"

#include "core/modelopt/HfQuantConfig.hpp"
#include "core/modelopt/ModelOptQuant.hpp"

#include <functional>
#include <stdexcept>
#include <string>

namespace mo = mimirmind::core::modelopt;

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

int main() {
    return mm::test::run();
}