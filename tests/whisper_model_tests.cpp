// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

// Pure-CPU tests for the Whisper config parse + arch gate
// (runtime/audio/WhisperModel). The weight loader itself (WhisperModel::load)
// needs a real checkpoint + a ComputeOps backend and is exercised on-box; the
// config parsing and the isWhisperConfig routing gate are host-testable here.

#include "TestFramework.hpp"

#include "runtime/audio/WhisperModel.hpp"

using namespace mimirmind::runtime::audio;

namespace {

// Trimmed openai/whisper-tiny config.json (the fields the loader reads).
const char* kWhisperTinyConfig = R"JSON({
  "model_type": "whisper",
  "architectures": ["WhisperForConditionalGeneration"],
  "num_mel_bins": 80,
  "d_model": 384,
  "encoder_layers": 4,
  "decoder_layers": 4,
  "encoder_attention_heads": 6,
  "decoder_attention_heads": 6,
  "encoder_ffn_dim": 1536,
  "decoder_ffn_dim": 1536,
  "vocab_size": 51865,
  "max_source_positions": 1500,
  "max_target_positions": 448,
  "bos_token_id": 50257,
  "eos_token_id": 50257,
  "decoder_start_token_id": 50258,
  "pad_token_id": 50257
})JSON";

const char* kRobertaConfig = R"JSON({
  "model_type": "xlm-roberta",
  "architectures": ["XLMRobertaForSequenceClassification"],
  "hidden_size": 1024,
  "num_hidden_layers": 24
})JSON";

} // namespace

TEST(whisper_arch_gate_accepts_whisper) {
    EXPECT_TRUE(isWhisperConfig(kWhisperTinyConfig));
}

TEST(whisper_arch_gate_rejects_non_whisper) {
    EXPECT_TRUE(!isWhisperConfig(kRobertaConfig));
}

TEST(whisper_arch_gate_rejects_garbage) {
    EXPECT_TRUE(!isWhisperConfig("not json at all {{{"));
    EXPECT_TRUE(!isWhisperConfig(""));
}

TEST(whisper_arch_gate_by_architectures_only) {
    // No model_type, but an architectures entry starting with "Whisper".
    const char* cfg = R"JSON({"architectures":["WhisperForAudioClassification"],
        "d_model":384,"encoder_layers":4,"decoder_layers":4,
        "encoder_attention_heads":6,"decoder_attention_heads":6,
        "encoder_ffn_dim":1536,"decoder_ffn_dim":1536,"vocab_size":51865})JSON";
    EXPECT_TRUE(isWhisperConfig(cfg));
}

TEST(whisper_config_parse_tiny) {
    const WhisperConfig c = parseWhisperConfig(kWhisperTinyConfig);
    EXPECT_EQ(c.numMelBins, static_cast<std::size_t>(80));
    EXPECT_EQ(c.dModel, static_cast<std::size_t>(384));
    EXPECT_EQ(c.encoderLayers, static_cast<std::size_t>(4));
    EXPECT_EQ(c.decoderLayers, static_cast<std::size_t>(4));
    EXPECT_EQ(c.encoderHeads, static_cast<std::size_t>(6));
    EXPECT_EQ(c.decoderHeads, static_cast<std::size_t>(6));
    EXPECT_EQ(c.encoderFfn, static_cast<std::size_t>(1536));
    EXPECT_EQ(c.decoderFfn, static_cast<std::size_t>(1536));
    EXPECT_EQ(c.vocab, static_cast<std::size_t>(51865));
    EXPECT_EQ(c.maxSourcePositions, static_cast<std::size_t>(1500));
    EXPECT_EQ(c.maxTargetPositions, static_cast<std::size_t>(448));
    EXPECT_EQ(c.decoderStartTokenId, 50258);
    // head_dim = d_model / heads = 384 / 6 = 64.
    EXPECT_EQ(c.encoderHeadDim(), static_cast<std::size_t>(64));
    EXPECT_EQ(c.decoderHeadDim(), static_cast<std::size_t>(64));
}

TEST(whisper_config_defaults_for_missing_optionals) {
    // Minimal required fields only; optionals must fall back to Whisper defaults.
    const char* cfg = R"JSON({"model_type":"whisper","d_model":512,
        "encoder_layers":6,"decoder_layers":6,"encoder_attention_heads":8,
        "decoder_attention_heads":8,"encoder_ffn_dim":2048,"decoder_ffn_dim":2048,
        "vocab_size":51865})JSON";
    const WhisperConfig c = parseWhisperConfig(cfg);
    EXPECT_EQ(c.numMelBins, static_cast<std::size_t>(80));
    EXPECT_EQ(c.maxSourcePositions, static_cast<std::size_t>(1500));
    EXPECT_EQ(c.maxTargetPositions, static_cast<std::size_t>(448));
    EXPECT_EQ(c.encoderHeadDim(), static_cast<std::size_t>(64));   // 512/8
}
