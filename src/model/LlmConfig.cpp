#include "model/LlmConfig.hpp"

#include "model/GgufReader.hpp"
#include "runtime/Log.hpp"

#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>

namespace mimirmind::model {

namespace {

template <typename T>
std::optional<T> asNumeric(const MetadataValue& v) {
    return std::visit([](const auto& x) -> std::optional<T> {
        using U = std::decay_t<decltype(x)>;
        if constexpr (std::is_arithmetic_v<U>) {
            return static_cast<T>(x);
        } else {
            return std::nullopt;
        }
    }, v);
}

std::optional<std::string> asString(const MetadataValue& v) {
    if (std::holds_alternative<std::string>(v)) {
        return std::get<std::string>(v);
    }
    return std::nullopt;
}

template <typename T>
std::optional<T> getNum(const GgufReader& reader, const std::string& key) {
    const auto* v = reader.findMetadata(key);
    return v ? asNumeric<T>(*v) : std::nullopt;
}

} // namespace

void LlmConfig::parseFromGguf(const GgufReader& reader) {
    // --- Architecture ----------------------------------------------------

    const auto* archV = reader.findMetadata("general.architecture");
    if (archV == nullptr) {
        MM_LOG_ERROR("config", "general.architecture missing — not a model file?");
        throw std::runtime_error("LlmConfig: general.architecture missing");
    }
    auto archStr = asString(*archV);
    if (!archStr) {
        throw std::runtime_error("LlmConfig: general.architecture not a string");
    }
    architecture = *archStr;
    MM_LOG_INFO("config", "architecture = '{}'", architecture);

    const std::string& a = architecture;

    auto requireU32 = [&](const std::string& suffix) -> std::uint32_t {
        const std::string key = a + "." + suffix;
        if (auto v = getNum<std::uint32_t>(reader, key)) {
            MM_LOG_INFO("config", "{} = {}", key, *v);
            return *v;
        }
        MM_LOG_ERROR("config", "required key '{}' missing or non-numeric", key);
        throw std::runtime_error("LlmConfig: missing required key " + key);
    };

    auto optionalU32 = [&](const std::string& suffix,
                           std::uint32_t fallback) -> std::uint32_t {
        const std::string key = a + "." + suffix;
        if (auto v = getNum<std::uint32_t>(reader, key)) {
            MM_LOG_INFO("config", "{} = {}", key, *v);
            return *v;
        }
        MM_LOG_INFO("config", "{} not set — defaulting to {}", key, fallback);
        return fallback;
    };

    auto optionalF32 = [&](const std::string& suffix, float fallback) -> float {
        const std::string key = a + "." + suffix;
        if (auto v = getNum<float>(reader, key)) {
            MM_LOG_INFO("config", "{} = {}", key, *v);
            return *v;
        }
        MM_LOG_INFO("config", "{} not set — defaulting to {}", key, fallback);
        return fallback;
    };

    // --- Required --------------------------------------------------------

    blockCount        = requireU32("block_count");
    contextLength     = requireU32("context_length");
    embeddingLength   = requireU32("embedding_length");
    feedForwardLength = requireU32("feed_forward_length");
    headCount         = requireU32("attention.head_count");
    headCountKv       = optionalU32("attention.head_count_kv", headCount);

    // --- Optional --------------------------------------------------------

    keyLength     = optionalU32("attention.key_length",   0);
    valueLength   = optionalU32("attention.value_length", keyLength);
    rmsNormEps    = optionalF32("attention.layer_norm_rms_epsilon", 1e-6F);
    ropeFreqBase  = optionalF32("rope.freq_base", 10000.0F);
    slidingWindow = optionalU32("attention.sliding_window", 0);

    // --- Architecture-specific shape corrections -------------------------

    // Gemma 4 26B-A4B's metadata is wrong on two critical fields:
    //   key_length      claims 512 but real head_dim is 256
    //   head_count_kv   is unset (defaults to head_count=16) but real
    //                   is 8 (GQA 2:1)
    // Both are recoverable from tensor shapes loaded later, but
    // LlmConfig is parsed *before* the matmul launches that would crash
    // on the inflated q_dim/kv_dim. Override here so downstream code
    // stays generic.
    if (architecture == "gemma4") {
        if (const auto* qNorm = reader.findTensor("blk.0.attn_q_norm.weight");
            qNorm != nullptr && !qNorm->dimensions.empty()) {
            const auto inferred =
                static_cast<std::uint32_t>(qNorm->dimensions[0]);
            if (inferred > 0 && inferred != keyLength) {
                MM_LOG_WARN("config",
                            "gemma4 override: key_length {} -> {} "
                            "(from attn_q_norm.weight dim[0])",
                            keyLength, inferred);
                keyLength   = inferred;
                valueLength = inferred;
            }
        }
        if (const auto* kW = reader.findTensor("blk.0.attn_k.weight");
            kW != nullptr && kW->dimensions.size() >= 2 && headDim() > 0) {
            const auto inferredKv =
                static_cast<std::uint32_t>(kW->dimensions[1] / headDim());
            if (inferredKv > 0 && inferredKv != headCountKv) {
                MM_LOG_WARN("config",
                            "gemma4 override: head_count_kv {} -> {} "
                            "(from attn_k.weight dim[1]={} / head_dim={})",
                            headCountKv, inferredKv,
                            kW->dimensions[1], headDim());
                headCountKv = inferredKv;
            }
        }
    }

    // --- Sanity ----------------------------------------------------------

    if (headCount == 0 || embeddingLength % headCount != 0) {
        MM_LOG_WARN("config",
                    "head_count={} does not evenly divide embedding_length={} "
                    "— headDim() derivation will be off",
                    headCount, embeddingLength);
    }
    if (headCountKv > headCount) {
        MM_LOG_WARN("config",
                    "head_count_kv={} > head_count={} — implausible",
                    headCountKv, headCount);
    }

    MM_LOG_INFO("config",
                "summary — arch={} blocks={} ctx={} d_model={} ff={} "
                "heads={} kv_heads={} head_dim={} rope_base={} rms_eps={}",
                architecture, blockCount, contextLength, embeddingLength,
                feedForwardLength, headCount, headCountKv, headDim(),
                ropeFreqBase, rmsNormEps);
}

} // namespace mimirmind::model