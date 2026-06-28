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