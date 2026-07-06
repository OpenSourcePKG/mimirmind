#include "model/LlmConfig.hpp"

#include "model/GgufReader.hpp"
#include "runtime/Log.hpp"

#include <cstring>
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

    keyLength       = optionalU32("attention.key_length",   0);
    valueLength     = optionalU32("attention.value_length", keyLength);

    // Gemma 4 ships separate full / SWA head_dim. `key_length` already
    // holds the FULL value; `key_length_swa` is the SWA one.
    keyLengthSwa   = optionalU32("attention.key_length_swa",   0);
    valueLengthSwa = optionalU32("attention.value_length_swa", keyLengthSwa);

    // head_count_kv may be an array (one entry per layer) on Gemma 4 —
    // SWA layers and full-attention layers can have different KV head
    // counts (e.g. 8 / 2). Parse element type matching GgufValueType:
    //   UInt32=4, Int32=5 (most common writers).
    if (const auto* v = reader.findMetadata(a + ".attention.head_count_kv");
        v != nullptr && std::holds_alternative<GgufArray>(*v)) {
        const auto& arr = std::get<GgufArray>(*v);
        const std::size_t elemBytes = valueElementWidth(arr.elementType);
        if (arr.raw.size() == arr.count * elemBytes &&
            (arr.elementType == GgufValueType::UInt32 ||
             arr.elementType == GgufValueType::Int32)) {
            headCountKvPerLayer.resize(arr.count);
            for (std::uint64_t i = 0; i < arr.count; ++i) {
                std::uint32_t value = 0;
                std::memcpy(&value, arr.raw.data() + i * elemBytes,
                            sizeof(std::uint32_t));
                headCountKvPerLayer[i] = value;
            }
            // Pick the most-common entry for the global headCountKv so
            // any code paths that still use the scalar see a reasonable
            // default (matches what we'd get with the legacy scalar key).
            std::uint32_t modeVal   = headCountKvPerLayer[0];
            std::uint32_t modeCount = 1;
            for (auto v2 : headCountKvPerLayer) {
                std::uint32_t c = 0;
                for (auto v3 : headCountKvPerLayer) if (v3 == v2) ++c;
                if (c > modeCount) { modeVal = v2; modeCount = c; }
            }
            MM_LOG_INFO("config",
                        "{}.attention.head_count_kv = array[{}], most-common "
                        "value {} ({} layers)",
                        a, arr.count, modeVal, modeCount);
            headCountKv = modeVal;
        } else {
            MM_LOG_WARN("config",
                        "{}.attention.head_count_kv array has unexpected "
                        "type/size (type={}, count={}, raw={} bytes) — ignored",
                        a, static_cast<int>(arr.elementType),
                        arr.count, arr.raw.size());
        }
    }
    rmsNormEps      = optionalF32("attention.layer_norm_rms_epsilon", 1e-6F);
    ropeFreqBase    = optionalF32("rope.freq_base", 10000.0F);
    ropeFreqBaseSwa = optionalF32("rope.freq_base_swa", ropeFreqBase);
    slidingWindow   = optionalU32("attention.sliding_window", 0);

    // Sliding-window-attention pattern. Stored in GGUF as an array of
    // bools (one per layer) under `<arch>.attention.sliding_window_pattern`.
    // Older Gemma-3 GGUFs used a uniform "every nth layer is SWA" scheme;
    // Gemma 4 spells out the full per-layer mask. Absent ⇒ no SWA.
    if (const auto* swp = reader.findMetadata(a + ".attention.sliding_window_pattern");
        swp != nullptr && std::holds_alternative<GgufArray>(*swp)) {
        const auto& arr = std::get<GgufArray>(*swp);
        if (arr.elementType == GgufValueType::Bool &&
            arr.raw.size() == arr.count) {
            slidingWindowPattern.resize(arr.count);
            for (std::uint64_t i = 0; i < arr.count; ++i) {
                slidingWindowPattern[i] = (arr.raw[i] != 0);
            }
            std::size_t swaCount = 0;
            for (bool s : slidingWindowPattern) if (s) ++swaCount;
            MM_LOG_INFO("config",
                        "{}.attention.sliding_window_pattern = [{} entries, "
                        "{} SWA / {} full]",
                        a, slidingWindowPattern.size(),
                        swaCount, slidingWindowPattern.size() - swaCount);
        } else {
            MM_LOG_WARN("config",
                        "{}.attention.sliding_window_pattern present but "
                        "unexpected (elementType={}, count={}, bytes={}) — ignoring",
                        a, static_cast<int>(arr.elementType),
                        arr.count, arr.raw.size());
        }
    } else {
        MM_LOG_INFO("config",
                    "{}.attention.sliding_window_pattern not set — all layers "
                    "treated as full attention", a);
    }

    // MoE hyperparameters. Absent (0) means the model isn't MoE; Gemma 4
    // 26B-A4B sets both. If used_count is 0 but expert_count is set we
    // default top-K to 4 (common Mixtral/Qwen-MoE convention).
    expertCount     = optionalU32("expert_count",      0);
    expertUsedCount = optionalU32("expert_used_count", 0);
    if (expertCount > 0 && expertUsedCount == 0) {
        expertUsedCount = 4;
        MM_LOG_WARN("config",
                    "{}.expert_used_count not set, defaulting top-K to {}",
                    a, expertUsedCount);
    }

    // Gemma 4: number of trailing layers that reuse an earlier layer's
    // K/V cache. In E4B this is 18 (n_layer_kv_from_start = 42 - 18 = 24).
    sharedKvLayers  = optionalU32("attention.shared_kv_layers", 0);
    if (sharedKvLayers > 0) {
        MM_LOG_INFO("config",
                    "{}.attention.shared_kv_layers = {} (last {} block(s) "
                    "reuse K/V from an earlier layer)",
                    a, sharedKvLayers, sharedKvLayers);
    }

    // --- Architecture-specific shape corrections -------------------------

    // Gemma 4 26B-A4B's metadata declares two distinct head dimensions
    // (`key_length` = head_dim_full, `key_length_swa` = head_dim_swa).
    // We trust the metadata when both arrays are set. Otherwise the
    // legacy tensor-shape override below picks the SWA value from
    // attn_q_norm.weight — fine for models without the SWA split.
    if (architecture == "gemma4") {
        // E4B specifically has UNIFORM head_count_kv (scalar 2) across
        // all layers but PER-LAYER-VARYING head_dim (256 SWA / 512 full).
        // The old check required headCountKvPerLayer to be non-empty as
        // proof of "per-layer split" which is what 26B-A4B has. That
        // check missed E4B's split and triggered the destructive override
        // below — pinning key_length to 256 globally, which then wrecked
        // the 7 full-attention layers that actually want head_dim=512.
        // The distinction we really care about is "do we have separate
        // key_length + key_length_swa AND a sliding_window_pattern telling
        // us which layer uses which head_dim?" — that's enough.
        const bool haveSplit = (keyLength > 0 && keyLengthSwa > 0 &&
                                 !slidingWindowPattern.empty());
        if (!haveSplit) {
            if (const auto* qNorm = reader.findTensor("blk.0.attn_q_norm.weight");
                qNorm != nullptr && !qNorm->dimensions.empty()) {
                const auto inferred =
                    static_cast<std::uint32_t>(qNorm->dimensions[0]);
                if (inferred > 0 && inferred != keyLength) {
                    MM_LOG_WARN("config",
                                "gemma4 override (no SWA split): key_length "
                                "{} -> {} (from attn_q_norm.weight dim[0])",
                                keyLength, inferred);
                    keyLength   = inferred;
                    valueLength = inferred;
                }
            }
        } else {
            MM_LOG_INFO("config",
                        "gemma4 per-layer attention dims active: full "
                        "head_dim={}, swa head_dim={}",
                        keyLength, keyLengthSwa);
        }
        // Only fall back to tensor-shape inference for headCountKv when
        // metadata really doesn't give us anything trustworthy. When we
        // have the SWA/full split (haveSplit) the metadata's
        // head_count_kv is authoritative — running the divide-by-head_dim
        // inference here would be wrong anyway, because block 0's attn_k
        // uses head_dim_swa (256 for E4B) while headDim() returns the
        // full-attention value (512). That's the bug that pinned E4B's
        // kv_heads to 1 in one earlier build.
        if (!haveSplit && headCountKvPerLayer.empty()) {
            if (const auto* kW = reader.findTensor("blk.0.attn_k.weight");
                kW != nullptr && kW->dimensions.size() >= 2 && headDim() > 0) {
                const auto inferredKv =
                    static_cast<std::uint32_t>(kW->dimensions[1] / headDim());
                if (inferredKv > 0 && inferredKv != headCountKv) {
                    MM_LOG_WARN("config",
                                "gemma4 override (no array): head_count_kv "
                                "{} -> {} (from attn_k.weight dim[1]={} / "
                                "head_dim={})",
                                headCountKv, inferredKv,
                                kW->dimensions[1], headDim());
                    headCountKv = inferredKv;
                }
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
                "heads={} kv_heads={} head_dim={}/{} rope_base={}/{} "
                "rms_eps={} experts={} top_k={} swa_layers={} kv_per_layer={}",
                architecture, blockCount, contextLength, embeddingLength,
                feedForwardLength, headCount, headCountKv,
                headDim(), keyLengthSwa,
                ropeFreqBase, ropeFreqBaseSwa, rmsNormEps,
                expertCount, expertUsedCount,
                slidingWindowPattern.size(),
                headCountKvPerLayer.size());
}

} // namespace mimirmind::model