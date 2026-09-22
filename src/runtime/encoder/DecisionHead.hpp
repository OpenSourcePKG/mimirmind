// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstddef>
#include <filesystem>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace mimirmind::runtime::encoder {

/**
 * One trained "System-One" typed-decision head (8.23): a frozen linear probe on
 * top of the resident bge-m3 / EncoderRunner sentence embedding. Given the
 * `hidden`-dim L2-normalized CLS embedding of a text it produces a single
 * typed answer — `logits = W·e + b`, then softmax with a calibration
 * temperature — in one host-side matvec (num_classes is tiny), instead of a
 * full autoregressive LLM turn.
 *
 * The head is trained OFFLINE from Pegenaut chat traces (behavioural cloning of
 * the tool/route/RAG decision the LLM already made); MimirMind only loads and
 * evaluates it. The input feature is exactly what /v1/embeddings returns for
 * the same text, so the offline trainer can harvest features through that
 * endpoint with no bespoke extraction path.
 *
 * On-disk contract (one directory per head; format chosen so a numpy trainer
 * emits it with `arr.astype('<f4').tofile(...)` — exact IEEE-754 bytes, clean
 * engine/Python parity, no safetensors dependency):
 *   - `head.json`   : { "name": "tool",
 *                       "labels": ["db_query","web_search","none"],
 *                       "hidden": 1024,
 *                       "temperature": 1.0,      // optional, default 1.0
 *                       "threshold": 0.0,        // optional, default 0.0
 *                       "encoder": "bge-m3" }    // optional, informational
 *   - `weight.f32`  : little-endian float32, num_classes*hidden, row-major
 *                     (row c = the weight vector for class c)
 *   - `bias.f32`    : little-endian float32, num_classes
 *
 * Host-side only: no device allocation, no ComputeOps. Not stateful — a loaded
 * head is immutable and safe to evaluate from any thread (the shared encoder in
 * front of it is what needs serialising, not this).
 */
class DecisionHead {
public:
    /// One class probability, label kept alongside so callers preserve the
    /// trained label order in the response `dist`.
    struct Prob {
        std::string label;
        float       p{0.0F};
    };

    /// The typed decision for one head over one embedding.
    struct Result {
        std::string       choice;              // argmax label
        float             p{0.0F};             // probability of `choice`
        std::vector<Prob> dist;                // full distribution, trained order
        bool              confident{false};    // p >= threshold
    };

    /// In-memory head definition for the upload/reload path (8.23 head push):
    /// the same fields the on-disk contract carries, weights inline. `weight`
    /// is row-major `[labels.size() * hidden]`, `bias` is `[labels.size()]`.
    struct Spec {
        std::string              name;
        std::vector<std::string> labels;
        std::size_t              hidden{0};
        float                    temperature{1.0F};
        float                    threshold{0.0F};
        std::string              encoder;        // optional, informational
        std::vector<float>       weight;
        std::vector<float>       bias;
    };

    /// Load a head from `dir` (head.json + weight.f32 + bias.f32). Throws
    /// std::runtime_error on any missing/malformed file or shape mismatch.
    [[nodiscard]] static DecisionHead loadFromDir(const std::filesystem::path& dir);

    /// Serialize a validated Spec to `dir` as the on-disk contract (head.json +
    /// weight.f32 + bias.f32), creating `dir` if needed. Validates every field
    /// and the weight/bias shapes; throws std::runtime_error on any mismatch so
    /// a bad push never writes a half-formed head. Round-trips with loadFromDir.
    static void writeToDir(const std::filesystem::path& dir, const Spec& spec);

    /// The question key this head answers (e.g. "tool"). Matched against the
    /// request `questions` object.
    [[nodiscard]] const std::string& name() const noexcept { return _name; }

    /// Expected embedding dimensionality (must equal the encoder hidden size).
    [[nodiscard]] std::size_t hidden() const noexcept { return _hidden; }

    /// The trained label set, in order.
    [[nodiscard]] const std::vector<std::string>& labels() const noexcept { return _labels; }

    /// Calibration temperature applied to logits before softmax.
    [[nodiscard]] float temperature() const noexcept { return _temperature; }

    /// Confidence threshold below which the caller should fall back to the LLM.
    [[nodiscard]] float threshold() const noexcept { return _threshold; }

    /// Evaluate the head on one sentence embedding. `emb.size()` must equal
    /// hidden(); throws std::invalid_argument otherwise.
    [[nodiscard]] Result decide(std::span<const float> emb) const;

private:
    std::string              _name;
    std::vector<std::string> _labels;
    std::size_t              _hidden{0};
    float                    _temperature{1.0F};
    float                    _threshold{0.0F};
    std::vector<float>       _weight;   // [num_classes * hidden], row-major
    std::vector<float>       _bias;     // [num_classes]
};

} // namespace mimirmind::runtime::encoder
