// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/gguf/GgufTypes.hpp"
#include "runtime/encoder/laya/LayaDecisionHead.hpp"
#include "runtime/encoder/laya/LayaDecisionRunner.hpp"
#include "runtime/encoder/laya/LayaEncoderModel.hpp"
#include "runtime/encoder/laya/LayaEncoderRunner.hpp"
#include "runtime/encoder/laya/LayaTokenizer.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace mimirmind::compute {
class ComputeOps;
class ComputeMatmul;
} // namespace mimirmind::compute

namespace mimirmind::runtime::encoder::laya {

/**
 * End-to-end Laya typed-decision: tokenizer + build_sequence/render_options +
 * backbone + decision head + decode. One LayaSystemOne owns the whole native
 * stack for one checkpoint (sequential option layout, B=1), mirroring the
 * upstream Agent.predict path. The serving handler (POST /v1/systemone) builds
 * questions and calls decide() per question.
 */
class LayaSystemOne {
public:
    enum class QType { Choice = 0, Score = 1, Noul = 2 };

    struct Question {
        QType type{QType::Choice};
        std::string instructions;
        // choice: ordered (key, description) criteria.
        std::vector<std::pair<std::string, std::string>> choice;
        // score: ordered level descriptions.
        std::vector<std::string> levels;
        // noul: false/true descriptions + labels (default "false"/"true").
        std::string noulFalse, noulTrue;
        std::string noulFalseLabel{"false"}, noulTrueLabel{"true"};
    };

    struct Answer {
        QType type{QType::Choice};
        std::string choice;                                 // choice: winning key
        float score{0.0F};                                  // score: expected level
        float noul{0.0F};                                   // noul: P(true)
        std::vector<std::pair<std::string, float>> probabilities; // key/index -> p
        float confidence{0.0F};
        float answerConfidence{0.0F};
        float actProbability{0.0F};
    };

    LayaSystemOne(compute::ComputeOps& ops, compute::ComputeMatmul& matmul,
                  std::size_t maxLen = 1024, std::size_t headMaxLen = 256);

    /// Load encoder (prefix `encoder.`), decision head, and tokenizer from `modelDir`.
    void load(std::string_view modelDir,
              core::gguf::GgmlType matmulType = core::gguf::GgmlType::F32,
              std::size_t nHeadLayers = 2);

    [[nodiscard]] Answer decide(std::string_view state, const Question& q);

    /// Render a question's option texts (label-index order), as upstream render_options.
    [[nodiscard]] static std::vector<std::string> renderOptions(const Question& q);

private:
    compute::ComputeOps&    _ops;
    compute::ComputeMatmul& _mm;
    std::size_t _maxLen;
    std::size_t _headMaxLen;

    LayaEncoderModel  _model;
    LayaDecisionHead  _head;
    LayaTokenizer     _tok;
    std::unique_ptr<LayaEncoderRunner>  _enc;
    std::unique_ptr<LayaDecisionRunner> _dec;
};

} // namespace mimirmind::runtime::encoder::laya
