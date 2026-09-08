// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "model/ToolCall.hpp"

#include <cstdint>
#include <memory>
#include <span>

namespace mimirmind::model {

class Tokenizer;

/**
 * Roadmap 8.19.13.2 — grammar-constrained tool decoding (xgrammar-backed).
 *
 * A per-slot decode-time constraint that guarantees the STRUCTURE of a
 * Qwen3-Coder XML tool call is valid, whatever free-text format the model would
 * otherwise prefer: the function NAME is forced to an offered tool name (kills
 * `read_file` when only `Read` is offered), and every parameter KEY is forced
 * to a key in that tool's JSON schema (kills `filename` vs `file_path`).
 *
 * Design (see decisions/2026-09-08-grammar-constrained-tool-decoding): the
 * open-vocabulary / boundary problems that make a hand-rolled string automaton
 * fragile are exactly what a proper token-level grammar automaton solves — the
 * same machinery vLLM uses. We vendor xgrammar (MIT, NOT llama.cpp/ggml) and
 * drive it from the tool wrapper's EBNF. The `<tool_call>` opener is a single
 * special token in this checkpoint, so it triggers the matcher robustly; from
 * there xgrammar masks each step's logits to the grammar. VALUE regions stay
 * free; the wrapper/closing are enforced.
 *
 * Host-side, applied to the read-back logits row in the 8.19.5 per-slot sampler
 * BEFORE the categorical draw. `active()` is false when no tools were offered
 * (or nothing compiled), so greedy / no-tool paths are untouched. Move-only;
 * one per slot. Pimpl keeps xgrammar out of this header.
 */
class ToolCallConstraint {
public:
    ToolCallConstraint() = default;
    ToolCallConstraint(std::span<const ToolSpec> tools, const Tokenizer& tok);
    ~ToolCallConstraint();
    ToolCallConstraint(ToolCallConstraint&&) noexcept;
    ToolCallConstraint& operator=(ToolCallConstraint&&) noexcept;
    ToolCallConstraint(const ToolCallConstraint&)            = delete;
    ToolCallConstraint& operator=(const ToolCallConstraint&) = delete;

    [[nodiscard]] bool active() const noexcept;

    /// Mask (to -inf) every disallowed next-token logit for the current state.
    /// No-op unless inside a tool call.
    void maskLogits(float* logits, std::size_t vocab) const;

    /// Advance the automaton with the just-sampled token.
    void advance(std::int32_t token);

    /// Reset to the start (call at the start of a response / on reseed).
    void reset() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace mimirmind::model
