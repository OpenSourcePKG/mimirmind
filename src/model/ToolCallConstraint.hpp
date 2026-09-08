// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "model/ToolCall.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace mimirmind::model {

class Tokenizer;

/**
 * Roadmap 8.19.13.2 — token-mask grammar constraint for tool calls.
 *
 * A per-slot decode-time automaton that guarantees the STRUCTURE of a
 * Qwen3-Coder XML tool call (`<tool_call>\n<function=NAME>\n<parameter=KEY>\n
 * VALUE\n</parameter>…</function>\n</tool_call>`) is valid, whatever free-text
 * format the model would otherwise prefer:
 *   - the function NAME is forced to be one of the offered tool names (kills
 *     `read_file` when only `Read` is offered — a post-hoc normalizer cannot
 *     close the open alias space),
 *   - a parameter KEY is forced to be a key in that tool's JSON schema (kills
 *     `filename` vs `file_path`, and invented keys).
 *
 * Scope of THIS increment: the NAME and KEY regions only. Once the model has
 * emitted the `<function=` / `<parameter=` opener, the mask permits only tokens
 * that spell an offered name / schema key (plus its `>` closer); VALUE regions
 * and the wrapper are left free (the existing parser consumes them). Forcing a
 * call vs prose (the ```bash-relapse case) is a later increment.
 *
 * Implementation: pure token-id tries built from the CANONICAL tokenisation of
 * `<function=NAME>` / `<parameter=KEY>` per offered tool — no vocab scan, no
 * byte-level piece table. The longest common token prefix across a group is the
 * marker (detected against the emitted-token ring); the per-name/-key suffixes
 * form the trie. Assumes the model tokenises the wrapper canonically (true once
 * the tools prompt elicits the XML form, b93cc45).
 *
 * Host-side, applied to the read-back logits row in the 8.19.5 per-slot
 * sampler BEFORE the categorical draw. `active()` is false when no tools were
 * offered, so greedy / no-tool paths are untouched. Move-only; one per slot.
 */
class ToolCallConstraint {
public:
    ToolCallConstraint() = default;
    /// Build from the request's offered tools. Inactive when `tools` is empty
    /// or nothing tokenised usefully.
    ToolCallConstraint(std::span<const ToolSpec> tools, const Tokenizer& tok);

    [[nodiscard]] bool active() const noexcept { return _active; }

    /// Zero (to -inf) every disallowed next-token logit for the current state.
    /// No-op unless the automaton is inside a NAME or KEY region.
    void maskLogits(float* logits, std::size_t vocab) const;

    /// Advance the automaton with the just-sampled token.
    void advance(std::int32_t token);

    /// Reset to the start (call at the beginning of a response / on reseed).
    void reset() noexcept;

private:
    enum class State { Free, InName, InKey };

    // A trie node over token ids: children[token] -> next node index; terminal
    // marks the end of a valid name/key (its `>` closer is a child edge).
    struct Node {
        std::unordered_map<std::int32_t, std::size_t> next;
        int  toolForName{-1};   // tool index if this node completes a name
        bool terminal{false};   // reached the `>` closer -> region ends here
    };

    // Build a trie (returned as a node vector, node 0 = root) from a set of
    // token sequences; `marker` receives the longest common token prefix.
    struct Trie {
        std::vector<Node>         nodes;
        std::vector<std::int32_t> marker;   // shared opener token prefix
        bool empty() const { return nodes.empty(); }
    };
    static Trie buildTrie(const std::vector<std::vector<std::int32_t>>& seqs,
                          const std::vector<int>& toolForSeq);

    bool  _active{false};
    Trie  _nameTrie;                       // <function=NAME>
    std::vector<Trie> _keyTrie;            // per-tool <parameter=KEY>
    std::vector<std::string> _toolNames;   // index -> name (diagnostics)

    // Rolling ring of recently emitted token ids, long enough to detect the
    // longest marker.
    std::vector<std::int32_t> _ring;
    std::size_t               _ringMax{0};

    State       _state{State::Free};
    std::size_t _node{0};       // current trie node within a region
    int         _curTool{-1};   // tool whose key-trie is active in InKey
    const Trie* _activeTrie{nullptr};

    // True if the emitted-token ring currently ends with `marker`.
    [[nodiscard]] bool ringEndsWith(const std::vector<std::int32_t>& marker) const;
};

} // namespace mimirmind::model
