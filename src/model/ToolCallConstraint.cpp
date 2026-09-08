// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "model/ToolCallConstraint.hpp"

#include "model/Tokenizer.hpp"

#include <nlohmann/json.hpp>

#include <limits>

namespace mimirmind::model {

using nlohmann::json;

namespace {

// Strip `marker` from the front of `seq`; returns false if it is not a prefix.
bool stripPrefix(const std::vector<std::int32_t>& seq,
                 const std::vector<std::int32_t>& marker,
                 std::vector<std::int32_t>&       out) {
    if (seq.size() <= marker.size()) { return false; }
    for (std::size_t i = 0; i < marker.size(); ++i) {
        if (seq[i] != marker[i]) { return false; }
    }
    out.assign(seq.begin() + static_cast<std::ptrdiff_t>(marker.size()),
               seq.end());
    return true;
}

// Ordered schema-property names of a tool from its OpenAI tool JSON.
std::vector<std::string> schemaKeys(const std::string& toolJson) {
    std::vector<std::string> keys;
    const json t = json::parse(toolJson, nullptr, false);
    if (t.is_discarded() || !t.contains("function")
        || !t["function"].is_object()
        || !t["function"].contains("parameters")
        || !t["function"]["parameters"].is_object()
        || !t["function"]["parameters"].contains("properties")
        || !t["function"]["parameters"]["properties"].is_object()) {
        return keys;
    }
    const json& props = t["function"]["parameters"]["properties"];
    for (auto it = props.begin(); it != props.end(); ++it) {
        keys.push_back(it.key());
    }
    return keys;
}

} // namespace

ToolCallConstraint::Trie
ToolCallConstraint::buildTrie(
        const std::vector<std::vector<std::int32_t>>& seqs,
        const std::vector<int>& toolForSeq) {
    Trie trie;
    if (seqs.empty()) { return trie; }
    trie.nodes.emplace_back();   // root
    for (std::size_t s = 0; s < seqs.size(); ++s) {
        std::size_t node = 0;
        for (const std::int32_t tk : seqs[s]) {
            auto& next = trie.nodes[node].next;
            const auto it = next.find(tk);
            if (it == next.end()) {
                const std::size_t child = trie.nodes.size();
                trie.nodes.emplace_back();
                trie.nodes[node].next.emplace(tk, child);
                node = child;
            } else {
                node = it->second;
            }
        }
        trie.nodes[node].terminal = true;
        if (s < toolForSeq.size()) {
            trie.nodes[node].toolForName = toolForSeq[s];
        }
    }
    return trie;
}

ToolCallConstraint::ToolCallConstraint(std::span<const ToolSpec> tools,
                                       const Tokenizer& tok) {
    if (tools.empty()) { return; }

    const std::vector<std::int32_t> funcMarker =
        tok.encode("<function=", /*addBos=*/false);
    const std::vector<std::int32_t> paramMarker =
        tok.encode("<parameter=", /*addBos=*/false);
    if (funcMarker.empty() || paramMarker.empty()) { return; }

    // Name trie: <function=NAME> per offered tool.
    std::vector<std::vector<std::int32_t>> nameSeqs;
    std::vector<int>                       nameTool;
    _keyTrie.resize(tools.size());
    _toolNames.resize(tools.size());
    for (std::size_t i = 0; i < tools.size(); ++i) {
        _toolNames[i] = tools[i].name;
        const auto full =
            tok.encode("<function=" + tools[i].name + ">", /*addBos=*/false);
        std::vector<std::int32_t> suffix;
        if (stripPrefix(full, funcMarker, suffix)) {
            nameSeqs.push_back(std::move(suffix));
            nameTool.push_back(static_cast<int>(i));
        }
        // Per-tool key trie: <parameter=KEY> for each schema key.
        std::vector<std::vector<std::int32_t>> keySeqs;
        std::vector<int>                       keyTool;
        for (const std::string& k : schemaKeys(tools[i].toolJson)) {
            const auto kf =
                tok.encode("<parameter=" + k + ">", /*addBos=*/false);
            std::vector<std::int32_t> ks;
            if (stripPrefix(kf, paramMarker, ks)) {
                keySeqs.push_back(std::move(ks));
                keyTool.push_back(static_cast<int>(i));
            }
        }
        _keyTrie[i] = buildTrie(keySeqs, keyTool);
        _keyTrie[i].marker = paramMarker;
    }
    _nameTrie = buildTrie(nameSeqs, nameTool);
    _nameTrie.marker = funcMarker;
    if (_nameTrie.empty()) { return; }   // nothing usable

    _ringMax = funcMarker.size();
    for (const auto& kt : _keyTrie) {
        _ringMax = std::max(_ringMax, kt.marker.size());
    }
    _ring.reserve(_ringMax);
    _active = true;
}

void ToolCallConstraint::reset() noexcept {
    _state = State::Free;
    _node = 0;
    _curTool = -1;
    _activeTrie = nullptr;
    _ring.clear();
}

bool ToolCallConstraint::ringEndsWith(
        const std::vector<std::int32_t>& marker) const {
    if (marker.empty() || _ring.size() < marker.size()) { return false; }
    const std::size_t off = _ring.size() - marker.size();
    for (std::size_t i = 0; i < marker.size(); ++i) {
        if (_ring[off + i] != marker[i]) { return false; }
    }
    return true;
}

void ToolCallConstraint::maskLogits(float* logits, std::size_t vocab) const {
    if (!_active || _state == State::Free || _activeTrie == nullptr
        || _node >= _activeTrie->nodes.size()) {
        return;
    }
    const auto& next = _activeTrie->nodes[_node].next;
    if (next.empty()) { return; }   // nothing to constrain (defensive)
    constexpr float kNeg = -std::numeric_limits<float>::infinity();
    for (std::size_t t = 0; t < vocab; ++t) {
        if (next.find(static_cast<std::int32_t>(t)) == next.end()) {
            logits[t] = kNeg;
        }
    }
}

void ToolCallConstraint::advance(std::int32_t token) {
    if (!_active) { return; }

    if (_state == State::Free) {
        _ring.push_back(token);
        if (_ring.size() > _ringMax) {
            _ring.erase(_ring.begin(),
                        _ring.begin()
                            + static_cast<std::ptrdiff_t>(_ring.size() - _ringMax));
        }
        // Enter a NAME region on <function=.
        if (ringEndsWith(_nameTrie.marker)) {
            _state = State::InName;
            _activeTrie = &_nameTrie;
            _node = 0;
            _ring.clear();
            return;
        }
        // Enter a KEY region on <parameter= (keys of the last-named tool).
        if (_curTool >= 0 && _curTool < static_cast<int>(_keyTrie.size())
            && !_keyTrie[_curTool].empty()
            && ringEndsWith(_keyTrie[_curTool].marker)) {
            _state = State::InKey;
            _activeTrie = &_keyTrie[_curTool];
            _node = 0;
            _ring.clear();
            return;
        }
        return;
    }

    // In a NAME or KEY region: follow the trie.
    const auto& nodes = _activeTrie->nodes;
    const auto it = nodes[_node].next.find(token);
    if (it == nodes[_node].next.end()) {
        // Should not happen (masked), but never get stuck.
        reset();
        return;
    }
    _node = it->second;
    if (_state == State::InName && nodes[_node].toolForName >= 0) {
        _curTool = nodes[_node].toolForName;   // remember for the key trie
    }
    if (nodes[_node].terminal || nodes[_node].next.empty()) {
        _state = State::Free;
        _activeTrie = nullptr;
        _node = 0;
        _ring.clear();
    }
}

} // namespace mimirmind::model
