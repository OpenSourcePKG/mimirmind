// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/gguf/GgufTypes.hpp"

#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <vector>

namespace mimirmind::core::ipc {

/**
 * Metadata for one large USM chunk that Munin exports over IPC. The
 * handle-fd for chunk `i` arrives via SCM_RIGHTS in the i-th HANDLE
 * frame that follows the manifest; the receiver walks chunks by index
 * and reconstructs its own base pointer via `zeMemOpenIpcHandle`.
 *
 * `bytes` carries the actual used footprint (not the raw chunk size)
 * so the worker can sanity-check its imported chunk against the payload
 * declared in the manifest.
 */
struct ChunkDesc {
    std::uint32_t chunkIndex{0};
    std::uint64_t bytes{0};
};

/**
 * One tensor's metadata in the manifest that Munin sends to an
 * attaching mimirmind worker. `chunkIndex` names the owning chunk from
 * the manifest's `chunks` list; `chunkOffset` is the byte offset inside
 * that chunk. The worker computes its pointer as
 * `importedChunkBases[chunkIndex] + chunkOffset` — no per-tensor IPC
 * handle is emitted.
 */
struct ManifestEntry {
    std::string                       name{};      // e.g. "blk.5.attn_q.weight"
    ::mimirmind::core::gguf::GgmlType type{
        ::mimirmind::core::gguf::GgmlType::Unknown};
    std::vector<std::uint64_t>        dims{};      // e.g. [2560, 512]
    std::uint64_t                     bytes{0};    // total size in bytes
    std::uint32_t                     chunkIndex{0};
    std::uint64_t                     chunkOffset{0};
};

/**
 * Top-level manifest sent from Munin to a mimirmind worker on attach.
 * Serialized as compact JSON.
 *
 * `protocolVersion` guards against wire-format drift between Munin and
 * mimirmind builds. The worker fails the attach hard on mismatch —
 * silent semantic drift is what M-Munin exists to prevent, so we
 * refuse to guess.
 *
 * `modelFingerprint` is a cheap identity check on the underlying GGUF
 * (header hash + tensor-count + size-sum is the current plan; upgrading
 * to full-file SHA is a Tier-3 follow-up). The worker knows what it
 * expected from its own config.json; a mismatch means "Munin has a
 * different model loaded than you thought — refuse to attach".
 */
struct TensorManifest {
    /// Bump when the wire format changes in a way old receivers can't
    /// parse. Cross-version compatibility is not a design goal — the
    /// intended workflow is that Munin and its worker deploy together.
    ///
    /// v1 (retired): per-tensor `handleIndex`, one IPC handle per tensor.
    /// v2 (current): explicit `chunks` list, tensors carry
    ///               `chunkIndex + chunkOffset`. One IPC handle per
    ///               chunk. Enables Multi-Model-Attach on Xe-LPG
    ///               (M-Munin.1a ADR).
    static constexpr std::uint32_t kCurrentProtocolVersion = 2;

    std::uint32_t              protocolVersion{kCurrentProtocolVersion};
    std::string                modelId{};           // "google_gemma-4-E4B-it-Q4_K_M"
    std::string                modelFingerprint{};  // opaque identity check
    std::vector<ChunkDesc>     chunks{};            // one entry per exported chunk
    std::vector<ManifestEntry> tensors{};

    [[nodiscard]] std::string toJson() const;

    /// Parse a JSON blob into a TensorManifest. Returns error string on
    /// malformed input, version mismatch, or missing required fields.
    /// The receiver's caller decides how to log and whether to close
    /// the connection.
    [[nodiscard]] static std::expected<TensorManifest, std::string>
    fromJson(std::string_view j);
};

} // namespace mimirmind::core::ipc