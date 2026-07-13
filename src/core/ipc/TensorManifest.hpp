#pragma once

#include "core/gguf/GgufTypes.hpp"

#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <vector>

namespace mimirmind::core::ipc {

/**
 * One tensor's metadata in the manifest that Munin sends to an
 * attaching mimirmind worker. `handleIndex` is the position of this
 * tensor's IPC handle in the sequence of HANDLE frames that follow
 * the manifest — the receiver walks the same order and matches by
 * index.
 */
struct ManifestEntry {
    std::string                       name{};      // e.g. "blk.5.attn_q.weight"
    ::mimirmind::core::gguf::GgmlType type{
        ::mimirmind::core::gguf::GgmlType::Unknown};
    std::vector<std::uint64_t>        dims{};      // e.g. [2560, 512]
    std::uint64_t                     bytes{0};    // total size in bytes
    std::uint32_t                     handleIndex{0};
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
    static constexpr std::uint32_t kCurrentProtocolVersion = 1;

    std::uint32_t              protocolVersion{kCurrentProtocolVersion};
    std::string                modelId{};           // "google_gemma-4-E4B-it-Q4_K_M"
    std::string                modelFingerprint{};  // opaque identity check
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