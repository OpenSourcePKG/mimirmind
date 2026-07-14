#pragma once

#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <vector>

namespace mimirmind::core::ipc {

/**
 * Wire protocol on top of `core::ipc::UnixSocketFrame` for the Munin
 * <-> mimirmind-worker channel. One request per attach connection; the
 * connection stays open afterwards so Munin can observe worker
 * disconnect as an implicit detach.
 *
 * Request frame (worker -> Munin): a single JSON envelope, no fds.
 *   `{"op": "healthz"}`
 *   `{"op": "attach",  "modelId": "google_gemma-4-E4B-it-Q4_K_M"}`
 *
 * Response frames (Munin -> worker):
 *   - healthz:  1 JSON frame (HealthzResponse::toJson).
 *   - attach:   1 JSON frame (TensorManifest::toJson, wire v2),
 *               then N frames of {64-byte handle payload, 1 SCM_RIGHTS fd},
 *               where N == manifest.chunks.size(). The frame at index
 *               i carries the IPC handle for `chunks[i]`; the worker
 *               resolves each tensor's pointer as
 *               `chunkBases[t.chunkIndex] + t.chunkOffset`.
 *   - error:    1 JSON frame `{"error": "<message>"}`.
 *
 * The kOp* constants are the on-wire values. Bumping the wire format is
 * done via `TensorManifest::kCurrentProtocolVersion` (already-versioned)
 * — this layer is intentionally schema-loose so healthz can grow fields
 * without breaking older workers.
 */
namespace op {
inline constexpr std::string_view kHealthz = "healthz";
inline constexpr std::string_view kAttach  = "attach";
} // namespace op

struct RequestEnvelope {
    std::string op{};
    std::string modelId{};  // only meaningful for op == "attach"

    [[nodiscard]] static std::expected<RequestEnvelope, std::string>
    fromJson(std::string_view j);
};

struct ModelSummaryWire {
    std::string   id{};
    std::string   fingerprint{};
    std::uint64_t totalBytes{0};
    std::uint32_t tensorCount{0};
};

/**
 * Body of the healthz response. `governorOwner` is "munin" for Munin
 * itself and "worker" when the daemon has been asked to hand ownership
 * off — the ownership handoff is Schritt 8 in the M-Munin roll-out, so
 * for Schritt 7 the field is always "munin".
 */
struct HealthzResponse {
    std::uint32_t                 protocolVersion{1};
    std::string                   status{"ok"};
    std::string                   governorOwner{"munin"};
    std::uint32_t                 pid{0};
    std::vector<ModelSummaryWire> models{};

    [[nodiscard]] std::string toJson() const;

    [[nodiscard]] static std::expected<HealthzResponse, std::string>
    fromJson(std::string_view j);
};

/**
 * Serialised error frame body. Kept as a helper so all error paths use
 * the same on-wire shape.
 */
[[nodiscard]] std::string makeErrorJson(std::string_view message);

/**
 * Parse an error frame body. Returns std::nullopt when the JSON does
 * not carry an `error` field (i.e. it is not an error envelope).
 */
[[nodiscard]] std::expected<std::string, std::string>
parseErrorJson(std::string_view j);

} // namespace mimirmind::core::ipc