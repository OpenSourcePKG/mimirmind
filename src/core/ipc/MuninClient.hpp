#pragma once

#include "core/gguf/GgufReader.hpp"
#include "core/ipc/TensorManifest.hpp"
#include "core/ipc/WireOps.hpp"
#include "core/l0/L0Context.hpp"

#include <expected>
#include <string>
#include <string_view>
#include <vector>

namespace mimirmind::core::ipc {

/**
 * Client-side wrapper for the M-Munin wire protocol. Represents one
 * open connection to a running Munin daemon. Two operations:
 *
 *   - `healthz(path)`  — probe: is Munin up, which models does it
 *                        hold, who owns the governor? Opens a fresh
 *                        connection each call and closes it on return.
 *
 *   - `attach(path, id)` — attach flow: opens a persistent connection,
 *                          sends the attach request, reads the manifest,
 *                          imports N SCM_RIGHTS handles via
 *                          `IpcImporter::importOne`, and hands back a
 *                          ready-to-use vector of `GgufTensor` structs
 *                          where each `usmPtr` field points at Munin-
 *                          owned USM in this worker's address space.
 *
 * After `attach()` succeeds the socket stays open — the worker keeps
 * this MuninClient alive for its whole run, so Munin observes the
 * eventual peer-close as an implicit detach and can prune its
 * bookkeeping.
 *
 * Not thread-safe; construct one per worker. Blocking I/O throughout.
 * L0 context is a reference — must outlive this object.
 */
class MuninClient {
public:
    explicit MuninClient(::mimirmind::core::l0::L0Context& l0);
    ~MuninClient();

    MuninClient(const MuninClient&)            = delete;
    MuninClient& operator=(const MuninClient&) = delete;
    MuninClient(MuninClient&&)                 = delete;
    MuninClient& operator=(MuninClient&&)      = delete;

    [[nodiscard]] static std::expected<HealthzResponse, std::string>
    healthz(std::string_view socketPath) noexcept;

    /**
     * Successful attach payload. The `tensors` vector is ready to be
     * moved into `WeightsMap(std::vector<GgufTensor>)`. The manifest is
     * kept alongside so the caller can log the model id + fingerprint
     * and — critically — verify the fingerprint against an expected
     * value before proceeding to compute.
     */
    struct AttachResult {
        TensorManifest                   manifest;
        std::vector<::mimirmind::core::gguf::GgufTensor> tensors;
    };

    [[nodiscard]] std::expected<AttachResult, std::string>
    attach(std::string_view socketPath, std::string_view modelId) noexcept;

    /// Diagnostic: session-alive check. False before attach() succeeds
    /// or after a detach.
    [[nodiscard]] bool isAttached() const noexcept { return _sessionFd >= 0; }

    /// Explicit detach: closes the session socket. Idempotent. Called
    /// implicitly by the destructor; callers only need this when they
    /// want to release the Munin-side session before the client object
    /// itself goes out of scope.
    void detach() noexcept;

private:
    ::mimirmind::core::l0::L0Context& _l0;
    int                               _sessionFd{-1};
};

} // namespace mimirmind::core::ipc