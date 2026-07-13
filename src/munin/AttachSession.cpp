#include "munin/AttachSession.hpp"

#include "core/ipc/IpcExporter.hpp"
#include "core/ipc/TensorManifest.hpp"
#include "core/ipc/UnixSocketFrame.hpp"
#include "core/ipc/WireOps.hpp"
#include "core/log/Log.hpp"

#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstddef>
#include <cstring>
#include <span>
#include <string>
#include <vector>

namespace mimirmind::munin {

using ::mimirmind::core::ipc::Frame;
using ::mimirmind::core::ipc::HealthzResponse;
using ::mimirmind::core::ipc::IpcExporter;
using ::mimirmind::core::ipc::makeErrorJson;
using ::mimirmind::core::ipc::ModelSummaryWire;
using ::mimirmind::core::ipc::RequestEnvelope;
using ::mimirmind::core::ipc::TensorManifest;
using ::mimirmind::core::ipc::UnixSocketFrame;
namespace op = ::mimirmind::core::ipc::op;

namespace {

std::span<const std::byte> asBytes(const std::string& s) {
    return {reinterpret_cast<const std::byte*>(s.data()), s.size()};
}

std::string_view asStringView(std::span<const std::byte> b) {
    return {reinterpret_cast<const char*>(b.data()), b.size()};
}

} // namespace

AttachSession::AttachSession(int                                        fd,
                             pid_t                                      peerPid,
                             std::uint32_t                              sessionId,
                             const ModelStore&                          store,
                             ::mimirmind::core::l0::L0Context&          l0)
    : _fd{fd},
      _peerPid{peerPid},
      _sessionId{sessionId},
      _store{store},
      _l0{l0} {}

AttachSession::~AttachSession() {
    if (_fd >= 0) {
        ::close(_fd);
        _fd = -1;
    }
}

std::string AttachSession::attachedModelId() const {
    return _attachedModelId;
}

void AttachSession::run() noexcept {
    MM_LOG_INFO("munin",
                "session#{} peer-pid={} connected", _sessionId, _peerPid);

    // Read one request envelope. Cap payload — the envelope is a small
    // JSON object; anything above 64 KiB is malicious or wrong.
    auto req = UnixSocketFrame::recv(_fd, /*maxPayloadBytes=*/64ULL * 1024);
    if (!req) {
        MM_LOG_WARN("munin",
                    "session#{} recv failed: {}",
                    _sessionId, req.error());
        return;
    }
    if (!req->fds.empty()) {
        // The worker never sends fds to Munin. Anything present is an
        // out-of-spec peer — close the fds so we don't leak them, then
        // refuse the request.
        for (int f : req->fds) {
            ::close(f);
        }
        sendErrorAndClose("request must not carry SCM_RIGHTS fds");
        return;
    }

    auto env = RequestEnvelope::fromJson(asStringView(req->payload));
    if (!env) {
        sendErrorAndClose(env.error());
        return;
    }

    if (env->op == op::kHealthz) {
        handleHealthz();
        return;
    }
    if (env->op == op::kAttach) {
        if (env->modelId.empty()) {
            sendErrorAndClose("attach: 'modelId' is required");
            return;
        }
        if (handleAttach(env->modelId)) {
            // Attached — keep the connection open until the worker
            // detaches by closing its end.
            waitForPeerClose();
        }
        return;
    }

    std::string msg{"unknown op '"};
    msg.append(env->op).append("'");
    sendErrorAndClose(msg);
}

void AttachSession::sendErrorAndClose(std::string_view msg) noexcept {
    MM_LOG_WARN("munin",
                "session#{}: {}", _sessionId, msg);
    const std::string body = makeErrorJson(msg);
    (void)UnixSocketFrame::send(_fd, asBytes(body));
    // Caller returns after this — destructor closes the fd.
}

bool AttachSession::handleHealthz() noexcept {
    HealthzResponse r{};
    r.pid = static_cast<std::uint32_t>(::getpid());
    for (const auto& s : _store.summaries()) {
        ModelSummaryWire w{};
        w.id          = s.id;
        w.fingerprint = s.fingerprint;
        w.totalBytes  = s.totalBytes;
        w.tensorCount = s.tensorCount;
        r.models.push_back(std::move(w));
    }

    const std::string body = r.toJson();
    if (auto s = UnixSocketFrame::send(_fd, asBytes(body)); !s) {
        MM_LOG_WARN("munin",
                    "session#{}: healthz send failed: {}",
                    _sessionId, s.error());
        return false;
    }
    MM_LOG_INFO("munin",
                "session#{}: healthz ok, {} model(s) reported",
                _sessionId, r.models.size());
    return true;
}

bool AttachSession::handleAttach(std::string_view modelId) noexcept {
    const LoadedModel* lm = _store.find(modelId);
    if (lm == nullptr) {
        std::string msg{"attach: no model with id '"};
        msg.append(modelId).append("' loaded in Munin");
        sendErrorAndClose(msg);
        return false;
    }

    // Build the manifest first — the client relies on this to know how
    // many HANDLE frames to expect.
    TensorManifest manifest = lm->buildManifest();
    const std::string manifestJson = manifest.toJson();

    // Export every tensor's IPC handle up front, before touching the
    // socket. If any fails, we abort the attach cleanly with an error
    // frame; the client sees no partial handoff.
    struct Exported {
        std::array<std::byte, 64> bytes{};
        int                       fd{-1};
    };
    std::vector<Exported> exports;
    exports.reserve(lm->reader->tensorCount());
    const auto& ts = lm->reader->tensors();
    for (std::size_t i = 0; i < ts.size(); ++i) {
        void* usmPtr = ts[i].usmPtr;
        if (usmPtr == nullptr) {
            std::string msg{"attach: tensor '"};
            msg.append(ts[i].name)
               .append("' has no USM pointer — Munin load is inconsistent");
            sendErrorAndClose(msg);
            return false;
        }
        auto e = IpcExporter::exportOne(_l0, usmPtr);
        if (!e) {
            std::string msg{"attach: IpcExporter failed for tensor '"};
            msg.append(ts[i].name).append("': ").append(e.error());
            sendErrorAndClose(msg);
            return false;
        }
        Exported x{};
        x.bytes = e->bytes;
        x.fd    = e->fd;
        exports.push_back(x);
    }

    // Send the manifest first. This tells the worker how many HANDLE
    // frames to expect and in what order.
    if (auto s = UnixSocketFrame::send(_fd, asBytes(manifestJson)); !s) {
        MM_LOG_WARN("munin",
                    "session#{}: attach manifest send failed: {}",
                    _sessionId, s.error());
        return false;
    }

    // Then one HANDLE frame per tensor: 64-byte payload + one SCM_RIGHTS
    // fd. The kernel dup's the fd across the socket into the worker's fd
    // table; the fd number we sent is only meaningful in our process, so
    // the worker's IpcImporter patches the received fd back into the
    // handle bytes before calling zeMemOpenIpcHandle.
    for (std::size_t i = 0; i < exports.size(); ++i) {
        const int fd = exports[i].fd;
        const auto payload =
            std::span<const std::byte>{exports[i].bytes.data(),
                                       exports[i].bytes.size()};
        const int fds[1] = {fd};
        if (auto s = UnixSocketFrame::send(_fd, payload, std::span<const int>{fds, 1});
            !s) {
            MM_LOG_WARN("munin",
                        "session#{}: attach handle[{}] '{}' send failed: {}",
                        _sessionId, i, ts[i].name, s.error());
            return false;
        }
    }

    _attachedModelId = std::string{modelId};
    MM_LOG_INFO("munin",
                "session#{}: attach ok, model='{}' tensors={} fingerprint='{}'",
                _sessionId, _attachedModelId, exports.size(), lm->fingerprint);
    return true;
}

void AttachSession::waitForPeerClose() noexcept {
    // The worker keeps the connection open for its lifetime; when it
    // closes we get EOF. We're not expecting any further protocol
    // messages, so any received bytes are treated as a protocol error
    // and we tear down. A shutdown from Munin closes _fd from the
    // SocketServer, which manifests here as recv-error and returns.
    for (;;) {
        if (_stopRequested.load()) {
            return;
        }
        char buf[64];
        const ssize_t r = ::recv(_fd, buf, sizeof(buf), 0);
        if (r == 0) {
            MM_LOG_INFO("munin",
                        "session#{}: peer closed (model='{}') — implicit detach",
                        _sessionId, _attachedModelId);
            return;
        }
        if (r < 0) {
            if (errno == EINTR) {
                continue;
            }
            MM_LOG_WARN("munin",
                        "session#{}: recv error: {} (errno={})",
                        _sessionId, std::strerror(errno), errno);
            return;
        }
        // Data received after a successful attach: protocol misuse.
        MM_LOG_WARN("munin",
                    "session#{}: unexpected {} bytes from attached worker "
                    "(model='{}') — tearing down",
                    _sessionId, r, _attachedModelId);
        return;
    }
}

} // namespace mimirmind::munin