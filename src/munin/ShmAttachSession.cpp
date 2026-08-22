// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "munin/ShmAttachSession.hpp"

#include "core/ipc/ShmIpcExporter.hpp"
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

using ::mimirmind::core::ipc::HealthzResponse;
using ::mimirmind::core::ipc::IpcHandle;
using ::mimirmind::core::ipc::makeErrorJson;
using ::mimirmind::core::ipc::ModelSummaryWire;
using ::mimirmind::core::ipc::RequestEnvelope;
using ::mimirmind::core::ipc::ShmIpcExporter;
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

ShmAttachSession::ShmAttachSession(int                  fd,
                                   pid_t                peerPid,
                                   std::uint32_t        sessionId,
                                   const ShmModelStore& store)
    : _fd{fd}, _peerPid{peerPid}, _sessionId{sessionId}, _store{store} {}

ShmAttachSession::~ShmAttachSession() {
    if (_fd >= 0) {
        ::close(_fd);
        _fd = -1;
    }
}

std::string ShmAttachSession::attachedModelId() const {
    return _attachedModelId;
}

void ShmAttachSession::run() noexcept {
    MM_LOG_INFO("munin",
                "shm-session#{} peer-pid={} connected", _sessionId, _peerPid);

    auto req = UnixSocketFrame::recv(_fd, /*maxPayloadBytes=*/64ULL * 1024);
    if (!req) {
        MM_LOG_WARN("munin",
                    "shm-session#{} recv failed: {}", _sessionId, req.error());
        return;
    }
    if (!req->fds.empty()) {
        for (int f : req->fds) ::close(f);
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
            waitForPeerClose();
        }
        return;
    }

    std::string msg{"unknown op '"};
    msg.append(env->op).append("'");
    sendErrorAndClose(msg);
}

void ShmAttachSession::sendErrorAndClose(std::string_view msg) noexcept {
    MM_LOG_WARN("munin", "shm-session#{}: {}", _sessionId, msg);
    const std::string body = makeErrorJson(msg);
    (void)UnixSocketFrame::send(_fd, asBytes(body));
}

bool ShmAttachSession::handleHealthz() noexcept {
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
                    "shm-session#{}: healthz send failed: {}",
                    _sessionId, s.error());
        return false;
    }
    MM_LOG_INFO("munin",
                "shm-session#{}: healthz ok, {} model(s) reported",
                _sessionId, r.models.size());
    return true;
}

bool ShmAttachSession::handleAttach(std::string_view modelId) noexcept {
    const ShmLoadedModel* lm = _store.find(modelId);
    if (lm == nullptr) {
        std::string msg{"attach: no model with id '"};
        msg.append(modelId).append("' loaded in Munin");
        sendErrorAndClose(msg);
        return false;
    }
    if (!lm->chunks) {
        std::string msg{"attach: model '"};
        msg.append(lm->id).append("' has no ShmChunkAllocator — Munin load is inconsistent");
        sendErrorAndClose(msg);
        return false;
    }
    if (lm->reader == nullptr || lm->reader->tensorCount() == 0) {
        std::string msg{"attach: model '"};
        msg.append(lm->id).append("' has no tensors — Munin load is inconsistent");
        sendErrorAndClose(msg);
        return false;
    }

    const TensorManifest manifest = lm->buildManifest();
    const std::string    manifestJson = manifest.toJson();

    // Export every chunk up front so a failure aborts cleanly before we
    // touch the socket. The fds are BORROWED from the allocator (kept open
    // for the model's lifetime) — SCM_RIGHTS dups them into the worker.
    const std::vector<int>           memfds   = lm->chunks->chunkMemfds();
    const std::vector<std::uint64_t> mapBytes = lm->chunks->chunkMapBytes();
    ShmIpcExporter exporter{std::span<const int>{memfds},
                            std::span<const std::uint64_t>{mapBytes}};

    const std::uint32_t nChunks = lm->chunks->chunkCount();
    std::vector<IpcHandle> handles;
    handles.reserve(nChunks);
    for (std::uint32_t i = 0; i < nChunks; ++i) {
        auto h = exporter.exportChunk(i, lm->chunks->chunkBase(i),
                                      lm->chunks->chunkUsedBytes(i));
        if (!h) {
            std::string msg{"attach: ShmIpcExporter failed for chunk["};
            msg.append(std::to_string(i)).append("]: ").append(h.error());
            sendErrorAndClose(msg);
            return false;
        }
        handles.push_back(*h);
    }

    // Manifest first — tells the worker how many HANDLE frames to expect.
    if (auto s = UnixSocketFrame::send(_fd, asBytes(manifestJson)); !s) {
        MM_LOG_WARN("munin",
                    "shm-session#{}: attach manifest send failed: {}",
                    _sessionId, s.error());
        return false;
    }

    // Then one HANDLE frame per chunk: 64-byte payload + one SCM_RIGHTS memfd.
    for (std::size_t i = 0; i < handles.size(); ++i) {
        const int fds[1] = {handles[i].fd};
        const auto payload = std::span<const std::byte>{handles[i].bytes.data(),
                                                        handles[i].bytes.size()};
        if (auto s = UnixSocketFrame::send(_fd, payload, std::span<const int>{fds, 1});
            !s) {
            MM_LOG_WARN("munin",
                        "shm-session#{}: attach chunk-handle[{}] send failed: {}",
                        _sessionId, i, s.error());
            return false;
        }
    }

    _attachedModelId = std::string{modelId};
    MM_LOG_INFO("munin",
                "shm-session#{}: attach ok, model='{}' tensors={} chunks={} "
                "fingerprint='{}'",
                _sessionId, _attachedModelId,
                lm->reader->tensorCount(), handles.size(), lm->fingerprint);
    return true;
}

void ShmAttachSession::waitForPeerClose() noexcept {
    for (;;) {
        if (_stopRequested.load()) {
            return;
        }
        char buf[64];
        const ssize_t r = ::recv(_fd, buf, sizeof(buf), 0);
        if (r == 0) {
            MM_LOG_INFO("munin",
                        "shm-session#{}: peer closed (model='{}') — implicit detach",
                        _sessionId, _attachedModelId);
            return;
        }
        if (r < 0) {
            if (errno == EINTR) {
                continue;
            }
            MM_LOG_WARN("munin",
                        "shm-session#{}: recv error: {} (errno={})",
                        _sessionId, std::strerror(errno), errno);
            return;
        }
        MM_LOG_WARN("munin",
                    "shm-session#{}: unexpected {} bytes from attached worker "
                    "(model='{}') — tearing down",
                    _sessionId, r, _attachedModelId);
        return;
    }
}

} // namespace mimirmind::munin
