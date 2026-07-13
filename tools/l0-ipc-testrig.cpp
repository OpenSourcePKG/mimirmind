// L0-IPC-Testrig — does Level Zero IPC actually work on Meteor Lake?
//
// This is the go/no-go gate for M-Munin (see
// Memory/mimirmind/decisions/2026-07-13-m-munin-scope.md). If two
// processes on the same host can share a USM allocation via
// zeMemGetIpcHandle / zeMemOpenIpcHandle, we can build the persistent
// model-daemon. If not, we fall back to hot-reload-only (Option B in
// research/munin-persistent-model-daemon.md) and the ADR is void.
//
// Usage (run BOTH commands, roughly concurrently):
//
//   ./l0_ipc_testrig owner    <socket> [--kind shared|host|device]
//   ./l0_ipc_testrig attacher <socket> [--kind shared|host|device]
//
// Both processes must agree on --kind (shared/host is CPU-derefable;
// device is not — attacher only exercises the IPC handshake, no
// buffer content is verified).
//
// Or use the tools/l0-ipc-testrig.sh helper which launches both in
// sequence for each kind and reports PASS/FAIL per variant.
//
// Protocol:
//   1. Owner: zeInit -> zeContextCreate -> zeMemAllocShared (64 MiB) ->
//      fill with owner-pattern -> zeMemGetIpcHandle -> listen on Unix
//      socket, accept ONE connection, send handle bytes + dma_buf fd
//      via SCM_RIGHTS -> wait for attacher-done -> verify buffer now
//      contains attacher's pattern in first 4 KiB, owner's pattern
//      everywhere else -> report PASS or FAIL.
//   2. Attacher: zeInit -> connect Unix socket -> recvmsg for handle
//      bytes + SCM_RIGHTS fd -> patch handle with received fd ->
//      zeMemOpenIpcHandle -> verify owner-pattern is visible -> write
//      attacher-pattern to first 4 KiB -> zeMemCloseIpcHandle -> notify
//      owner -> report PASS or FAIL.
//
// What PASS proves:
//   - Level Zero IPC is functional on Meteor Lake iGPU with the
//     current i915/level-zero-gpu combo (both sides can allocate + share)
//   - The dma_buf FD encoding in ze_ipc_mem_handle_t works with
//     SCM_RIGHTS transport
//   - Attacher can read owner's writes (data visible before attach)
//   - Owner can read attacher's writes (data visible after attach's
//     zeMemCloseIpcHandle + attacher exit)
//
// What PASS does NOT prove:
//   - Cross-container IPC (would need a second test with shared bind-mount)
//   - Robustness under owner crash mid-transaction
//   - Cross-context IPC (both processes create their own zeContext here)
//
// Both should probably follow if the basic same-container test passes,
// but we gate M-Munin decisions on this test first.

#include <level_zero/ze_api.h>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

namespace {

enum class AllocKind {
    Shared,
    Host,
    Device,
};

const char* allocKindName(AllocKind k) {
    switch (k) {
        case AllocKind::Shared: return "shared";
        case AllocKind::Host:   return "host";
        case AllocKind::Device: return "device";
    }
    return "?";
}

// Decode the ze_result_t codes we're most likely to hit into a
// readable tag. Not exhaustive — just the ones we care about
// for IPC diagnosis.
const char* zeResultName(ze_result_t r) {
    switch (r) {
        case ZE_RESULT_SUCCESS:                             return "SUCCESS";
        case ZE_RESULT_NOT_READY:                           return "NOT_READY";
        case ZE_RESULT_ERROR_UNINITIALIZED:                 return "UNINITIALIZED";
        case ZE_RESULT_ERROR_DEVICE_LOST:                   return "DEVICE_LOST";
        case ZE_RESULT_ERROR_INVALID_ARGUMENT:              return "INVALID_ARGUMENT";
        case ZE_RESULT_ERROR_INVALID_NULL_HANDLE:           return "INVALID_NULL_HANDLE";
        case ZE_RESULT_ERROR_INVALID_NULL_POINTER:          return "INVALID_NULL_POINTER";
        case ZE_RESULT_ERROR_UNSUPPORTED_FEATURE:           return "UNSUPPORTED_FEATURE";
        case ZE_RESULT_ERROR_UNSUPPORTED_VERSION:           return "UNSUPPORTED_VERSION";
        case ZE_RESULT_ERROR_INVALID_ENUMERATION:           return "INVALID_ENUMERATION";
        case ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION:       return "UNSUPPORTED_ENUMERATION";
        case ZE_RESULT_ERROR_UNSUPPORTED_SIZE:              return "UNSUPPORTED_SIZE";
        case ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT:         return "UNSUPPORTED_ALIGNMENT";
        case ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT:return "INVALID_SYNCHRONIZATION_OBJECT";
        case ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY:            return "OUT_OF_HOST_MEMORY";
        case ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY:          return "OUT_OF_DEVICE_MEMORY";
        case ZE_RESULT_ERROR_MODULE_BUILD_FAILURE:          return "MODULE_BUILD_FAILURE";
        case ZE_RESULT_ERROR_UNKNOWN:                       return "UNKNOWN";
        default:                                            return "(other)";
    }
}

// Kernel-space canonical addresses on x86_64 Linux start at
// 0xffff_8000_0000_0000. If zeMemOpenIpcHandle returns a pointer in
// that range, dereferencing it segfaults — the driver mis-mapped the
// dma_buf and there's nothing we can do with the pointer.
bool looksLikeKernelPointer(const void* p) {
    const auto v = reinterpret_cast<std::uintptr_t>(p);
    return (v & 0xffff'0000'0000'0000ULL) == 0xffff'0000'0000'0000ULL;
}

// 64 MiB is large enough to force a genuine allocation path (not a
// fast-path shortcut) but small enough that fill loops complete in
// well under a second.
constexpr std::size_t   kBufferBytes      = 64ULL * 1024ULL * 1024ULL;
constexpr std::size_t   kBufferWords      = kBufferBytes / sizeof(std::uint32_t);

// First 4 KiB (1024 words) is what the attacher rewrites. The rest
// must stay owner-owned when the owner checks post-attach.
constexpr std::size_t   kAttacherWriteWords = 1024;

constexpr std::uint32_t kOwnerPattern      = 0xA5A5A5A5U;
constexpr std::uint32_t kAttacherPattern   = 0x5A5A5A5AU;

#define ZE_CHECK(call, tag)                                           \
    do {                                                              \
        const ze_result_t _r = (call);                                \
        if (_r != ZE_RESULT_SUCCESS) {                                \
            std::fprintf(stderr, "[%s] %s -> 0x%x\n",                 \
                         (tag), #call, static_cast<unsigned>(_r));    \
            return 1;                                                 \
        }                                                             \
    } while (false)

int initLevelZero(ze_driver_handle_t&  drv,
                  ze_device_handle_t&  dev,
                  ze_context_handle_t& ctx,
                  const char*          tag) {
    ZE_CHECK(zeInit(0), tag);

    std::uint32_t drvCount = 0;
    ZE_CHECK(zeDriverGet(&drvCount, nullptr), tag);
    if (drvCount == 0) {
        std::fprintf(stderr, "[%s] no L0 drivers\n", tag);
        return 1;
    }
    std::vector<ze_driver_handle_t> drivers(drvCount);
    ZE_CHECK(zeDriverGet(&drvCount, drivers.data()), tag);
    drv = drivers[0];

    std::uint32_t devCount = 0;
    ZE_CHECK(zeDeviceGet(drv, &devCount, nullptr), tag);
    if (devCount == 0) {
        std::fprintf(stderr, "[%s] no L0 devices\n", tag);
        return 1;
    }
    std::vector<ze_device_handle_t> devs(devCount);
    ZE_CHECK(zeDeviceGet(drv, &devCount, devs.data()), tag);
    // Meteor Lake with no dGPU: dev 0 is the iGPU. If a system ever
    // hosts a discrete GPU alongside, this pick may need refining;
    // the test rig deliberately keeps that concern out of scope.
    dev = devs[0];

    ze_device_properties_t devProps{};
    devProps.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    ZE_CHECK(zeDeviceGetProperties(dev, &devProps), tag);
    std::fprintf(stderr,
                 "[%s] driver=%p device=%p name='%s' vendorId=0x%x deviceId=0x%x\n",
                 tag,
                 static_cast<void*>(drv),
                 static_cast<void*>(dev),
                 devProps.name,
                 static_cast<unsigned>(devProps.vendorId),
                 static_cast<unsigned>(devProps.deviceId));

    ze_context_desc_t ctxDesc{};
    ctxDesc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
    ZE_CHECK(zeContextCreate(drv, &ctxDesc, &ctx), tag);
    return 0;
}

int listenUnix(const std::string& path) {
    ::unlink(path.c_str());
    const int s = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (s < 0) {
        std::perror("socket");
        return -1;
    }
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);
    if (::bind(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        std::perror("bind");
        ::close(s);
        return -1;
    }
    if (::listen(s, 1) < 0) {
        std::perror("listen");
        ::close(s);
        return -1;
    }
    return s;
}

int connectUnix(const std::string& path) {
    const int s = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (s < 0) {
        std::perror("socket");
        return -1;
    }
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);
    if (::connect(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        std::perror("connect");
        ::close(s);
        return -1;
    }
    return s;
}

// Send handle.data + one fd via SCM_RIGHTS. Returns 0 on success.
int sendIpcHandle(int sock, const ze_ipc_mem_handle_t& handle, int fd) {
    char cmsgBuf[CMSG_SPACE(sizeof(int))]{};
    iovec iov{};
    iov.iov_base = const_cast<char*>(handle.data);
    iov.iov_len  = sizeof(handle.data);
    msghdr msg{};
    msg.msg_iov         = &iov;
    msg.msg_iovlen      = 1;
    msg.msg_control     = cmsgBuf;
    msg.msg_controllen  = sizeof(cmsgBuf);
    cmsghdr* c = CMSG_FIRSTHDR(&msg);
    c->cmsg_level = SOL_SOCKET;
    c->cmsg_type  = SCM_RIGHTS;
    c->cmsg_len   = CMSG_LEN(sizeof(int));
    std::memcpy(CMSG_DATA(c), &fd, sizeof(int));
    const ssize_t n = ::sendmsg(sock, &msg, 0);
    if (n < 0) {
        std::perror("sendmsg");
        return -1;
    }
    if (static_cast<std::size_t>(n) != sizeof(handle.data)) {
        std::fprintf(stderr,
                     "sendmsg: short write %zd of %zu\n",
                     n, sizeof(handle.data));
        return -1;
    }
    return 0;
}

// Receive handle.data + one fd via SCM_RIGHTS. Sets outFd = -1 on no
// control message; caller may then abort.
int recvIpcHandle(int sock, ze_ipc_mem_handle_t& handle, int& outFd) {
    char cmsgBuf[CMSG_SPACE(sizeof(int))]{};
    iovec iov{};
    iov.iov_base = handle.data;
    iov.iov_len  = sizeof(handle.data);
    msghdr msg{};
    msg.msg_iov        = &iov;
    msg.msg_iovlen     = 1;
    msg.msg_control    = cmsgBuf;
    msg.msg_controllen = sizeof(cmsgBuf);
    const ssize_t n = ::recvmsg(sock, &msg, 0);
    if (n < 0) {
        std::perror("recvmsg");
        return -1;
    }
    outFd = -1;
    for (cmsghdr* c = CMSG_FIRSTHDR(&msg); c != nullptr; c = CMSG_NXTHDR(&msg, c)) {
        if (c->cmsg_level == SOL_SOCKET && c->cmsg_type == SCM_RIGHTS) {
            std::memcpy(&outFd, CMSG_DATA(c), sizeof(int));
            break;
        }
    }
    return 0;
}

void dumpHandle(const char* tag, const ze_ipc_mem_handle_t& handle) {
    std::fprintf(stderr, "[%s] ipc handle bytes: ", tag);
    for (std::size_t i = 0; i < sizeof(handle.data); ++i) {
        std::fprintf(stderr, "%02x", static_cast<unsigned char>(handle.data[i]));
    }
    int firstAsInt = 0;
    std::memcpy(&firstAsInt, handle.data, sizeof(int));
    std::fprintf(stderr, "  (first int=%d)\n", firstAsInt);
}

int runOwner(const std::string& sockPath, AllocKind kind) {
    ze_driver_handle_t  drv = nullptr;
    ze_device_handle_t  dev = nullptr;
    ze_context_handle_t ctx = nullptr;
    if (initLevelZero(drv, dev, ctx, "owner") != 0) {
        std::fprintf(stderr, "RESULT: FAIL (initLevelZero)\n");
        return 1;
    }
    std::fprintf(stderr, "[owner] alloc kind: %s\n", allocKindName(kind));

    ze_device_mem_alloc_desc_t devDesc{};
    devDesc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;

    ze_host_mem_alloc_desc_t hostDesc{};
    hostDesc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
    // BIAS_INITIAL_PLACEMENT is a hint. Only applies to Shared+Host.
    hostDesc.flags = ZE_HOST_MEM_ALLOC_FLAG_BIAS_INITIAL_PLACEMENT;

    void* ptr = nullptr;
    ze_result_t allocRes = ZE_RESULT_SUCCESS;
    switch (kind) {
        case AllocKind::Shared:
            allocRes = zeMemAllocShared(ctx, &devDesc, &hostDesc,
                                        kBufferBytes, /*alignment=*/64,
                                        dev, &ptr);
            break;
        case AllocKind::Host:
            allocRes = zeMemAllocHost(ctx, &hostDesc,
                                      kBufferBytes, /*alignment=*/64,
                                      &ptr);
            break;
        case AllocKind::Device:
            allocRes = zeMemAllocDevice(ctx, &devDesc,
                                        kBufferBytes, /*alignment=*/64,
                                        dev, &ptr);
            break;
    }
    if (allocRes != ZE_RESULT_SUCCESS) {
        std::fprintf(stderr,
                     "[owner] alloc(%s) -> %s (0x%x)\n",
                     allocKindName(kind), zeResultName(allocRes),
                     static_cast<unsigned>(allocRes));
        std::fprintf(stderr, "RESULT: FAIL (alloc)\n");
        return 1;
    }
    std::fprintf(stderr, "[owner] allocated %zu bytes at %p\n", kBufferBytes, ptr);

    // Fill with owner pattern — only when the pointer is CPU-derefable.
    // Device allocations return a device-only address that the CPU
    // cannot touch; leave the buffer untouched and let the attacher
    // side just verify the IPC handshake.
    const bool cpuDerefable = (kind != AllocKind::Device);
    if (cpuDerefable) {
        auto* words = static_cast<std::uint32_t*>(ptr);
        for (std::size_t i = 0; i < kBufferWords; ++i) {
            words[i] = kOwnerPattern;
        }
    } else {
        std::fprintf(stderr,
                     "[owner] device-mem: skipping CPU fill (device-only pointer)\n");
    }

    // Extract IPC handle.
    ze_ipc_mem_handle_t ipcHandle{};
    const ze_result_t ipcRes = zeMemGetIpcHandle(ctx, ptr, &ipcHandle);
    if (ipcRes != ZE_RESULT_SUCCESS) {
        std::fprintf(stderr,
                     "[owner] zeMemGetIpcHandle -> 0x%x\n",
                     static_cast<unsigned>(ipcRes));
        std::fprintf(stderr, "RESULT: FAIL (zeMemGetIpcHandle)\n");
        return 1;
    }
    dumpHandle("owner", ipcHandle);

    // On Linux+i915+level-zero-gpu, the handle typically encodes a
    // dma_buf fd in the first sizeof(int) bytes. We pass that fd out
    // of band via SCM_RIGHTS so the receiving process gets a valid
    // fd in ITS own descriptor table.
    int senderFd = -1;
    std::memcpy(&senderFd, ipcHandle.data, sizeof(int));
    std::fprintf(stderr, "[owner] sender-side fd from handle = %d\n", senderFd);

    // Listen + accept + send handle.
    const int listenFd = listenUnix(sockPath);
    if (listenFd < 0) {
        std::fprintf(stderr, "RESULT: FAIL (listenUnix)\n");
        return 1;
    }
    std::fprintf(stderr, "[owner] listening on %s, waiting for attacher…\n", sockPath.c_str());
    const int clientFd = ::accept(listenFd, nullptr, nullptr);
    if (clientFd < 0) {
        std::perror("accept");
        std::fprintf(stderr, "RESULT: FAIL (accept)\n");
        return 1;
    }
    std::fprintf(stderr, "[owner] attacher connected\n");

    if (sendIpcHandle(clientFd, ipcHandle, senderFd) != 0) {
        std::fprintf(stderr, "RESULT: FAIL (sendmsg)\n");
        return 1;
    }
    std::fprintf(stderr, "[owner] handle sent, waiting for attacher-done…\n");

    // Wait for attacher's single-byte "done" signal.
    char done = 0;
    const ssize_t r = ::read(clientFd, &done, 1);
    if (r != 1) {
        std::fprintf(stderr,
                     "[owner] read done: r=%zd errno=%d\n", r, errno);
        std::fprintf(stderr, "RESULT: FAIL (owner missed attacher-done)\n");
        return 1;
    }

    bool passOK = true;
    if (cpuDerefable) {
        // Verify: first 4 KiB should now be attacher pattern, rest
        // untouched (still owner pattern).
        const auto* words = static_cast<const std::uint32_t*>(ptr);
        bool attacherWritesVisible = true;
        for (std::size_t i = 0; i < kAttacherWriteWords; ++i) {
            if (words[i] != kAttacherPattern) {
                attacherWritesVisible = false;
                std::fprintf(stderr,
                             "[owner] word %zu = 0x%08x (expected attacher 0x%08x)\n",
                             i, words[i], kAttacherPattern);
                break;
            }
        }
        bool ownerRegionIntact = true;
        for (std::size_t i = kAttacherWriteWords; i < kBufferWords; i += 4096) {
            if (words[i] != kOwnerPattern) {
                ownerRegionIntact = false;
                std::fprintf(stderr,
                             "[owner] word %zu = 0x%08x (expected owner 0x%08x)\n",
                             i, words[i], kOwnerPattern);
                break;
            }
        }
        std::fprintf(stderr,
                     "[owner] attacherWritesVisible=%d ownerRegionIntact=%d\n",
                     static_cast<int>(attacherWritesVisible),
                     static_cast<int>(ownerRegionIntact));
        passOK = attacherWritesVisible && ownerRegionIntact;
    } else {
        std::fprintf(stderr,
                     "[owner] device-mem: skipping CPU verify (attacher handshake only)\n");
    }

    // Cleanup.
    ::close(clientFd);
    ::close(listenFd);
    ::unlink(sockPath.c_str());
    zeMemFree(ctx, ptr);
    zeContextDestroy(ctx);

    if (passOK) {
        std::fprintf(stderr, "RESULT: PASS (owner, kind=%s)\n", allocKindName(kind));
        return 0;
    }
    std::fprintf(stderr, "RESULT: FAIL (owner post-attach verify, kind=%s)\n",
                 allocKindName(kind));
    return 1;
}

int runAttacher(const std::string& sockPath, AllocKind kind) {
    ze_driver_handle_t  drv = nullptr;
    ze_device_handle_t  dev = nullptr;
    ze_context_handle_t ctx = nullptr;
    if (initLevelZero(drv, dev, ctx, "attacher") != 0) {
        std::fprintf(stderr, "RESULT: FAIL (initLevelZero)\n");
        return 1;
    }
    std::fprintf(stderr, "[attacher] alloc kind (expected): %s\n", allocKindName(kind));

    const int sockFd = connectUnix(sockPath);
    if (sockFd < 0) {
        std::fprintf(stderr, "RESULT: FAIL (connectUnix)\n");
        return 1;
    }

    ze_ipc_mem_handle_t ipcHandle{};
    int recvedFd = -1;
    if (recvIpcHandle(sockFd, ipcHandle, recvedFd) != 0) {
        std::fprintf(stderr, "RESULT: FAIL (recvmsg)\n");
        return 1;
    }
    dumpHandle("attacher", ipcHandle);
    std::fprintf(stderr, "[attacher] SCM_RIGHTS fd = %d\n", recvedFd);
    if (recvedFd < 0) {
        std::fprintf(stderr, "RESULT: FAIL (no fd in SCM_RIGHTS)\n");
        return 1;
    }

    // The sender's fd number is meaningless in our process. Patch the
    // handle's first int with the fd number we actually received.
    int senderFdEmbedded = 0;
    std::memcpy(&senderFdEmbedded, ipcHandle.data, sizeof(int));
    if (senderFdEmbedded != recvedFd) {
        std::fprintf(stderr,
                     "[attacher] patching handle: sender-embedded fd=%d -> local recv fd=%d\n",
                     senderFdEmbedded, recvedFd);
        std::memcpy(ipcHandle.data, &recvedFd, sizeof(int));
    }

    void* ptr = nullptr;
    const ze_result_t openRes = zeMemOpenIpcHandle(ctx, dev, ipcHandle,
                                                    /*flags=*/0, &ptr);
    if (openRes != ZE_RESULT_SUCCESS) {
        std::fprintf(stderr,
                     "[attacher] ERROR zeMemOpenIpcHandle -> %s (0x%x)\n",
                     zeResultName(openRes),
                     static_cast<unsigned>(openRes));
        std::fprintf(stderr, "RESULT: FAIL (zeMemOpenIpcHandle)\n");
        return 1;
    }
    std::fprintf(stderr, "[attacher] OK    zeMemOpenIpcHandle ptr=%p\n", ptr);

    // Sanity: kernel-space pointer means the driver mis-mapped the
    // dma_buf; dereferencing would segfault. Report and abort cleanly.
    if (looksLikeKernelPointer(ptr)) {
        std::fprintf(stderr,
                     "[attacher] KERNEL-SPACE POINTER (%p) — driver mapped dma_buf outside user VA range\n",
                     ptr);
        std::fprintf(stderr, "RESULT: FAIL (bad-ptr, kind=%s)\n", allocKindName(kind));
        (void)zeMemCloseIpcHandle(ctx, ptr);
        char done = 0;   // 0 = "I failed"
        if (::write(sockFd, &done, 1) != 1) { /* best-effort */ }
        return 1;
    }

    bool passOK = true;
    if (kind == AllocKind::Device) {
        // Device-only pointer — the CPU cannot read/write it. The
        // fact that zeMemOpenIpcHandle succeeded and returned a
        // non-kernel address is the pass criterion for this variant.
        std::fprintf(stderr,
                     "[attacher] device-mem: skipping CPU read/write (device-only pointer)\n");
    } else {
        // Spot-check owner pattern is visible. First 64 bytes (16 words)
        // is enough to catch obvious "we got a completely different
        // buffer" failures.
        const auto* words = static_cast<const std::uint32_t*>(ptr);
        for (std::size_t i = 0; i < 16; ++i) {
            if (words[i] != kOwnerPattern) {
                passOK = false;
                std::fprintf(stderr,
                             "[attacher] word %zu = 0x%08x (expected 0x%08x)\n",
                             i, words[i], kOwnerPattern);
                break;
            }
        }
        std::fprintf(stderr, "[attacher] owner-pattern visible: %d\n",
                     static_cast<int>(passOK));

        // Write attacher pattern to first 4 KiB. Owner will verify this
        // after we close + signal done.
        auto* mwords = static_cast<std::uint32_t*>(ptr);
        for (std::size_t i = 0; i < kAttacherWriteWords; ++i) {
            mwords[i] = kAttacherPattern;
        }
        std::fprintf(stderr, "[attacher] wrote attacher pattern to first %zu bytes\n",
                     kAttacherWriteWords * sizeof(std::uint32_t));
    }

    ZE_CHECK(zeMemCloseIpcHandle(ctx, ptr), "attacher");

    // Signal done.
    char done = 1;
    if (::write(sockFd, &done, 1) != 1) {
        std::perror("write done");
    }
    ::close(sockFd);
    zeContextDestroy(ctx);

    if (passOK) {
        std::fprintf(stderr, "RESULT: PASS (attacher, kind=%s)\n", allocKindName(kind));
        return 0;
    }
    std::fprintf(stderr, "RESULT: FAIL (attacher owner-pattern mismatch, kind=%s)\n",
                 allocKindName(kind));
    return 1;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "Usage:\n"
                     "  %s owner    <socket-path> [--kind shared|host|device]\n"
                     "  %s attacher <socket-path> [--kind shared|host|device]\n",
                     argv[0], argv[0]);
        return 2;
    }
    const std::string_view mode = argv[1];
    const std::string      sockPath = argv[2];
    AllocKind kind = AllocKind::Shared;
    for (int i = 3; i < argc; ++i) {
        const std::string_view a = argv[i];
        if (a == "--kind" && i + 1 < argc) {
            const std::string_view v = argv[++i];
            if      (v == "shared") kind = AllocKind::Shared;
            else if (v == "host")   kind = AllocKind::Host;
            else if (v == "device") kind = AllocKind::Device;
            else {
                std::fprintf(stderr, "unknown --kind: %s\n", argv[i]);
                return 2;
            }
        } else {
            std::fprintf(stderr, "unknown arg: %s\n", argv[i]);
            return 2;
        }
    }
    if (mode == "owner")    return runOwner(sockPath, kind);
    if (mode == "attacher") return runAttacher(sockPath, kind);
    std::fprintf(stderr, "unknown mode: %s\n", argv[1]);
    return 2;
}