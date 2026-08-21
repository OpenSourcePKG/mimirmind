// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// CUDA-IPC-Testrig — does CUDA IPC actually work on DGX Spark GB10?
//
// This is the go/no-go gate for the M-Munin CUDA/GB10 port (roadmap #6).
// The L0 Munin (Meteor Lake) shares USM cross-process via
// zeMemGetIpcHandle; the CUDA port needs the equivalent. GB10 is
// Grace-Blackwell UNIFIED memory (LPDDR5x, no discrete device RAM), and
// mimirmind allocates its weights with cudaMallocManaged (GpuOps:
// device-only cudaMalloc segfaults host reads on this part). Classic
// cudaIpcGetMemHandle is documented to support only cudaMalloc (device)
// memory, NOT managed — so the whole port hinges on WHICH mechanism can
// export the memory a long-lived Munin would hold. This rig answers that
// empirically, the exact analogue of tools/l0-ipc-testrig.cpp.
//
// Four mechanisms, selected by --kind:
//   device  : cudaMalloc + cudaIpcGetMemHandle / cudaIpcOpenMemHandle.
//             Opaque 64-byte handle, sent verbatim over the socket (no FD).
//   managed : cudaMallocManaged + cudaIpcGetMemHandle. Expected to fail on
//             most CUDA (managed unsupported); tested because GB10 unified
//             memory MIGHT behave differently. Documents the result.
//   vmm     : low-level VMM — cuMemCreate + cuMemExportToShareableHandle
//             (CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR); the FD is passed
//             via SCM_RIGHTS, attacher cuMemImportFromShareableHandle +
//             cuMemMap. The robust modern path (Dynamo / AnchorTP use it).
//   shm     : *** THE CHOSEN M-Munin path *** (ADR 2026-08-14,
//             "M-Munin.CUDA via POSIX-shm"). Owner is HOST-ONLY (models Munin,
//             which never touches the GPU): memfd_create + ftruncate + mmap =
//             plain host RAM, filled with a host loop, FD passed via
//             SCM_RIGHTS. Attacher mmaps the SAME memfd and hands the RAW
//             host pointer STRAIGHT to a real CUDA kernel — no
//             cudaHostRegister, no cudaMalloc, no cudaMemcpy — relying on
//             GB10 pageableMemAccess + hostPageTables (verified step 1a,
//             d352055). That kernel deref is the linchpin the whole shm plan
//             rests on; this is its first end-to-end empirical proof. The
//             kernel read-verifies the owner pattern and writes the attacher
//             pattern into the poke window; the owner re-reads host-side to
//             confirm the kernel's writes are coherent back on the host.
//
// This file is CUDA (.cu) — the shm path carries a real __global__ kernel, so
// it must be compiled by nvcc (the earlier device/managed/vmm-only version
// was a plain .cpp buildable with g++; that no longer holds).
//
// Usage (run BOTH, roughly concurrently — or use tools/cuda-ipc-testrig.sh):
//   ./cuda_ipc_testrig owner    <socket> --kind device|managed|vmm|shm
//   ./cuda_ipc_testrig attacher <socket> --kind device|managed|vmm|shm
//
// Protocol (mirrors the L0 rig):
//   Owner allocates 64 MiB, fills an owner-pattern, exports the handle,
//   listens on a Unix socket, accepts ONE attacher, sends the handle
//   (bytes for device/managed; SCM_RIGHTS FD for vmm), waits for the
//   attacher to finish, then verifies the first 4 KiB now hold the
//   attacher-pattern and the rest still hold the owner-pattern -> PASS/FAIL.
//   Attacher imports the handle, verifies it sees the owner-pattern, writes
//   its own pattern into the first 4 KiB, signals done -> PASS/FAIL.
//
// What PASS proves: same-host cross-process CUDA memory sharing works on
// GB10 for that kind, with reads AND writes visible both ways -> M-Munin
// CUDA port is viable on that mechanism. What it does NOT prove:
// cross-container (needs a shared IPC namespace / same --ipc), owner-crash
// robustness, cross-CUDA-context nuances beyond the single-GPU same-driver
// case exercised here.

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/un.h>
#include <unistd.h>

#ifndef MFD_CLOEXEC
#define MFD_CLOEXEC 0x0001U  // not always exposed by <sys/mman.h> without _GNU_SOURCE
#endif

namespace {

constexpr std::size_t kSize      = 64ull * 1024 * 1024; // 64 MiB
constexpr std::size_t kPokeBytes = 4096;                // attacher window
constexpr std::uint32_t kOwnerPat    = 0xA5A5A5A5u;
constexpr std::uint32_t kAttacherPat = 0x5A5A5A5Au;

[[noreturn]] void die(const std::string& msg) {
    std::fprintf(stderr, "FAIL: %s\n", msg.c_str());
    std::exit(2);
}

void ckRt(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        die(std::string(what) + ": " + cudaGetErrorString(e));
    }
}

void ckDrv(CUresult r, const char* what) {
    if (r != CUDA_SUCCESS) {
        const char* s = nullptr;
        cuGetErrorString(r, &s);
        die(std::string(what) + ": " + (s ? s : "unknown"));
    }
}

// ---- Unix-socket helpers (blocking, one-shot) ---------------------------

int listenSocket(const std::string& path) {
    ::unlink(path.c_str());
    const int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) die("socket()");
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);
    if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) die("bind()");
    if (::listen(fd, 1) < 0) die("listen()");
    return fd;
}

int connectSocket(const std::string& path) {
    const int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) die("socket()");
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);
    // Retry a few seconds so owner/attacher launch order does not matter.
    for (int i = 0; i < 200; ++i) {
        if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) return fd;
        ::usleep(25000);
    }
    die("connect() timed out");
}

void writeAll(int fd, const void* buf, std::size_t n) {
    const auto* p = static_cast<const std::uint8_t*>(buf);
    while (n > 0) {
        const ssize_t w = ::write(fd, p, n);
        if (w <= 0) die("write()");
        p += w;
        n -= static_cast<std::size_t>(w);
    }
}

void readAll(int fd, void* buf, std::size_t n) {
    auto* p = static_cast<std::uint8_t*>(buf);
    while (n > 0) {
        const ssize_t r = ::read(fd, p, n);
        if (r <= 0) die("read()");
        p += r;
        n -= static_cast<std::size_t>(r);
    }
}

// Send one FD over an fd-stream via SCM_RIGHTS (for the VMM path).
void sendFd(int sock, int fd) {
    char dummy = 'F';
    iovec iov{&dummy, 1};
    char cbuf[CMSG_SPACE(sizeof(int))]{};
    msghdr msg{};
    msg.msg_iov     = &iov;
    msg.msg_iovlen  = 1;
    msg.msg_control = cbuf;
    msg.msg_controllen = sizeof(cbuf);
    cmsghdr* cm = CMSG_FIRSTHDR(&msg);
    cm->cmsg_level = SOL_SOCKET;
    cm->cmsg_type  = SCM_RIGHTS;
    cm->cmsg_len   = CMSG_LEN(sizeof(int));
    std::memcpy(CMSG_DATA(cm), &fd, sizeof(int));
    if (::sendmsg(sock, &msg, 0) < 0) die("sendmsg(SCM_RIGHTS)");
}

int recvFd(int sock) {
    char dummy = 0;
    iovec iov{&dummy, 1};
    char cbuf[CMSG_SPACE(sizeof(int))]{};
    msghdr msg{};
    msg.msg_iov     = &iov;
    msg.msg_iovlen  = 1;
    msg.msg_control = cbuf;
    msg.msg_controllen = sizeof(cbuf);
    if (::recvmsg(sock, &msg, 0) < 0) die("recvmsg(SCM_RIGHTS)");
    cmsghdr* cm = CMSG_FIRSTHDR(&msg);
    if (cm == nullptr || cm->cmsg_type != SCM_RIGHTS) die("no SCM_RIGHTS fd received");
    int fd = -1;
    std::memcpy(&fd, CMSG_DATA(cm), sizeof(int));
    return fd;
}

enum class Kind { Device, Managed, Vmm, Shm };

Kind parseKind(std::string_view s) {
    if (s == "device")  return Kind::Device;
    if (s == "managed") return Kind::Managed;
    if (s == "vmm")     return Kind::Vmm;
    if (s == "shm")     return Kind::Shm;
    die("unknown --kind (expected device|managed|vmm|shm)");
}

// A host staging buffer for filling / verifying device memory (works for
// all kinds: managed is host-derefable but we go through cudaMemcpy so the
// same code path verifies device and vmm too).
std::vector<std::uint32_t> hostFilled(std::uint32_t pat) {
    std::vector<std::uint32_t> h(kSize / 4, pat);
    return h;
}

// =====================================================================
//  Classic runtime IPC (device / managed)
// =====================================================================

int runOwnerRuntime(const std::string& sock, Kind kind) {
    ckRt(cudaSetDevice(0), "cudaSetDevice");
    void* ptr = nullptr;
    if (kind == Kind::Managed) {
        ckRt(cudaMallocManaged(&ptr, kSize), "cudaMallocManaged");
    } else {
        ckRt(cudaMalloc(&ptr, kSize), "cudaMalloc");
    }
    auto h = hostFilled(kOwnerPat);
    ckRt(cudaMemcpy(ptr, h.data(), kSize, cudaMemcpyHostToDevice), "H2D fill");
    ckRt(cudaDeviceSynchronize(), "sync fill");

    cudaIpcMemHandle_t handle{};
    const cudaError_t ge = cudaIpcGetMemHandle(&handle, ptr);
    if (ge != cudaSuccess) {
        std::fprintf(stderr,
            "FAIL: cudaIpcGetMemHandle (kind=%s): %s\n",
            kind == Kind::Managed ? "managed" : "device",
            cudaGetErrorString(ge));
        return 2;  // clean documented failure (esp. expected for managed)
    }

    const int lfd = listenSocket(sock);
    const int cfd = ::accept(lfd, nullptr, nullptr);
    if (cfd < 0) die("accept()");
    writeAll(cfd, &handle, sizeof(handle));

    // Wait for the attacher to finish writing its window.
    char done = 0;
    readAll(cfd, &done, 1);

    // Verify: first 4 KiB = attacher-pattern, rest = owner-pattern.
    std::vector<std::uint32_t> back(kSize / 4, 0);
    ckRt(cudaMemcpy(back.data(), ptr, kSize, cudaMemcpyDeviceToHost), "D2H verify");
    ckRt(cudaDeviceSynchronize(), "sync verify");
    bool ok = true;
    for (std::size_t i = 0; i < kPokeBytes / 4; ++i)
        if (back[i] != kAttacherPat) { ok = false; break; }
    for (std::size_t i = kPokeBytes / 4; ok && i < back.size(); i += 4096)
        if (back[i] != kOwnerPat) { ok = false; break; }

    writeAll(cfd, "X", 1);  // release attacher
    ::close(cfd);
    ::close(lfd);
    ::unlink(sock.c_str());
    std::printf(ok ? "PASS owner (kind=%s): attacher writes visible\n"
                   : "FAIL owner (kind=%s): attacher writes NOT visible\n",
                kind == Kind::Managed ? "managed" : "device");
    return ok ? 0 : 2;
}

int runAttacherRuntime(const std::string& sock, Kind kind) {
    ckRt(cudaSetDevice(0), "cudaSetDevice");
    const int cfd = connectSocket(sock);
    cudaIpcMemHandle_t handle{};
    readAll(cfd, &handle, sizeof(handle));

    void* ptr = nullptr;
    const cudaError_t oe =
        cudaIpcOpenMemHandle(&ptr, handle, cudaIpcMemLazyEnablePeerAccess);
    if (oe != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaIpcOpenMemHandle (kind=%s): %s\n",
                     kind == Kind::Managed ? "managed" : "device",
                     cudaGetErrorString(oe));
        return 2;
    }

    // Verify owner-pattern is visible, then write attacher-pattern window.
    std::vector<std::uint32_t> back(kSize / 4, 0);
    ckRt(cudaMemcpy(back.data(), ptr, kSize, cudaMemcpyDeviceToHost), "D2H read");
    ckRt(cudaDeviceSynchronize(), "sync read");
    bool seen = true;
    for (std::size_t i = 0; i < back.size(); i += 4096)
        if (back[i] != kOwnerPat) { seen = false; break; }

    auto poke = hostFilled(kAttacherPat);
    ckRt(cudaMemcpy(ptr, poke.data(), kPokeBytes, cudaMemcpyHostToDevice), "H2D poke");
    ckRt(cudaDeviceSynchronize(), "sync poke");

    writeAll(cfd, "D", 1);          // tell owner we are done
    char x = 0; readAll(cfd, &x, 1); // wait for release
    ckRt(cudaIpcCloseMemHandle(ptr), "cudaIpcCloseMemHandle");
    ::close(cfd);
    std::printf(seen ? "PASS attacher (kind=%s): owner data visible\n"
                     : "FAIL attacher (kind=%s): owner data NOT visible\n",
                kind == Kind::Managed ? "managed" : "device");
    return seen ? 0 : 2;
}

// =====================================================================
//  VMM driver IPC (POSIX FD via SCM_RIGHTS)
// =====================================================================

CUmemAllocationProp vmmProp() {
    CUmemAllocationProp prop{};
    prop.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id   = 0;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    return prop;
}

std::size_t vmmGranularity(const CUmemAllocationProp& prop) {
    std::size_t g = 0;
    ckDrv(cuMemGetAllocationGranularity(&g, &prop,
                                        CU_MEM_ALLOC_GRANULARITY_MINIMUM),
          "cuMemGetAllocationGranularity");
    return g;
}

void vmmSetAccess(CUdeviceptr dptr, std::size_t sz) {
    CUmemAccessDesc acc{};
    acc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    acc.location.id   = 0;
    acc.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    ckDrv(cuMemSetAccess(dptr, sz, &acc, 1), "cuMemSetAccess");
}

int runOwnerVmm(const std::string& sock) {
    ckDrv(cuInit(0), "cuInit");
    CUdevice dev; ckDrv(cuDeviceGet(&dev, 0), "cuDeviceGet");
    CUcontext ctx; ckDrv(cuCtxCreate(&ctx, nullptr, 0, dev), "cuCtxCreate");

    const CUmemAllocationProp prop = vmmProp();
    const std::size_t gran = vmmGranularity(prop);
    const std::size_t sz   = ((kSize + gran - 1) / gran) * gran;

    CUmemGenericAllocationHandle mh;
    ckDrv(cuMemCreate(&mh, sz, &prop, 0), "cuMemCreate");

    int fd = -1;
    ckDrv(cuMemExportToShareableHandle(
              &fd, mh, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0),
          "cuMemExportToShareableHandle");

    CUdeviceptr dptr;
    ckDrv(cuMemAddressReserve(&dptr, sz, 0, 0, 0), "cuMemAddressReserve");
    ckDrv(cuMemMap(dptr, sz, 0, mh, 0), "cuMemMap");
    vmmSetAccess(dptr, sz);

    auto h = hostFilled(kOwnerPat);
    ckDrv(cuMemcpyHtoD(dptr, h.data(), kSize), "HtoD fill");
    ckDrv(cuCtxSynchronize(), "sync fill");

    const int lfd = listenSocket(sock);
    const int cfd = ::accept(lfd, nullptr, nullptr);
    if (cfd < 0) die("accept()");
    writeAll(cfd, &sz, sizeof(sz));  // attacher needs the padded size
    sendFd(cfd, fd);

    char done = 0; readAll(cfd, &done, 1);

    std::vector<std::uint32_t> back(kSize / 4, 0);
    ckDrv(cuMemcpyDtoH(back.data(), dptr, kSize), "DtoH verify");
    ckDrv(cuCtxSynchronize(), "sync verify");
    bool ok = true;
    for (std::size_t i = 0; i < kPokeBytes / 4; ++i)
        if (back[i] != kAttacherPat) { ok = false; break; }
    for (std::size_t i = kPokeBytes / 4; ok && i < back.size(); i += 4096)
        if (back[i] != kOwnerPat) { ok = false; break; }

    writeAll(cfd, "X", 1);
    ::close(fd); ::close(cfd); ::close(lfd);
    ::unlink(sock.c_str());
    std::printf(ok ? "PASS owner (kind=vmm): attacher writes visible\n"
                   : "FAIL owner (kind=vmm): attacher writes NOT visible\n");
    return ok ? 0 : 2;
}

int runAttacherVmm(const std::string& sock) {
    ckDrv(cuInit(0), "cuInit");
    CUdevice dev; ckDrv(cuDeviceGet(&dev, 0), "cuDeviceGet");
    CUcontext ctx; ckDrv(cuCtxCreate(&ctx, nullptr, 0, dev), "cuCtxCreate");

    const int cfd = connectSocket(sock);
    std::size_t sz = 0; readAll(cfd, &sz, sizeof(sz));
    const int fd = recvFd(cfd);

    CUmemGenericAllocationHandle mh;
    ckDrv(cuMemImportFromShareableHandle(
              &mh, reinterpret_cast<void*>(static_cast<std::intptr_t>(fd)),
              CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR),
          "cuMemImportFromShareableHandle");

    CUdeviceptr dptr;
    ckDrv(cuMemAddressReserve(&dptr, sz, 0, 0, 0), "cuMemAddressReserve");
    ckDrv(cuMemMap(dptr, sz, 0, mh, 0), "cuMemMap");
    vmmSetAccess(dptr, sz);

    std::vector<std::uint32_t> back(kSize / 4, 0);
    ckDrv(cuMemcpyDtoH(back.data(), dptr, kSize), "DtoH read");
    ckDrv(cuCtxSynchronize(), "sync read");
    bool seen = true;
    for (std::size_t i = 0; i < back.size(); i += 4096)
        if (back[i] != kOwnerPat) { seen = false; break; }

    auto poke = hostFilled(kAttacherPat);
    ckDrv(cuMemcpyHtoD(dptr, poke.data(), kPokeBytes), "HtoD poke");
    ckDrv(cuCtxSynchronize(), "sync poke");

    writeAll(cfd, "D", 1);
    char x = 0; readAll(cfd, &x, 1);
    ::close(fd); ::close(cfd);
    std::printf(seen ? "PASS attacher (kind=vmm): owner data visible\n"
                     : "FAIL attacher (kind=vmm): owner data NOT visible\n");
    return seen ? 0 : 2;
}

// =====================================================================
//  POSIX-shm (memfd) — THE CHOSEN M-Munin path
//
//  Owner never touches the GPU (models Munin): it only holds host RAM in a
//  memfd and hands the FD to the worker. The worker (attacher) mmaps the
//  same memfd and passes the RAW host pointer directly to a real kernel,
//  relying on GB10 pageableMemAccess + hostPageTables (no cudaHostRegister).
// =====================================================================

int makeMemfd(std::size_t sz) {
    // syscall wrapper so we do not depend on a glibc that exposes
    // memfd_create() in <sys/mman.h> (avoids _GNU_SOURCE ordering issues).
    const long fd = ::syscall(SYS_memfd_create, "munin-shm", MFD_CLOEXEC);
    if (fd < 0) die("memfd_create()");
    if (::ftruncate(static_cast<int>(fd), static_cast<off_t>(sz)) != 0)
        die("ftruncate(memfd)");
    return static_cast<int>(fd);
}

// The linchpin kernel: dereference an mmap'd shm host pointer DIRECTLY on the
// SMs. Read-verify every word outside the poke window against expectPat
// (flagging any mismatch), and write pokePat into the poke window. Reads skip
// the poke window so there is no read/write race on the same location.
__global__ void shmVerifyAndPoke(std::uint32_t* p, std::size_t nWords,
                                 std::size_t pokeWords, std::uint32_t expectPat,
                                 std::uint32_t pokePat, int* mismatch) {
    const std::size_t idx    = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = idx; i < nWords; i += stride) {
        if (i < pokeWords) {
            p[i] = pokePat;               // kernel WRITE through the shm pointer
        } else if (p[i] != expectPat) {   // kernel READ through the shm pointer
            atomicExch(mismatch, 1);
        }
    }
}

int runOwnerShm(const std::string& sock) {
    // HOST-ONLY, exactly like Munin: no CUDA calls at all on this side.
    const int mfd = makeMemfd(kSize);
    void* map = ::mmap(nullptr, kSize, PROT_READ | PROT_WRITE, MAP_SHARED, mfd, 0);
    if (map == MAP_FAILED) die("mmap(owner)");
    auto* words = static_cast<std::uint32_t*>(map);
    for (std::size_t i = 0; i < kSize / 4; ++i) words[i] = kOwnerPat;

    const int lfd = listenSocket(sock);
    const int cfd = ::accept(lfd, nullptr, nullptr);
    if (cfd < 0) die("accept()");
    sendFd(cfd, mfd);                 // hand the memfd to the worker

    char done = 0; readAll(cfd, &done, 1);  // wait for the worker's kernel

    // Verify host-side that the worker's KERNEL writes are coherent back on
    // the host: first 4 KiB = attacher-pattern, rest still owner-pattern.
    bool ok = true;
    for (std::size_t i = 0; i < kPokeBytes / 4; ++i)
        if (words[i] != kAttacherPat) { ok = false; break; }
    for (std::size_t i = kPokeBytes / 4; ok && i < kSize / 4; i += 4096)
        if (words[i] != kOwnerPat) { ok = false; break; }

    writeAll(cfd, "X", 1);
    ::munmap(map, kSize);
    ::close(mfd); ::close(cfd); ::close(lfd);
    ::unlink(sock.c_str());
    std::printf(ok ? "PASS owner (kind=shm): kernel writes visible host-side\n"
                   : "FAIL owner (kind=shm): kernel writes NOT visible host-side\n");
    return ok ? 0 : 2;
}

int runAttacherShm(const std::string& sock) {
    ckRt(cudaSetDevice(0), "cudaSetDevice");
    const int cfd = connectSocket(sock);
    const int mfd = recvFd(cfd);

    void* map = ::mmap(nullptr, kSize, PROT_READ | PROT_WRITE, MAP_SHARED, mfd, 0);
    if (map == MAP_FAILED) die("mmap(attacher)");
    auto* words = static_cast<std::uint32_t*>(map);

    // Sanity: shm is coherent host<->host before we bring in the GPU.
    bool hostSeen = true;
    for (std::size_t i = 0; i < kSize / 4; i += 4096)
        if (words[i] != kOwnerPat) { hostSeen = false; break; }

    // *** THE ACTUAL TEST ***: pass the raw mmap'd host pointer straight to a
    // kernel. No cudaHostRegister / cudaMalloc / cudaMemcpy. If GB10 cannot
    // deref host page tables on the SMs, this faults with an illegal address
    // -> clean go/no-go FAIL.
    int* mismatch = nullptr;
    ckRt(cudaMallocManaged(&mismatch, sizeof(int)), "cudaMallocManaged(flag)");
    *mismatch = 0;

    const std::size_t nWords    = kSize / 4;
    const std::size_t pokeWords = kPokeBytes / 4;
    constexpr int kBlock = 256;
    const int grid = 1024;  // grid-stride; covers the whole buffer
    shmVerifyAndPoke<<<grid, kBlock>>>(words, nWords, pokeWords,
                                       kOwnerPat, kAttacherPat, mismatch);
    const cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) die(std::string("kernel launch: ") + cudaGetErrorString(le));
    const cudaError_t se = cudaDeviceSynchronize();
    if (se != cudaSuccess) {
        std::fprintf(stderr,
            "FAIL attacher (kind=shm): kernel deref of shm pointer faulted: %s\n",
            cudaGetErrorString(se));
        return 2;
    }
    const bool kernelSeen = (*mismatch == 0);

    writeAll(cfd, "D", 1);
    char x = 0; readAll(cfd, &x, 1);
    cudaFree(mismatch);
    ::munmap(map, kSize);
    ::close(mfd); ::close(cfd);

    const bool ok = hostSeen && kernelSeen;
    std::printf(ok ? "PASS attacher (kind=shm): kernel read owner data via shm pointer\n"
                   : "FAIL attacher (kind=shm): host_seen=%d kernel_seen=%d\n",
                static_cast<int>(hostSeen), static_cast<int>(kernelSeen));
    return ok ? 0 : 2;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: %s owner|attacher <socket> [--kind device|managed|vmm]\n",
            argv[0]);
        return 1;
    }
    const std::string role = argv[1];
    const std::string sock = argv[2];
    Kind kind = Kind::Device;
    for (int i = 3; i < argc; ++i) {
        if (std::string_view(argv[i]) == "--kind" && i + 1 < argc) {
            kind = parseKind(argv[++i]);
        }
    }

    if (role == "owner") {
        if (kind == Kind::Vmm) return runOwnerVmm(sock);
        if (kind == Kind::Shm) return runOwnerShm(sock);
        return runOwnerRuntime(sock, kind);
    }
    if (role == "attacher") {
        if (kind == Kind::Vmm) return runAttacherVmm(sock);
        if (kind == Kind::Shm) return runAttacherShm(sock);
        return runAttacherRuntime(sock, kind);
    }
    std::fprintf(stderr, "unknown role '%s' (owner|attacher)\n", role.c_str());
    return 1;
}
