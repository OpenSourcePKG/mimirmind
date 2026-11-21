// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Hugin coexist rig — roadmap 7.0 de-risk on DGX Spark GB10.
//
// Hugin (the adaptation companion) must run next to a serving mimirmind on the
// SAME base weights: Kolibri-1 is ~40 GiB, so a second materialized copy does not
// fit beside serving. This rig answers the two questions that decide the design:
//
//   bw      Can a kernel read weights that live in a memfd (POSIX shm, shareable
//           across processes) as fast as from the allocations the engine uses
//           today? Streams one buffer per allocation kind and reports GB/s:
//             device   cudaMalloc
//             managed  cudaMallocManaged (+ ReadMostly / PreferredLocation advice)
//             shm      memfd_create + mmap, raw host pointer straight to the kernel
//                      (GB10 pageableMemAccess + hostPageTables)
//             shmreg   the same memfd mapping, cudaHostRegister'ed
//
//   owner / attacher
//           Two CUDA processes on ONE memfd weight buffer. The owner models the
//           serving engine: a loop of "steps", each one a full checksum kernel over
//           the buffer, compared with the expected checksum published in a shared
//           control page. The attacher models Hugin: it attaches the same memfd,
//           makes its own scratch allocation, runs its own read kernels (forward
//           reads must be byte-exact), and patches slices of the buffer from the
//           GPU. With --quiesce it asks the owner to park at a step boundary
//           before every patch (the batcher quiesce point); without it the patch
//           races the owner's reads, and the owner must SEE torn reads. That
//           proves the checksum can catch them.
//
// The checksum is an order-independent 64-bit sum of 32-bit words, so it is
// deterministic regardless of the block schedule.
//
// Usage:
//   hugin_coexist_rig bw <GiB> [iters]
//   hugin_coexist_rig owner    <socket> <GiB> <seconds>
//   hugin_coexist_rig attacher <socket> <patches> [--quiesce] [--scratch-gib N]
//   (tools/hugin-coexist-rig.sh runs the matrix.)
//
// Build (no GPU needed): nvcc -std=c++17 -O2 -o hugin_coexist_rig tools/hugin-coexist-rig.cu
// Run: inside a --gpus container on the GB10 host, prod serving STOPPED (only
// one GPU container on this box), total memory far below the pool limit.

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <thread>
#include <vector>

#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace {

using Clock = std::chrono::steady_clock;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "FATAL %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

void die(const char* what) {
    std::fprintf(stderr, "FATAL %s: %s\n", what, std::strerror(errno));
    std::exit(2);
}

// ---------------------------------------------------------------- kernels

__global__ void sumKernel(const uint4* __restrict__ p, std::size_t n16, unsigned long long* out) {
    unsigned long long acc = 0;
    for (std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x; i < n16;
         i += std::size_t(gridDim.x) * blockDim.x) {
        const uint4 v = p[i];
        acc += std::uint64_t(v.x) + v.y + v.z + v.w;
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if ((threadIdx.x & 31) == 0) atomicAdd(out, acc);
}

__global__ void fillKernel(std::uint32_t* p, std::size_t n, std::uint32_t seed) {
    for (std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x; i < n;
         i += std::size_t(gridDim.x) * blockDim.x) {
        p[i] = std::uint32_t(i) * 2654435761u ^ seed;
    }
}

int smCount() {
    int dev = 0, sms = 0;
    ck(cudaGetDevice(&dev), "cudaGetDevice");
    ck(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev), "sm count");
    return sms;
}

unsigned long long gpuSum(const void* p, std::size_t bytes, unsigned long long* dOut, cudaStream_t s) {
    ck(cudaMemsetAsync(dOut, 0, sizeof(*dOut), s), "memset sum");
    sumKernel<<<smCount() * 8, 256, 0, s>>>(static_cast<const uint4*>(p), bytes / 16, dOut);
    ck(cudaGetLastError(), "sumKernel launch");
    unsigned long long h = 0;
    ck(cudaMemcpyAsync(&h, dOut, sizeof(h), cudaMemcpyDeviceToHost, s), "sum D2H");
    ck(cudaStreamSynchronize(s), "sum sync");
    return h;
}

std::uint64_t hostSum(const std::uint32_t* p, std::size_t n) {
    std::uint64_t a = 0;
    for (std::size_t i = 0; i < n; ++i) a += p[i];
    return a;
}

void fill(std::uint32_t* p, std::size_t n, std::uint32_t seed) {
    fillKernel<<<smCount() * 8, 256>>>(p, n, seed);
    ck(cudaGetLastError(), "fill launch");
    ck(cudaDeviceSynchronize(), "fill sync");
}

int memfdOf(std::size_t bytes, const char* name) {
    const int fd = static_cast<int>(syscall(SYS_memfd_create, name, 0));
    if (fd < 0) die("memfd_create");
    if (ftruncate(fd, static_cast<off_t>(bytes)) != 0) die("ftruncate");
    return fd;
}

void* mapFd(int fd, std::size_t bytes) {
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) die("mmap");
    return p;
}

// ---------------------------------------------------------------- bw mode

double streamGBs(const void* p, std::size_t bytes, int iters) {
    unsigned long long* dOut = nullptr;
    ck(cudaMalloc(&dOut, sizeof(*dOut)), "malloc sum");
    cudaStream_t s;
    ck(cudaStreamCreate(&s), "stream");
    const unsigned long long ref = gpuSum(p, bytes, dOut, s);   // warm-up (first touch / page tables)
    cudaEvent_t a, b;
    ck(cudaEventCreate(&a), "ev");
    ck(cudaEventCreate(&b), "ev");
    ck(cudaEventRecord(a, s), "rec");
    for (int i = 0; i < iters; ++i) {
        ck(cudaMemsetAsync(dOut, 0, sizeof(*dOut), s), "memset");
        sumKernel<<<smCount() * 8, 256, 0, s>>>(static_cast<const uint4*>(p), bytes / 16, dOut);
    }
    ck(cudaEventRecord(b, s), "rec");
    ck(cudaEventSynchronize(b), "sync");
    float ms = 0;
    ck(cudaEventElapsedTime(&ms, a, b), "elapsed");
    unsigned long long last = 0;
    ck(cudaMemcpy(&last, dOut, sizeof(last), cudaMemcpyDeviceToHost), "D2H");
    if (last != ref) std::printf("  WARN checksum drift %llu vs %llu\n", last, ref);
    cudaEventDestroy(a); cudaEventDestroy(b); cudaStreamDestroy(s); cudaFree(dOut);
    return double(bytes) * iters / (ms * 1e-3) / 1e9;
}

int runBw(double gib, int iters) {
    const std::size_t bytes = std::size_t(gib * double(1ULL << 30)) & ~std::size_t(4095);
    const std::size_t n = bytes / 4;
    std::printf("bw: %.2f GiB per buffer, %d iters, %d SMs\n", gib, iters, smCount());
    {
        void* p = nullptr;
        ck(cudaMalloc(&p, bytes), "cudaMalloc");
        fill(static_cast<std::uint32_t*>(p), n, 1);
        std::printf("RESULT bw device   %.1f GB/s\n", streamGBs(p, bytes, iters));
        cudaFree(p);
    }
    {
        void* p = nullptr;
        ck(cudaMallocManaged(&p, bytes, cudaMemAttachGlobal), "cudaMallocManaged");
        int dev = 0;
        cudaGetDevice(&dev);
        cudaMemLocation loc{};
        loc.type = cudaMemLocationTypeDevice;
        loc.id   = dev;
        cudaMemAdvise(p, bytes, cudaMemAdviseSetReadMostly, loc);
        cudaMemAdvise(p, bytes, cudaMemAdviseSetPreferredLocation, loc);
        fill(static_cast<std::uint32_t*>(p), n, 1);
        std::printf("RESULT bw managed  %.1f GB/s\n", streamGBs(p, bytes, iters));
        cudaFree(p);
    }
    {
        const int fd = memfdOf(bytes, "bw-shm");
        void* p = mapFd(fd, bytes);
        auto* w = static_cast<std::uint32_t*>(p);
        for (std::size_t i = 0; i < n; ++i) w[i] = std::uint32_t(i) * 2654435761u ^ 1u;   // host fill, as Munin would
        std::printf("RESULT bw shm      %.1f GB/s\n", streamGBs(p, bytes, iters));
        const cudaError_t r = cudaHostRegister(p, bytes, cudaHostRegisterDefault);
        if (r == cudaSuccess) {
            std::printf("RESULT bw shmreg   %.1f GB/s\n", streamGBs(p, bytes, iters));
            cudaHostUnregister(p);
        } else {
            std::printf("RESULT bw shmreg   n/a (cudaHostRegister: %s)\n", cudaGetErrorString(r));
            cudaGetLastError();
        }
        munmap(p, bytes);
        close(fd);
    }
    return 0;
}

// ---------------------------------------------------------------- shared control page

struct Control {
    std::atomic<std::uint64_t> expected;      // checksum of the weight buffer the owner must see
    std::atomic<std::uint32_t> quiesceReq;    // attacher -> owner: park at the next step boundary
    std::atomic<std::uint32_t> quiesceAck;    // owner -> attacher: parked, no kernel in flight
    std::atomic<std::uint32_t> attacherDone;
    std::atomic<std::uint32_t> ownerDone;
    std::atomic<std::uint64_t> steps;
    std::atomic<std::uint64_t> mismatches;
};
static_assert(sizeof(Control) <= 4096, "control page");
constexpr std::size_t kCtl = 4096;

int listenOn(const char* path) {
    const int s = socket(AF_UNIX, SOCK_STREAM, 0);
    if (s < 0) die("socket");
    sockaddr_un a{};
    a.sun_family = AF_UNIX;
    std::snprintf(a.sun_path, sizeof(a.sun_path), "%s", path);
    unlink(path);
    if (bind(s, reinterpret_cast<sockaddr*>(&a), sizeof(a)) != 0) die("bind");
    if (listen(s, 1) != 0) die("listen");
    return s;
}

int connectTo(const char* path) {
    for (int i = 0; i < 200; ++i) {
        const int s = socket(AF_UNIX, SOCK_STREAM, 0);
        sockaddr_un a{};
        a.sun_family = AF_UNIX;
        std::snprintf(a.sun_path, sizeof(a.sun_path), "%s", path);
        if (connect(s, reinterpret_cast<sockaddr*>(&a), sizeof(a)) == 0) return s;
        close(s);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    die("connect");
    return -1;
}

void sendFd(int sock, int fd, std::uint64_t bytes) {
    msghdr m{};
    iovec io{&bytes, sizeof(bytes)};
    m.msg_iov = &io;
    m.msg_iovlen = 1;
    alignas(cmsghdr) char buf[CMSG_SPACE(sizeof(int))] = {};
    m.msg_control = buf;
    m.msg_controllen = sizeof(buf);
    cmsghdr* c = CMSG_FIRSTHDR(&m);
    c->cmsg_level = SOL_SOCKET;
    c->cmsg_type = SCM_RIGHTS;
    c->cmsg_len = CMSG_LEN(sizeof(int));
    std::memcpy(CMSG_DATA(c), &fd, sizeof(int));
    if (sendmsg(sock, &m, 0) < 0) die("sendmsg");
}

int recvFd(int sock, std::uint64_t* bytes) {
    msghdr m{};
    iovec io{bytes, sizeof(*bytes)};
    m.msg_iov = &io;
    m.msg_iovlen = 1;
    alignas(cmsghdr) char buf[CMSG_SPACE(sizeof(int))] = {};
    m.msg_control = buf;
    m.msg_controllen = sizeof(buf);
    if (recvmsg(sock, &m, 0) < 0) die("recvmsg");
    cmsghdr* c = CMSG_FIRSTHDR(&m);
    if (c == nullptr || c->cmsg_type != SCM_RIGHTS) die("no fd");
    int fd = -1;
    std::memcpy(&fd, CMSG_DATA(c), sizeof(int));
    return fd;
}

// ---------------------------------------------------------------- owner (serving engine)

int runOwner(const char* sockPath, double gib, double seconds) {
    const std::size_t wBytes = std::size_t(gib * double(1ULL << 30)) & ~std::size_t(4095);
    const std::size_t total = kCtl + wBytes;
    const int fd = memfdOf(total, "coexist-weights");
    auto* base = static_cast<std::uint8_t*>(mapFd(fd, total));
    auto* ctl = new (base) Control{};
    auto* w = reinterpret_cast<std::uint32_t*>(base + kCtl);
    const std::size_t n = wBytes / 4;
    for (std::size_t i = 0; i < n; ++i) w[i] = std::uint32_t(i) * 2654435761u ^ 7u;
    ctl->expected.store(hostSum(w, n));
    std::printf("owner: %.2f GiB weights in memfd, expected %llu\n", gib, (unsigned long long)ctl->expected.load());

    const int ls = listenOn(sockPath);
    const int s = accept(ls, nullptr, nullptr);
    if (s < 0) die("accept");
    sendFd(s, fd, total);

    unsigned long long* dOut = nullptr;
    ck(cudaMalloc(&dOut, sizeof(*dOut)), "malloc");
    cudaStream_t st;
    ck(cudaStreamCreate(&st), "stream");
    const auto t0 = Clock::now();
    std::uint64_t steps = 0, bad = 0, parks = 0;
    double bwSum = 0;
    while (std::chrono::duration<double>(Clock::now() - t0).count() < seconds || !ctl->attacherDone.load()) {
        if (ctl->quiesceReq.load(std::memory_order_acquire) != 0) {   // step boundary: park
            ctl->quiesceAck.store(1, std::memory_order_release);
            while (ctl->quiesceReq.load(std::memory_order_acquire) != 0) std::this_thread::yield();
            ctl->quiesceAck.store(0, std::memory_order_release);
            ++parks;
        }
        const std::uint64_t want = ctl->expected.load(std::memory_order_acquire);
        const auto a = Clock::now();
        const unsigned long long got = gpuSum(w, wBytes, dOut, st);
        bwSum += double(wBytes) / std::chrono::duration<double>(Clock::now() - a).count() / 1e9;
        ++steps;
        // A patch landing mid-step changes `expected` too: re-read it before calling a mismatch torn.
        // Without --quiesce a fully applied patch whose new checksum is not published yet also counts:
        // the owner served bytes nobody had vouched for, which is the hazard quiesce removes.
        if (got != want && got != ctl->expected.load(std::memory_order_acquire)) ++bad;
        ctl->steps.store(steps);
        ctl->mismatches.store(bad);
        if (std::chrono::duration<double>(Clock::now() - t0).count() > seconds * 4 + 120) break;   // safety
    }
    ctl->ownerDone.store(1);
    std::printf("RESULT owner steps %llu mismatches %llu parks %llu mean step-bw %.1f GB/s\n",
                (unsigned long long)steps, (unsigned long long)bad, (unsigned long long)parks, bwSum / double(steps));
    close(s);
    close(ls);
    unlink(sockPath);
    return 0;
}

// ---------------------------------------------------------------- attacher (Hugin)

__global__ void patchKernel(std::uint32_t* p, std::size_t n, std::uint32_t seed) {
    for (std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x; i < n;
         i += std::size_t(gridDim.x) * blockDim.x) {
        p[i] = p[i] * 1664525u + seed;   // a "weight edit" computed on the GPU from the current value
    }
}

int runAttacher(const char* sockPath, int patches, bool quiesce, double scratchGib) {
    const int s = connectTo(sockPath);
    std::uint64_t total = 0;
    const int fd = recvFd(s, &total);
    auto* base = static_cast<std::uint8_t*>(mapFd(fd, total));
    auto* ctl = reinterpret_cast<Control*>(base);
    auto* w = reinterpret_cast<std::uint32_t*>(base + kCtl);
    const std::size_t wBytes = total - kCtl, n = wBytes / 4;
    std::printf("attacher: mapped %.2f GiB, quiesce=%d, scratch %.1f GiB\n", double(wBytes) / double(1ULL << 30),
                quiesce ? 1 : 0, scratchGib);

    // Hugin's own GPU working set next to the shared weights.
    void* scratch = nullptr;
    const std::size_t scratchBytes = std::size_t(scratchGib * double(1ULL << 30)) & ~std::size_t(15);
    if (scratchBytes > 0) {
        ck(cudaMalloc(&scratch, scratchBytes), "scratch");
        fill(static_cast<std::uint32_t*>(scratch), scratchBytes / 4, 3);
    }
    unsigned long long* dOut = nullptr;
    ck(cudaMalloc(&dOut, sizeof(*dOut)), "malloc");
    cudaStream_t st;
    ck(cudaStreamCreate(&st), "stream");

    int readBad = 0;
    const std::size_t slice = n / 64;
    for (int k = 0; k < patches; ++k) {
        // Forward read of the shared weights (byte-exact against the published checksum)
        // between edits, while the owner keeps serving.
        for (int r = 0; r < 3; ++r) {
            const std::uint64_t want = ctl->expected.load(std::memory_order_acquire);
            if (gpuSum(w, wBytes, dOut, st) != want) ++readBad;
            if (scratch) (void)gpuSum(scratch, scratchBytes, dOut, st);
        }
        if (quiesce) {
            while (ctl->quiesceAck.load(std::memory_order_acquire) != 0) std::this_thread::yield();   // previous park released
            ctl->quiesceReq.store(1, std::memory_order_release);
            while (ctl->quiesceAck.load(std::memory_order_acquire) == 0) std::this_thread::yield();
        }
        std::uint32_t* sl = w + std::size_t(k % 64) * slice;
        const std::uint64_t before = hostSum(sl, slice);
        patchKernel<<<smCount() * 4, 256, 0, st>>>(sl, slice, 0x9e3779b9u + k);
        ck(cudaGetLastError(), "patch launch");
        ck(cudaStreamSynchronize(st), "patch sync");
        const std::uint64_t after = hostSum(sl, slice);   // host sees the GPU write (coherence)
        ctl->expected.store(ctl->expected.load() - before + after, std::memory_order_release);
        if (quiesce) ctl->quiesceReq.store(0, std::memory_order_release);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    // Final full verification: host and GPU agree with the published checksum.
    const std::uint64_t want = ctl->expected.load();
    const std::uint64_t h = hostSum(w, n);
    const unsigned long long g = gpuSum(w, wBytes, dOut, st);
    ctl->attacherDone.store(1);
    while (ctl->ownerDone.load() == 0) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    std::printf("RESULT attacher patches %d read-mismatches %d final host %s gpu %s owner-steps %llu owner-mismatches %llu\n",
                patches, readBad, h == want ? "OK" : "BAD", g == want ? "OK" : "BAD",
                (unsigned long long)ctl->steps.load(), (unsigned long long)ctl->mismatches.load());
    close(s);
    return (h == want && g == want) ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: bw <GiB> [iters] | owner <sock> <GiB> <sec> | attacher <sock> <patches> [--quiesce] [--scratch-gib N]\n");
        return 2;
    }
    const std::string mode = argv[1];
    if (mode == "bw" && argc >= 3) return runBw(std::atof(argv[2]), argc >= 4 ? std::atoi(argv[3]) : 20);
    if (mode == "owner" && argc >= 5) return runOwner(argv[2], std::atof(argv[3]), std::atof(argv[4]));
    if (mode == "attacher" && argc >= 4) {
        bool q = false;
        double scratch = 0;
        for (int i = 4; i < argc; ++i) {
            if (std::string(argv[i]) == "--quiesce") q = true;
            else if (std::string(argv[i]) == "--scratch-gib" && i + 1 < argc) scratch = std::atof(argv[++i]);
        }
        return runAttacher(argv[2], std::atoi(argv[3]), q, scratch);
    }
    std::fprintf(stderr, "bad arguments\n");
    return 2;
}
