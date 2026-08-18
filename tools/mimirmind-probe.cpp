// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// mimirmind-probe — offline probe-driven config generation (M-Probe.1).
//
// This is increment 1: the HW-fingerprint generator + the config-artefact
// contract. It enumerates the active compute backend + host, derives a
// stable hardware fingerprint, and writes
//
//     configs/hw/{fingerprint}/probe-result.json
//
// The runtime reads that artefact *lookup-only* (match on fingerprint, fall
// back to defaults on a miss) — no runtime autotune (that is a CLR landmine;
// see the M-Probe.1 design note). The backend x kernel-variant x shape sweep,
// the CPU-gold correctness gate, and the median/IQR margin guards
// (M-Probe.1.B-F) land in later increments; until then the artefact carries
// an empty `ops` table and `probe_status: "fingerprint-only"`, which the
// runtime loader treats as "use defaults".
//
// Runs standalone — no HTTP server, no model load. When the compute backend
// cannot be constructed (e.g. no /dev/dri), it degrades to a host-only
// fingerprint rather than failing, so the artefact contract still holds.
//
// Build:  cmake --build build --target mimirmind_probe
// Run:    ./build/mimirmind_probe [--print] [--out <dir>] [--model <id>]

#include "core/backend/BackendRegistry.hpp"
#include "core/backend/ComputeBackend.hpp"
#include "core/backend/ComputeContext.hpp"

#include <nlohmann/json.hpp>

#include <sys/utsname.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

// The op-sweep (M-Probe.1.B/C/D) drives concrete backend ops, so it is only
// compiled when the Level Zero backend is linked in (CMake sets the macro).
#ifdef MIMIRMIND_PROBE_HAS_L0
#include "compute/l0/GpuMatmul.hpp"
#include "compute/l0/GpuOps.hpp"
#include "compute/quant/Q8_0.hpp"
#include "core/gpu/l0/L0ComputeContext.hpp"
#include "core/gpu/l0/UsmAllocator.hpp"
#include "core/gguf/GgufTypes.hpp"
#endif

namespace {

using json = nlohmann::json;
namespace be = mimirmind::core::backend;

// FNV-1a 64-bit — the fingerprint is a stable cache key, not a security
// digest, so a fast dependency-free hash is the right tool (no OpenSSL link).
std::uint64_t fnv1a64(std::string_view s) {
    std::uint64_t h = 1469598103934665603ULL;
    for (char ch : s) {
        h ^= static_cast<unsigned char>(ch);
        h *= 1099511628211ULL;
    }
    return h;
}

std::string toHex16(std::uint64_t v) {
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(v));
    return std::string{buf};
}

// First "model name" line from /proc/cpuinfo (Linux). Empty on failure.
std::string cpuModel() {
    std::ifstream f{"/proc/cpuinfo"};
    std::string line;
    while (std::getline(f, line)) {
        const auto colon = line.find(':');
        if (colon != std::string::npos &&
            line.compare(0, 10, "model name") == 0) {
            std::string v = line.substr(colon + 1);
            const auto b = v.find_first_not_of(" \t");
            const auto e = v.find_last_not_of(" \t");
            return (b == std::string::npos) ? std::string{} : v.substr(b, e - b + 1);
        }
    }
    return {};
}

unsigned cpuThreads() {
    const unsigned hc = std::thread::hardware_concurrency();
    return hc ? hc : 0U;
}

// MemTotal from /proc/meminfo, rounded to whole GiB (a bucket — small RAM
// jitter across boots must not change the fingerprint).
std::uint64_t ramGiB() {
    std::ifstream f{"/proc/meminfo"};
    std::string key;
    std::uint64_t kb = 0;
    if (f >> key >> kb) {
        if (key == "MemTotal:") {
            return (kb + (512ULL * 1024ULL)) / (1024ULL * 1024ULL);  // kB -> GiB, rounded
        }
    }
    return 0;
}

std::string kernelRelease() {
    struct utsname u{};
    if (::uname(&u) == 0) {
        return std::string{u.release};
    }
    return {};
}

std::string iso8601UtcNow() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    struct tm tmv{};
    ::gmtime_r(&t, &tmv);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tmv);
    return std::string{buf};
}

const char* deviceKindName(be::DeviceKind k) {
    switch (k) {
        case be::DeviceKind::GpuIntegrated: return "GpuIntegrated";
        case be::DeviceKind::GpuDiscrete:   return "GpuDiscrete";
        case be::DeviceKind::Npu:           return "Npu";
        case be::DeviceKind::Cpu:           return "Cpu";
        case be::DeviceKind::Unknown:       return "Unknown";
    }
    return "Unknown";
}

// The five neutral feature flags the fingerprint + artefact care about.
struct FeatureBit { be::BackendFeature f; const char* name; };
const FeatureBit kFeatures[] = {
    {be::BackendFeature::MutableCommandLists, "MutableCommandLists"},
    {be::BackendFeature::IntegerDotProduct,   "IntegerDotProduct"},
    {be::BackendFeature::IpcHandleExport,     "IpcHandleExport"},
    {be::BackendFeature::UnifiedMemoryHost,   "UnifiedMemoryHost"},
    {be::BackendFeature::MatrixEngine,        "MatrixEngine"},
};

// --- Op-sweep (M-Probe.1.B/C/D) ----------------------------------------------
//
// Increment 2a: an L0 F32 matmul sweep across M-buckets with an inline
// double-accumulate reference gate + median/IQR margin. F32 needs no quant
// encoder and its reference is exact, so this is the safe first op. Quant
// dtypes (Q8_0 has a row encoder; Q4_K/Q6_K need encoders that don't exist
// yet) and the cross-backend axis (needs a neutral op-invocation seam that
// ComputeContext does not expose today) are the remaining 4.6.2 work.

using Clock = std::chrono::steady_clock;

double percentile(std::vector<double>& v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const double idx = (p / 100.0) * static_cast<double>(v.size() - 1);
    const auto lo = static_cast<std::size_t>(idx);
    const auto hi = std::min(lo + 1, v.size() - 1);
    const double frac = idx - static_cast<double>(lo);
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}

#ifdef MIMIRMIND_PROBE_HAS_L0

// Reference: Y[M,N] = X[M,K] * W[N,K]^T, F64 accumulate (the correctness gold).
void refMatmulF32(const float* W, const float* X, float* Y,
                  std::size_t N, std::size_t K, std::size_t M) {
    for (std::size_t m = 0; m < M; ++m) {
        for (std::size_t n = 0; n < N; ++n) {
            double acc = 0.0;
            const float* wr = W + n * K;
            const float* xr = X + m * K;
            for (std::size_t k = 0; k < K; ++k) {
                acc += static_cast<double>(xr[k]) * static_cast<double>(wr[k]);
            }
            Y[m * N + n] = static_cast<float>(acc);
        }
    }
}

// Max |gpu - gold| / max|gold| over all elements.
double relError(const float* gpu, const float* gold, std::size_t n) {
    double maxAbsDiff = 0.0, maxAbsGold = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        maxAbsDiff = std::max(maxAbsDiff, std::fabs(static_cast<double>(gpu[i]) -
                                                    static_cast<double>(gold[i])));
        maxAbsGold = std::max(maxAbsGold, std::fabs(static_cast<double>(gold[i])));
    }
    return maxAbsDiff / (maxAbsGold + 1e-9);
}

// Run the L0 matmul sweep (F32 + Q8_0) and return the `ops` array. Throws on
// device construction failure (caller degrades to no-sweep).
nlohmann::json runL0MatmulSweep() {
    namespace cl0 = mimirmind::core::l0;
    namespace ml0 = mimirmind::compute::l0;
    namespace mq  = mimirmind::compute::quant;
    using mimirmind::core::gguf::GgmlType;

    nlohmann::json ops = nlohmann::json::array();

    cl0::L0ComputeContext ctx{};
    ml0::GpuOps           gops{ctx};   // defaults => Q8_0 native layout
    ml0::GpuMatmul        gmm{ctx, gops};
    cl0::UsmAllocator&    usm = ctx.allocator();

    constexpr std::size_t N = 2048, K = 2048;  // representative hidden dim (K%32==0)
    const std::size_t Ms[] = {1, 16, 64, 256};
    constexpr int kWarmup = 3, kReps = 25;
    constexpr double kTol = 1e-2;   // 1% — separates accum jitter from a bug

    // Deterministic random F32 weights [N,K]; the quant cases encode from these
    // and the reference matmul runs on the *dequantised* weights so the gate
    // measures kernel correctness, not the quantisation error itself.
    std::mt19937 rng{0x9e3779b9U};
    std::uniform_real_distribution<float> dist{-0.05F, 0.05F};
    std::vector<float> wF32(N * K);
    for (float& v : wF32) v = dist(rng);

    struct DtypeCase { GgmlType type; const char* name; };
    const DtypeCase cases[] = {
        {GgmlType::F32,  "F32"},
        {GgmlType::Q8_0, "Q8_0"},
    };

    for (const DtypeCase& c : cases) {
        if (!gmm.supports(c.type)) {
            ops.push_back({{"op", "matmul"}, {"dtype", c.name},
                           {"backend", "LevelZero"}, {"status", "unsupported"}});
            continue;
        }

        // Build the weight buffer in the dtype's layout + the F32 reference
        // weights the gold matmul reads.
        std::vector<float> wRef;   // [N,K] F32 — exactly what the kernel sees
        std::size_t wBytes = 0;
        void*       wUsm   = nullptr;

        if (c.type == GgmlType::F32) {
            wRef   = wF32;
            wBytes = wF32.size() * sizeof(float);
            wUsm   = usm.allocate(wBytes);
            std::memcpy(wUsm, wF32.data(), wBytes);
        } else {  // Q8_0 — encode per row (native 34-byte blocks), then dequant
            const std::size_t rowBytes = mq::Q8_0::rowBytes(K);
            wBytes = N * rowBytes;
            std::vector<std::uint8_t> wQ8(wBytes);
            for (std::size_t n = 0; n < N; ++n) {
                mq::Q8_0::quantizeRow(wF32.data() + n * K, K, wQ8.data() + n * rowBytes);
            }
            wRef.resize(N * K);
            mq::Q8_0::instance().dequantToF32(wQ8.data(), N * K, wRef.data());
            wUsm = usm.allocate(wBytes);
            std::memcpy(wUsm, wQ8.data(), wBytes);
        }

        for (std::size_t M : Ms) {
            std::vector<float> xHost(M * K);
            for (float& v : xHost) v = dist(rng);
            const std::size_t xBytes = xHost.size() * sizeof(float);
            const std::size_t yBytes = M * N * sizeof(float);
            const std::size_t sBytes = M * N * sizeof(float);  // generous scratch

            void* xUsm = usm.allocate(xBytes);
            void* yUsm = usm.allocate(yBytes);
            void* sUsm = usm.allocate(sBytes);
            std::memcpy(xUsm, xHost.data(), xBytes);
            std::memset(yUsm, 0, yBytes);

            auto call = [&]() {
                gmm.matmul(c.type, wUsm, N, K,
                           static_cast<const float*>(xUsm), M,
                           static_cast<float*>(yUsm), static_cast<float*>(sUsm));
            };

            for (int w = 0; w < kWarmup; ++w) call();
            std::vector<double> ms;
            ms.reserve(kReps);
            for (int r = 0; r < kReps; ++r) {
                const auto t0 = Clock::now();
                call();
                ms.push_back(std::chrono::duration<double, std::milli>(
                                 Clock::now() - t0).count());
            }

            std::vector<float> yGold(M * N);
            refMatmulF32(wRef.data(), xHost.data(), yGold.data(), N, K, M);
            const double relErr =
                relError(static_cast<const float*>(yUsm), yGold.data(), M * N);

            const double p50 = percentile(ms, 50.0);
            const double iqr = percentile(ms, 75.0) - percentile(ms, 25.0);
            std::string status = "ok";
            if (relErr >= kTol) status = "parity_fail";   // gate before perf
            else if (p50 > 0.0 && iqr / p50 > 0.15) status = "inconclusive";

            ops.push_back({
                {"op", "matmul"},
                {"dtype", c.name},
                {"backend", "LevelZero"},
                {"shape", {{"N", N}, {"K", K}, {"M", M}}},
                {"median_ms", p50},
                {"iqr_ms", iqr},
                {"reps", kReps},
                {"parity_max_rel_diff", relErr},
                {"status", status},
            });

            usm.deallocate(xUsm, xBytes);
            usm.deallocate(yUsm, yBytes);
            usm.deallocate(sUsm, sBytes);
        }
        usm.deallocate(wUsm, wBytes);
    }
    return ops;
}

#endif  // MIMIRMIND_PROBE_HAS_L0

struct Options {
    bool        printOnly = false;
    bool        sweep     = false;
    std::string outRoot   = "configs";
    std::string modelId   = "unspecified";
};

void printUsage() {
    std::puts(
        "mimirmind-probe — HW fingerprint + probe config artefact (M-Probe.1)\n"
        "\n"
        "Usage: mimirmind_probe [--print] [--sweep] [--out <dir>] [--model <id>]\n"
        "\n"
        "  --print, -p     write the artefact to stdout instead of a file\n"
        "  --sweep         run the op sweep (L0 F32 matmul x M-bucket) and fill\n"
        "                  ops[]; without it the artefact is fingerprint-only\n"
        "  --out <dir>     config root (default: configs) ->\n"
        "                  <dir>/hw/{fingerprint}/probe-result.json\n"
        "  --model <id>    tag the artefact with this model id\n"
        "  --help, -h      this help\n");
}

bool parseArgs(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        const std::string_view a{argv[i]};
        if (a == "--help" || a == "-h") { printUsage(); return false; }
        else if (a == "--print" || a == "-p") { o.printOnly = true; }
        else if (a == "--sweep") { o.sweep = true; }
        else if (a == "--out" && i + 1 < argc) { o.outRoot = argv[++i]; }
        else if (a == "--model" && i + 1 < argc) { o.modelId = argv[++i]; }
        else {
            std::fprintf(stderr, "unknown arg: %.*s\n",
                         static_cast<int>(a.size()), a.data());
            printUsage();
            return false;
        }
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    Options opt;
    if (!parseArgs(argc, argv, opt)) {
        // --help returns success; a bad arg is caught above with usage. Both
        // paths here are "stop, not an error worth a non-zero code".
        return 0;
    }

    // --- Host block (always available) -------------------------------------
    const std::string hostCpu     = cpuModel();
    const unsigned    hostThreads = cpuThreads();
    const std::uint64_t hostRamGiB = ramGiB();
    const std::string hostKernel  = kernelRelease();

    json hw;
    hw["host"] = {
        {"cpu_model", hostCpu},
        {"cpu_threads", hostThreads},
        {"ram_gib", hostRamGiB},
        {"kernel", hostKernel},
    };

    // Canonical fingerprint string — only STABLE identity fields. Timing,
    // timestamps and mutable counters never enter it.
    std::ostringstream fp;
    fp << "host|" << hostCpu << '|' << hostThreads << '|' << hostRamGiB
       << '|' << hostKernel;

    // --- Available-backends probe (no construction) ------------------------
    json avail = json::array();
    for (const auto& p : be::BackendRegistry::probeAll()) {
        avail.push_back({
            {"kind", be::BackendRegistry::name(p.kind)},
            {"compiled", p.compiledIn},
            {"available", p.available},
            {"detail", p.detail},
        });
    }
    hw["available_backends"] = std::move(avail);

    // --- Active backend block (construct + observe, degrade on failure) -----
    const be::BackendKind kind = be::BackendRegistry::autoSelect();
    json backendJson;
    try {
        std::unique_ptr<be::ComputeContext> ctx =
            be::BackendRegistry::createContext(kind);
        const be::ComputeBackend&    b    = ctx->backend();
        const be::BackendDeviceInfo& info = b.deviceInfo();

        char pci[16];
        std::snprintf(pci, sizeof(pci), "0x%04x:0x%04x", info.vendorId, info.deviceId);

        json features;
        for (const auto& fb : kFeatures) {
            features[fb.name] = b.hasFeature(fb.f);
        }

        backendJson = {
            {"kind", be::BackendRegistry::name(b.kind())},
            {"device_name", info.name},
            {"uuid", info.uuid},
            {"pci", pci},
            {"device_kind", deviceKindName(info.kind)},
            {"compute_units", info.numComputeUnits},
            {"total_local_mem_gib",
             static_cast<double>(info.totalLocalMem) / (1024.0 * 1024.0 * 1024.0)},
            {"max_alloc_gib",
             static_cast<double>(info.maxMemAllocSize) / (1024.0 * 1024.0 * 1024.0)},
            {"core_clock_mhz", info.coreClockRate},
            {"bandwidth_gbps", ctx->bandwidthGBps()},
            {"features", features},
        };

        // Fingerprint from stable device identity — NOT clock/mem which can
        // jitter with governor state, and NOT bandwidth (a heuristic).
        fp << "|dev|" << be::BackendRegistry::name(b.kind()) << '|' << info.name
           << '|' << info.uuid << '|' << pci << '|' << info.numComputeUnits;
        for (const auto& fb : kFeatures) {
            fp << (b.hasFeature(fb.f) ? '1' : '0');
        }
    } catch (const std::exception& e) {
        // No device (e.g. no /dev/dri): host-only fingerprint, backend noted.
        backendJson = {
            {"kind", "none"},
            {"error", e.what()},
        };
        fp << "|dev|none";
        std::fprintf(stderr,
                     "[warn] backend '%s' could not be constructed (%s) — "
                     "emitting a host-only fingerprint\n",
                     be::BackendRegistry::name(kind), e.what());
    }
    hw["backend"] = std::move(backendJson);

    const std::string fingerprint = toHex16(fnv1a64(fp.str()));

    // --- Op sweep (optional) -----------------------------------------------
    // "fingerprint-only" = no measurements (runtime uses defaults).
    // "swept-partial"    = some ops measured, but NOT the full
    //                      backend x kernel x dtype matrix (F32 matmul only
    //                      today) — the runtime must still default anything
    //                      not present in ops[].
    json ops = json::array();
    std::string probeStatus = "fingerprint-only";
    if (opt.sweep) {
#ifdef MIMIRMIND_PROBE_HAS_L0
        try {
            ops = runL0MatmulSweep();
            if (!ops.empty()) probeStatus = "swept-partial";
        } catch (const std::exception& e) {
            std::fprintf(stderr,
                         "[warn] op sweep failed (%s) — fingerprint-only\n",
                         e.what());
        }
#else
        std::fprintf(stderr,
                     "[warn] --sweep requested but this build has no L0 sweep "
                     "support — fingerprint-only\n");
#endif
    }

    // --- Artefact ----------------------------------------------------------
    json artefact = {
        {"schema", "mimirmind-probe-result/v1"},
        {"generated_at", iso8601UtcNow()},
        {"fingerprint", fingerprint},
        {"model_id", opt.modelId},
        {"probe_status", probeStatus},
        {"hardware", std::move(hw)},
        {"ops", std::move(ops)},
    };

    const std::string dump = artefact.dump(2);

    if (opt.printOnly) {
        std::puts(dump.c_str());
        return 0;
    }

    namespace fs = std::filesystem;
    const fs::path dir = fs::path{opt.outRoot} / "hw" / fingerprint;
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) {
        std::fprintf(stderr, "[error] cannot create %s: %s\n",
                     dir.c_str(), ec.message().c_str());
        return 1;
    }
    const fs::path out = dir / "probe-result.json";
    std::ofstream os{out, std::ios::trunc};
    if (!os) {
        std::fprintf(stderr, "[error] cannot write %s\n", out.c_str());
        return 1;
    }
    os << dump << '\n';
    os.close();

    std::printf("mimirmind-probe: fingerprint=%s  backend=%s  ->  %s\n",
                fingerprint.c_str(),
                be::BackendRegistry::name(kind),
                out.c_str());
    return 0;
}
