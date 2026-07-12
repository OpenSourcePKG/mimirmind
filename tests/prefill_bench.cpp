// Prefill-attention bench-runner for the M9.8b long-context path.
//
// Not a unit test — measures wall-clock latency for a single
// GpuOps::attentionAsync dispatch at a caller-configurable T_k and
// prints a Markdown row suitable for pasting into
// Memory/mimirmind/research/perf-regression-ledger.md, plus one NDJSON
// row for programmatic aggregation.
//
// Requires:
//   - A Level Zero device (Intel iGPU) reachable via /dev/dri.
//   - SPV kernels under /usr/local/share/mimirmind/spv.
//
// Usage:
//   prefill_bench [--tk N] [--tq N] [--warmups N] [--iters N]
//                 [--nheads N] [--nkvheads N] [--headdim N]
//                 [--label STR]
//
// Defaults reproduce the Gemma 4 26B-A4B full-attention geometry from
// gpu_tests attention_prefill_flash_maxContext24k
// (nHeads=16, nKvHeads=4, headDim=256, T_q=32, T_k=24576).
//
// Run via:
//   docker compose run --rm mimirmind /usr/local/bin/prefill_bench --tk 24576
//   docker compose run --rm mimirmind /usr/local/bin/prefill_bench --tk 32768

#include "compute/GpuOps.hpp"
#include "runtime/CommandQueue.hpp"
#include "runtime/Config.hpp"
#include "runtime/L0Context.hpp"
#include "runtime/Log.hpp"
#include "runtime/UsmAllocator.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Args {
    std::size_t tk       = 24576;
    std::size_t tq       = 32;
    std::size_t warmups  = 3;
    std::size_t iters    = 5;
    std::size_t nHeads   = 16;
    std::size_t nKvHeads = 4;
    std::size_t headDim  = 256;
    std::string label    = "flash_prefill_gemma4_26B_ctx24k";
};

void printHelp() {
    std::cout <<
        "prefill_bench — wall-clock benchmark for GpuOps::attentionAsync\n"
        "                (M9.8b long-context path)\n"
        "\n"
        "Options:\n"
        "  --tk N        keys per query (default 24576)\n"
        "  --tq N        query rows (default 32)\n"
        "  --warmups N   warm-up iterations (default 3)\n"
        "  --iters N     measured iterations (default 5)\n"
        "  --nheads N    query heads (default 16 — Gemma 4 26B-A4B)\n"
        "  --nkvheads N  KV heads (default 4)\n"
        "  --headdim N   per-head embedding (default 256)\n"
        "  --label STR   ledger label (default flash_prefill_gemma4_26B_ctx24k)\n"
        "  -h, --help    print this and exit\n";
}

Args parseArgs(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "prefill_bench: missing value for " << k << "\n";
                std::exit(2);
            }
            return argv[++i];
        };
        if      (k == "--tk"       ) a.tk       = std::stoull(next());
        else if (k == "--tq"       ) a.tq       = std::stoull(next());
        else if (k == "--warmups"  ) a.warmups  = std::stoull(next());
        else if (k == "--iters"    ) a.iters    = std::stoull(next());
        else if (k == "--nheads"   ) a.nHeads   = std::stoull(next());
        else if (k == "--nkvheads" ) a.nKvHeads = std::stoull(next());
        else if (k == "--headdim"  ) a.headDim  = std::stoull(next());
        else if (k == "--label"    ) a.label    = next();
        else if (k == "-h" || k == "--help") {
            printHelp();
            std::exit(0);
        } else {
            std::cerr << "prefill_bench: unknown arg " << k << "\n";
            printHelp();
            std::exit(2);
        }
    }
    if (a.tk == 0 || a.tq == 0 || a.iters == 0 || a.nHeads == 0
        || a.nKvHeads == 0 || a.headDim == 0) {
        std::cerr << "prefill_bench: T_k, T_q, iters and head geometry "
                     "must all be positive\n";
        std::exit(2);
    }
    if (a.nHeads % a.nKvHeads != 0) {
        std::cerr << "prefill_bench: nHeads (" << a.nHeads
                  << ") must be a positive multiple of nKvHeads ("
                  << a.nKvHeads << ")\n";
        std::exit(2);
    }
    if (a.tk < a.tq) {
        std::cerr << "prefill_bench: T_k (" << a.tk
                  << ") must be >= T_q (" << a.tq << ")\n";
        std::exit(2);
    }
    return a;
}

double percentile(std::vector<double>& sorted, double p) {
    // sorted must already be sorted ascending. p in [0, 100].
    if (sorted.empty()) return 0.0;
    const std::size_t n   = sorted.size();
    const std::size_t idx = std::min<std::size_t>(
        n - 1, static_cast<std::size_t>((p / 100.0) * n));
    return sorted[idx];
}

} // namespace

int main(int argc, char** argv) {
    const Args args = parseArgs(argc, argv);

    mimirmind::runtime::L0Context    ctx;
    mimirmind::runtime::UsmAllocator usm{ctx};
    mimirmind::runtime::CommandQueue queue{ctx};
    mimirmind::compute::GpuOps       ops{ctx, usm, queue};

    // Allocation sizes match the runAttentionParity helper in
    // gpu_tests.cpp — same F32 QKV layout regardless of KV dtype
    // (KV block quantisation would need a separate pack step).
    const std::size_t qBytes = args.tq * args.nHeads   * args.headDim
                                       * sizeof(float);
    const std::size_t kBytes = args.tk * args.nKvHeads * args.headDim
                                       * sizeof(float);
    const std::size_t vBytes = kBytes;
    const std::size_t oBytes = qBytes;

    void* qUsm = usm.allocate(qBytes);
    void* kUsm = usm.allocate(kBytes);
    void* vUsm = usm.allocate(vBytes);
    void* oUsm = usm.allocate(oBytes);

    // Deterministic ramp inputs. Same generator as the gpu_tests
    // fixture (see runAttentionParity), so results are comparable
    // across runs on the same host.
    const std::size_t qN  = qBytes / sizeof(float);
    const std::size_t kvN = kBytes / sizeof(float);
    std::vector<float> qHost(qN), kHost(kvN), vHost(kvN);
    for (std::size_t i = 0; i < qN;  ++i)
        qHost[i] = static_cast<float>((i * 7  + 1) % 17) * 0.125F - 1.0F;
    for (std::size_t i = 0; i < kvN; ++i)
        kHost[i] = static_cast<float>((i * 11 + 3) % 19) * 0.125F - 1.25F;
    for (std::size_t i = 0; i < kvN; ++i)
        vHost[i] = static_cast<float>((i * 13 + 5) % 23) * 0.0625F - 0.75F;
    std::memcpy(qUsm, qHost.data(), qBytes);
    std::memcpy(kUsm, kHost.data(), kBytes);
    std::memcpy(vUsm, vHost.data(), vBytes);
    std::memset(oUsm, 0, oBytes);

    const float       scale = 1.0F / std::sqrt(
        static_cast<float>(args.headDim));
    const std::size_t positionOffset = args.tk - args.tq;

    // Warm-ups (SPV cache, USM residency, driver command-list build).
    for (std::size_t w = 0; w < args.warmups; ++w) {
        ops.attentionAsync(
            static_cast<const float*>(qUsm),
            kUsm, vUsm,
            args.tq, args.tk,
            args.nHeads, args.nKvHeads, args.headDim,
            positionOffset, scale,
            static_cast<float*>(oUsm));
        queue.flush();
    }

    // Measured iterations.
    std::vector<double> wallMs;
    wallMs.reserve(args.iters);
    for (std::size_t r = 0; r < args.iters; ++r) {
        const auto t0 = Clock::now();
        ops.attentionAsync(
            static_cast<const float*>(qUsm),
            kUsm, vUsm,
            args.tq, args.tk,
            args.nHeads, args.nKvHeads, args.headDim,
            positionOffset, scale,
            static_cast<float*>(oUsm));
        queue.flush();
        const auto t1 = Clock::now();
        const double ms = std::chrono::duration<double, std::milli>(
            t1 - t0).count();
        wallMs.push_back(ms);
    }

    usm.deallocate(qUsm, qBytes);
    usm.deallocate(kUsm, kBytes);
    usm.deallocate(vUsm, vBytes);
    usm.deallocate(oUsm, oBytes);

    // Stats.
    std::vector<double> sorted = wallMs;
    std::sort(sorted.begin(), sorted.end());
    const double p50 = percentile(sorted, 50.0);
    const double p95 = percentile(sorted, 95.0);
    const double minv = sorted.front();
    const double maxv = sorted.back();
    double mean = 0.0;
    for (const double x : wallMs) mean += x;
    mean /= static_cast<double>(wallMs.size());

    // KV footprint estimate (Gemma-4-A4B assumes 30 layers; we only
    // measure a single dispatch here, so the "KV bytes / token"
    // reported below is per-layer, per-token, both KV heads).
    const std::size_t kvBytesPerToken =
        2 * args.nKvHeads * args.headDim * sizeof(float);

    // Human-readable summary.
    std::cout << "prefill_bench " << args.label << ":\n"
              << "  T_q="        << args.tq
              << " T_k="         << args.tk
              << " nHeads="      << args.nHeads
              << " nKvHeads="    << args.nKvHeads
              << " headDim="     << args.headDim << "\n"
              << "  warmups="    << args.warmups
              << " iters="       << args.iters   << "\n"
              << "  wall p50="   << p50 << " ms"
              <<     " p95="     << p95 << " ms"
              <<     " min="     << minv << " ms"
              <<     " max="     << maxv << " ms"
              <<     " mean="    << mean << " ms\n"
              << "  per-layer KV bytes / token (KV F32) = "
              << kvBytesPerToken << "\n";

    // Ledger row (Markdown) — matches the pipe-format the ledger uses.
    std::cout << "\nLedger row (paste into perf-regression-ledger.md):\n"
              << "| " << args.label
              << " | T_q=" << args.tq
              <<   " T_k=" << args.tk
              <<   " (nH="   << args.nHeads
              <<   ",nKV="   << args.nKvHeads
              <<   ",hd="    << args.headDim << ")"
              << " | p50="  << p50 << " ms"
              <<   ", p95=" << p95 << " ms"
              <<   " (min " << minv << " / max " << maxv
              <<   ", n="   << wallMs.size() << ")"
              << " | (fill status) |\n";

    // NDJSON row — one line, machine-parseable.
    std::cout << "\nNDJSON row:\n"
              << "{\"kind\":\"prefill_bench\","
              <<   "\"label\":\""    << args.label << "\","
              <<   "\"T_q\":"        << args.tq << ","
              <<   "\"T_k\":"        << args.tk << ","
              <<   "\"nHeads\":"     << args.nHeads << ","
              <<   "\"nKvHeads\":"   << args.nKvHeads << ","
              <<   "\"headDim\":"    << args.headDim << ","
              <<   "\"warmups\":"    << args.warmups << ","
              <<   "\"iters\":"      << args.iters << ","
              <<   "\"p50_ms\":"     << p50 << ","
              <<   "\"p95_ms\":"     << p95 << ","
              <<   "\"min_ms\":"     << minv << ","
              <<   "\"max_ms\":"     << maxv << ","
              <<   "\"mean_ms\":"    << mean << ","
              <<   "\"kv_bytes_per_token_per_layer\":"
              <<   kvBytesPerToken
              << "}\n";

    return 0;
}
