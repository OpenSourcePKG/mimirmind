// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "munin/ShmDaemon.hpp"

#include <cstdio>
#include <string>
#include <string_view>

namespace {

constexpr const char* kUsage =
    "usage: munin-shm --config PATH [--socket PATH] [--log-file PATH] [--log-level LEVEL]\n"
    "\n"
    "The M-Munin.CUDA persistent model-memory daemon (POSIX-shm transport for\n"
    "DGX Spark / GB10). Loads models listed in the config's\n"
    "`models[loadOnStart:true]` into memfd host RAM, exposes an AF_UNIX socket\n"
    "for attaching mimirmind workers (which mmap the memfds and hand the\n"
    "pointers straight to CUDA kernels), and holds the tensors across worker\n"
    "restarts. No GPU SDK required in the daemon itself.\n"
    "\n"
    "Options:\n"
    "  --config PATH        Path to config.json (same schema as mimirmind uses).\n"
    "  --socket PATH        AF_UNIX path to bind. Default: /var/run/munin/munin.sock\n"
    "  --log-file PATH      Overrides `server.log.file` from config.\n"
    "  --log-level LEVEL    Overrides `server.log.level` from config.\n"
    "                       One of: trace | debug | info | warn | error | off\n"
    "\n"
    "Signals: SIGINT/SIGTERM triggers a clean shutdown (socket unlink,\n"
    "sessions torn down, memfds released). SIGPIPE is ignored.\n";

bool eq(std::string_view a, std::string_view b) noexcept {
    return a == b;
}

bool takeValue(int argc, char** argv, int& i, const char* name, std::string& out) {
    if (i + 1 >= argc) {
        std::fprintf(stderr, "munin-shm: '%s' requires a value\n", name);
        return false;
    }
    out = argv[i + 1];
    i += 2;
    return true;
}

} // namespace

int main(int argc, char** argv) {
    mimirmind::munin::CliOptions opts;

    for (int i = 1; i < argc;) {
        const std::string_view a{argv[i]};
        if (eq(a, "--help") || eq(a, "-h")) {
            std::fputs(kUsage, stdout);
            return 0;
        }
        if (eq(a, "--config")) {
            if (!takeValue(argc, argv, i, "--config", opts.configPath)) return 2;
            continue;
        }
        if (eq(a, "--socket")) {
            if (!takeValue(argc, argv, i, "--socket", opts.socketPath)) return 2;
            continue;
        }
        if (eq(a, "--log-file")) {
            if (!takeValue(argc, argv, i, "--log-file", opts.logFile)) return 2;
            continue;
        }
        if (eq(a, "--log-level")) {
            if (!takeValue(argc, argv, i, "--log-level", opts.logLevel)) return 2;
            continue;
        }
        std::fprintf(stderr, "munin-shm: unknown argument '%s'\n\n", argv[i]);
        std::fputs(kUsage, stderr);
        return 2;
    }

    if (opts.configPath.empty()) {
        std::fprintf(stderr, "munin-shm: --config is required\n\n");
        std::fputs(kUsage, stderr);
        return 2;
    }

    return mimirmind::munin::ShmDaemon::run(opts);
}
