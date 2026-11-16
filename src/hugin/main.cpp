// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

// Hugin — mimirmind's adaptation companion (roadmap 7). This binary is the
// MCP server Claude talks to over stdio. stdout carries ONLY protocol
// messages; every log line goes to stderr.

#include "hugin/ForwardTools.hpp"
#include "hugin/InspectTools.hpp"
#include "hugin/McpServer.hpp"

#include <cstdio>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>

#include <unistd.h>

namespace {

constexpr const char* kVersion = "0.2.0";

void usage() {
    std::cerr << "usage: hugin [--model <file.gguf | nvfp4-checkpoint-dir>] [--tokenizer-gguf <file.gguf>]\n"
                 "  Runs an MCP server on stdin/stdout (newline-delimited JSON-RPC 2.0).\n"
                 "  Register with Claude Code, e.g.:\n"
                 "    claude mcp add hugin -- /path/to/hugin --model /path/to/model.gguf\n";
}

} // namespace

int main(int argc, char** argv) {
    std::string model;
    std::string tokenizerGguf;
    for (int i = 1; i < argc; ++i) {
        const std::string_view a = argv[i];
        if (a == "--model" && i + 1 < argc) {
            model = argv[++i];
        } else if (a == "--tokenizer-gguf" && i + 1 < argc) {
            tokenizerGguf = argv[++i];
        } else if (a == "--help" || a == "-h") {
            usage();
            return 0;
        } else {
            std::cerr << "hugin: unknown argument '" << a << "'\n";
            usage();
            return 2;
        }
    }

    // The engine is linked in (forward tools) and parts of it print to
    // stdout. stdout must carry ONLY protocol messages, so the protocol gets
    // a private duplicate of fd 1 and fd 1 itself is pointed at stderr.
    std::fflush(stdout);
    const int protoFd = ::dup(STDOUT_FILENO);
    if (protoFd < 0 || ::dup2(STDERR_FILENO, STDOUT_FILENO) < 0) {
        std::cerr << "hugin: fatal: cannot isolate stdout\n";
        return 1;
    }
    std::ofstream proto("/proc/self/fd/" + std::to_string(protoFd));
    if (!proto) {
        std::cerr << "hugin: fatal: cannot open protocol stream\n";
        return 1;
    }

    try {
        mimirmind::hugin::McpServer    server{"hugin", kVersion};
        mimirmind::hugin::InspectTools inspect{model, tokenizerGguf};
        mimirmind::hugin::ForwardTools forwardTools{[&inspect] {
            return mimirmind::hugin::ForwardTools::Target{inspect.modelPath(), inspect.tokenizerGguf()};
        }};
        inspect.registerOn(server);
        forwardTools.registerOn(server);
        std::cerr << "hugin " << kVersion << ": MCP server ready on stdio"
                  << (model.empty() ? std::string{} : " (model '" + model + "')") << "\n";
        server.run(std::cin, proto);
    } catch (const std::exception& e) {
        std::cerr << "hugin: fatal: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
