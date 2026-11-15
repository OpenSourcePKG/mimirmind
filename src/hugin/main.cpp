// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

// Hugin — mimirmind's adaptation companion (roadmap 7). This binary is the
// MCP server Claude talks to over stdio. stdout carries ONLY protocol
// messages; every log line goes to stderr.

#include "hugin/InspectTools.hpp"
#include "hugin/McpServer.hpp"

#include <exception>
#include <iostream>
#include <string>
#include <string_view>

namespace {

constexpr const char* kVersion = "0.1.0";

void usage() {
    std::cerr << "usage: hugin [--model <file.gguf>]\n"
                 "  Runs an MCP server on stdin/stdout (newline-delimited JSON-RPC 2.0).\n"
                 "  Register with Claude Code, e.g.:\n"
                 "    claude mcp add hugin -- /path/to/hugin --model /path/to/model.gguf\n";
}

} // namespace

int main(int argc, char** argv) {
    std::string model;
    for (int i = 1; i < argc; ++i) {
        const std::string_view a = argv[i];
        if (a == "--model" && i + 1 < argc) {
            model = argv[++i];
        } else if (a == "--help" || a == "-h") {
            usage();
            return 0;
        } else {
            std::cerr << "hugin: unknown argument '" << a << "'\n";
            usage();
            return 2;
        }
    }

    try {
        mimirmind::hugin::McpServer    server{"hugin", kVersion};
        mimirmind::hugin::InspectTools inspect{model};
        inspect.registerOn(server);
        std::cerr << "hugin " << kVersion << ": MCP server ready on stdio"
                  << (model.empty() ? std::string{} : " (model '" + model + "')") << "\n";
        server.run(std::cin, std::cout);
    } catch (const std::exception& e) {
        std::cerr << "hugin: fatal: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
