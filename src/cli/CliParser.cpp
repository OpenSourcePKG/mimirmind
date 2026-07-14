// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "cli/CliParser.hpp"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>

namespace mimirmind::cli {

const char* const kBanner =
    "+------------------------------------------------------------+\n"
    "|                          Mimirmind                         |\n"
    "|   M1-M5 GPU matmul + M7a InferenceEngine (Envoy -> Mimir)  |\n"
    "+------------------------------------------------------------+\n";

const char* const kUsage =
    "Usage:\n"
    "  mimirmind [smoke|serve|parity] [options]\n"
    "\n"
    "Modes:\n"
    "  smoke              Run M1-M5 diagnostics + end-to-end generate (default)\n"
    "  serve              Start the OpenAI-compatible HTTP server\n"
    "  parity             Tensor-parity test: run llama.cpp + mimirmind on\n"
    "                     the same prompt, dump per-block hidden state, diff\n"
    "\n"
    "Options:\n"
    "  --config PATH      config.json file (default ./config.json).\n"
    "                     Fails hard if the file is missing or malformed —\n"
    "                     copy config.example.json and edit for your host.\n"
    "  --model PATH       GGUF model file (overrides models[<defaultModel>].path)\n"
    "  --port N           HTTP port for serve mode (overrides server.port)\n"
    "  --log-level LEVEL  trace|debug|info|warn|error|off (overrides server.log.level)\n"
    "  --log-file PATH    Append log lines here (overrides server.log.file)\n"
    "  --prompt TEXT      Prompt for smoke generate (default \"Hello, world!\")\n"
    "  --max-new N        Max new tokens for smoke generate (default 20)\n"
    "  --temperature F    Sampling temperature, 0 = greedy (default 0)\n"
    "  --top-k N          Top-K cutoff, 0 = disabled (default 0)\n"
    "  --top-p F          Top-P (nucleus) cutoff, 1.0 = disabled (default 1.0)\n"
    "  --seed N           RNG seed, 0 = random_device (default 0)\n"
    "  --chat             Wrap --prompt as a user message via the chat template\n"
    "  --system TEXT      System message for --chat (default: Qwen2.5 default)\n"
    "  --dump-dir PATH    (parity only) directory for reference-oracle dumps\n"
    "                     — sets diagnostics.parityDump for the subprocess.\n"
    "  --attach URI       (serve only) attach to a running Munin daemon\n"
    "                     instead of loading weights locally. URI form is\n"
    "                     'unix:/path/to/munin.sock'. Governor ownership\n"
    "                     stays with Munin; the worker skips thermal /\n"
    "                     governor / fan install and the governor flock.\n"
    "  -h, --help         Show this help and exit\n";

bool parseArgs(int argc, char** argv, CliArgs& out) {
    int idx = 1;

    if (argc > 1 && argv[1][0] != '-') {
        const std::string_view m{argv[1]};
        if (m == "smoke") {
            out.mode = Mode::Smoke;
        } else if (m == "serve") {
            out.mode = Mode::Serve;
        } else if (m == "parity") {
            out.mode = Mode::Parity;
        } else if (m == "-h" || m == "--help") {
            std::cout << kUsage;
            std::exit(0);
        } else {
            std::cerr << "unknown mode '" << m << "'\n" << kUsage;
            return false;
        }
        idx = 2;
    }

    auto needValue = [&](const char* flag) -> const char* {
        if (idx + 1 >= argc) {
            std::cerr << flag << " requires a value\n";
            return nullptr;
        }
        return argv[++idx];
    };

    for (; idx < argc; ++idx) {
        const std::string_view a{argv[idx]};
        if (a == "-h" || a == "--help") {
            std::cout << kUsage;
            std::exit(0);
        } else if (a == "--config") {
            const char* v = needValue("--config");
            if (v == nullptr) return false;
            out.configPath = v;
        } else if (a == "--model") {
            const char* v = needValue("--model");
            if (v == nullptr) return false;
            out.modelPath = v;
        } else if (a == "--prompt") {
            const char* v = needValue("--prompt");
            if (v == nullptr) return false;
            out.prompt = v;
        } else if (a == "--max-new") {
            const char* v = needValue("--max-new");
            if (v == nullptr) return false;
            out.maxNew = static_cast<std::size_t>(std::strtoull(v, nullptr, 10));
        } else if (a == "--port") {
            const char* v = needValue("--port");
            if (v == nullptr) return false;
            out.port = static_cast<std::uint16_t>(std::strtoul(v, nullptr, 10));
        } else if (a == "--log-level") {
            const char* v = needValue("--log-level");
            if (v == nullptr) return false;
            out.logLevel = v;
        } else if (a == "--log-file") {
            const char* v = needValue("--log-file");
            if (v == nullptr) return false;
            out.logFile = v;
        } else if (a == "--temperature") {
            const char* v = needValue("--temperature");
            if (v == nullptr) return false;
            out.temperature = std::strtof(v, nullptr);
        } else if (a == "--top-k") {
            const char* v = needValue("--top-k");
            if (v == nullptr) return false;
            out.topK = static_cast<std::size_t>(std::strtoull(v, nullptr, 10));
        } else if (a == "--top-p") {
            const char* v = needValue("--top-p");
            if (v == nullptr) return false;
            out.topP = std::strtof(v, nullptr);
        } else if (a == "--seed") {
            const char* v = needValue("--seed");
            if (v == nullptr) return false;
            out.seed = std::strtoull(v, nullptr, 10);
        } else if (a == "--chat") {
            out.chat = true;
        } else if (a == "--system") {
            const char* v = needValue("--system");
            if (v == nullptr) return false;
            out.systemMessage = v;
        } else if (a == "--dump-dir") {
            const char* v = needValue("--dump-dir");
            if (v == nullptr) return false;
            out.dumpDir = v;
        } else if (a == "--attach") {
            const char* v = needValue("--attach");
            if (v == nullptr) return false;
            // Only unix: URIs supported for now. Reject anything else
            // up-front so a mistyped tcp: URI does not silently mean
            // "no socket path" and drop into standalone load.
            const std::string_view uri{v};
            constexpr std::string_view kPrefix{"unix:"};
            if (uri.substr(0, kPrefix.size()) != kPrefix) {
                std::cerr << "--attach: only 'unix:PATH' URIs are supported "
                             "(got '" << uri << "')\n";
                return false;
            }
            out.attachSocket = std::string{uri.substr(kPrefix.size())};
            if (out.attachSocket.empty()) {
                std::cerr << "--attach: path after 'unix:' is empty\n";
                return false;
            }
        } else {
            std::cerr << "unknown argument '" << a << "'\n" << kUsage;
            return false;
        }
    }

    return true;
}

} // namespace mimirmind::cli