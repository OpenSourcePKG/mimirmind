// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "mimirmind/ParityMode.hpp"

#include "mimirmind/CliArgs.hpp"

#include "core/config/Config.hpp"
#include "model/Tokenizer.hpp"
#include "runtime/InferenceEngine.hpp"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace mimirmind::cli {

int runParity(const CliArgs& argsIn, const ::mimirmind::core::config::Config& cfg) {
    if (argsIn.modelPath.empty()) {
        std::cerr << "parity: --model PATH is required "
                     "(fill models[].path in config.json)\n";
        return 2;
    }

    const std::string dumpRoot = "/tmp/dumps";
    const std::string llamaDir = dumpRoot + "/llama";
    const std::string mimirPfx = dumpRoot + "/mimir";

    // Wipe + recreate dump dirs.
    std::filesystem::remove_all(dumpRoot);
    std::filesystem::create_directories(llamaDir);

    auto shellQuote = [](const std::string& s) {
        std::string out{"'"};
        for (char c : s) {
            if (c == '\'') { out += "'\\''"; } else { out += c; }
        }
        out += "'";
        return out;
    };

    // --- 1. llama-parity-dump --------------------------------------------
    {
        std::cout << "\n=== [1/3] llama-parity-dump (reference oracle) ===\n";
        std::string cmd = "llama-parity-dump";
        cmd += " --model "    + shellQuote(argsIn.modelPath);
        cmd += " --prompt "   + shellQuote(argsIn.prompt);
        cmd += " --dump-dir " + shellQuote(llamaDir);
        cmd += " --n-predict 0";  // prefill-only, no generation
        std::cout << "$ " << cmd << "\n" << std::flush;
        const int rc = std::system(cmd.c_str());
        if (rc != 0) {
            std::cerr << "parity: llama-parity-dump exited " << rc << "\n";
            return 1;
        }
    }

    // --- 2. mimirmind own forward dump -----------------------------------
    {
        std::cout << "\n=== [2/3] mimirmind dump ===\n" << std::flush;

        // Override diagnostics.parityDump for the parity subcommand so
        // the engine writes per-stage bin files under the dump prefix.
        ::mimirmind::core::config::Config parityCfg = cfg;
        parityCfg.diagnostics.parityDump = mimirPfx;

        ::mimirmind::runtime::InferenceEngine engine{parityCfg};
        engine.loadModel(argsIn.modelPath);

        const auto promptIds = engine.tokenizer().encode(argsIn.prompt, true);
        ::mimirmind::runtime::GenerateParams gp{};
        gp.maxNewTokens = 1;  // only need prefill; one token is fine
        engine.generate(std::span<const std::int32_t>{promptIds.data(),
                                                       promptIds.size()},
                        gp);
    }

    // --- 3. parity-diff ---------------------------------------------------
    {
        std::cout << "\n=== [3/3] parity-diff ===\n" << std::flush;
        const std::string cmd = "parity-diff " +
            shellQuote(llamaDir) + " " + shellQuote(mimirPfx);
        std::cout << "$ " << cmd << "\n" << std::flush;
        return std::system(cmd.c_str());
    }
}

} // namespace mimirmind::cli