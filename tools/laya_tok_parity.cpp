// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// laya_tok_parity — encode text lines with the native LayaTokenizer and print
// the token ids, for parity against the python AutoTokenizer (roadmap 8.27.4).
// Pure CPU (no CUDA used). Usage: laya_tok_parity <model_dir> <lines_file>

#include "runtime/encoder/laya/LayaTokenizer.hpp"

#include <cstdio>
#include <fstream>
#include <string>

int main(int argc, char** argv) {
    using mimirmind::runtime::encoder::laya::LayaTokenizer;
    if (argc < 3) {
        std::fprintf(stderr, "usage: laya_tok_parity <model_dir> <lines_file>\n");
        return 2;
    }
    try {
        LayaTokenizer tok;
        tok.load(argv[1]);
        std::fprintf(stderr, "vocab=%zu cls=%d sep=%d mask=%d pad=%d unk=%d\n",
                     tok.vocabSize(), tok.clsId(), tok.sepId(), tok.maskId(),
                     tok.padId(), tok.unkId());
        std::ifstream in(argv[2]);
        if (!in) { std::fprintf(stderr, "cannot read %s\n", argv[2]); return 2; }
        std::string line;
        while (std::getline(in, line)) {
            const auto ids = tok.encode(line);
            bool first = true;
            for (int id : ids) {
                std::printf(first ? "%d" : " %d", id);
                first = false;
            }
            std::printf("\n");
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "laya_tok_parity: %s\n", e.what());
        return 1;
    }
    return 0;
}
