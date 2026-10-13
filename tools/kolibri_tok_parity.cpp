// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// kolibri_tok_parity — encode text lines with the native mimirmind Tokenizer
// (byte-level GPT-2/Qwen BPE loaded from tokenizer.json) and print the token
// ids, for parity against the HF `tokenizers`/transformers oracle on a DE+EN
// corpus (roadmap 5.34.3 / I-2). Also checks the decode(encode(x))==x
// round-trip. Pure CPU, no model/CUDA load. Usage:
//   kolibri_tok_parity <model_dir> <lines_file>
// stdout: one space-separated id list per input line (addBos=false).
// stderr: per-line round-trip failures + a final summary.

#include "model/Tokenizer.hpp"

#include <cstdio>
#include <fstream>
#include <span>
#include <string>

int main(int argc, char** argv) {
    using mimirmind::model::Tokenizer;
    if (argc < 3) {
        std::fprintf(stderr, "usage: kolibri_tok_parity <model_dir> <lines_file>\n");
        return 2;
    }
    try {
        Tokenizer tok;
        tok.loadFromHfJson(argv[1]);
        std::fprintf(stderr, "bos=%d eos=%d pad=%d unk=%d\n",
                     tok.bosId(), tok.eosId(), tok.padId(), tok.unknownId());
        std::ifstream in(argv[2]);
        if (!in) { std::fprintf(stderr, "cannot read %s\n", argv[2]); return 2; }
        std::string line;
        std::size_t nLines = 0, nRtFail = 0;
        while (std::getline(in, line)) {
            ++nLines;
            const auto ids = tok.encode(line, /*addBos=*/false);
            bool first = true;
            for (int id : ids) {
                std::printf(first ? "%d" : " %d", id);
                first = false;
            }
            std::printf("\n");
            // Round-trip: decode must reproduce the input byte-for-byte.
            const std::string rt =
                tok.decode(std::span<const std::int32_t>(ids), /*skipSpecial=*/false);
            if (rt != line) {
                ++nRtFail;
                std::fprintf(stderr, "RT_FAIL line %zu: in=[%s] out=[%s]\n",
                             nLines, line.c_str(), rt.c_str());
            }
        }
        std::fprintf(stderr, "lines=%zu round_trip_fail=%zu\n", nLines, nRtFail);
        return (nRtFail == 0) ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "kolibri_tok_parity: %s\n", e.what());
        return 1;
    }
}
