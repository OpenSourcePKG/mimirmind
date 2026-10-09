// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// laya_parity — run the native LayaEncoderModel/LayaEncoderRunner backbone on a
// fixed token-id sequence and print the mean-pooled embedding, for parity
// against the upstream python ModernBERT reference (roadmap 8.27.2).
//
// Box-only (needs a CUDA device). Usage:
//   laya_parity <model_dir> [ids_file]
//     <model_dir>  dir with model.safetensors (+ encoder/config.json), weights
//                  under the `encoder.` prefix (the Laya checkpoint layout).
//     [ids_file]   optional whitespace-separated int token ids; default is a
//                  short fixed sequence. Keep T <= local window (128) — the
//                  forward is exact there (see LayaEncoderRunner header).
//
// Output (stdout): one header line, then the H mean-pool floats space-separated,
// so laya_ref.py can diff them.

#include "compute/cuda/GpuMatmul.hpp"
#include "compute/cuda/GpuOps.hpp"
#include "core/gguf/GgufTypes.hpp"
#include "core/gpu/cuda/CudaComputeContext.hpp"
#include "runtime/encoder/laya/LayaEncoderModel.hpp"
#include "runtime/encoder/laya/LayaEncoderRunner.hpp"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    using namespace mimirmind;
    if (argc < 2) {
        std::fprintf(stderr, "usage: laya_parity <model_dir> [ids_file]\n");
        return 2;
    }
    const std::string dir = argv[1];

    std::vector<std::int32_t> ids;
    if (argc >= 3) {
        std::ifstream in(argv[2]);
        if (!in) {
            std::fprintf(stderr, "laya_parity: cannot read ids file %s\n", argv[2]);
            return 2;
        }
        long v;
        while (in >> v) {
            ids.push_back(static_cast<std::int32_t>(v));
        }
    } else {
        ids = {1, 500, 1000, 2000, 2};   // fixed short in-vocab sequence
    }
    if (ids.empty()) {
        std::fprintf(stderr, "laya_parity: no token ids\n");
        return 2;
    }

    try {
        core::cuda::CudaComputeContext ctx{};
        compute::cuda::GpuOps          ops{ctx};
        compute::cuda::GpuMatmul       mm{ctx, ops};

        runtime::encoder::laya::LayaEncoderModel model;
        model.load(dir, ops, core::gguf::GgmlType::F32, "encoder.");
        const auto& c = model.config();
        std::fprintf(stderr,
                     "loaded: hidden=%zu layers=%zu heads=%zu ffn=%zu vocab=%zu "
                     "meanPool=%d weightMiB=%.1f\n",
                     c.hidden, c.numLayers, c.heads, c.ffn, c.vocab,
                     static_cast<int>(c.meanPooling),
                     static_cast<double>(model.weightBytes()) / (1024.0 * 1024.0));

        runtime::encoder::laya::LayaEncoderRunner runner{model, ops, mm};
        const std::vector<float> emb = runner.embed(ids);

        std::printf("H=%zu T=%zu\n", emb.size(), ids.size());
        for (float f : emb) {
            std::printf("%.6f ", f);
        }
        std::printf("\n");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "laya_parity: %s\n", e.what());
        return 1;
    }
    return 0;
}
