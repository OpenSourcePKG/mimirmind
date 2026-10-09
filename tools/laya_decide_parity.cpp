// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// laya_decide_parity — run the native Laya decision head (backbone +
// LayaDecisionHead/Runner) on a dumped (ids, markers, qtype) and print the raw
// per-option logits + act_probability, for parity against the python oracle
// (roadmap 8.27.3). Box-only (CUDA).
//
// Usage: laya_decide_parity <model_dir> <dump_file>
//   dump_file lines:  "qtype <q>" / "ids <i0> <i1> ..." / "markers <m0> ..."

#include "compute/cuda/GpuMatmul.hpp"
#include "compute/cuda/GpuOps.hpp"
#include "core/gguf/GgufTypes.hpp"
#include "core/gpu/cuda/CudaComputeContext.hpp"
#include "runtime/encoder/laya/LayaDecisionHead.hpp"
#include "runtime/encoder/laya/LayaDecisionRunner.hpp"
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
    if (argc < 3) {
        std::fprintf(stderr, "usage: laya_decide_parity <model_dir> <dump_file>\n");
        return 2;
    }
    const std::string dir = argv[1];
    std::int32_t qtype = 0;
    std::vector<std::int32_t> ids;
    std::vector<std::int32_t> markers;
    {
        std::ifstream in(argv[2]);
        if (!in) { std::fprintf(stderr, "cannot read %s\n", argv[2]); return 2; }
        std::string line;
        while (std::getline(in, line)) {
            std::istringstream ss(line);
            std::string tag;
            ss >> tag;
            if (tag == "qtype") { ss >> qtype; }
            else if (tag == "ids") { long v; while (ss >> v) ids.push_back(static_cast<std::int32_t>(v)); }
            else if (tag == "markers") { long v; while (ss >> v) markers.push_back(static_cast<std::int32_t>(v)); }
        }
    }
    if (ids.empty() || markers.empty()) { std::fprintf(stderr, "empty ids/markers\n"); return 2; }

    try {
        core::cuda::CudaComputeContext ctx{};
        compute::cuda::GpuOps          ops{ctx};
        compute::cuda::GpuMatmul       mm{ctx, ops};

        runtime::encoder::laya::LayaEncoderModel model;
        model.load(dir, ops, core::gguf::GgmlType::F32, "encoder.");
        runtime::encoder::laya::LayaDecisionHead head;
        head.load(dir, ops, model.config().hidden, /*nHeadLayers=*/2, core::gguf::GgmlType::F32);

        runtime::encoder::laya::LayaEncoderRunner enc{model, ops, mm};
        runtime::encoder::laya::LayaDecisionRunner dec{model, enc, head, ops, mm};

        const auto r = dec.decide(ids, markers, qtype);
        std::printf("qtype=%d T=%zu k=%zu\n", qtype, ids.size(), markers.size());
        std::printf("logits:");
        for (float f : r.logits) { std::printf(" %.6f", f); }
        std::printf("\nactProb: %.6f\n", r.actProbability);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "laya_decide_parity: %s\n", e.what());
        return 1;
    }
    return 0;
}
