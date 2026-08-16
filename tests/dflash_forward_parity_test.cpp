// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// DFlash draft-forward parity (Phase 3). Validates the C++ DFlashDraftRunner
// against the z-lab reference golden (tools/dflash_golden.py). Incremental:
//   P3.1 — materializeContext (fc -> hidden_norm) vs golden ctx_proj.bin
// The golden is an F32 reference; our impl uses the BF16 fc weight, so the
// check is tolerance-based (not bit-exact). Run on the box (needs a CUDA dev):
//   ./dflash_forward_parity_test <drafter_dir> <golden_dir>

#include "runtime/dflash/DFlashDraftModel.hpp"
#include "runtime/dflash/DFlashDraftRunner.hpp"

#include "compute/cuda/GpuMatmul.hpp"
#include "compute/cuda/GpuOps.hpp"
#include "core/gpu/cuda/CudaComputeContext.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using mimirmind::compute::cuda::GpuMatmul;
using mimirmind::compute::cuda::GpuOps;
using mimirmind::core::cuda::CudaComputeContext;
using mimirmind::runtime::dflash::DFlashDraftModel;
using mimirmind::runtime::dflash::DFlashDraftRunner;

namespace {

// Fixed by tools/dflash_golden.py (seeded controlled inputs).
constexpr std::size_t kCtx  = 8;
constexpr std::size_t kBs   = 8;
constexpr std::size_t kH    = 2048;
constexpr std::size_t kTaps = 8;

std::vector<float> readBin(const std::string& path, std::size_t n) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        printf("  cannot open %s\n", path.c_str());
        return {};
    }
    std::vector<float> v(n);
    in.read(reinterpret_cast<char*>(v.data()),
            static_cast<std::streamsize>(n * sizeof(float)));
    if (static_cast<std::size_t>(in.gcount()) != n * sizeof(float)) {
        printf("  short read on %s\n", path.c_str());
        return {};
    }
    return v;
}

} // namespace

int main(int argc, char** argv) {
    const std::string drafter = (argc > 1)
        ? argv[1] : "/opt/mimirmind/models/qwen3.6-35b-a3b-dflash-drafter";
    const std::string golden  = (argc > 2) ? argv[2] : "/opt/mimirmind/dflash_golden";
    printf("DFlash forward-parity test\n  drafter=%s\n  golden=%s\n",
           drafter.c_str(), golden.c_str());

    CudaComputeContext ctx{};
    GpuOps  ops{ctx};
    GpuMatmul mm{ctx, ops};

    DFlashDraftModel model;
    try {
        model.load(drafter, ops);
    } catch (const std::exception& e) {
        printf("load FAILED: %s\n", e.what());
        return 90;
    }
    DFlashDraftRunner runner{model, ops, mm};

    // ---- P3.1: materializeContext (fc -> hidden_norm) ----------------------
    const std::vector<float> th  = readBin(golden + "/target_hidden.bin", kCtx * kTaps * kH);
    const std::vector<float> ref = readBin(golden + "/ctx_proj.bin", kCtx * kH);
    if (th.empty() || ref.empty()) {
        printf("RESULT: FAIL (missing golden bins)\n");
        return 91;
    }

    mimirmind::compute::ComputeBuffer thBuf  = ops.allocate(th.size() * sizeof(float));
    mimirmind::compute::ComputeBuffer outBuf = ops.allocate(kCtx * kH * sizeof(float));
    ops.uploadHostBytes(thBuf.get(), th.data(), th.size() * sizeof(float));

    runner.materializeContext(static_cast<const float*>(thBuf.get()), kCtx,
                              static_cast<float*>(outBuf.get()));
    ops.flush();

    std::vector<float> got(kCtx * kH);
    ops.readbackToHost(got.data(), outBuf.get(), got.size() * sizeof(float));

    double maxAbs = 0.0, sumAbs = 0.0, maxRel = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double a = std::fabs(static_cast<double>(got[i]) - ref[i]);
        maxAbs = std::max(maxAbs, a);
        sumAbs += a;
        const double denom = std::fabs(static_cast<double>(ref[i])) + 1e-3;
        maxRel = std::max(maxRel, a / denom);
    }
    const double meanAbs = sumAbs / static_cast<double>(got.size());
    printf("ctx_proj parity: maxAbs=%.4g meanAbs=%.4g maxRel=%.4g\n",
           maxAbs, meanAbs, maxRel);
    const bool pass1 = (maxAbs < 0.06) && (meanAbs < 0.01);
    printf("  P3.1 materializeContext: %s\n", pass1 ? "PASS" : "FAIL");

    // ---- P3.2: full 6-layer draft forward vs golden out.bin ----------------
    const std::vector<float> noise = readBin(golden + "/noise_embedding.bin", kBs * kH);
    const std::vector<float> outRef = readBin(golden + "/out.bin", kBs * kH);
    bool pass2 = false;
    if (noise.empty() || outRef.empty()) {
        printf("  P3.2: FAIL (missing noise/out golden)\n");
    } else {
        mimirmind::compute::ComputeBuffer noiseBuf = ops.allocate(noise.size() * sizeof(float));
        mimirmind::compute::ComputeBuffer fwdBuf   = ops.allocate(kBs * kH * sizeof(float));
        ops.uploadHostBytes(noiseBuf.get(), noise.data(), noise.size() * sizeof(float));
        runner.draftForward(static_cast<const float*>(noiseBuf.get()),
                            static_cast<const float*>(thBuf.get()),
                            kBs, kCtx, static_cast<float*>(fwdBuf.get()));
        std::vector<float> fwd(kBs * kH);
        ops.readbackToHost(fwd.data(), fwdBuf.get(), fwd.size() * sizeof(float));

        double mA = 0.0, sA = 0.0;
        for (std::size_t i = 0; i < fwd.size(); ++i) {
            const double a = std::fabs(static_cast<double>(fwd[i]) - outRef[i]);
            mA = std::max(mA, a);
            sA += a;
        }
        const double meanA = sA / static_cast<double>(fwd.size());
        printf("out parity:      maxAbs=%.4g meanAbs=%.4g\n", mA, meanA);
        printf("  got[0,:4]= %.5f %.5f %.5f %.5f\n", fwd[0], fwd[1], fwd[2], fwd[3]);
        printf("  ref[0,:4]= %.5f %.5f %.5f %.5f\n", outRef[0], outRef[1], outRef[2], outRef[3]);
        // 6 bf16 layers accumulate more error than one fc; F32 host attention.
        pass2 = (mA < 0.25) && (meanA < 0.04);
        printf("  P3.2 draftForward: %s\n", pass2 ? "PASS" : "FAIL");
    }

    const bool pass = pass1 && pass2;
    printf("\nRESULT: %s (P3.1+P3.2 draft forward)\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
