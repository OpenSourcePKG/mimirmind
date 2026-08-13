// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// DFlash drafter device-upload validation (Phase 1b). Loads the BF16
// safetensors checkpoint via DFlashDraftModel onto the CUDA device, asserts
// the derived architecture dims, that every weight got a non-null device
// handle, and that a round-tripped tensor (norm.weight, read back from device)
// is byte-identical to the host safetensors bytes — proving the H2D upload
// path. Runs on the box (needs a CUDA device):
//   ./dflash_device_upload_test /opt/mimirmind/models/qwen3.6-35b-a3b-dflash-drafter

#include "runtime/dflash/DFlashDraftModel.hpp"

#include "compute/cuda/GpuOps.hpp"
#include "core/gpu/cuda/CudaComputeContext.hpp"
#include "core/safetensors/SafetensorsModel.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <string>
#include <vector>

namespace fe = mimirmind::core::safetensors;
using mimirmind::compute::cuda::GpuOps;
using mimirmind::core::cuda::CudaComputeContext;
using mimirmind::runtime::dflash::DFlashDraftModel;

namespace {

int g_fail = 0;

void expectEq(const char* what, std::size_t got, std::size_t want) {
    if (got != want) {
        printf("  MISMATCH %s: got %zu want %zu\n", what, got, want);
        ++g_fail;
    }
}

void expectNonNull(const char* what, const void* p) {
    if (p == nullptr) {
        printf("  NULL device ptr: %s\n", what);
        ++g_fail;
    }
}

} // namespace

int main(int argc, char** argv) {
    const std::string path = (argc > 1)
        ? argv[1] : "/opt/mimirmind/models/qwen3.6-35b-a3b-dflash-drafter";
    printf("DFlash device-upload test: %s\n", path.c_str());

    CudaComputeContext ctx{};
    GpuOps ops{ctx};

    DFlashDraftModel model;
    try {
        model.load(path, ops);
    } catch (const std::exception& e) {
        printf("load FAILED: %s\n", e.what());
        return 90;
    }

    const auto& c = model.config();
    printf("derived config: layers=%zu hidden=%zu headDim=%zu nQHeads=%zu "
           "nKvHeads=%zu inter=%zu taps=%zu\n",
           c.numLayers, c.hidden, c.headDim, c.nQHeads, c.nKvHeads, c.inter, c.taps);
    printf("uploaded %.1f MiB to device\n", model.uploadedBytes() / 1048576.0);

    // Expected Qwen3.6-35B-A3B-DFlash architecture.
    expectEq("numLayers", c.numLayers, 6);
    expectEq("hidden", c.hidden, 2048);
    expectEq("headDim", c.headDim, 128);
    expectEq("nQHeads", c.nQHeads, 32);
    expectEq("nKvHeads", c.nKvHeads, 8);
    expectEq("inter", c.inter, 6144);
    expectEq("taps", c.taps, 8);
    expectEq("layerCount", model.layerCount(), 6);

    // Every layer weight must have a non-null device handle.
    for (std::size_t l = 0; l < model.layerCount(); ++l) {
        const auto& w = model.layer(l);
        expectNonNull("qProj", w.qProj);       expectNonNull("kProj", w.kProj);
        expectNonNull("vProj", w.vProj);       expectNonNull("oProj", w.oProj);
        expectNonNull("qNorm", w.qNorm);       expectNonNull("kNorm", w.kNorm);
        expectNonNull("inputLn", w.inputLn);   expectNonNull("postAttnLn", w.postAttnLn);
        expectNonNull("gateProj", w.gateProj); expectNonNull("upProj", w.upProj);
        expectNonNull("downProj", w.downProj);
    }
    expectNonNull("fc", model.fc());
    expectNonNull("hiddenNorm", model.hiddenNorm());
    expectNonNull("norm", model.norm());

    // Round-trip proofs of the H2D upload.
    {
        fe::SafetensorsModel sm;
        sm.open(path);
        // (a) A BF16 LINEAR weight round-trips byte-for-byte (uploaded verbatim).
        const std::span<const std::uint8_t> qh =
            sm.tensorBytes("layers.0.self_attn.q_proj.weight");
        std::vector<std::uint8_t> qd(qh.size());
        ops.readbackToHost(qd.data(), model.layer(0).qProj, qd.size());
        if (qh.empty() || std::memcmp(qh.data(), qd.data(), qh.size()) != 0) {
            printf("  ROUND-TRIP MISMATCH on q_proj bf16 (%zu bytes)\n", qh.size());
            ++g_fail;
        } else {
            printf("round-trip OK: q_proj %zu bf16 bytes byte-identical device<->host\n",
                   qh.size());
        }
        // (b) An F32 RMSNorm weight equals the host BF16 widened to F32.
        const std::span<const std::uint8_t> nh = sm.tensorBytes("norm.weight");
        const std::size_t n = nh.size() / sizeof(std::uint16_t);
        const auto* nbf = reinterpret_cast<const std::uint16_t*>(nh.data());
        std::vector<float> nd(n);
        ops.readbackToHost(nd.data(), model.norm(), n * sizeof(float));
        bool ok = !nh.empty();
        for (std::size_t i = 0; i < n && ok; ++i) {
            const std::uint32_t bits = static_cast<std::uint32_t>(nbf[i]) << 16;
            float ref;
            std::memcpy(&ref, &bits, sizeof(float));
            if (nd[i] != ref) { ok = false; }
        }
        if (!ok) {
            printf("  NORM F32-WIDEN MISMATCH on norm.weight\n");
            ++g_fail;
        } else {
            printf("norm.weight F32-widen OK: %zu elems match host bf16->f32\n", n);
        }
    }

    printf("\nRESULT: %s (%d problems)\n",
           g_fail == 0 ? "PASS — DFlash weights uploaded to device" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
