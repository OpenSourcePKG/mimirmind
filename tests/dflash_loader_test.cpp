// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// DFlash drafter loader validation (Phase 1). Opens the bf16 safetensors
// checkpoint via SafetensorsModel and asserts the FULL expected tensor
// inventory of a DFlashDraftModel (6 qwen3 layers + fc/hidden_norm/norm,
// no embed/lm_head — borrowed from the target) with exact shapes and BF16
// dtype. This encodes the loader contract and is runnable against the real
// checkpoint on the box:
//   ./dflash_loader_test /opt/mimirmind/models/qwen3.6-35b-a3b-dflash-drafter
//
// CPU-only (links mimirmind_core_common); no CUDA / device upload here.

#include "core/safetensors/SafetensorsModel.hpp"
#include "core/safetensors/SafetensorsHeader.hpp"
#include "core/safetensors/SafetensorsDtype.hpp"

#include <cstdio>
#include <string>
#include <vector>
#include <cstdint>

namespace fe = mimirmind::core::safetensors;

namespace {

// DFlash drafter architecture (Qwen3.6-35B-A3B-DFlash, from config.json).
constexpr int kLayers = 6;
constexpr int kHidden = 2048;   // == target text hidden_size
constexpr int kHeadDim = 128;
constexpr int kNQ = 32, kNKv = 8;          // GQA 4:1
constexpr int kInter = 6144;
constexpr int kTaps = 8;                   // target_layer_ids [1,6,11,16,22,27,32,37]

int g_fail = 0;

void check(fe::SafetensorsModel& sm, const std::string& name,
           std::vector<std::uint64_t> want) {
    const auto* t = sm.find(name);
    if (!t) { printf("  MISSING: %s\n", name.c_str()); ++g_fail; return; }
    if (t->dtype != fe::SafetensorsDtype::BF16) {
        printf("  DTYPE!=BF16: %s (%d)\n", name.c_str(), (int)t->dtype); ++g_fail; return;
    }
    if (t->shape != want) {
        printf("  SHAPE MISMATCH: %s got [", name.c_str());
        for (auto d : t->shape) printf("%llu ", (unsigned long long)d);
        printf("] want [");
        for (auto d : want) printf("%llu ", (unsigned long long)d);
        printf("]\n"); ++g_fail; return;
    }
}

}  // namespace

int main(int argc, char** argv) {
    const std::string path = (argc > 1)
        ? argv[1] : "/opt/mimirmind/models/qwen3.6-35b-a3b-dflash-drafter";
    printf("DFlash loader test: %s\n", path.c_str());

    fe::SafetensorsModel sm;
    try { sm.open(path); }
    catch (const std::exception& e) { printf("open FAILED: %s\n", e.what()); return 90; }
    printf("opened: %zu tensors, %zu shards, declared %.1f MiB\n",
           sm.tensorCount(), sm.shardCount(), sm.declaredTotalSize() / 1048576.0);

    const std::uint64_t H = kHidden, Q = (std::uint64_t)kNQ * kHeadDim,   // 4096
                        KV = (std::uint64_t)kNKv * kHeadDim,              // 1024
                        I = kInter, HD = kHeadDim;
    for (int L = 0; L < kLayers; ++L) {
        std::string p = "layers." + std::to_string(L) + ".";
        check(sm, p + "self_attn.q_proj.weight", {Q, H});
        check(sm, p + "self_attn.k_proj.weight", {KV, H});
        check(sm, p + "self_attn.v_proj.weight", {KV, H});
        check(sm, p + "self_attn.o_proj.weight", {H, Q});
        check(sm, p + "self_attn.q_norm.weight", {HD});
        check(sm, p + "self_attn.k_norm.weight", {HD});
        check(sm, p + "input_layernorm.weight", {H});
        check(sm, p + "post_attention_layernorm.weight", {H});
        check(sm, p + "mlp.gate_proj.weight", {I, H});
        check(sm, p + "mlp.up_proj.weight", {I, H});
        check(sm, p + "mlp.down_proj.weight", {H, I});
    }
    // DFlash-specific top-level tensors.
    check(sm, "fc.weight", {H, (std::uint64_t)kTaps * H});   // [2048, 16384]
    check(sm, "hidden_norm.weight", {H});
    check(sm, "norm.weight", {H});

    // No embed/lm_head (borrowed from target).
    for (const char* n : {"embed_tokens.weight", "lm_head.weight",
                          "model.embed_tokens.weight"}) {
        if (sm.find(n)) { printf("  UNEXPECTED (should be borrowed): %s\n", n); ++g_fail; }
    }

    // Param + byte accounting.
    std::uint64_t params = 0, bytes = 0;
    for (const auto* t : sm.tensors()) { params += t->nelements; bytes += t->nbytes; }
    printf("total: %.1f M params, %.1f MiB\n", params / 1e6, bytes / 1048576.0);

    printf("\nRESULT: %s (%d problems)\n",
           g_fail == 0 ? "PASS — full DFlash tensor inventory validated" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
