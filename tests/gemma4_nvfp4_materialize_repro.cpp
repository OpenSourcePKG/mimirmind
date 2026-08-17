// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Minimal repro for the GB10 illegal-access during Gemma4 NVFP4 BF16
// materialisation (roadmap 5.12 blocker). Isolated from the serve harness so it
// loads + materialises + exits in seconds. Two parts:
//   A) SYNTHETIC: a big (~1.875 GiB) Managed alloc, then a small one, then a
//      widen kernel on the small one — tests the "large-alloc shadow" theory in
//      pure isolation, with mitigation variants (full memset, prefetch, order).
//   B) REAL: loadNvfp4Model(gemma) then widen SPECIFIC norm tensors one-by-one
//      with a device sync after each, to see exactly which uploaded tensor a
//      kernel cannot read.
//   ./gemma4_nvfp4_materialize_repro [/opt/mimirmind/models/gemma-4-12b-it-nvfp4]

#include "compute/cuda/CudaMaterializerOps.hpp"
#include "compute/cuda/GpuOps.hpp"
#include "core/gpu/cuda/CudaComputeContext.hpp"
#include "core/modelopt/Gemma4Materializer.hpp"
#include "core/safetensors/SafetensorsDtype.hpp"
#include "core/safetensors/SafetensorsModel.hpp"
#include "runtime/nvfp4/ComputeOpsUploader.hpp"
#include "runtime/nvfp4/Gemma4Config.hpp"
#include "runtime/nvfp4/NvFp4Materializer.hpp"
#include "runtime/nvfp4/NvFp4Model.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using mimirmind::compute::ComputeBuffer;
using mimirmind::compute::cuda::CudaMaterializerOps;
using mimirmind::compute::cuda::GpuOps;
using mimirmind::core::cuda::CudaComputeContext;
using mimirmind::core::safetensors::SafetensorsDtype;
using mimirmind::runtime::nvfp4::ComputeOpsUploader;

namespace {

bool synced(const char* what) {
    const cudaError_t rc = cudaDeviceSynchronize();
    if (rc != cudaSuccess) {
        printf("  [FAULT] %s: %s\n", what, cudaGetErrorString(rc));
        // Clear the sticky error so subsequent probes can still run (best
        // effort — an illegal access usually poisons the whole context).
        cudaGetLastError();
        return false;
    }
    printf("  [ ok  ] %s\n", what);
    return true;
}

constexpr std::uint64_t kEmbedBytes = 2013265920ull; // 262144*3840*2 = 1.875 GiB

// Fill a device buffer's first `n` BF16 elements with 1.0 (0x3F80) via H2D.
void fillBf16(GpuOps& ops, void* dev, std::size_t n) {
    std::vector<std::uint16_t> host(n, 0x3F80);
    ops.uploadHostBytes(dev, host.data(), n * 2);
}

} // namespace

int main(int argc, char** argv) {
    // Unbuffered stdout: a hard CUDA fault aborts the process without flushing,
    // so fully-buffered piped output is lost (the single-prefix REPRO_PREFIX=k
    // runs printed nothing). Force every printf to hit the pipe immediately.
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    // When bisecting the REAL plan (REPRO_PREFIX / REPRO_E), skip the synthetic
    // diagnostic parts (A/C/D/F/G). Their repeated multi-GiB Managed alloc/free
    // churn contaminates the context and produces spurious faults in PART E — so
    // an isolated PART E is the only trustworthy read of executeMaterialization.
    const bool bisect =
        std::getenv("REPRO_PREFIX") != nullptr || std::getenv("REPRO_E") != nullptr
        || std::getenv("REPRO_STEPWISE") != nullptr
        || std::getenv("REPRO_E_WIDENLOOP") != nullptr;

    const std::string dir =
        (argc > 1) ? argv[1] : "/opt/mimirmind/models/gemma-4-12b-it-nvfp4";
    printf("gemma4 nvfp4 materialize repro — %s\n", dir.c_str());

    CudaComputeContext ctx{};
    GpuOps ops{ctx};
    CudaMaterializerOps devOps{ctx, ops};
    synced("startup");

    auto widen = [&](const char* label, const void* src, std::size_t n) {
        ComputeBuffer dst = ops.allocate(n * 4);
        devOps.widenToF32(dst.get(), src, SafetensorsDtype::BF16, n);
        char msg[256];
        std::snprintf(msg, sizeof(msg), "widen %s (n=%zu src=%p dst=%p)", label,
                      n, src, dst.get());
        return synced(msg);
    };

    // ================= PART A: synthetic isolation =================
    if (!bisect) {
    printf("\n=== PART A: synthetic alloc-pattern probes ===\n");
    {
        // A0: small-only widen (baseline, no big alloc before it).
        ComputeBuffer s0 = ops.allocate(7680);
        fillBf16(ops, s0.get(), 3840);
        widen("A0 small-only", s0.get(), 3840);

        // A1: big alloc, then small alloc, then widen small (the suspect).
        ComputeBuffer big = ops.allocate(kEmbedBytes);
        synced("A1 big alloc");
        ComputeBuffer s1 = ops.allocate(7680);
        fillBf16(ops, s1.get(), 3840);
        widen("A1 small-after-big", s1.get(), 3840);

        // A2: same but memset the WHOLE small buffer first (touch every page).
        ComputeBuffer s2 = ops.allocate(7680);
        cudaMemset(s2.get(), 0, 7680);
        synced("A2 memset small");
        fillBf16(ops, s2.get(), 3840);
        widen("A2 small-after-big + full memset", s2.get(), 3840);

        // A4: memset the WHOLE big buffer (touch all its pages) then widen small.
        ComputeBuffer big2 = ops.allocate(kEmbedBytes);
        cudaMemset(big2.get(), 0, kEmbedBytes);
        synced("A4 memset whole big");
        ComputeBuffer s4 = ops.allocate(7680);
        fillBf16(ops, s4.get(), 3840);
        widen("A4 small-after-memset-big", s4.get(), 3840);
    }

    } // end if(!bisect) PART A

    // ================= PART B: real gemma load =================
    printf("\n=== PART B: real gemma load + per-tensor widen ===\n");
    ComputeOpsUploader up{ops};
    mimirmind::runtime::nvfp4::NvFp4Model model =
        mimirmind::runtime::nvfp4::loadNvfp4Model(dir, up);
    synced("B loadNvfp4Model");

    // REPRO_WIDENLOOP=N: right after load, widen the SAME norm tensor N times
    // (fresh dst freed each, sync+check each). If this faults around a fixed
    // count independent of tensor/keep/size, the root cause is a kernel-launch /
    // context resource threshold on this GB10 box — not the Gemma plan.
    if (const char* wl = std::getenv("REPRO_WIDENLOOP")) {
        using Dt = mimirmind::core::safetensors::SafetensorsDtype;
        const int N = std::atoi(wl);
        const auto* nrm = model.find("model.language_model.norm.weight");
        const auto* iln =
            model.find("model.language_model.layers.0.input_layernorm.weight");
        const bool addone = std::getenv("REPRO_WL_ADDONE") != nullptr;
        // REPRO_SM_FIRST: open a second SafetensorsModel (mmap ~10 GB of shards)
        // BEFORE the known-good pair loop. If the loop then faults, the mmap
        // coexisting with the CUDA managed model is the poison.
        mimirmind::core::safetensors::SafetensorsModel smPoison;
        if (std::getenv("REPRO_SM_FIRST") != nullptr) {
            smPoison.open(dir);
            printf("  [SM_FIRST] opened safetensors mmap before widenloop\n");
        }
        printf("  WIDENLOOP N=%d pair(norm=%p, input_ln=%p) dtypes(%d,%d) addone=%d\n",
               N, nrm ? nrm->devPtr : nullptr, iln ? iln->devPtr : nullptr,
               nrm ? static_cast<int>(nrm->dtype) : -1,
               iln ? static_cast<int>(iln->dtype) : -1, (int)addone);
        // REPRO_WL_PROBE: do a readF32 (cudaMemcpy D2H on the default stream)
        // right before each widen — exactly what the stepwise src-probe does.
        // If WIDENLOOP (otherwise clean) now faults, that D2H-before-kernel is the
        // GB10 trigger.
        const bool wlprobe = std::getenv("REPRO_WL_PROBE") != nullptr;
        for (int j = 0; j < N && nrm && iln; ++j) {
            // widen norm (step-1 analogue)
            ComputeBuffer dn = ops.allocate(3840 * 4);
            if (wlprobe) { (void)devOps.readF32(nrm->devPtr); cudaDeviceSynchronize(); }
            devOps.widenToF32(dn.get(), nrm->devPtr, nrm->dtype, 3840);
            if (addone) devOps.addOneInPlaceF32(dn.get(), 3840);
            cudaError_t rc = cudaDeviceSynchronize();
            if (rc != cudaSuccess) {
                printf("  WIDENLOOP FAULT at iter %d NORM: %s\n", j,
                       cudaGetErrorString(rc)); cudaGetLastError(); break;
            }
            // widen input_layernorm (step-3 analogue)
            ComputeBuffer di = ops.allocate(3840 * 4);
            if (wlprobe) { (void)devOps.readF32(iln->devPtr); cudaDeviceSynchronize(); }
            devOps.widenToF32(di.get(), iln->devPtr, iln->dtype, 3840);
            if (addone) devOps.addOneInPlaceF32(di.get(), 3840);
            rc = cudaDeviceSynchronize();
            if (rc != cudaSuccess) {
                printf("  WIDENLOOP FAULT at iter %d INPUT_LN: %s\n", j,
                       cudaGetErrorString(rc)); cudaGetLastError(); break;
            }
            if (j < 6) printf("  widenloop[%d] pair ok\n", j);
        }
        printf("  WIDENLOOP done\n");
    }

    const char* names[] = {
        "model.language_model.norm.weight",
        "model.language_model.layers.0.input_layernorm.weight",
        "model.language_model.layers.1.input_layernorm.weight",
        "model.language_model.layers.5.input_layernorm.weight",
        "model.language_model.layers.47.input_layernorm.weight",
        "model.language_model.layers.0.post_attention_layernorm.weight",
        "model.language_model.layers.0.self_attn.q_norm.weight",
    };
    for (const char* nm : names) {
        const auto* t = model.find(nm);
        if (t == nullptr || t->devPtr == nullptr) {
            printf("  [skip] %s (missing/null)\n", nm);
            continue;
        }
        widen(nm, t->devPtr, t->nbytes / 2);
    }

    // ================= PART C: replicate the real executeMaterialization
    // prefix — the two ~1.875 GiB Bf16Copy (token_embd / output) via
    // appendMemoryCopy, KEPT alive, then a norm widen. This is the one thing
    // PART B did NOT do.
    if (!bisect) {
    printf("\n=== PART C: 2x 2GiB Bf16Copy (kept) then norm widen ===\n");
    {
        const auto* embed  = model.find("model.language_model.embed_tokens.weight");
        const auto* lmhead = model.find("lm_head.weight");
        const auto* iln    = model.find(
            "model.language_model.layers.0.input_layernorm.weight");
        if (embed && lmhead && iln) {
            ComputeBuffer te = ops.allocate(embed->nbytes);
            devOps.copyBytes(te.get(), embed->devPtr, embed->nbytes);
            synced("C token_embd Bf16Copy 2GiB");
            ComputeBuffer outb = ops.allocate(lmhead->nbytes);
            devOps.copyBytes(outb.get(), lmhead->devPtr, lmhead->nbytes);
            synced("C output Bf16Copy 2GiB");
            widen("C input_layernorm after 2x 2GiB copy", iln->devPtr,
                  iln->nbytes / 2);

            // C2: same but NO sync between the copies + widen (batched, like the
            // real path), one final sync.
            printf("  -- C2: batched (no intermediate sync) --\n");
            ComputeBuffer te2 = ops.allocate(embed->nbytes);
            devOps.copyBytes(te2.get(), embed->devPtr, embed->nbytes);
            ComputeBuffer out2 = ops.allocate(lmhead->nbytes);
            devOps.copyBytes(out2.get(), lmhead->devPtr, lmhead->nbytes);
            ComputeBuffer wd = ops.allocate((iln->nbytes / 2) * 4);
            devOps.widenToF32(wd.get(), iln->devPtr, SafetensorsDtype::BF16,
                              iln->nbytes / 2);
            synced("C2 batched copies+widen, final sync");
        }
    }
    } // end if(!bisect) PART C

    if (!bisect) {
    // ================= PART D: NVFP4 dequant of real tensors (incl. the
    // unusual full-attention shapes: q_proj 8192 rows, k_proj 512 rows/1 KV).
    // KEEP every dst (match executeMaterialization's accumulation). Sync after
    // each; the first FAULT names the corrupting dequant.
    printf("\n=== PART D: NVFP4 dequant (kept dsts) ===\n");
    std::vector<ComputeBuffer> keep;  // accumulate like the real materialiser
    auto dequantOne = [&](const std::string& base) {
        const auto* pk = model.find(base + ".weight_packed");
        const auto* bs = model.find(base + ".weight_scale");
        const auto* gs = model.find(base + ".weight_global_scale");
        if (!pk || !bs || !gs || pk->shape.size() != 2) {
            printf("  [skip] %s\n", base.c_str());
            return;
        }
        const std::uint64_t rows = pk->shape[0];
        const std::uint64_t in   = pk->shape[1] * 2;
        const float gRaw = devOps.readF32(gs->devPtr);
        const float g    = 1.0F / gRaw;   // compressed-tensors reciprocal
        ComputeBuffer dst = ops.allocate(rows * in * 2);
        devOps.dequantNvfp4(pk->devPtr, bs->devPtr, g, rows, in, dst.get());
        char msg[256];
        std::snprintf(msg, sizeof(msg), "dequant %s (rows=%llu in=%llu gRaw=%.1f)",
                      base.c_str(), (unsigned long long)rows,
                      (unsigned long long)in, gRaw);
        synced(msg);
        keep.push_back(std::move(dst));
    };
    const char* L0 = "model.language_model.layers.0.";
    const char* L5 = "model.language_model.layers.5."; // full-attention layer
    dequantOne(std::string(L0) + "self_attn.q_proj");
    dequantOne(std::string(L0) + "self_attn.k_proj");
    dequantOne(std::string(L0) + "self_attn.v_proj");
    dequantOne(std::string(L0) + "self_attn.o_proj");
    dequantOne(std::string(L0) + "mlp.gate_proj");
    dequantOne(std::string(L0) + "mlp.up_proj");
    dequantOne(std::string(L0) + "mlp.down_proj");
    dequantOne(std::string(L5) + "self_attn.q_proj");   // 8192 rows
    dequantOne(std::string(L5) + "self_attn.k_proj");   // 512 rows / 1 KV
    dequantOne(std::string(L5) + "self_attn.o_proj");   // in=8192
    // Then a norm widen AFTER the dequants (does a dequant corrupt it?).
    {
        const auto* iln = model.find(
            "model.language_model.layers.0.input_layernorm.weight");
        if (iln) widen("D input_layernorm after dequants", iln->devPtr,
                       iln->nbytes / 2);
    }
    } // end if(!bisect) PART D

    // ================= PART E: the REAL executeMaterialization, and a
    // prefix bisection to localise the faulting step in this fast harness.
    printf("\n=== PART E: real plan + executeMaterialization prefix bisection ===\n");
    {
        std::ifstream f(dir + "/config.json");
        std::stringstream ss; ss << f.rdbuf();
        auto cfg = mimirmind::runtime::nvfp4::parseGemma4SafetensorsConfig(ss.str());
        mimirmind::core::safetensors::SafetensorsModel sm;
        sm.open(dir);
        mimirmind::core::modelopt::Gemma4Arch arch;
        arch.numLayers = static_cast<int>(cfg.blockCount);
        arch.isFullAttn.assign(cfg.blockCount, false);
        for (std::size_t L = 0; L < cfg.blockCount; ++L) {
            const bool sliding = (L < cfg.slidingWindowPattern.size())
                                     ? static_cast<bool>(cfg.slidingWindowPattern[L])
                                     : true;
            arch.isFullAttn[L] = !sliding;
        }
        const auto steps =
            mimirmind::core::modelopt::planGemma4Materialization(sm, arch);
        printf("  plan has %zu steps\n", steps.size());

        // REPRO_CLOSESM: release the planning SafetensorsModel's ~10 GB of shard
        // mmaps BEFORE running the materialisation kernels. PART B/D run kernels
        // fine and never open a second mmap; PART E does. If closing it clears the
        // fault, the trigger is the checkpoint mmap coexisting with CUDA unified
        // addressing on GB10 — and the production fix is to close the safetensors
        // handle before executeMaterialization (or plan from already-loaded meta).
        if (std::getenv("REPRO_CLOSESM") != nullptr) {
            sm.close();
            printf("  [CLOSESM] released safetensors mmaps before materialise\n");
        }

        // Prefix bisection: run executeMaterialization on the first `k` steps,
        // syncing after. The smallest k that faults pins the culprit (and shows
        // whether it needs accumulation up to k).
        auto tryPrefix = [&](std::size_t k) -> bool {
            std::vector<mimirmind::core::modelopt::MaterializationStep> pref(
                steps.begin(), steps.begin() + k);
            try {
                auto mats = mimirmind::runtime::nvfp4::executeMaterialization(
                    pref, model, devOps);
                const cudaError_t rc = cudaDeviceSynchronize();
                if (rc != cudaSuccess) {
                    printf("  prefix[0:%zu] FAULT: %s (last step '%s')\n", k,
                           cudaGetErrorString(rc), steps[k - 1].ggufName.c_str());
                    cudaGetLastError();
                    return false;
                }
                printf("  prefix[0:%zu] ok (last '%s')\n", k,
                       steps[k - 1].ggufName.c_str());
                return true;
            } catch (const std::exception& e) {
                printf("  prefix[0:%zu] THREW: %s\n", k, e.what());
                return false;
            }
        };
        // Linear scan in coarse chunks first to find the faulting region, then
        // narrow. (A fresh context per prefix would be ideal but the fault
        // poisons the context; so run smallest→largest and stop at the first
        // fault — each prefix is independent allocations, freed on return.)
        // Single-prefix mode (REPRO_PREFIX=k): run exactly ONE prefix in this
        // fresh process — the clean way to bisect (each fault poisons the
        // context, so multiple prefixes in one process are unreliable).
        // Stepwise executor (REPRO_STEPWISE=k): faithfully replays the real
        // executeMaterialization ops in order, but cudaDeviceSynchronize()s AFTER
        // EACH step, keeping every buffer. This is the race-vs-deterministic
        // discriminator: if per-step sync makes the fault vanish (all steps ok
        // where prefix[0:k] faulted) -> a no-sync / Managed-migration ordering
        // issue (fix = a strategic flush/prefetch). If it still faults, it names
        // the exact deterministic step. Also proves whether a per-step flush is a
        // viable production fix.
        namespace mo2 = mimirmind::core::modelopt;
        auto stepwise = [&](std::size_t k) {
            std::vector<ComputeBuffer> hold;
            hold.reserve(k);
            std::vector<void*> devKeep;  // REPRO_DEVDST: raw cudaMalloc big dsts
            const bool devdst = std::getenv("REPRO_DEVDST") != nullptr;
            for (std::size_t i = 0; i < k && i < steps.size(); ++i) {
                const mo2::MaterializationStep& step = steps[i];
                const std::size_t elemBytes = step.outF32 ? 4 : 2;
                const std::size_t outBytes =
                    static_cast<std::size_t>(step.totalElems) * elemBytes;
                // REPRO_NOTRACE: disable ALL per-step instrumentation (memGetInfo
                // + src-probe readF32 + prints) — isolates whether the harness
                // trace (esp. the readF32 D2H on the default stream right before
                // the widen), which WIDENLOOP lacks, triggers the fault.
                const bool trace =
                    (i < 6) && std::getenv("REPRO_NOTRACE") == nullptr;
                // REPRO_DEVDST: allocate the two big Bf16Copy dsts as DEVICE
                // memory (cudaMalloc) instead of Managed — removing them from the
                // managed working set. If step 3 then survives, the trigger is the
                // GPU-mappable Managed cap, and the production fix is device (or
                // pinned) allocation for the large matmul weights.
                const bool bigCopy = step.sources.size() == 1
                    && step.sources[0].kind == mo2::SourceKind::Bf16Copy;
                // REPRO_SKIPBIG: skip the two big Bf16Copy steps entirely. If the
                // remaining 665 steps then materialise clean, the two ~1.875 GiB
                // allocations are the sole trigger and the fix is to handle them
                // separately (materialise last / device-allocate / stream-free).
                if (bigCopy && std::getenv("REPRO_SKIPBIG") != nullptr) {
                    if (i < 6) printf("  [SKIPBIG] skipped step %zu '%s'\n", i,
                                      step.ggufName.c_str());
                    continue;
                }
                const bool alldev = std::getenv("REPRO_ALLDEV") != nullptr;
                ComputeBuffer buf;
                std::uint8_t* dstBase = nullptr;
                if ((devdst && bigCopy) || alldev) {
                    void* dp = nullptr;
                    const cudaError_t mrc = cudaMalloc(&dp, outBytes);
                    if (trace) {
                        printf("  [DEVDST] cudaMalloc %zu -> %p (%s)\n", outBytes, dp,
                               cudaGetErrorString(mrc));
                    }
                    dstBase = static_cast<std::uint8_t*>(dp);
                    devKeep.push_back(dp);
                } else {
                    buf = ops.allocate(outBytes);
                    dstBase = static_cast<std::uint8_t*>(buf.get());
                }
                for (const mo2::MaterializationSource& s : step.sources) {
                    void* dst = dstBase + s.dstElemOffset * elemBytes;
                    const auto* w = model.find(s.hfWeightName);
                    if (w == nullptr) {
                        printf("  [MISS] step %zu src '%s'\n", i,
                               s.hfWeightName.c_str());
                        continue;
                    }
                    if (trace) {
                        std::size_t mfree = 0, mtot = 0;
                        cudaMemGetInfo(&mfree, &mtot);
                        cudaGetLastError();
                        printf("  step %zu kind=%d src=%p dst=%p out=%zu n=%llu "
                               "free=%zuMiB tot=%zuMiB '%s'\n",
                               i, static_cast<int>(s.kind), w->devPtr, (void*)dstBase,
                               outBytes, (unsigned long long)(s.rows * s.in),
                               mfree >> 20, mtot >> 20,
                               s.hfWeightName.c_str());
                    }
                    if (s.kind == mo2::SourceKind::Nvfp4) {
                        const auto* bs = model.find(s.blockScaleName);
                        const auto* gs = model.find(s.globalScaleName);
                        float g = devOps.readF32(gs->devPtr);
                        if (s.globalIsReciprocal) g = 1.0F / g;
                        devOps.dequantNvfp4(w->devPtr, bs->devPtr, g, s.rows, s.in, dst);
                    } else if (s.kind == mo2::SourceKind::Bf16Copy) {
                        const auto* sb = static_cast<const std::uint8_t*>(w->devPtr)
                                         + s.srcElemOffset * 2;
                        // REPRO_NOCOPY: allocate the big dst but SKIP the D2D copy,
                        // to separate "the 1.875 GiB Managed alloc" from "the big
                        // Managed D2D copy" as the fault trigger.
                        if (std::getenv("REPRO_NOCOPY") == nullptr) {
                            devOps.copyBytes(dst, sb, s.rows * s.in * 2);
                        } else if (trace) {
                            printf("    [NOCOPY] skipped D2D copy of %zu bytes\n",
                                   (std::size_t)(s.rows * s.in * 2));
                        }
                    } else { // Bf16Passthrough
                        // REPRO_PREFETCH: pull the source pages device-resident
                        // right before the kernel. If this clears the fault, the
                        // root cause is fault-driven Managed eviction of model
                        // pages under big-allocation pressure, and the production
                        // fix is a device prefetch / SetPreferredLocation on load.
                        if (std::getenv("REPRO_PREFETCH") != nullptr) {
                            const std::size_t pfBytes =
                                static_cast<std::size_t>(s.rows * s.in) * 2;
                            cudaMemLocation loc{};
                            loc.type = cudaMemLocationTypeDevice;
                            loc.id   = 0;
                            const cudaError_t prc =
                                cudaMemPrefetchAsync(w->devPtr, pfBytes, loc, 0, nullptr);
                            const cudaError_t psc = cudaDeviceSynchronize();
                            if (trace) {
                                printf("    [PREFETCH] async=%s sync=%s\n",
                                       cudaGetErrorString(prc), cudaGetErrorString(psc));
                            }
                            cudaGetLastError();
                        }
                        // REPRO_ADVISE: establish a persistent GPU mapping for the
                        // source (SetAccessedBy device) so the kernel reads it over
                        // the coherent link WITHOUT a fault-in migration.
                        if (std::getenv("REPRO_ADVISE") != nullptr) {
                            const std::size_t adBytes =
                                static_cast<std::size_t>(s.rows * s.in) * 2;
                            cudaMemLocation loc{};
                            loc.type = cudaMemLocationTypeDevice;
                            loc.id   = 0;
                            const cudaError_t arc = cudaMemAdvise(
                                w->devPtr, adBytes, cudaMemAdviseSetAccessedBy, loc);
                            if (trace) {
                                printf("    [ADVISE] SetAccessedBy=%s\n",
                                       cudaGetErrorString(arc));
                            }
                            cudaGetLastError();
                        }
                        // REPRO_DSTZERO: populate the fresh dst page device-side
                        // before the kernel writes it. If this clears the fault,
                        // the illegal access is the kernel's WRITE to a fresh
                        // Managed dst under big-allocation pressure, not the read.
                        if (std::getenv("REPRO_DSTZERO") != nullptr) {
                            const cudaError_t zrc =
                                cudaMemset(dst, 0, static_cast<std::size_t>(
                                    s.rows * s.in) * elemBytes);
                            const cudaError_t zsc = cudaDeviceSynchronize();
                            if (trace) {
                                printf("    [DSTZERO] memset=%s sync=%s\n",
                                       cudaGetErrorString(zrc), cudaGetErrorString(zsc));
                            }
                            cudaGetLastError();
                        }
                        // REPRO_HOSTTOUCH: fault the fresh dst pages in FROM THE
                        // HOST (a CPU memset on the Managed pointer) before the GPU
                        // kernel writes them. Host access always works in these
                        // repros; if this clears the fault, the production fix is to
                        // host-touch (or device-allocate) materialisation outputs.
                        if (std::getenv("REPRO_HOSTTOUCH") != nullptr) {
                            std::memset(dstBase, 0, outBytes);
                            if (trace) printf("    [HOSTTOUCH] CPU memset %zu bytes\n",
                                              outBytes);
                        }
                        // REPRO_SRCDEV: make the SOURCE pure device memory — D2D
                        // copy the Managed model tensor into a fresh cudaMalloc
                        // device buffer, then widen FROM that. Combined with
                        // REPRO_ALLDEV (device dst), NEITHER kernel operand is
                        // Managed. If step 3 then survives, the fault is the GPU
                        // kernel touching Managed pages, and the fix is to
                        // materialise from/to device memory on GB10.
                        const void* wsrc = w->devPtr;
                        if (std::getenv("REPRO_SRCDEV") != nullptr) {
                            const std::size_t sb2 =
                                static_cast<std::size_t>(s.rows * s.in) * 2;
                            void* dsrc = nullptr;
                            cudaMalloc(&dsrc, sb2);
                            cudaMemcpy(dsrc, w->devPtr, sb2, cudaMemcpyDefault);
                            cudaDeviceSynchronize();
                            if (trace) printf("    [SRCDEV] copied %zu B to device %p (%s)\n",
                                              sb2, dsrc, cudaGetErrorString(cudaGetLastError()));
                            wsrc = dsrc;
                            devKeep.push_back(dsrc);
                        }
                        // Probe the SOURCE pointer first (a tiny D2H read). If THIS
                        // faults, the model tensor's devPtr itself is unreadable at
                        // this point (eviction / aliasing) — not a kernel-arg bug.
                        if (trace) {
                            try {
                                const float probe = devOps.readF32(w->devPtr);
                                const cudaError_t prc = cudaDeviceSynchronize();
                                printf("    src-probe readF32=%.4g rc=%s\n", probe,
                                       cudaGetErrorString(prc));
                                cudaGetLastError();
                            } catch (const std::exception& e) {
                                printf("    src-probe THREW: %s\n", e.what());
                                cudaGetLastError();
                            }
                        }
                        devOps.widenToF32(dst, wsrc, w->dtype, s.rows * s.in);
                    }
                }
                // Sync after the fill (before the post-transform) to attribute the
                // fault to the fill vs the post-transform kernel.
                {
                    const cudaError_t rcf = cudaDeviceSynchronize();
                    if (rcf != cudaSuccess) {
                        printf("  STEPWISE FAULT at step %zu FILL ('%s' kind=%d): %s\n",
                               i, step.ggufName.c_str(),
                               static_cast<int>(step.sources[0].kind),
                               cudaGetErrorString(rcf));
                        cudaGetLastError();
                        return;
                    }
                }
                // REPRO_NOADDONE: skip the in-place post-transform. If step 3 then
                // survives, the widen+addOne (in-place RMW on Managed) sequence is
                // the trigger.
                if (std::getenv("REPRO_NOADDONE") == nullptr) {
                    if (step.postTransform == mo2::PostTransform::AddOne) {
                        devOps.addOneInPlaceF32(dstBase, step.totalElems);
                    } else if (step.postTransform == mo2::PostTransform::NegExp) {
                        devOps.negExpInPlaceF32(dstBase, step.totalElems);
                    }
                }
                // REPRO_NOKEEP: free each output immediately (do not accumulate).
                // If step 3 then survives, the trigger is holding the prior fresh
                // Managed buffer alive while allocating+touching the next.
                if (std::getenv("REPRO_NOKEEP") == nullptr) {
                    hold.push_back(std::move(buf));
                }
                const cudaError_t rc = cudaDeviceSynchronize();
                if (rc != cudaSuccess) {
                    printf("  STEPWISE FAULT at step %zu POST ('%s' kind=%d): %s\n", i,
                           step.ggufName.c_str(),
                           static_cast<int>(step.sources[0].kind),
                           cudaGetErrorString(rc));
                    cudaGetLastError();
                    return;
                }
                // REPRO_STATS: read back the materialised buffer and print value
                // stats (min/max/absmean/nan) — a norm should be ~1.0 after AddOne
                // (Gemma stores ~0-centred), a projection ~O(0.01-0.1). Catches a
                // wrong global scale / missing AddOne / bad layout without a GGUF
                // oracle.
                if (std::getenv("REPRO_STATS") != nullptr && i < 22) {
                    const std::size_t n = static_cast<std::size_t>(step.totalElems);
                    double mn = 1e30, mx = -1e30, sabs = 0.0; std::size_t nan = 0;
                    if (step.outF32) {
                        std::vector<float> h(n);
                        ops.readbackToHost(h.data(), dstBase, n * 4);
                        for (float v : h) {
                            if (std::isnan(v) || std::isinf(v)) { ++nan; continue; }
                            mn = std::min(mn, (double)v); mx = std::max(mx, (double)v);
                            sabs += std::fabs((double)v);
                        }
                    } else {
                        std::vector<std::uint16_t> h(n);
                        ops.readbackToHost(h.data(), dstBase, n * 2);
                        for (std::uint16_t bits : h) {
                            std::uint32_t u = (std::uint32_t)bits << 16; float v;
                            std::memcpy(&v, &u, 4);
                            if (std::isnan(v) || std::isinf(v)) { ++nan; continue; }
                            mn = std::min(mn, (double)v); mx = std::max(mx, (double)v);
                            sabs += std::fabs((double)v);
                        }
                    }
                    printf("  STATS[%zu] '%s' %s min=%.4g max=%.4g absmean=%.4g nan=%zu/%zu\n",
                           i, step.ggufName.c_str(), step.outF32 ? "F32" : "BF16",
                           mn, mx, sabs / (double)(n ? n : 1), nan, n);
                }
                if (i < 20 || (i % 64) == 0) {
                    printf("  stepwise[%zu] ok ('%s')\n", i, step.ggufName.c_str());
                }
            }
            printf("  STEPWISE all %zu steps ok (per-step sync)\n", k);
        };

        // REPRO_WARMUP: mimic PART A — a big Managed alloc + full memset + free —
        // right before materialising. Non-bisect runs (which run PART A) never
        // fault; bisect runs (PART A skipped) do. If this warm-up clears the
        // fault, the trigger is a COLD managed/allocator state that a large
        // touched-then-freed allocation initialises, and the production fix is a
        // one-time warm-up before executeMaterialization.
        if (std::getenv("REPRO_WARMUP") != nullptr) {
            ComputeBuffer w1 = ops.allocate(kEmbedBytes);
            cudaMemset(w1.get(), 0, kEmbedBytes);
            cudaDeviceSynchronize();
            printf("  [WARMUP] 1.875 GiB alloc+memset+free done (rc=%s)\n",
                   cudaGetErrorString(cudaGetLastError()));
        }

        // REPRO_E_WIDENLOOP: run the EXACT clean WIDENLOOP pair loop, but HERE in
        // PART E (same position as stepwise, after plan-build / sm.open), via
        // model.find. If this faults, the PART E context is the poison, not the
        // stepwise loop body. If clean, the stepwise body itself is the trigger.
        if (std::getenv("REPRO_E_WIDENLOOP") != nullptr) {
            using Dt = mimirmind::core::safetensors::SafetensorsDtype;
            const auto* nrm = model.find("model.language_model.norm.weight");
            const auto* iln =
                model.find("model.language_model.layers.0.input_layernorm.weight");
            printf("  E_WIDENLOOP pair(norm=%p, input_ln=%p)\n",
                   nrm ? nrm->devPtr : nullptr, iln ? iln->devPtr : nullptr);
            bool efault = false;
            for (int j = 0; j < 10 && nrm && iln && !efault; ++j) {
                ComputeBuffer dn = ops.allocate(3840 * 4);
                devOps.widenToF32(dn.get(), nrm->devPtr, nrm->dtype, 3840);
                devOps.addOneInPlaceF32(dn.get(), 3840);
                cudaError_t rc = cudaDeviceSynchronize();
                if (rc != cudaSuccess) { printf("  E_WIDENLOOP FAULT iter %d NORM: %s\n",
                    j, cudaGetErrorString(rc)); cudaGetLastError(); efault = true; break; }
                ComputeBuffer di = ops.allocate(3840 * 4);
                devOps.widenToF32(di.get(), iln->devPtr, iln->dtype, 3840);
                devOps.addOneInPlaceF32(di.get(), 3840);
                rc = cudaDeviceSynchronize();
                if (rc != cudaSuccess) { printf("  E_WIDENLOOP FAULT iter %d INPUT_LN: %s\n",
                    j, cudaGetErrorString(rc)); cudaGetLastError(); efault = true; break; }
                printf("  e_widenloop[%d] pair ok\n", j);
            }
            if (!efault) printf("  E_WIDENLOOP all ok (PART E context is NOT the poison)\n");
        }

        if (const char* sw = std::getenv("REPRO_STEPWISE")) {
            std::size_t k = static_cast<std::size_t>(std::atoi(sw));
            if (k == 0 || k > steps.size()) k = steps.size();
            printf("  REPRO_STEPWISE=%zu\n", k);
            stepwise(k);
        } else if (const char* pk = std::getenv("REPRO_PREFIX")) {
            const std::size_t k = static_cast<std::size_t>(std::atoi(pk));
            printf("  REPRO_PREFIX=%zu\n", k);
            tryPrefix(k);
        } else if (std::getenv("REPRO_E") != nullptr) {
            for (std::size_t k = 1; k <= steps.size(); k += 1) {
                if (!tryPrefix(k)) {
                    printf("  --> FIRST FAULTING prefix ends at step %zu ('%s')\n",
                           k, steps[k - 1].ggufName.c_str());
                    break;
                }
            }
        } else {
            printf("  (PART E bisection skipped — set REPRO_E=1 or REPRO_PREFIX=k)\n");
        }
    }

    // ================= PART F: minimise the [0:4] trigger =================
    if (!bisect) {
    printf("\n=== PART F: minimise the faulting combination ===\n");
    {
        const auto* nrm = model.find("model.language_model.norm.weight");
        const auto* iln = model.find(
            "model.language_model.layers.0.input_layernorm.weight");
        const auto* emb = model.find("model.language_model.embed_tokens.weight");
        const auto* lmh = model.find("lm_head.weight");
        auto widenAddOneKeep = [&](std::vector<ComputeBuffer>& hold, const char* lbl,
                                   const void* src, std::size_t n, bool addOne,
                                   bool syncNow) {
            ComputeBuffer d = ops.allocate(n * 4);
            devOps.widenToF32(d.get(), src, SafetensorsDtype::BF16, n);
            if (addOne) devOps.addOneInPlaceF32(d.get(), n);
            if (syncNow) synced(lbl);
            hold.push_back(std::move(d));
        };

        // F1: two widen+addOne (norm, input_layernorm), dsts KEPT, no sync between.
        { std::vector<ComputeBuffer> h;
          widenAddOneKeep(h, "F1 a", nrm->devPtr, 3840, true, false);
          widenAddOneKeep(h, "F1 b", iln->devPtr, 3840, true, false);
          synced("F1 two widen+addOne kept, no intermediate sync"); }

        // F2: same but sync after each.
        { std::vector<ComputeBuffer> h;
          widenAddOneKeep(h, "F2 a", nrm->devPtr, 3840, true, true);
          widenAddOneKeep(h, "F2 b", iln->devPtr, 3840, true, true);
          synced("F2 two widen+addOne kept, sync each"); }

        // F3: exact [0:4] by hand — copy(emb), widen+addOne(norm), copy(lmh),
        // widen+addOne(iln), all KEPT, no intermediate sync.
        if (emb && lmh) { std::vector<ComputeBuffer> h;
          { ComputeBuffer d = ops.allocate(emb->nbytes);
            devOps.copyBytes(d.get(), emb->devPtr, emb->nbytes); h.push_back(std::move(d)); }
          widenAddOneKeep(h, "F3 norm", nrm->devPtr, 3840, true, false);
          { ComputeBuffer d = ops.allocate(lmh->nbytes);
            devOps.copyBytes(d.get(), lmh->devPtr, lmh->nbytes); h.push_back(std::move(d)); }
          widenAddOneKeep(h, "F3 iln", iln->devPtr, 3840, true, false);
          synced("F3 exact [0:4] by hand"); }

        // F4: F1 but WITHOUT addOne (just two widens).
        { std::vector<ComputeBuffer> h;
          widenAddOneKeep(h, "F4 a", nrm->devPtr, 3840, false, false);
          widenAddOneKeep(h, "F4 b", iln->devPtr, 3840, false, false);
          synced("F4 two widens NO addOne kept, no sync"); }
    }
    } // end if(!bisect) PART F

    if (!bisect) {
    // ================= PART G: the readF32-during-in-flight-copy hazard.
    // executeMaterialization does NOT sync between steps; its first sync-like op
    // is readF32 (cudaMemcpy on the LEGACY DEFAULT stream), which does NOT wait
    // on the non-blocking _ctx.stream() where the 2 GiB D2D copies are still in
    // flight. Hypothesis: that concurrency faults on GB10 managed memory.
    printf("\n=== PART G: readF32 with vs without a preceding sync ===\n");
    {
        const auto* nrm = model.find("model.language_model.norm.weight");
        const auto* iln = model.find(
            "model.language_model.layers.0.input_layernorm.weight");
        const auto* emb = model.find("model.language_model.embed_tokens.weight");
        const auto* lmh = model.find("lm_head.weight");
        const auto* gs  = model.find(
            "model.language_model.layers.0.self_attn.q_proj.weight_global_scale");
        auto batched4 = [&](std::vector<ComputeBuffer>& h) {
            { ComputeBuffer d = ops.allocate(emb->nbytes);
              devOps.copyBytes(d.get(), emb->devPtr, emb->nbytes); h.push_back(std::move(d)); }
            { ComputeBuffer d = ops.allocate(3840 * 4);
              devOps.widenToF32(d.get(), nrm->devPtr, SafetensorsDtype::BF16, 3840);
              devOps.addOneInPlaceF32(d.get(), 3840); h.push_back(std::move(d)); }
            { ComputeBuffer d = ops.allocate(lmh->nbytes);
              devOps.copyBytes(d.get(), lmh->devPtr, lmh->nbytes); h.push_back(std::move(d)); }
            { ComputeBuffer d = ops.allocate(3840 * 4);
              devOps.widenToF32(d.get(), iln->devPtr, SafetensorsDtype::BF16, 3840);
              devOps.addOneInPlaceF32(d.get(), 3840); h.push_back(std::move(d)); }
        };
        // G_ok: 4 batched ops, cudaDeviceSynchronize, THEN readF32.
        try {
            std::vector<ComputeBuffer> h; batched4(h);
            cudaDeviceSynchronize();
            const float g = devOps.readF32(gs->devPtr);
            printf("  [ ok  ] G_ok readF32-after-sync = %.1f\n", g);
        } catch (const std::exception& e) {
            printf("  [FAULT] G_ok: %s\n", e.what());
        }
        cudaGetLastError();
        // G_bad: 4 batched ops, NO sync, readF32 immediately (copies in flight).
        try {
            std::vector<ComputeBuffer> h; batched4(h);
            const float g = devOps.readF32(gs->devPtr);
            printf("  [ ok  ] G_bad readF32-no-sync = %.1f (NO fault)\n", g);
        } catch (const std::exception& e) {
            printf("  [FAULT] G_bad readF32 while copies in flight: %s\n", e.what());
        }
    }
    } // end if(!bisect) PART G

    // ================= PART H: the one combination PART C/G did NOT cover —
    // a LARGE dequant kernel writing a FRESH Managed dst while the two ~1.9 GiB
    // D2D copies are still IN FLIGHT on the non-blocking stream. PART C only did
    // a tiny 3840-elem widen after the copies; PART G only did a tiny readF32.
    // On GB10 a big fresh-Managed-buffer kernel-write concurrent with in-flight
    // 2 GiB Managed D2D copies is untested — and is exactly what step 4
    // (q_proj dequant, ~62 MB) does right after steps 0/2 in the real plan.
    // Run ONE variant per fresh process (an illegal-address sticky error is not
    // clearable — it poisons the context, so H1 and H2 in the same process would
    // both fault). REPRO_H=1 -> no-sync (the suspect); REPRO_H=2 -> flush() fix.
    printf("\n=== PART H: big dequant to fresh Managed dst, copies in flight ===\n");
    if (const char* hv = std::getenv("REPRO_H")) {
        const int variant = std::atoi(hv);
        const auto* embed  = model.find("model.language_model.embed_tokens.weight");
        const auto* lmhead = model.find("lm_head.weight");
        const auto* pk = model.find("model.language_model.layers.0.self_attn.q_proj.weight_packed");
        const auto* bs = model.find("model.language_model.layers.0.self_attn.q_proj.weight_scale");
        const auto* gs = model.find("model.language_model.layers.0.self_attn.q_proj.weight_global_scale");
        if (embed && lmhead && pk && bs && gs && pk->shape.size() == 2) {
            const std::uint64_t rows = pk->shape[0];
            const std::uint64_t in   = pk->shape[1] * 2;

            // Queue both 2 GiB D2D copies on the non-blocking stream, then (for
            // the fix variant) flush, then read the global scale and launch the
            // big q_proj dequant to a FRESH Managed dst. Mirrors the real
            // executeMaterialization step 0 -> step 2 -> step 4 ordering.
            std::vector<ComputeBuffer> h;
            { ComputeBuffer d = ops.allocate(embed->nbytes);
              devOps.copyBytes(d.get(), embed->devPtr, embed->nbytes); h.push_back(std::move(d)); }
            { ComputeBuffer d = ops.allocate(lmhead->nbytes);
              devOps.copyBytes(d.get(), lmhead->devPtr, lmhead->nbytes); h.push_back(std::move(d)); }
            if (variant == 2) {
                ops.flush();   // <- candidate fix: drain the big copies first
                printf("  REPRO_H=2 (flush-fix): flushed after copies\n");
            } else {
                printf("  REPRO_H=1 (suspect): NO sync between copies and dequant\n");
            }
            const float gRaw = devOps.readF32(gs->devPtr);  // first sync point
            ComputeBuffer qd = ops.allocate(rows * in * 2);
            devOps.dequantNvfp4(pk->devPtr, bs->devPtr, 1.0F / gRaw, rows, in, qd.get());
            const bool ok = synced("H big dequant (rows*in fresh Managed dst)");
            printf("  H VERDICT (variant %d): %s\n", variant, ok ? "ok" : "FAULT");
        } else {
            printf("  [skip] PART H (missing tensors)\n");
        }
    } else {
        printf("  (PART H skipped — set REPRO_H=1 [suspect] or REPRO_H=2 [flush fix])\n");
    }

    printf("\nrepro done.\n");
    return 0;
}
