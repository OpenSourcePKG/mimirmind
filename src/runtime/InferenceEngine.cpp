// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/InferenceEngine.hpp"

#include "compute/Embedding.hpp"
#include "core/backend/BackendRegistry.hpp"
#include "core/gguf/GgufTypes.hpp"
#include "core/gguf/TensorFingerprint.hpp"
#include "core/config/Config.hpp"
#include "runtime/FanController.hpp"
#include "runtime/Lcp.hpp"
#include "core/log/Log.hpp"
#include "runtime/GpuClockGovernor.hpp"
#include "runtime/PerfRegressionDetector.hpp"
#include "runtime/PowerMonitor.hpp"
#include "runtime/SystemMonitor.hpp"
#include "runtime/ThermalGuard.hpp"
#include "runtime/arch/ArchBackend.hpp"

#ifdef MIMIRMIND_HAVE_HIP
#include "compute/hip/GpuMatmul.hpp"
#include "compute/hip/GpuOps.hpp"
#include "core/gpu/hip/HipComputeContext.hpp"
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace mimirmind::runtime {

namespace {

/// One-shot diagnostic: log the suffix names + quant types of one
/// transformer block. Truth for the architecture handler.
void logBlockTensorInventory(const core::gguf::GgufReader& reader,
                             std::size_t              blockIdx) {
    const std::string prefix = "blk." + std::to_string(blockIdx) + ".";
    std::size_t hits = 0;
    for (const auto& t : reader.tensors()) {
        if (t.name.compare(0, prefix.size(), prefix) != 0) {
            continue;
        }
        ++hits;
        std::string dims;
        for (std::size_t i = 0; i < t.dimensions.size(); ++i) {
            if (i > 0) {
                dims += ',';
            }
            dims += std::to_string(t.dimensions[i]);
        }
        MM_LOG_INFO("inventory",
                    "  {} type={} dims=[{}] bytes={}",
                    t.name, core::gguf::typeInfo(t.type).name,
                    dims, t.nbytes);
    }
    MM_LOG_INFO("inventory", "block {} has {} tensor(s)", blockIdx, hits);
}

/// Dump every top-level (non-blk.*) tensor plus a canonical set of
/// per-block suffix names sorted by frequency. Answers "does this GGUF
/// carry the AltUp / Laurel / PLE side tensors we need for Gemma 4 E-Series"
/// without swamping the log with 720 lines. Cheap enough to leave in
/// for any architecture — non-E-series models just show their own set.
void logTensorTaxonomy(const core::gguf::GgufReader& reader) {
    std::size_t totalCount = 0;
    std::size_t topLevelCount = 0;
    std::map<std::string, std::size_t> suffixCount;
    for (const auto& t : reader.tensors()) {
        ++totalCount;
        if (t.name.compare(0, 4, "blk.") != 0) {
            std::string dims;
            for (std::size_t i = 0; i < t.dimensions.size(); ++i) {
                if (i > 0) dims += ',';
                dims += std::to_string(t.dimensions[i]);
            }
            MM_LOG_INFO("taxonomy",
                        "top-level: {} type={} dims=[{}] bytes={}",
                        t.name, core::gguf::typeInfo(t.type).name, dims, t.nbytes);
            ++topLevelCount;
            continue;
        }
        // blk.<N>.<suffix>  →  strip the block index, keep the suffix.
        const auto dot1 = t.name.find('.', 4);
        if (dot1 == std::string::npos) continue;
        const auto suffix = t.name.substr(dot1 + 1);
        ++suffixCount[suffix];
    }
    MM_LOG_INFO("taxonomy",
                "== {} tensors total, {} top-level, {} distinct per-block suffixes ==",
                totalCount, topLevelCount, suffixCount.size());
    for (const auto& [suffix, count] : suffixCount) {
        MM_LOG_INFO("taxonomy", "per-block suffix: {} × {}", suffix, count);
    }
}

// Schicht 4 — factory helpers. `_computeCtx` is picked via
// BackendRegistry::autoSelect (respects MIMIRMIND_BACKEND env var; falls
// back to LevelZero, which keeps existing setups untouched). `_ops` and
// `_gmm` factories switch on the resulting kind and forward to the
// matching concrete ctor. Compile-time guards drop each branch when the
// corresponding backend isn't built in — a config that asks for HIP on
// an L0-only build throws with a clear diagnostic.

std::unique_ptr<core::backend::ComputeContext>
makeComputeContext(const Config& cfg, core::backend::BackendKind kind) {
    switch (kind) {
        case core::backend::BackendKind::LevelZero:
#ifdef MIMIRMIND_HAVE_L0
            return std::make_unique<core::l0::L0ComputeContext>(
                core::l0::L0ComputeContext::Options{
                    .spvDirOverride    = std::string{cfg.runtime.spvDir.value_or("")},
                    .usmProbeTotalGiB  = cfg.runtime.usmProbeTotalGib,
                    .usmKindOverride   = std::nullopt});
#else
            throw std::runtime_error{
                "InferenceEngine: LevelZero requested but not compiled in "
                "(MIMIRMIND_ENABLE_L0=OFF at build time)"};
#endif

        case core::backend::BackendKind::Hip:
#ifdef MIMIRMIND_HAVE_HIP
            (void)cfg;  // HIP path takes no config knobs yet.
            return std::make_unique<core::hip::HipComputeContext>();
#else
            throw std::runtime_error{
                "InferenceEngine: HIP requested but not compiled in "
                "(MIMIRMIND_ENABLE_HIP=OFF at build time)"};
#endif

        default:
            throw std::runtime_error{
                "InferenceEngine: no supported compute backend for kind "
                + std::to_string(static_cast<int>(kind))};
    }
}

std::unique_ptr<compute::ComputeOps>
makeGpuOps(core::backend::ComputeContext&        ctx,
           const core::config::FeatureSettings&  features) {
    switch (ctx.kind()) {
        case core::backend::BackendKind::LevelZero:
#ifdef MIMIRMIND_HAVE_L0
            return std::make_unique<compute::l0::GpuOps>(
                static_cast<core::l0::L0ComputeContext&>(ctx),
                features.flashPrefill,
                features.flashPrefillGqaQ8,
                features.flashPrefillKTileQ8,
                features.q8_0Reorder);
#else
            break;
#endif

        case core::backend::BackendKind::Hip:
#ifdef MIMIRMIND_HAVE_HIP
            return std::make_unique<compute::hip::GpuOps>(
                static_cast<core::hip::HipComputeContext&>(ctx),
                features.flashPrefill,
                features.flashPrefillGqaQ8,
                features.flashPrefillKTileQ8,
                features.q8_0Reorder);
#else
            break;
#endif

        default:
            break;
    }
    throw std::runtime_error{
        "InferenceEngine::makeGpuOps: no compute::*::GpuOps impl for the "
        "resolved backend kind"};
}

std::unique_ptr<compute::ComputeMatmul>
makeGpuMatmul(core::backend::ComputeContext& ctx,
              compute::ComputeOps&           ops) {
    switch (ctx.kind()) {
        case core::backend::BackendKind::LevelZero:
#ifdef MIMIRMIND_HAVE_L0
            return std::make_unique<compute::l0::GpuMatmul>(
                static_cast<core::l0::L0ComputeContext&>(ctx),
                static_cast<compute::l0::GpuOps&>(ops));
#else
            break;
#endif

        case core::backend::BackendKind::Hip:
#ifdef MIMIRMIND_HAVE_HIP
            return std::make_unique<compute::hip::GpuMatmul>(
                static_cast<core::hip::HipComputeContext&>(ctx),
                static_cast<compute::hip::GpuOps&>(ops));
#else
            break;
#endif

        default:
            break;
    }
    throw std::runtime_error{
        "InferenceEngine::makeGpuMatmul: no compute::*::GpuMatmul impl "
        "for the resolved backend kind"};
}

} // namespace

InferenceEngine::InferenceEngine(const Config& cfg)
    : _cfg{cfg},
      _computeCtx{makeComputeContext(
          cfg,
          core::backend::BackendRegistry::autoSelect(
              core::backend::BackendKind::LevelZero))},
      _ops{makeGpuOps(*_computeCtx, cfg.features)},
      _gmm{makeGpuMatmul(*_computeCtx, *_ops)},
      _opProfiler{} {
    const auto kind = _computeCtx->kind();
    MM_LOG_INFO("engine",
                "InferenceEngine: runtime bound to backend '{}'",
                core::backend::BackendRegistry::name(kind));

    // OpProfiler wiring — the L0 timing pipeline needs the L0
    // CommandQueue; on HIP we emplace a no-op instance (Schicht 5.3
    // default ctor) so arch-backend deref of `_opProfiler.value()`
    // stays valid regardless of backend. Every `mark/finish/dump`
    // call is behind an `if (!_enabled) return` early-return anyway.
    if (kind == core::backend::BackendKind::LevelZero) {
        _opProfiler.emplace(l0ComputeContext().queue(),
                            cfg.diagnostics.traceOpTimes);
        MM_LOG_INFO("engine", "InferenceEngine: probing USM limits");
        allocator().probeLimits();
    } else {
        _opProfiler.emplace();  // disabled no-op profiler for non-L0
    }

    if (cfg.diagnostics.traceBlock0) {
        _traceBlock0 = true;
        MM_LOG_INFO("engine",
                    "diagnostics.traceBlock0=true — block-0 trace enabled "
                    "for first forward only");
    }

    // Per-decode-token NDJSON telemetry sink. Opens the file truncating;
    // the engine never rotates so a long-running process keeps appending.
    // Operators should rotate externally if they care.
    if (!cfg.diagnostics.traceDecodeFile.empty()) {
        const std::string& path = cfg.diagnostics.traceDecodeFile;
        _decodeTrace = std::fopen(path.c_str(), "w");
        if (_decodeTrace != nullptr) {
            MM_LOG_INFO("engine",
                        "diagnostics.traceDecodeFile={} — per-token "
                        "decode trace enabled (wall_ms, cap_mhz, pkg_c)",
                        path);
        } else {
            MM_LOG_WARN("engine",
                        "diagnostics.traceDecodeFile={} — could not open "
                        "for writing, decode trace disabled",
                        path);
        }
    }
}

InferenceEngine::~InferenceEngine() {
    if (_decodeTrace != nullptr) {
        std::fclose(_decodeTrace);
        _decodeTrace = nullptr;
    }
}

// Schicht 4 — L0-only downcast of the polymorphic `_computeCtx`. The
// public `ctx()` / `allocator()` / `computeContext()` accessors route
// through here; each throws with a clear diagnostic on non-L0 runs so
// the failure surfaces at the call site rather than as UB from a bad
// static_cast.
core::l0::L0ComputeContext& InferenceEngine::l0ComputeContext() {
    if (_computeCtx == nullptr ||
        _computeCtx->kind() != core::backend::BackendKind::LevelZero) {
        throw std::runtime_error{
            "InferenceEngine::l0ComputeContext: runtime is not bound to "
            "LevelZero — L0-only accessor called on a "
            + std::string{core::backend::BackendRegistry::name(
                  _computeCtx ? _computeCtx->kind()
                              : core::backend::BackendKind::Unknown)}
            + " backend"};
    }
    return static_cast<core::l0::L0ComputeContext&>(*_computeCtx);
}

const core::l0::L0ComputeContext& InferenceEngine::l0ComputeContext() const {
    if (_computeCtx == nullptr ||
        _computeCtx->kind() != core::backend::BackendKind::LevelZero) {
        throw std::runtime_error{
            "InferenceEngine::l0ComputeContext: runtime is not bound to "
            "LevelZero — L0-only accessor called on a "
            + std::string{core::backend::BackendRegistry::name(
                  _computeCtx ? _computeCtx->kind()
                              : core::backend::BackendKind::Unknown)}
            + " backend"};
    }
    return static_cast<const core::l0::L0ComputeContext&>(*_computeCtx);
}

const core::gguf::WeightsMap& InferenceEngine::weights() const {
    if (!_weights.has_value()) {
        throw std::runtime_error("InferenceEngine: no model loaded");
    }
    return *_weights;
}

void InferenceEngine::loadModel(std::string_view ggufPath) {
    if (_modelLoaded) {
        throw std::runtime_error("InferenceEngine: model already loaded");
    }

    MM_LOG_INFO("engine", "loadModel: opening '{}'", ggufPath);
    _reader.open(ggufPath);

    _config.parseFromGguf(_reader);
    _tokenizer.loadFromGguf(_reader);

    MM_LOG_INFO("engine", "loadModel: copying tensors into USM");
    _reader.loadTensors(*_ops);

    _weights.emplace(_reader);

    finalizeLoad();
}

void InferenceEngine::loadModelAttached(
        std::string_view                     ggufPath,
        const core::ipc::TensorManifest&     manifest,
        std::span<void* const>               chunkBases) {
    if (_modelLoaded) {
        throw std::runtime_error("InferenceEngine: model already loaded");
    }
    if (manifest.tensors.empty()) {
        throw std::runtime_error("InferenceEngine::loadModelAttached: "
                                 "manifest.tensors is empty — nothing to attach");
    }

    MM_LOG_INFO("engine",
                "loadModelAttached: opening '{}' (header only)", ggufPath);
    _reader.open(ggufPath);
    _config.parseFromGguf(_reader);
    _tokenizer.loadFromGguf(_reader);

    // Refuse-on-drift check. The local GGUF header must produce the
    // same fingerprint Munin advertised for this model — otherwise
    // the tensor list we just accepted from Munin describes a
    // different model than our config/tokenizer expects. The whole
    // point of the manifest protocol is to catch this at handshake
    // time rather than during the first decode.
    const std::string localFingerprint =
        core::gguf::tensorFingerprint(_reader);
    if (localFingerprint != manifest.modelFingerprint) {
        throw std::runtime_error(
            std::string{"InferenceEngine::loadModelAttached: fingerprint "
                        "mismatch — Munin advertised '"} +
            manifest.modelFingerprint +
            "' for this model but the local GGUF hashes to '" +
            localFingerprint +
            "'. Refusing attach. Either the wrong file is mounted on "
            "the worker or Munin loaded a different revision. Restart "
            "Munin with the correct file or rebuild the worker with a "
            "matching config.");
    }

    MM_LOG_INFO("engine",
                "loadModelAttached: fingerprint match ({}), materialising {} "
                "attached tensors from {} chunk base(s) (Munin-owned USM)",
                localFingerprint, manifest.tensors.size(), chunkBases.size());
    _weights.emplace(
        core::gguf::WeightsMap::fromAttachedChunked(manifest, chunkBases));

    finalizeLoad();
}

void InferenceEngine::finalizeLoad() {
    // One-shot architecture diagnostic. For gemma4 also dump the first
    // shared-KV block (block 5 in the standard 26B-A4B layout) — we
    // need to see if its Q tensor has a different output dim than block 0,
    // which would tell us full-attention layers use head_dim_full while
    // SWA layers use head_dim_swa.
    logBlockTensorInventory(_reader, 0);
    if (_config.architecture == "gemma4" && _config.blockCount > 5) {
        logBlockTensorInventory(_reader, 5);
    }
    logTensorTaxonomy(_reader);

    // M5i.B: Try to fuse per-block attn_q/k/v weights so the QKV
    // projections can dispatch as one matmul. The class no-ops when
    // `features.fusedQkv` is false or when a block doesn't qualify
    // (missing tensors, mismatched types).
    // Gemma 4 E4B (and any future arch with shared-KV layers) skips
    // fusion for the trailing reuse-KV blocks. E4B also mixes attn_q
    // (Q6_K/Q4_K) with attn_k (Q5_K) per own-KV block, which would kill
    // fusion entirely — the requant-to-Q8_0 path rescues those blocks
    // at load time (quality neutral-to-better, ~70 MiB overhead on E4B).
    const bool requantToQ8_0 =
        (_config.architecture == "gemma4") && (_config.sharedKvLayers > 0);
    // Schicht 5.4 — FusedQkvWeights + load-time diagnostics are L0-only.
    // On HIP `_fusedQkv` stays nullptr (downstream code already handles
    // "fusion off"). selfTest / autotune / autotune-KTile-Q8 stay skipped
    // — they exercise the L0 SPV kernels via UsmAllocator; the HIP-side
    // equivalents don't exist yet and aren't blockers for a load attempt.
    if (_computeCtx->kind() == core::backend::BackendKind::LevelZero) {
        _fusedQkv = std::make_unique<model::FusedQkvWeights>(
            *_weights, allocator(), _config.blockCount,
            _cfg.features.fusedQkv,
            _config.sharedKvLayers, requantToQ8_0,
            /*q8_0ReorderEnabled=*/
            _cfg.features.q8_0Reorder != core::config::TriState::Disable);

        // M8.K.Q8_0-Reorder Phase 5b — register every fused-QKV block that
        // grew a reorder-layout copy with GpuOps so the /v1/system/status
        // .kernels.q8_0_reorder counter reflects the real number of
        // tensors on the reorder path. One note per block keeps the
        // tensor-count accurate (~24 fused blocks in E4B post-shared-KV
        // filtering) instead of collapsing them into a single hit.
        for (std::size_t b = 0; b < _fusedQkv->numBlocks(); ++b) {
            const auto* blk = _fusedQkv->at(b);
            if (blk == nullptr || blk->reorderUsmPtr == nullptr) continue;
            _ops->noteQ8_0ReorderApplied(
                blk->reorderBytes,
                "fused_qkv[" + std::to_string(b) + "]");
        }

        // M5i.D: Load-time self-tests of the GPU compute path. selfTest
        // verifies every non-matmul GPU op against a CPU reference on a
        // tiny fixed input — catches broken SPV loads and driver bugs on
        // unfamiliar iGPU µarchs. autotune runs a matvec-vs-GEMM micro-
        // bench (with its own parity gate) and pins the dispatch decision
        // per QuantType. Honours `features.gemm` / `features.dp4a`.
        l0Ops().selfTest(allocator());
        l0Gmm().autotune(allocator(), _config.embeddingLength, _cfg.features);
    } else {
        MM_LOG_INFO("engine",
                    "non-L0 backend ({}) — skipping FusedQkvWeights, "
                    "GPU-op self-test, and matmul autotune (L0-only paths)",
                    core::backend::BackendRegistry::name(_computeCtx->kind()));
    }

    // Pick the arch backend now that weights are available. Returns
    // nullptr for unsupported architectures so generate() can refuse
    // gracefully with the original architecture string in the error.
    // Schicht 5.1 — arch backends now consume ComputeOps / ComputeMatmul
    // through the base. `_ops` / `_gmm` are unique_ptr<Base>, deref-passes
    // the referenced instance. OpProfiler is still L0-only (guarded on
    // ctor); if a HIP model-load path lands (Schicht 5.2+ around
    // GgufReader), the profiler needs its own backend-neutral variant.
    _backend = arch::createArchBackend(
        _config.architecture, _config, *_weights, _fusedQkv.get(),
        *_ops, *_gmm, _opProfiler.value(), _cfg.features.moeGroup,
        _cfg.features.moeFusedDown != core::config::TriState::Disable);

    _modelLoaded = true;
    // Defensive: a previous model's KV state must not survive into the
    // new model. The current API throws on loadModel-while-loaded so
    // this is theoretical, but it keeps the invariant local.
    resetCache();
    MM_LOG_INFO("engine",
                "loadModel: ready — arch={} blocks={} d_model={} ff={} heads={} kv={}",
                _config.architecture, _config.blockCount,
                _config.embeddingLength, _config.feedForwardLength,
                _config.headCount, _config.headCountKv);
}

void InferenceEngine::resetCache() noexcept {
    if (_kvCache != nullptr) {
        _kvCache->reset();
    }
    _cachedTokens.clear();
}

void InferenceEngine::setKvDtype(KvDtype dtype) {
    // F32 is unconditionally safe — bit-identical to pre-M10.2 behaviour.
    if (dtype == KvDtype::F32) {
        _kvDtype = dtype;
        return;
    }

    // FP16 and Q8_0 both require the model to route every own-KV block
    // through the fused-QKV → qkv_split_*[fp16|q8_0] path. Raw fp32
    // matmul writes and fp32 bias adds on non-fp32 K/V slots would
    // silently corrupt the cache. See F32-only-Pfade comments in
    // Qwen2Backend.cpp / GemmaBaseBackend.cpp (M10.2 Commit 5). This
    // guard catches the mismatch at startup instead of shipping
    // garbage tokens.
    // Q8_0 additionally requires kvDim % 32 == 0 per own-KV layer
    // because storage is block-based (32 elements per block); the
    // KvCache ctor also asserts this but the engine-side guard here
    // surfaces the specific dtype-vs-model mismatch with a clearer
    // error before the cache is even constructed.
    const char* dtypeName =
        (dtype == KvDtype::FP16) ? "FP16"
                                 : "Q8_0";
    if (!_modelLoaded) {
        throw std::runtime_error(
            std::string{"InferenceEngine::setKvDtype("} + dtypeName +
            "): loadModel() must run first — the guard needs to "
            "inspect fused-QKV and attn_k/v.bias coverage per block");
    }

    const auto kvSource = _backend->kvSourceLayerPerLayer();
    const bool hasSharedKvMap = !kvSource.empty();
    const auto kvDimPerLayer  = _backend->kvDimPerLayer();
    constexpr std::size_t kQ8BlockElements = 32;
    for (std::size_t b = 0; b < _config.blockCount; ++b) {
        // Shared-KV blocks never touch the K/V projection themselves —
        // they alias an earlier block's slot. The earlier own-KV block
        // is what actually matters, and it will be validated on its own
        // iteration.
        const bool ownsKv = !hasSharedKvMap || kvSource[b] == b;
        if (!ownsKv) {
            continue;
        }
        // M10.2 Phase 1a Commit 5 relaxation: Q8_0 routes both fused and
        // non-fused K/V writes through the fp32 staging buffers in
        // BlockBuffers (kFp32Scratch / vFp32Scratch), then folds each
        // row into a Q8_0 block via kv_quant_commit_q8_0. The Q8_0 slot
        // never sees a raw fp32 matmul, so fused-QKV is NOT required
        // for correctness under Q8_0. FP16 still needs fused-QKV
        // because a raw fp32 K/V matmul would land bytes directly in
        // an fp16-typed slot and corrupt it — the FP16 write path has
        // no staging redirect.
        if (dtype == KvDtype::FP16 &&
            (_fusedQkv == nullptr || _fusedQkv->find(b) == nullptr)) {
            throw std::runtime_error(
                std::string{"InferenceEngine::setKvDtype("} + dtypeName +
                "): block " + std::to_string(b) +
                " has no fused-QKV entry — raw fp32 matmul would corrupt "
                "the fp16 K/V slot. Rebuild the model with fused-QKV weights "
                "(see FusedQkvWeights) or set runtime.kvDtype='q8_0' in "
                "config.json (which routes non-fused writes through fp32 "
                "staging) or leave runtime.kvDtype unset for the f32 default.");
        }
        if (_weights->findBlock(b, "attn_k.bias") != nullptr) {
            throw std::runtime_error(
                std::string{"InferenceEngine::setKvDtype("} + dtypeName +
                "): block " + std::to_string(b) +
                " carries attn_k.bias — the fp32 addBiasAsync would "
                "corrupt the K slot. Only bias-less K/V projections "
                "(Qwen 2.5+ style, Gemma family) are safe on non-F32 "
                "KV today.");
        }
        if (_weights->findBlock(b, "attn_v.bias") != nullptr) {
            throw std::runtime_error(
                std::string{"InferenceEngine::setKvDtype("} + dtypeName +
                "): block " + std::to_string(b) +
                " carries attn_v.bias — the fp32 addBiasAsync would "
                "corrupt the V slot. Only bias-less K/V projections "
                "(Qwen 2.5+ style, Gemma family) are safe on non-F32 "
                "KV today.");
        }
        if (dtype == KvDtype::Q8_0 &&
            b < kvDimPerLayer.size() &&
            kvDimPerLayer[b] % kQ8BlockElements != 0)
        {
            throw std::runtime_error(
                "InferenceEngine::setKvDtype(Q8_0): block " +
                std::to_string(b) + " kvDim=" +
                std::to_string(kvDimPerLayer[b]) +
                " is not a multiple of 32 — Q8_0 stores 32-element "
                "blocks so misaligned rows would straddle block "
                "boundaries. Use dtype=fp16 for this model.");
        }
    }

    _kvDtype = dtype;
    const std::size_t ownKvBlocks =
        _config.blockCount -
        (hasSharedKvMap ? _config.sharedKvLayers : 0);
    if (dtype == KvDtype::FP16) {
        MM_LOG_INFO("engine",
                    "setKvDtype: fp16 enablement guard passed — "
                    "{} own-KV blocks all fused, no attn_k/v.bias tensors",
                    ownKvBlocks);
    } else {
        // Q8_0 guard is intentionally lighter: fused-QKV is not required,
        // the non-fused path stages through kvKFp32Scratch/kvVFp32Scratch.
        std::size_t nFused = 0;
        for (std::size_t b = 0; b < _config.blockCount; ++b) {
            const bool ownsKv =
                !hasSharedKvMap || kvSource[b] == b;
            if (!ownsKv) continue;
            if (_fusedQkv != nullptr && _fusedQkv->find(b) != nullptr) {
                ++nFused;
            }
        }
        MM_LOG_INFO("engine",
                    "setKvDtype: q8_0 enablement guard passed — "
                    "{} own-KV blocks ({} fused, {} routed through fp32 "
                    "staging), no attn_k/v.bias tensors, "
                    "all kvDim are multiples of 32",
                    ownKvBlocks, nFused, ownKvBlocks - nFused);
    }

    // R5 — K-tile autotune for the packed Q8_0 GQA prefill kernel. Only
    // fires when features.flashPrefillKTileQ8 == 0 (autotune requested)
    // AND the model has a GQA shape AND we're actually running Q8_0 KV
    // storage. Guards inside `autotuneKTileQ8` early-out with an info
    // log for every other case. Uses SWA head_dim when available (most
    // Gemma 4 layers) since that's the geometry the packed kernel spends
    // most of its time on. L0-only — driven by the L0 kernel-bench path.
    if (_computeCtx->kind() == core::backend::BackendKind::LevelZero) {
        const std::size_t autotuneHeadDim = static_cast<std::size_t>(
            _config.keyLengthSwa > 0 ? _config.keyLengthSwa
                                     : _config.keyLength);
        l0Ops().autotuneKTileQ8(allocator(),
                             static_cast<std::size_t>(_config.headCount),
                             static_cast<std::size_t>(_config.headCountKv),
                             autotuneHeadDim,
                             _kvDtype);
    }
}

void InferenceEngine::ensureCapacity(std::size_t maxT, std::size_t Tp,
                                     std::size_t maxNew,
                                     std::size_t vocab_lm, std::size_t d_model) {
    // KvCache + cachedTokens: lifetime-critical. Allocate once at the
    // configured _maxContextTokens; never realloc on request growth.
    // This is the change that makes multi-turn prefix reuse actually
    // work — previously a growing prompt or max_tokens would trip the
    // realloc-and-reset path and lose every cached token.
    if (_kvCache == nullptr) {
        _kvCache = std::make_unique<KvCache>(
            *_ops, _maxContextTokens,
            _backend->kvDimPerLayer(),
            _backend->kvSourceLayerPerLayer(),
            _kvDtype);
        const char* dtypeLabel = "f32";
        if (_kvDtype == KvDtype::FP16) dtypeLabel = "fp16";
        else if (_kvDtype == KvDtype::Q8_0) dtypeLabel = "q8_0";
        MM_LOG_INFO("kvcache",
                    "pre-allocated for {} tokens dtype={} "
                    "(set via runtime.maxContextTokens / runtime.kvDtype)",
                    _maxContextTokens, dtypeLabel);
    }

    // Hard cap: a request that doesn't fit gets a clear error rather
    // than silently overflowing the cache. Operator can raise
    // `runtime.maxContextTokens` in config.json if the workload needs it.
    if (Tp + maxNew + 4 > _maxContextTokens) {
        throw std::runtime_error(
            "generate: request needs " + std::to_string(Tp + maxNew + 4) +
            " tokens of KV (prompt " + std::to_string(Tp) +
            " + max_new " + std::to_string(maxNew) +
            " + slack 4) but the engine is configured for "
            + std::to_string(_maxContextTokens) +
            " — raise runtime.maxContextTokens or shrink the request");
    }

    // BlockBuffers + scratch: purely transient, safe to realloc on
    // demand. scoreScratch inside BlockBuffers is sized to the cache
    // capacity (not the request chunk) so attention can scan the full
    // current KV length even when the chunk is just 1 decode token.
    const bool needScratchGrow =
        !_blockBuffers.has_value() ||
        maxT      > _cacheMaxT ||
        vocab_lm  > _cacheVocabLm;

    if (!needScratchGrow) {
        return;
    }

    const auto [qDimMax, kvDimMax] = _backend->maxQKVDims();
    const bool withFusedQkv =
        _fusedQkv != nullptr && _fusedQkv->anyFused();
    // M10.2 Phase 1a Commit 5 — Q8_0 KV requires a persistent fp32 K/V
    // workspace: rmsnorm_qkv + RoPE run fp32-in-place there and then
    // `kv_quant_commit_q8_0` folds the rows into the Q8_0 cache slot.
    const bool withKvFp32Scratch = (_kvDtype == KvDtype::Q8_0);
    _blockBuffers = allocBlockBuffers(*_ops, _config,
                                      maxT, _maxContextTokens,
                                      qDimMax, kvDimMax,
                                      withFusedQkv,
                                      withKvFp32Scratch);

    _xBufH      = _ops->allocate(maxT      * d_model  * sizeof(float));
    _normFinalH = _ops->allocate(d_model   * sizeof(float));
    _logitsH    = _ops->allocate(vocab_lm  * sizeof(float));
    _logitsScH  = _ops->allocate(d_model   * sizeof(float));

    _cacheMaxT     = maxT;
    _cacheVocabLm  = vocab_lm;
}

std::int32_t
InferenceEngine::sampleNext(const float*                   hidden,
                            std::size_t                    vocab_lm,
                            const core::gguf::GgufTensor&       outNorm,
                            const core::gguf::GgufTensor&       lmHead,
                            float*                         normScratch,
                            float*                         logits,
                            float*                         matmulScratch,
                            std::span<const std::int32_t>  recentTokens,
                            const compute::SamplingParams& sampling) {
    // Final-norm runs on the same queue as the residual-add that
    // produced `hidden`. The subsequent _gmm->matmul flushes and syncs,
    // so CPU argmax sees a fully-resolved logits buffer.
    //
    // Both Qwen and Gemma 4 use plain `w * rms_norm(x)` at runtime:
    // Gemma 3 stored (1+w) in HF and the converter shifts w_gguf =
    // w_hf + 1; Gemma 4's converter (gemma.py:621-623) returns
    // norm_shift = 0.0 because the HF reference uses standard weight
    // (init at 1.0) — so the GGUF weight is already the multiplicative
    // scale.
    _ops->rmsNormAsync(
        hidden, 1, _config.embeddingLength,
        static_cast<const float*>(outNorm.usmPtr),
        _config.rmsNormEps,
        normScratch);

    _gmm->matmul(
        lmHead.type, lmHead.usmPtr,
        vocab_lm, _config.embeddingLength,
        normScratch, 1,
        logits, matmulScratch);

    return _sampler.sample(
        std::span<const float>{logits, vocab_lm}, recentTokens, sampling);
}

std::vector<std::int32_t>
InferenceEngine::generate(std::span<const std::int32_t>   promptIds,
                          const GenerateParams&           params,
                          const TokenCallback&            onToken,
                          GenerateStats*                  outStats,
                          const PrefillCallback&          onPrefillDone,
                          const PrefillProgressCallback&  onPrefillProgress) {
    namespace cmp = mimirmind::compute;
    using clock = std::chrono::steady_clock;

    if (!_modelLoaded) {
        throw std::runtime_error("InferenceEngine::generate: no model loaded");
    }
    if (promptIds.empty()) {
        throw std::runtime_error("InferenceEngine::generate: empty prompt");
    }

    if (_backend == nullptr) {
        throw std::runtime_error(
            "generate: architecture '" + _config.architecture +
            "' is not recognised. Model is loaded into USM but the "
            "engine has no handler for it. See "
            "Memory/mimirmind/research/m8-gemma4-staging.md for the "
            "list of supported architectures.");
    }

    // M9.2 thermal admission. Throws ThermalLimitExceeded if any hard
    // limit in the configured profile is currently breached; ApiServer
    // turns that into HTTP 503 + Retry-After. Skips silently when no
    // guard is installed (op chose to run unprotected).
    if (_thermalGuard != nullptr) {
        _thermalGuard->checkAdmission();
    }

    // M9.11.b proactive fan boost. Ramp the chassis fan up before
    // prefill so the GPU clock governor has thermal headroom when
    // matmul starts pulling watts. RAII guard so an exception during
    // prefill/decode still releases the fan back to auto — leaving the
    // fan pinned at 100 % across an entire idle period would be loud
    // and would burn power.
    struct FanBoostGuard {
        FanController* fc;
        explicit FanBoostGuard(FanController* c) : fc{c} {
            if (fc != nullptr && fc->available()) {
                (void)fc->boost();
            }
        }
        ~FanBoostGuard() {
            if (fc != nullptr && fc->available()) {
                fc->releaseToAuto();
            }
        }
        FanBoostGuard(const FanBoostGuard&)            = delete;
        FanBoostGuard& operator=(const FanBoostGuard&) = delete;
    };
    FanBoostGuard fanBoost{_fanController};

    // Snapshot RAPL counters at the start so generate() can report how
    // much energy the request consumed. No-op when the monitor is
    // unavailable or absent.
    PowerMonitor::Snapshot powerStart{};
    if (_powerMonitor != nullptr && _powerMonitor->available()) {
        powerStart = _powerMonitor->snapshot();
    }

    const auto* tokEmb = _weights->find("token_embd.weight");
    if (tokEmb == nullptr) {
        tokEmb = _weights->find("tok_embeddings.weight");
    }
    const auto* outNorm = _weights->find("output_norm.weight");
    const auto* lmHead  = _weights->find("output.weight");
    if (lmHead == nullptr) {
        lmHead = _weights->find("token_embd.weight");
    }

    if (tokEmb == nullptr) {
        throw std::runtime_error("generate: token embedding tensor missing");
    }
    if (outNorm == nullptr ||
        outNorm->type != core::gguf::GgmlType::F32) {
        throw std::runtime_error(
            "generate: output_norm.weight missing or not F32");
    }
    if (lmHead == nullptr) {
        throw std::runtime_error("generate: lm_head tensor missing");
    }

    const std::size_t Tp        = promptIds.size();
    const std::size_t maxNew    = params.maxNewTokens;
    const std::size_t vocab_lm  = lmHead->dimensions.size() >= 2
                                    ? lmHead->dimensions[1]
                                    : _tokenizer.vocabSize();
    const std::size_t vocab_emb = tokEmb->dimensions.size() >= 2
                                    ? tokEmb->dimensions[1]
                                    : _tokenizer.vocabSize();
    const std::size_t maxT      = std::max<std::size_t>(Tp, 1);
    const std::size_t d_model   = _config.embeddingLength;

    ensureCapacity(maxT, Tp, maxNew, vocab_lm, d_model);
    KvCache&      cache   = *_kvCache;
    BlockBuffers& buffers = *_blockBuffers;

    float* const xBuf      = _xBufH     .as<float>();
    float* const normFinal = _normFinalH.as<float>();
    float* const logits    = _logitsH   .as<float>();
    float* const logitsSc  = _logitsScH .as<float>();

    // --- M9.1 prefix cache ----------------------------------------------
    //
    // _cachedTokens holds the ids whose K/V state is currently sitting in
    // `cache` from a previous generate() call. Compute how many leading
    // tokens of the new prompt match — those tokens can re-use the
    // existing KV rows.
    //
    // Clamp to Tp - 1: even on a perfect prefix match we still need to
    // re-run prefill for the final prompt token, because sampleNext()
    // reads its hidden state directly from xBuf (the cache only stores
    // K/V, not the hidden state that feeds the lm-head).
    std::size_t lcp = longestCommonPrefix(promptIds,
                                          std::span<const std::int32_t>{_cachedTokens});
    if (lcp >= Tp) {
        lcp = Tp - 1;
    }
    cache.truncate(lcp);
    const std::size_t prefillStart = lcp;
    const std::size_t prefillCount = Tp - lcp;

    // Gemma family scales the token embedding by sqrt(d_model) before
    // it enters the first block — the per-token vectors are otherwise
    // in the ~0.05 range and attention/FFN expects them at unit-ish
    // scale. Qwen/Llama don't do this. Backend tells us.
    const bool  embedScaleEnabled = _backend->scalesEmbedding();
    const float embedScale = embedScaleEnabled
        ? std::sqrt(static_cast<float>(d_model))
        : 1.0F;
    auto scaleEmbeddingIfNeeded = [&](float* dst, std::size_t T) {
        if (embedScaleEnabled && T > 0) {
            _ops->mulScalarAsync(dst, embedScale, T * d_model);
        }
    };

    // -- Prefill ---------------------------------------------------------
    //
    // Optional parity-test dump. `diagnostics.parityDump: "PREFIX"` in
    // config.json makes the backend write PREFIX-blk{N}-<stage>.bin at
    // multiple stages inside each block during prefill. Format matches
    // llama-parity-dump.

    if (!_cfg.diagnostics.parityDump.empty()) {
        const std::string& dumpPrefix = _cfg.diagnostics.parityDump;
        _backend->setParityDumpPrefix(dumpPrefix.c_str());
        // Log the token sequence so the operator can hand the same tokens
        // to llama-parity-dump and compare block-0 outputs.
        std::string idsCsv;
        const std::size_t nShow = std::min(promptIds.size(), std::size_t{48});
        for (std::size_t i = 0; i < nShow; ++i) {
            if (!idsCsv.empty()) idsCsv.push_back(',');
            idsCsv += std::to_string(promptIds[i]);
        }
        MM_LOG_INFO("parity",
                    "diagnostics.parityDump={} — prefill token count={}, "
                    "first {}: [{}]",
                    dumpPrefix, promptIds.size(), nShow, idsCsv);
        // dumpStage() sync-flushes the GPU and writes T*d_model*4 bytes per
        // stage. On a 42-block E4B with ~6 stages/block this scales to
        // multi-GB per request; a 3431-token prefill has been observed to
        // stall for >30 minutes with the flag left on in production.
        if (prefillCount > 256) {
            MM_LOG_WARN("parity",
                        "diagnostics.parityDump is set with prefillCount={} — "
                        "expect multi-GB synchronous disk writes and severely "
                        "degraded prefill throughput. Clear the field for "
                        "production traffic.",
                        prefillCount);
        }
    }

    std::vector<std::int32_t> generated;
    double                    preMs   = 0.0;
    double                    decMs   = 0.0;
    bool                      hitStop = false;
    bool                      aborted         = false;
    bool                      prefillAborted  = false;

    try {
        const auto preT0 = clock::now();
        const auto prefillIds = promptIds.subspan(prefillStart, prefillCount);
        cmp::embeddingLookup(
            tokEmb->type, tokEmb->usmPtr,
            d_model, vocab_emb,
            prefillIds, xBuf);
        scaleEmbeddingIfNeeded(xBuf, prefillCount);

        // Parity-dump the post-embed-scale hidden state so we can diff
        // it against llama.cpp's `inp_scaled` tensor. Written as
        // `<prefix>-blk0-input_scaled.bin` for parity-diff to pick up.
        // No-op unless `diagnostics.parityDump` is non-empty.
        if (!_cfg.diagnostics.parityDump.empty()) {
            const std::string& dumpPrefix = _cfg.diagnostics.parityDump;
            // scaleEmbeddingIfNeeded submitted `mulScalarAsync` — it's
            // still queued on the GPU. Flush before the CPU reads xBuf,
            // otherwise the dump captures the un-scaled embedding and
            // parity-diff reports a spurious factor-of-sqrt(n_embd)
            // divergence (RMSNorm inside block 0 cancels it so the
            // rest of the pipeline still runs on the scaled value).
            _ops->flush();
            const std::string fname =
                dumpPrefix + "-blk0-inp_scaled.bin";
            std::ofstream f(fname, std::ios::binary);
            if (f) {
                const std::uint32_t hdr[3] = {
                    0U,
                    static_cast<std::uint32_t>(prefillCount),
                    static_cast<std::uint32_t>(d_model),
                };
                f.write(reinterpret_cast<const char*>(hdr), sizeof(hdr));
                f.write(reinterpret_cast<const char*>(xBuf),
                        static_cast<std::streamsize>(prefillCount * d_model *
                                                     sizeof(float)));
            }
        }

        // Per-forward architecture hook. Non-E-series backends no-op;
        // Gemma4E4BBackend dequantizes its PLE slices AND runs the
        // per_layer_model_proj chain on the token embeddings before
        // the block loop starts.
        _backend->prepareForward(prefillIds, xBuf, prefillCount);

        for (std::uint32_t b = 0; b < _config.blockCount; ++b) {
            _backend->runBlock(b, xBuf, prefillCount, cache, buffers,
                               _traceBlock0);
            if (onPrefillProgress) {
                const auto now = clock::now();
                const double elapsedMs =
                    std::chrono::duration<double, std::milli>(now - preT0)
                        .count();
                const bool keepGoing = onPrefillProgress(PrefillProgress{
                    static_cast<std::size_t>(b) + 1,
                    static_cast<std::size_t>(_config.blockCount),
                    elapsedMs,
                });
                if (!keepGoing) {
                    // M7g client-cancel during prefill. Break at the next
                    // block barrier — the currently-executing block has
                    // already finished by the time we're here.
                    prefillAborted = true;
                    aborted        = true;
                    break;
                }
            }
        }

        // Compute the elapsed prefill time regardless of abort so the
        // operator can see how far the client got in outStats + logs.
        const auto preT1 = clock::now();
        preMs =
            std::chrono::duration<double, std::milli>(preT1 - preT0).count();

        if (prefillAborted) {
            // Partial KV writes are invalid without a matching commit.
            // Wiping keeps the next request honest and avoids any bogus
            // prefix-cache hits on the aborted prompt.
            resetCache();
        } else {
            cache.commit(prefillCount);
            if (onPrefillDone) {
                onPrefillDone(PrefillDone{Tp, prefillCount, preMs});
            }
        }

        _traceBlock0 = false;  // diagnostic done; mute for further calls

        // The rest of the try body — sampler seeding, first sample, decode
        // loop, per-token telemetry, perf-detector run-complete — is decode
        // work that a client cancelling during prefill has explicitly asked
        // us not to do (M7g). Wrap it in a single conditional so control
        // flow stays linear.
        if (!prefillAborted) {
        // Reseed the sampler per generate() call so deterministic seeds
        // produce reproducible streams. seed == 0 ⇒ random_device.
        _sampler.reseed(params.sampling.seed);

        auto isStop = [&](std::int32_t id) -> bool {
            if (id == _tokenizer.eosId()) {
                return true;
            }
            for (auto s : params.stopIds) {
                if (id == s) {
                    return true;
                }
            }
            return false;
        };

        // Sample first new token from the last prefill row. xBuf only
        // holds the freshly-prefilled suffix, so the last row sits at
        // (prefillCount - 1) * d_model — not (Tp - 1).
        //
        // M7f: seed the penalty window with the last stretch of the
        // prompt so an immediate first-token repeat (e.g. echoing the
        // prompt's last phrase) is discouraged as well. Sampler
        // subspans to `sampling.penaltyWindow` internally.
        const float* lastRow = xBuf + (prefillCount - 1) * d_model;
        std::int32_t nextId = sampleNext(lastRow, vocab_lm,
                                         *outNorm, *lmHead,
                                         normFinal, logits, logitsSc,
                                         promptIds,
                                         params.sampling);

        generated.reserve(maxNew);
        generated.push_back(nextId);

        if (onToken && !onToken(nextId)) {
            aborted = true;
        }

        // -- Decode loop -------------------------------------------------

        const auto decT0 = clock::now();

        // M-CLR.4: opt-in Command-List-Replay for the decode block loop.
        // Step 1 stays immediate (warms the KV cache with the first
        // sampled token). Step 2 records the block loop into a
        // persistent command list and executes it via replay. Steps 3+
        // update the shared USM curLen slot and re-execute the same
        // recorded list — no per-token appendLaunch, no per-token
        // barrier setup, no per-kernel arg binding.
        const bool clrEnvOn = _cfg.features.clr;
        // MoE models are NOT CLR-safe: `Gemma4MoeBackend::runBlock` picks
        // top-K experts on the host per record via `cmp::moeTopKRoute`
        // (reading the router matmul output on the CPU), and every
        // per-expert matmul dispatch captures a fixed expert weight
        // pointer + weight scale via setPtr/setValue at record time. At
        // replay for steps 3+ the routing decision is frozen to step-2's
        // hidden state — the recorded expert selection is stale, per-
        // token attention degrades into a repetition loop, and the
        // sampler emits noise tokens. Confirmed on L0_TARGET_HOST with
        // Gemma 4 26B-A4B-it-Q6_K under both KvDtype::F32 and
        // KvDtype::Q8_0. The router matmul is also synchronous
        // (`_gmm->matmul`, not `matmulAsync`) which flushes the recording
        // list mid-record — a second, subtler CLR breach. Both problems
        // are structural to MoE and would need a GPU-side gather +
        // stable-record refactor to fix cleanly; until that lands, MoE
        // decode runs in immediate mode.
        // Schicht 5.5 — CLR record/replay lives on the L0 CommandQueue.
        // HIP has no equivalent (hipGraph would be the door, but not
        // wired). Adding the kind-gate here disables every downstream
        // `l0Ops().queue()` / `curLenSlot()` call in the decode path
        // via the existing `if (clrEnabled)` blocks — one flag, all
        // sites covered.
        const bool clrEnabled =
            clrEnvOn && (_config.expertCount == 0) &&
            (_computeCtx->kind() == core::backend::BackendKind::LevelZero);
        if (clrEnvOn && _config.expertCount > 0) {
            MM_LOG_WARN("engine",
                        "features.clr=true requested but disabled "
                        "for this MoE model (expertCount={}) — MoE "
                        "routing bakes host-computed expert selections "
                        "into the recording, which stales at replay. "
                        "Immediate-mode decode used instead.",
                        _config.expertCount);
        }
        if (clrEnabled) {
            MM_LOG_INFO("engine",
                        "features.clr=true — decode uses record/replay "
                        "from step 2 on");
            l0Ops().queue().resetRecording();
            // Right-size the FlashAttention partial launch geometry to
            // what THIS generate() call could possibly need. `kFlashMaxKTiles`
            // is a coarse upper bound (32768 / 64 = 512 post-M9.8b) that
            // wastes ~511/512 work-groups per attention call at typical
            // chat context. Bound by prompt + max_new saves 30-90 ms/tok
            // on short-context E4B.
            const std::size_t maxCurLen =
                promptIds.size() + params.maxNewTokens;
            const std::size_t replayKTiles = std::min(
                (maxCurLen + compute::l0::GpuOps::kFlashKTileSize - 1) /
                    compute::l0::GpuOps::kFlashKTileSize,
                compute::l0::GpuOps::kFlashMaxKTiles);
            _ops->setReplayMaxKTiles(replayKTiles);
            MM_LOG_INFO("engine",
                        "features.clr right-sized flash launch "
                        "geometry to {} k-tiles (max curLen {})",
                        replayKTiles, maxCurLen);
        } else {
            _ops->setReplayMaxKTiles(0);
        }

        // Inter-token thermal pacing — consult guard every kPaceWindow
        // tokens so /sys reads don't dominate the inner loop. Window of
        // 4 keeps overhead under a millisecond per token at ~145 ms/tok
        // decode while still reacting to a fast temperature climb
        // within ~500 ms.
        constexpr std::size_t kPaceWindow = 4;
        // The GPU clock governor adjusts the iGPU max-freq cap; this
        // happens at a slower cadence than the per-token pacing
        // because a fresh sysfs write costs ~200 µs and reaches the
        // hardware on the next dispatch. 8 tokens at ~145 ms each is
        // ~1.2 s between adjustments — well within the package
        // thermal time constant.
        constexpr std::size_t kGovernorWindow = 8;

        for (std::size_t step = 1;
             !aborted && step < maxNew && cache.length() < _maxContextTokens;
             ++step)
        {
            if (isStop(nextId)) {
                hitStop = true;
                break;
            }

            if (_thermalGuard != nullptr && (step % kPaceWindow) == 0) {
                const auto pause = _thermalGuard->paceForCurrentReading();
                if (pause.count() > 0) {
                    std::this_thread::sleep_for(pause);
                }
            }

            if (_gpuGovernor != nullptr && !_gpuGovernor->pinned()
                && _governorMonitor != nullptr
                && (step % kGovernorWindow) == 0) {
                (void)_gpuGovernor->tick(*_governorMonitor);
            }

            const auto tokT0 = clock::now();

            std::array<std::int32_t, 1> oneId{nextId};
            cmp::embeddingLookup(
                tokEmb->type, tokEmb->usmPtr,
                d_model, vocab_emb,
                oneId, xBuf);
            scaleEmbeddingIfNeeded(xBuf, 1);

            _backend->prepareForward(
                std::span<const std::int32_t>{oneId}, xBuf, 1);

            // M-CLR.4: three modes for the block loop:
            //   step == 1                 → immediate (warm)
            //   step == 2 && clrEnabled   → record + replay-once
            //   step >  2 && recording    → update curLen slot + replay
            if (clrEnabled && l0Ops().queue().hasRecording()) {
                // Replay-only path. Update the shared USM slot so every
                // recorded rope / attention / qkv_split / rmsnorm_qkv
                // kernel sees the current KV-cache length.
                *l0Ops().curLenSlot() =
                    static_cast<std::int32_t>(cache.length());
                l0Ops().queue().replay();
            } else if (clrEnabled && step == 1) {
                // Step 1 records into the persistent list AND executes it
                // via a first replay(). Subsequent steps reuse the
                // recording without re-dispatching.
                //
                // `scaleEmbeddingIfNeeded` above dispatches `mulScalarAsync`
                // into the immediate list whenever the backend scales its
                // embeddings (all Gemma-4 variants do). `beginRecord()`
                // requires the immediate list to be idle — the E4B backend
                // used to mask this by flushing internally inside its
                // `prepareForward`, but MoE / Dense have a no-op default
                // and would trip the "immediate work is pending" throw.
                // Flush the immediate list explicitly so the invariant
                // holds independently of the backend.
                _ops->flush();
                l0Ops().queue().beginRecord();
                for (std::uint32_t b = 0; b < _config.blockCount; ++b) {
                    _backend->runBlock(b, xBuf, 1, cache, buffers, false);
                }
                l0Ops().queue().endRecord();
                l0Ops().queue().replay();
            } else {
                for (std::uint32_t b = 0; b < _config.blockCount; ++b) {
                    _backend->runBlock(b, xBuf, 1, cache, buffers, false);
                }
            }
            cache.commit(1);

            nextId = sampleNext(xBuf, vocab_lm,
                                *outNorm, *lmHead,
                                normFinal, logits, logitsSc,
                                std::span<const std::int32_t>{generated},
                                params.sampling);
            generated.push_back(nextId);

            // Per-token telemetry. Both the env-controlled NDJSON sink
            // and the in-process perf-regression detector (when set)
            // consume the same three numbers (wall_ms, cap_mhz, pkg_c),
            // so the sensor reads happen once.
            // sampleNext sync-waits on the lmHead matmul so tokT1 is
            // real wall-time including the whole layer chain.
            if (_decodeTrace != nullptr || _perfDetector != nullptr) {
                const auto tokT1 = clock::now();
                const double tokMs = std::chrono::duration<double, std::milli>(
                    tokT1 - tokT0).count();
                std::uint32_t cap = 0;
                double pkg = -1.0;
                if (_gpuGovernor != nullptr) {
                    cap = _gpuGovernor->currentCapMhz();
                }
                if (_governorMonitor != nullptr) {
                    const auto r = _governorMonitor->read();
                    if (r.package_temp_c.has_value()) {
                        pkg = static_cast<double>(*r.package_temp_c);
                    }
                }
                if (_decodeTrace != nullptr) {
                    std::fprintf(_decodeTrace,
                                 "{\"tok\":%zu,\"wall_ms\":%.3f,"
                                 "\"cap_mhz\":%u,\"pkg_c\":%.1f}\n",
                                 step, tokMs, cap, pkg);
                }
                if (_perfDetector != nullptr) {
                    _perfDetector->onDecodeToken(
                        PerfRegressionDetector::Sample{tokMs, cap, pkg});
                }
                // M8.K.0 diagnostic: emits a per-category share summary
                // every 50 tokens when diagnostics.traceOpTimes=true.
                // No-op otherwise. Only present on L0 today (see ctor).
                if (_opProfiler) {
                    _opProfiler->maybeDumpAndReset(step);
                }
            }

            if (onToken && !onToken(nextId)) {
                aborted = true;
            }
        }

        if (_decodeTrace != nullptr) {
            std::fflush(_decodeTrace);
        }

        const auto decT1 = clock::now();
        decMs =
            std::chrono::duration<double, std::milli>(decT1 - decT0).count();

        // Detector consults the ring buffer once per run — computes p50,
        // compares against the rolling baseline, alerts + persists.
        // Never throws (noexcept), so it stays outside the try body's
        // hot path but inside the outer generate() try where a failure
        // is at worst logged.
        if (_perfDetector != nullptr) {
            _perfDetector->onRunComplete(generated.size());
        }
        } // if (!prefillAborted) — M7g decode-phase wrapper
    } catch (...) {
        // Mid-flight failure leaves the KV state partially written.
        // Discarding the cache is cheap and keeps the next call honest.
        resetCache();
        throw;
    }

    // -- Update the prefix cache so the next generate() can resume -----
    //
    // Maintain the invariant `_cachedTokens.size() == cache.length()`.
    // cache.length() at this point is: prefillStart + prefillCount +
    // (decode steps that committed). The first sampled token is not
    // yet committed; each subsequent loop iteration commits exactly one
    // token (the previous step's sample) before sampling the next one.
    //
    // M7g — skip when prefill was aborted. resetCache() has already run
    // and cache.length() is 0; writing promptIds here anyway would make
    // the next request see a bogus prefix-cache hit on a prompt whose
    // KV was never actually committed.
    if (!prefillAborted) {
        const std::size_t finalLen   = cache.length();
        const std::size_t genFromCache = (finalLen > Tp) ? (finalLen - Tp) : 0;
        const std::size_t take       = std::min(genFromCache, generated.size());

        _cachedTokens.clear();
        _cachedTokens.reserve(finalLen);
        _cachedTokens.insert(_cachedTokens.end(),
                             promptIds.begin(), promptIds.end());
        _cachedTokens.insert(_cachedTokens.end(),
                             generated.begin(),
                             generated.begin() +
                                 static_cast<std::ptrdiff_t>(take));
    }

    if (outStats != nullptr) {
        outStats->promptTokens    = Tp;
        outStats->generatedTokens = generated.size();
        outStats->cachedTokens    = lcp;
        outStats->prefillMs       = preMs;
        outStats->decodeMs        = decMs;
        outStats->hitStop         = hitStop;

        if (_powerMonitor != nullptr && _powerMonitor->available() &&
            !powerStart.raw_energy_uj.empty()) {
            const auto powerEnd = _powerMonitor->snapshot();
            const auto joules   = _powerMonitor->energyBetween(powerStart, powerEnd);
            // The first discovered domain is the package socket (intel-rapl:0).
            // Report that as the headline figure. Operators who want the
            // per-sub-domain split can scrape /v1/system/status.
            if (!joules.empty()) {
                outStats->packageJoules = joules.front();
            }
        }
    }

    return generated;
}

std::vector<std::vector<float>>
InferenceEngine::forwardVerify(std::span<const std::int32_t> newTokens) {
    namespace cmp = mimirmind::compute;

    if (!_modelLoaded) {
        throw std::runtime_error(
            "InferenceEngine::forwardVerify: no model loaded");
    }
    if (newTokens.empty()) {
        return {};
    }
    if (_backend == nullptr) {
        throw std::runtime_error(
            "forwardVerify: architecture '" + _config.architecture +
            "' has no backend");
    }

    const auto* tokEmb = _weights->find("token_embd.weight");
    if (tokEmb == nullptr) {
        tokEmb = _weights->find("tok_embeddings.weight");
    }
    const auto* outNorm = _weights->find("output_norm.weight");
    const auto* lmHead  = _weights->find("output.weight");
    if (lmHead == nullptr) {
        lmHead = _weights->find("token_embd.weight");
    }
    if (tokEmb == nullptr) {
        throw std::runtime_error("forwardVerify: token embedding missing");
    }
    if (outNorm == nullptr || outNorm->type != core::gguf::GgmlType::F32) {
        throw std::runtime_error(
            "forwardVerify: output_norm.weight missing or not F32");
    }
    if (lmHead == nullptr) {
        throw std::runtime_error("forwardVerify: lm_head tensor missing");
    }

    const std::size_t N        = newTokens.size();
    const std::size_t vocab_lm = lmHead->dimensions.size() >= 2
                                    ? lmHead->dimensions[1]
                                    : _tokenizer.vocabSize();
    const std::size_t vocab_emb = tokEmb->dimensions.size() >= 2
                                     ? tokEmb->dimensions[1]
                                     : _tokenizer.vocabSize();
    const std::size_t d_model  = _config.embeddingLength;

    // Admission + scratch sizing. Reuse the same helper as generate();
    // Tp is where the KV would end up if we committed everything,
    // maxNew=0 because verify never extends past N.
    const std::size_t curLen = (_kvCache != nullptr) ? _kvCache->length() : 0;
    ensureCapacity(N, curLen + N, 0, vocab_lm, d_model);

    KvCache&      cache   = *_kvCache;
    BlockBuffers& buffers = *_blockBuffers;

    float* const xBuf      = _xBufH     .as<float>();
    float* const normFinal = _normFinalH.as<float>();
    float* const logits    = _logitsH   .as<float>();
    float* const logitsSc  = _logitsScH .as<float>();

    // Gemma-family sqrt(d_model) embedding scale, delegated to backend.
    const bool  embedScaleEnabled = _backend->scalesEmbedding();
    const float embedScale = embedScaleEnabled
        ? std::sqrt(static_cast<float>(d_model)) : 1.0F;

    cmp::embeddingLookup(tokEmb->type, tokEmb->usmPtr,
                         d_model, vocab_emb,
                         newTokens, xBuf);
    if (embedScaleEnabled) {
        _ops->mulScalarAsync(xBuf, embedScale, N * d_model);
    }

    _backend->prepareForward(newTokens, xBuf, N);

    // Batched block forward. runBlock writes provisional K/V rows to
    // [curLen, curLen+N); commit is deferred to commitVerified().
    for (std::uint32_t b = 0; b < _config.blockCount; ++b) {
        _backend->runBlock(b, xBuf, N, cache, buffers, false);
    }

    // Per-position logits. rmsNorm + lmHead matmul is cheap next to the
    // block chain, so no need to batch M=N here — a loop keeps the
    // scratch requirement at vocab_lm floats (M=1).
    std::vector<std::vector<float>> out;
    out.reserve(N);
    for (std::size_t i = 0; i < N; ++i) {
        const float* row = xBuf + i * d_model;
        _ops->rmsNormAsync(row, 1, d_model,
                          static_cast<const float*>(outNorm->usmPtr),
                          _config.rmsNormEps, normFinal);
        _gmm->matmul(lmHead->type, lmHead->usmPtr,
                    vocab_lm, d_model, normFinal, 1,
                    logits, logitsSc);
        out.emplace_back(logits, logits + vocab_lm);
    }
    return out;
}

void InferenceEngine::commitVerified(
    std::span<const std::int32_t> acceptedTokens) {
    if (acceptedTokens.empty()) {
        return;
    }
    if (_kvCache == nullptr) {
        throw std::runtime_error(
            "commitVerified: KV cache not allocated (forwardVerify never ran)");
    }
    _kvCache->commit(acceptedTokens.size());
    _cachedTokens.insert(_cachedTokens.end(),
                         acceptedTokens.begin(), acceptedTokens.end());
}

} // namespace mimirmind::runtime