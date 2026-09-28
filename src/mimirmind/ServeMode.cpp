// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "mimirmind/ServeMode.hpp"

#include "mimirmind/CliArgs.hpp"
#include "mimirmind/CliParser.hpp"
#include "mimirmind/ServerBootstrap.hpp"
#include "mimirmind/ServingBenchModes.hpp"

#ifdef MIMIRMIND_HAVE_L0
#include "compute/l0/GpuOps.hpp"
#endif
#include "core/backend/BackendPool.hpp"
#include "core/backend/BackendRegistry.hpp"
#include "core/config/Config.hpp"
#include "core/ipc/MuninClient.hpp"
#ifdef MIMIRMIND_HAVE_L0
#include "core/ipc/L0IpcImporter.hpp"
#endif
#ifdef MIMIRMIND_HAVE_CUDA
#include "core/ipc/ShmIpcImporter.hpp"
#endif
#include "core/log/Log.hpp"
#include "core/os/GovernorLock.hpp"
#include "core/security/ApiKeyStore.hpp"
#include "core/security/SelfSignedCert.hpp"
#include "model/Tokenizer.hpp"
#include "runtime/ComputeStack.hpp"
#include "runtime/InferenceEngine.hpp"
#include "runtime/encoder/RerankEngine.hpp"
#include "runtime/encoder/EmbedEngine.hpp"
#include "runtime/encoder/DecideEngine.hpp"
#include "runtime/audio/AudioEngine.hpp"
#include "runtime/audio/SpeakEngine.hpp"
#include "runtime/serving/ContinuousBatcher.hpp"
#include "runtime/nvfp4/ModelFormatResolver.hpp"
#include "runtime/perf/PerfRegressionDetector.hpp"
#include "runtime/spec/Drafter.hpp"
#include "runtime/spec/ModelDrafter.hpp"
#include "runtime/spec/NGramDrafter.hpp"
#include "runtime/thermal/FanController.hpp"
#include "runtime/thermal/GpuClockGovernor.hpp"
#include "runtime/thermal/PowerMonitor.hpp"
#include "runtime/thermal/SystemMonitor.hpp"
#include "runtime/thermal/ThermalGuard.hpp"
#include "runtime/thermal/ThermalProfile.hpp"
#include "server/ApiServer.hpp"
#include "server/AttachedModelProvider.hpp"  // M-Munin.3 pool provider (2b)

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace {

// Held in the SIGINT handler so a Ctrl-C asks the listener to drain.
std::atomic<::mimirmind::server::ApiServer*> g_runningServer{nullptr};

// Directory that auto-provisioned security files (cert, key, keyfile) live
// in: the directory of the loaded config file. Falls back to "." when the
// config path has no parent component (e.g. a bare "config.json").
std::filesystem::path securityDir(const std::string& configPath) {
    std::filesystem::path p{configPath};
    std::filesystem::path parent = p.parent_path();
    return parent.empty() ? std::filesystem::path{"."} : parent;
}

std::string readTextFile(const std::filesystem::path& path) {
    std::ifstream in{path, std::ios::binary};
    if (!in) return {};
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

bool writeTextFile(const std::filesystem::path& path, const std::string& body) {
    std::ofstream out{path, std::ios::binary | std::ios::trunc};
    if (!out) return false;
    out << body;
    return static_cast<bool>(out);
}

// Provision in-process TLS material into `scfg.tls`. Reuses operator-supplied
// cert/key files, else reuses a persisted self-signed pair next to the
// config, else mints a fresh self-signed pair (persisted when the dir is
// writable, otherwise kept in memory for this run only). Returns false only
// when TLS is enabled but no usable material could be produced.
bool provisionTls(const ::mimirmind::core::config::TlsSettings& in,
                  const std::string&                            host,
                  const std::filesystem::path&                  dir,
                  ::mimirmind::server::TlsConfig&               out) {
    namespace sec = ::mimirmind::core::security;

    out.enabled = in.enabled.value_or(true);   // secure-by-default
    if (!out.enabled) {
        MM_LOG_WARN("main",
                    "server.tls.enabled=false — HTTP will be served in "
                    "cleartext on {}", host);
        return true;
    }

    // (1) Operator-provided cert/key files win outright.
    if (!in.certFile.empty() || !in.keyFile.empty()) {
        const std::string cert = readTextFile(in.certFile);
        const std::string key  = readTextFile(in.keyFile);
        if (cert.empty() || key.empty()) {
            MM_LOG_ERROR("main",
                         "server.tls: certFile '{}' / keyFile '{}' set but "
                         "not both readable — cannot enable TLS",
                         in.certFile, in.keyFile);
            return false;
        }
        out.certPem    = cert;
        out.keyPem     = key;
        out.selfSigned = false;
        MM_LOG_INFO("main", "TLS: using operator cert '{}' + key '{}'",
                    in.certFile, in.keyFile);
        return true;
    }

    // (2) Reuse a persisted self-signed pair if present (idempotent).
    const std::filesystem::path certPath = dir / "mimir-cert.pem";
    const std::filesystem::path keyPath  = dir / "mimir-key.pem";
    {
        const std::string cert = readTextFile(certPath);
        const std::string key  = readTextFile(keyPath);
        if (!cert.empty() && !key.empty()) {
            out.certPem    = cert;
            out.keyPem     = key;
            out.selfSigned = true;
            MM_LOG_WARN("main",
                        "TLS: reusing SELF-SIGNED certificate {} — clients "
                        "must trust it explicitly (curl --insecure / import "
                        "the cert). Provide server.tls.certFile+keyFile for a "
                        "real cert.", certPath.string());
            return true;
        }
    }

    // (3) Mint a fresh self-signed pair.
    sec::SelfSignedCertParams params;
    params.commonName = sec::resolveHostname();
    const sec::SelfSignedCert gen = sec::generateSelfSignedCert(params);
    if (!gen.ok) {
        MM_LOG_ERROR("main", "TLS: self-signed generation failed: {}",
                     gen.error);
        return false;
    }
    out.certPem    = gen.certPem;
    out.keyPem     = gen.keyPem;
    out.selfSigned = true;

    const bool wroteCert = writeTextFile(certPath, gen.certPem);
    const bool wroteKey  = writeTextFile(keyPath, gen.keyPem);
    if (wroteCert && wroteKey) {
        std::error_code ec;
        std::filesystem::permissions(
            keyPath,
            std::filesystem::perms::owner_read | std::filesystem::perms::owner_write,
            std::filesystem::perm_options::replace, ec);
        MM_LOG_WARN("main",
                    "TLS: generated SELF-SIGNED certificate (CN={}) and saved "
                    "it to {} / {} — clients must trust it explicitly (curl "
                    "--insecure / import the cert). Provide "
                    "server.tls.certFile+keyFile for a real cert.",
                    params.commonName, certPath.string(), keyPath.string());
    } else {
        MM_LOG_WARN("main",
                    "TLS: generated SELF-SIGNED certificate (CN={}) but could "
                    "not persist to {} — using an EPHEMERAL in-memory cert "
                    "(regenerated every restart). Clients must trust it "
                    "explicitly.", params.commonName, dir.string());
    }
    return true;
}

// Provision bearer-token API keys into `scfg.auth`. Resolves the bind-aware
// enabled default, loads explicit keys[] + keyFile, and — when auth is on and
// no key is configured — auto-generates + persists one and logs it once.
void provisionAuth(const ::mimirmind::core::config::AuthSettings& in,
                   const std::string&                             host,
                   const std::filesystem::path&                   dir,
                   ::mimirmind::server::AuthConfig&               out) {
    namespace sec = ::mimirmind::core::security;

    // Policy: auth is ON by default regardless of bind address. An
    // internet-exposed listener must never answer (or accept) anything
    // without a valid key — no bind-aware exception. Only an explicit
    // server.auth.enabled=false opts out, and that is loudly warned.
    out.enabled = in.enabled.value_or(true);

    if (!out.enabled) {
        MM_LOG_WARN("main",
                    "server.auth.enabled=false — the API is UNAUTHENTICATED "
                    "on {} (every route is open, including /health)", host);
        return;
    }

    // Parse an entry shaped as `key` or `name:key[:tenantId[:role]]`.
    // `role` == "admin" grants access to the operator-only routes.
    auto parseEntry = [](const std::string& raw) -> sec::ApiKey {
        sec::ApiKey k;
        const auto c1 = raw.find(':');
        if (c1 == std::string::npos) {
            k.name = "config"; k.key = raw; k.tenantId = "default";
            return k;
        }
        k.name        = raw.substr(0, c1);
        const auto c2 = raw.find(':', c1 + 1);
        if (c2 == std::string::npos) {
            k.key = raw.substr(c1 + 1); k.tenantId = k.name;
        } else {
            k.key         = raw.substr(c1 + 1, c2 - c1 - 1);
            const auto c3 = raw.find(':', c2 + 1);
            if (c3 == std::string::npos) {
                k.tenantId = raw.substr(c2 + 1);
            } else {
                k.tenantId = raw.substr(c2 + 1, c3 - c2 - 1);
                k.isAdmin  = (raw.substr(c3 + 1) == "admin");
            }
        }
        if (k.tenantId.empty()) k.tenantId = k.name;
        return k;
    };

    for (const std::string& raw : in.keys) {
        sec::ApiKey k = parseEntry(raw);
        if (!k.key.empty()) out.store.add(std::move(k));
    }

    const std::filesystem::path keyfile =
        in.keyFile.empty() ? (dir / "mimir-apikeys.txt")
                           : std::filesystem::path{in.keyFile};
    for (sec::ApiKey& k : sec::ApiKeyStore::loadKeyFile(keyfile.string())) {
        out.store.add(std::move(k));
    }

    if (!out.store.empty()) {
        MM_LOG_INFO("main", "AUTH: enabled with {} API key(s)",
                    out.store.size());
        return;
    }

    if (!in.autoGenerateKey) {
        MM_LOG_WARN("main",
                    "AUTH: enabled but no key configured and "
                    "autoGenerateKey=false — every request will 401. Set "
                    "server.auth.keys or a keyFile.");
        return;
    }

    // Auto-generate + persist a single key, and log it once, in full. The
    // bootstrap key is admin so a bare `serve` can reach the operator-only
    // routes (/v1/admin/tenants, /metrics) with the key it just printed.
    sec::ApiKey k;
    k.name     = "auto";
    k.key      = sec::ApiKeyStore::generateKey();
    k.tenantId = "default";
    k.isAdmin  = true;
    if (k.key.empty()) {
        MM_LOG_ERROR("main", "AUTH: key auto-generation failed (CSPRNG) — "
                             "auth is enabled but has no usable key");
        return;
    }
    const bool persisted =
        sec::ApiKeyStore::saveKeyFile(keyfile.string(), {k});
    MM_LOG_WARN("main",
                "AUTH: no API key configured — generated one and saved it to "
                "{}.\n  ==> API key: {}\n  Send it as: Authorization: Bearer "
                "{}\n  Set server.auth.keys to manage keys explicitly.",
                persisted ? keyfile.string() : "(memory only — keyfile not "
                                               "writable)",
                k.key, k.key);
    out.store.add(std::move(k));
}

} // namespace

extern "C" void signalStop(int /*sig*/) {
    if (auto* s = g_runningServer.load(std::memory_order_acquire)) {
        s->stop();
    }
}

namespace mimirmind::cli {

int runServe(const CliArgs& args, const ::mimirmind::core::config::Config& cfg) {
    std::cout << kBanner;
    std::cout.flush();

    if (args.modelPath.empty()) {
        std::cerr << "serve: models[<defaultModel>].path is required "
                     "(fill it in config.json or pass --model PATH)\n";
        return 2;
    }

    // ---- M-Munin attached mode -----------------------------------------
    // Two things happen up-front:
    //   1. Standalone workers acquire the governor flock so a second
    //      standalone process on the same host fails fast rather than
    //      dueling over sysfs writes.
    //   2. Attached workers instead probe Munin's healthz to confirm the
    //      daemon is up and the models Munin holds cover our loadOnStart
    //      list. We refuse to start if any expected model is missing —
    //      failing at boot beats failing on the first request.
    const bool attachedMode = !args.attachSocket.empty();

#if !defined(MIMIRMIND_HAVE_L0) && !defined(MIMIRMIND_HAVE_CUDA)
    if (attachedMode) {
        std::cerr << "serve: --attach requested but this build has neither the "
                     "L0 nor the CUDA backend compiled in — attached mode needs "
                     "one of them (L0 IPC handles on Xe-LPG, or GB10 POSIX-shm "
                     "on CUDA). Rebuild with -DMIMIRMIND_ENABLE_L0=ON or "
                     "-DMIMIRMIND_ENABLE_CUDA=ON, or drop --attach.\n";
        return 2;
    }
#endif

    std::optional<::mimirmind::core::os::GovernorLock> governorLock;
    // These dev/bench hooks exit before the HTTP server starts, so they
    // need neither the thermal governor nor exclusive ownership — skip the
    // flock so they can run alongside a live serve/Munin worker (subject to
    // host memory). MIMIRMIND_L0_BATCH (the L0 synchronized batched-decode
    // parity+perf hook) is one of them: without this it would fail the
    // GovernorLock::tryAcquire while Munin holds the lock and never reach
    // the bench block below.
    const bool servingParity =
        std::getenv("MIMIRMIND_SERVING_PARITY") != nullptr ||
        std::getenv("MIMIRMIND_BATCH_BENCH")    != nullptr ||
        std::getenv("MIMIRMIND_L0_BATCH")       != nullptr;
    if (!attachedMode && !servingParity) {
        auto lk = ::mimirmind::core::os::GovernorLock::tryAcquire();
        if (!lk) {
            std::cerr << "serve: " << lk.error()
                      << "\nHint: if Munin is running, start this "
                         "worker with --attach unix:/var/run/munin/munin.sock "
                         "so it does not compete for governor ownership.\n";
            return 2;
        }
        governorLock = std::move(*lk);
        MM_LOG_INFO("main",
                    "serve: acquired governor flock at '{}'",
                    governorLock->path());
    }

    // Probed once before the per-model attach loop so a dead Munin does
    // not manifest as N confusing per-model attach errors. The whole
    // Munin/attach chain is L0-only; a HIP-only build already errored
    // out above if `--attach` was set, so this block is unreachable
    // and its `MuninClient` references would fail to compile without
    // the guard.
#if defined(MIMIRMIND_HAVE_L0) || defined(MIMIRMIND_HAVE_CUDA)
    if (attachedMode) {
        MM_LOG_INFO("main",
                    "serve: attached mode — probing Munin at '{}'",
                    args.attachSocket);
        auto hz = ::mimirmind::core::ipc::MuninClient::healthz(args.attachSocket);
        if (!hz) {
            std::cerr << "serve: Munin healthz failed at '"
                      << args.attachSocket << "': " << hz.error() << "\n";
            return 2;
        }
        if (hz->governorOwner != "munin") {
            std::cerr << "serve: refusing to attach — Munin reports "
                         "governor_owner='" << hz->governorOwner
                      << "', expected 'munin'. Standalone-worker "
                         "handoff back to Munin is not part of "
                         "Schritt 8-minimal (M-Munin ADR).\n";
            return 2;
        }
        MM_LOG_INFO("main",
                    "serve: Munin healthz ok — pid={} models={} owner={}",
                    hz->pid, hz->models.size(), hz->governorOwner);
        for (const auto& m : hz->models) {
            MM_LOG_INFO("main",
                        "  munin-model id='{}' fingerprint='{}' bytes={}",
                        m.id, m.fingerprint, m.totalBytes);
        }
    }
#endif

    // Load every loadOnStart:true model. Each gets its own InferenceEngine
    // (own L0 context, USM, autotune) — request dispatch picks the target
    // via `req.model`. Startup cost scales linearly with N (each model
    // runs its own selfTest + autotune pass), and USM is shared UMA-style
    // so N models × their footprint × ~1.2 must fit under
    // runtime.usmProbeTotalGib.
    auto applyRuntimeOverrides = [&](::mimirmind::runtime::InferenceEngine& e,
                                     const ::mimirmind::core::config::RuntimeSettings& rt) {
        if (rt.maxContextTokens.has_value() && *rt.maxContextTokens > 0) {
            e.setMaxContextTokens(*rt.maxContextTokens);
        }
        if (rt.kvDtype.has_value()) {
            const std::string_view v{*rt.kvDtype};
            if (v == "fp16")           e.setKvDtype(::mimirmind::runtime::KvDtype::FP16);
            else if (v == "q8_0")      e.setKvDtype(::mimirmind::runtime::KvDtype::Q8_0);
            else if (v == "f32" || v.empty())
                                       e.setKvDtype(::mimirmind::runtime::KvDtype::F32);
            else {
                MM_LOG_WARN("main",
                            "runtime.kvDtype='{}' unrecognised — falling "
                            "back to f32", v);
            }
        }
        // serving.kvDtype — paged serving-pool KV tier (config overrides the
        // MIMIRMIND_SERVING_KV_* env flags; unset → env/F32 fallback in the
        // serving path). Validated in Config parse to {f32,fp16,fp8}.
        if (cfg.serving.kvDtype.has_value() && !cfg.serving.kvDtype->empty()) {
            const std::string_view v{*cfg.serving.kvDtype};
            if (v == "fp8")       e.setServingKvDtype(::mimirmind::runtime::KvDtype::FP8_E4M3);
            else if (v == "fp16") e.setServingKvDtype(::mimirmind::runtime::KvDtype::FP16);
            else if (v == "f32")  e.setServingKvDtype(::mimirmind::runtime::KvDtype::F32);
            MM_LOG_INFO("main", "serving.kvDtype='{}' → paged serving-pool KV tier", v);
        }
    };

    std::vector<std::unique_ptr<::mimirmind::runtime::InferenceEngine>> ownedEngines;
    std::vector<::mimirmind::server::LoadedEngine> loadedEngines;
    // Rerank (cross-encoder) models: each owns its own compute stack. Declared
    // BEFORE the rerankers so the stacks (ctx + ops) outlive the RerankEngines
    // whose USM buffers free through those ops at teardown.
    std::vector<::mimirmind::runtime::ComputeStack> ownedRerankStacks;
    std::vector<std::unique_ptr<::mimirmind::runtime::encoder::RerankEngine>>
        ownedRerankers;
    std::vector<::mimirmind::server::LoadedReranker> loadedRerankers;
    // Embed (bi-encoder) models: same isolated-compute-stack lifetime rule as
    // the rerankers — stacks declared before the EmbedEngines they back.
    std::vector<::mimirmind::runtime::ComputeStack> ownedEmbedStacks;
    std::vector<std::unique_ptr<::mimirmind::runtime::encoder::EmbedEngine>>
        ownedEmbedders;
    std::vector<::mimirmind::server::LoadedEmbedder> loadedEmbedders;
    // Decide (8.23 System-One typed-decision) models: same isolated-compute-
    // stack lifetime rule — stacks declared before the DecideEngines they back.
    std::vector<::mimirmind::runtime::ComputeStack> ownedDecideStacks;
    std::vector<std::unique_ptr<::mimirmind::runtime::encoder::DecideEngine>>
        ownedDeciders;
    std::vector<::mimirmind::server::LoadedDecider> loadedDeciders;
    // Transcribe (Whisper-class ASR) models: same isolated-compute-stack
    // lifetime rule — stacks declared before the AudioEngines they back.
    std::vector<::mimirmind::runtime::ComputeStack> ownedTranscribeStacks;
    std::vector<std::unique_ptr<::mimirmind::runtime::audio::AudioEngine>>
        ownedTranscribers;
    std::vector<::mimirmind::server::LoadedTranscriber> loadedTranscribers;
    // Speak (Orpheus TTS) models: the acoustic InferenceEngine backbone is kept
    // alive alongside the SpeakEngine that borrows it (declared before the
    // SpeakEngines so it outlives them).
    std::vector<std::unique_ptr<::mimirmind::runtime::InferenceEngine>>
        ownedSpeakBackbones;
    std::vector<std::unique_ptr<::mimirmind::runtime::audio::SpeakEngine>>
        ownedSpeakers;
    std::vector<::mimirmind::server::LoadedSpeaker> loadedSpeakers;
    // In attached mode one client per loaded model is kept alive for the
    // whole worker run so Munin sees the peer-close (implicit detach) only at
    // shutdown. The transport is chosen at build time — L0 IPC handles on
    // Xe-LPG, GB10 POSIX-shm on CUDA — and the two client types are unrelated,
    // so they are parked type-erased.
    std::vector<std::shared_ptr<void>> attachedKeepAlive;

    // Attach engine `e` to Munin for model `m` over this build's transport and
    // materialise its WeightsMap via loadModelAttached. Returns the
    // (importer, client) keep-alive pair on success, std::nullopt (and logs)
    // on failure — the CALLER decides where the pair lives: the eager path
    // pushes it into the process-lifetime `attachedKeepAlive` vector below;
    // the M-Munin.3 pool-mode factory instead stores it inside the
    // PooledEngine slot it's building, so eviction correctly detaches THAT
    // model instead of leaking it into the process-wide vector forever. One
    // implementation shared by every attach site (chat, speak, pool
    // factory); the transport differs only in which client class is used —
    // loadModelAttached itself is backend-neutral (it takes void* chunk
    // bases).
    using AttachKeepAlive =
        std::pair<std::shared_ptr<void>, std::shared_ptr<void>>;
    auto attachEngine =
        [&](::mimirmind::runtime::InferenceEngine& e,
            const ::mimirmind::core::config::ModelEntry& m)
            -> std::optional<AttachKeepAlive> {
#if defined(MIMIRMIND_HAVE_L0) || defined(MIMIRMIND_HAVE_CUDA)
        // Pick the IPC transport at build time; the one MuninClient wire
        // implementation drives either via the IpcImporterBackend seam.
#ifdef MIMIRMIND_HAVE_L0
        auto importer =
            std::make_shared<::mimirmind::core::ipc::L0IpcImporter>(e.ctx());
#else
        auto importer = std::make_shared<::mimirmind::core::ipc::ShmIpcImporter>();
#endif
        auto client =
            std::make_shared<::mimirmind::core::ipc::MuninClient>(*importer);
        auto result = client->attach(args.attachSocket, m.id);
        if (!result) {
            std::cerr << "serve: attach for id='" << m.id
                      << "' failed: " << result.error() << "\n";
            return std::nullopt;
        }
        try {
            e.setModelIdHint(m.id);   // for the per-model profile overlay
            if (result->manifest.format == "nvfp4") {
                // GB10 shm attach of an NVFP4 checkpoint: the chunks hold the
                // raw safetensors shards; the engine reconstructs them and runs
                // the NVFP4 materialization.
                e.loadModelAttachedNvfp4(
                    m.path, m.tokenizerGguf, result->manifest,
                    std::span<void* const>{result->chunkBases});
            } else {
                e.loadModelAttached(
                    m.path, result->manifest,
                    std::span<void* const>{result->chunkBases});
            }
        } catch (const std::exception& x) {
            std::cerr << "serve: loadModelAttached('" << m.id
                      << "') failed: " << x.what() << "\n";
            return std::nullopt;
        }
        // The importer owns the imported mappings — keep it alive alongside
        // (and, being first in the pair, destroyed after) the client.
        return AttachKeepAlive{std::move(importer), std::move(client)};
#else
        (void)e;
        (void)m;
        std::cerr << "serve: attached mode is not supported in this build\n";
        return std::nullopt;
#endif
    };

    // Enumerate every compiled-in + runtime-available backend/device
    // once, up-front. Per-model config gets to pick its entry by token
    // (`models[].backend`) — enables dual-GPU deployments (target on
    // dGPU, draft on iGPU) without spawning two worker processes.
    ::mimirmind::core::backend::BackendPool backendPool;
    backendPool.discoverAll();

    // M-Munin.3 (full): hoisted up-front (normally computed after the loop,
    // see below) so the per-model loop can decide, WHILE iterating, whether
    // a given chat model is the anchor (stays eager — see the comment at its
    // registration site) or a pool candidate. Resolution logic must stay
    // identical to the post-loop copy futher down.
    const std::string defaultId = cfg.defaultModel.empty()
        ? cfg.defaultModelEntry().id
        : cfg.defaultModel;
    // M-Munin.3 (full): chat-model entries deferred to the worker-side
    // materialize/evict pool instead of eager loading, when
    // serving.modelPoolCapacity > 0 in attached mode. The DEFAULT model is
    // deliberately excluded — it stays eager so the process-wide ancillary
    // systems below (thermal guard, power monitor, fan controller, perf
    // detector, draft-model vocab check) keep a concrete anchor engine,
    // exactly as in the no-pool case. Populated inside the loop below;
    // consumed after it to build the AttachedModelProvider.
    std::vector<::mimirmind::core::config::ModelEntry> poolChatModels;

    for (const auto& m : cfg.models) {
        if (!m.loadOnStart) continue;
        ::mimirmind::core::backend::BackendKind engineKind{};
        try {
            const std::string token = m.backend.empty() ? std::string{"auto"} : m.backend;
            auto& entry = backendPool.selectByToken(token);
            engineKind  = entry.kind;
            MM_LOG_INFO("main",
                        "serve: model '{}' bound to backend '{}' via token '{}'",
                        m.id,
                        ::mimirmind::core::backend::BackendRegistry::name(entry.kind),
                        entry.token);
        } catch (const std::exception& x) {
            std::cerr << "serve: model '" << m.id
                      << "' backend='" << m.backend << "' cannot be "
                      << "resolved: " << x.what() << "\n";
            return 2;
        }
        // Rerank models take a separate path: a dense F32 cross-encoder
        // (EncoderModel + XLM-R tokenizer) behind /v1/rerank, with its own
        // isolated compute stack — not an autoregressive InferenceEngine.
        if (m.task == ::mimirmind::core::config::ModelTask::Rerank) {
            try {
                ownedRerankStacks.push_back(
                    ::mimirmind::runtime::makeComputeStack(cfg, engineKind));
                auto& stk = ownedRerankStacks.back();
                auto re = std::make_unique<
                    ::mimirmind::runtime::encoder::RerankEngine>(
                    m.path, *stk.ops, *stk.matmul);
                ::mimirmind::server::LoadedReranker lr{};
                lr.id     = m.id;
                lr.title  = m.title;
                lr.engine = re.get();
                loadedRerankers.push_back(std::move(lr));
                ownedRerankers.push_back(std::move(re));
                MM_LOG_INFO("main",
                            "serve: loaded rerank model '{}' (id='{}')",
                            m.path, m.id);
            } catch (const std::exception& x) {
                std::cerr << "serve: rerank model '" << m.id
                          << "' load failed: " << x.what() << "\n";
                return 2;
            }
            continue;
        }

        // Embed models: a dense F32 bi-encoder (EncoderModel without classifier
        // head + XLM-R tokenizer) behind /v1/embeddings, own isolated stack.
        if (m.task == ::mimirmind::core::config::ModelTask::Embed) {
            try {
                ownedEmbedStacks.push_back(
                    ::mimirmind::runtime::makeComputeStack(cfg, engineKind));
                auto& stk = ownedEmbedStacks.back();
                auto ee = std::make_unique<
                    ::mimirmind::runtime::encoder::EmbedEngine>(
                    m.path, *stk.ops, *stk.matmul);
                ::mimirmind::server::LoadedEmbedder le{};
                le.id     = m.id;
                le.title  = m.title;
                le.engine = ee.get();
                loadedEmbedders.push_back(std::move(le));
                ownedEmbedders.push_back(std::move(ee));
                MM_LOG_INFO("main",
                            "serve: loaded embed model '{}' (id='{}')",
                            m.path, m.id);
            } catch (const std::exception& x) {
                std::cerr << "serve: embed model '" << m.id
                          << "' load failed: " << x.what() << "\n";
                return 2;
            }
            continue;
        }

        // Decide models (8.23 System-One): the same bge-m3 encoder as an embed
        // model plus trained decision heads, behind /v1/decide, on its own
        // isolated compute stack — same lifetime rule as the embed branch.
        if (m.task == ::mimirmind::core::config::ModelTask::Decide) {
            try {
                ownedDecideStacks.push_back(
                    ::mimirmind::runtime::makeComputeStack(cfg, engineKind));
                auto& stk = ownedDecideStacks.back();
                auto de = std::make_unique<
                    ::mimirmind::runtime::encoder::DecideEngine>(
                    m.path, m.decideHeadsDir, *stk.ops, *stk.matmul);
                ::mimirmind::server::LoadedDecider ld{};
                ld.id     = m.id;
                ld.title  = m.title;
                ld.engine = de.get();
                MM_LOG_INFO("main",
                            "serve: loaded decide model '{}' (id='{}', {} heads)",
                            m.path, m.id, de->headCount());
                loadedDeciders.push_back(std::move(ld));
                ownedDeciders.push_back(std::move(de));
            } catch (const std::exception& x) {
                std::cerr << "serve: decide model '" << m.id
                          << "' load failed: " << x.what() << "\n";
                return 2;
            }
            continue;
        }

        // Transcribe models: a Whisper-class encoder-decoder ASR checkpoint
        // (AudioEngine = WhisperModel + tokenizer + WhisperRunner) behind
        // /v1/audio/transcriptions, on its own isolated compute stack — same
        // pattern as the rerank/embed paths.
        if (m.task == ::mimirmind::core::config::ModelTask::Transcribe) {
            try {
                ownedTranscribeStacks.push_back(
                    ::mimirmind::runtime::makeComputeStack(cfg, engineKind));
                auto& stk = ownedTranscribeStacks.back();
                auto ae = std::make_unique<
                    ::mimirmind::runtime::audio::AudioEngine>(
                    m.path, *stk.ops, *stk.matmul);
                ::mimirmind::server::LoadedTranscriber lt{};
                lt.id     = m.id;
                lt.title  = m.title;
                lt.engine = ae.get();
                loadedTranscribers.push_back(std::move(lt));
                ownedTranscribers.push_back(std::move(ae));
                MM_LOG_INFO("main",
                            "serve: loaded transcribe model '{}' (id='{}')",
                            m.path, m.id);
            } catch (const std::exception& x) {
                std::cerr << "serve: transcribe model '" << m.id
                          << "' load failed: " << x.what() << "\n";
                return 2;
            }
            continue;
        }

        // Speak (Orpheus TTS) models: a Llama-3.2 acoustic InferenceEngine plus
        // the SNAC codec decoder (SpeakEngine) behind /v1/audio/speech. The
        // backbone loads like a chat model (attached / NVFP4 / GGUF) but is a
        // generate() consumer only (no serving batcher). Requires a `codec`
        // path = the converted SNAC-24kHz safetensors (scripts/convert-snac.py).
        if (m.task == ::mimirmind::core::config::ModelTask::Speak) {
            if (m.codecPath.empty()) {
                std::cerr << "serve: speak model '" << m.id
                          << "' requires a 'codec' path (converted SNAC "
                             "safetensors)\n";
                return 2;
            }
            try {
                auto e = std::make_unique<::mimirmind::runtime::InferenceEngine>(
                    cfg, engineKind);
                if (attachedMode) {
                    auto ka = attachEngine(*e, m);
                    if (!ka) {
                        return 2;
                    }
                    attachedKeepAlive.push_back(std::move(ka->first));
                    attachedKeepAlive.push_back(std::move(ka->second));
                } else {
                    e->setModelIdHint(m.id);   // for the per-model profile overlay
                    if (runtime::nvfp4::resolveModelFormat(m.format, m.path)
                        == core::config::ModelFormat::Nvfp4) {
                        e->loadModelNvfp4(m.path, m.tokenizerGguf);
                    } else {
                        e->loadModel(m.path);
                    }
                }
                auto se = std::make_unique<
                    ::mimirmind::runtime::audio::SpeakEngine>(*e, m.codecPath);
                ::mimirmind::server::LoadedSpeaker ls{};
                ls.id     = m.id;
                ls.title  = m.title;
                ls.engine = se.get();
                loadedSpeakers.push_back(std::move(ls));
                ownedSpeakers.push_back(std::move(se));
                ownedSpeakBackbones.push_back(std::move(e));
                MM_LOG_INFO("main",
                            "serve: loaded speak model '{}' (id='{}', codec='{}')",
                            m.path, m.id, m.codecPath);
            } catch (const std::exception& x) {
                std::cerr << "serve: speak model '" << m.id
                          << "' load failed: " << x.what() << "\n";
                return 2;
            }
            continue;
        }

        // M-Munin.3 (full): non-default chat models defer to the pool
        // instead of eager-loading here. See `poolChatModels`'s comment
        // above for why the default is excluded from this.
        if (attachedMode && cfg.serving.modelPoolCapacity > 0 &&
            m.id != defaultId) {
            poolChatModels.push_back(m);
            MM_LOG_INFO("main",
                        "serve: model '{}' registered with the pool "
                        "(lazy materialize, capacity={})",
                        m.id, cfg.serving.modelPoolCapacity);
            continue;
        }

        auto e = std::make_unique<::mimirmind::runtime::InferenceEngine>(
            cfg, engineKind);

        if (attachedMode) {
            MM_LOG_INFO("main",
                        "serve: attaching to Munin for model '{}' "
                        "(local header from '{}')", m.id, m.path);
            auto ka = attachEngine(*e, m);
            if (!ka) {
                return 2;
            }
            attachedKeepAlive.push_back(std::move(ka->first));
            attachedKeepAlive.push_back(std::move(ka->second));
        } else {
            e->setModelIdHint(m.id);   // for the per-model profile overlay
            if (runtime::nvfp4::resolveModelFormat(m.format, m.path)
                == core::config::ModelFormat::Nvfp4) {
                MM_LOG_INFO("main", "serve: loading NVFP4 model '{}' (id='{}')",
                            m.path, m.id);
                e->loadModelNvfp4(m.path, m.tokenizerGguf);
            } else {
                MM_LOG_INFO("main", "serve: loading model '{}' (id='{}')",
                            m.path, m.id);
                e->loadModel(m.path);
            }
        }

        const auto& arch = e->config().architecture;
        if (arch != "qwen2" && arch != "llama" && arch != "gemma4" &&
            arch != "qwen35moe" && arch != "qwen4_exp") {
            const std::string msg =
                "serve: architecture '" + arch + "' (model id '" + m.id +
                "') is not implemented yet. See "
                "Memory/mimirmind/research/m8-gemma4-staging.md.";
            MM_LOG_ERROR("main", "{}", msg);
            std::cerr << msg << "\n";
            return 2;
        }
        // setKvDtype + setMaxContextTokens inspect loaded model state
        // (fused-QKV coverage, attn_k/v.bias presence per block), so
        // apply the per-model runtime overrides AFTER loadModel.
        applyRuntimeOverrides(*e, cfg.effectiveRuntime(m.id));
        if (const auto benchExit =
                ServingBenchModes::maybeRun(*e, arch, cfg)) {
            return *benchExit;
        }
        // M9.8b — cross-block sanity check on the effective runtime.
        // The plain-attention fallback in kernels/attention.cl holds
        // scores[ATTN_MAX_TK] in 64 KiB SLM, so if a caller forces the
        // plain path (features.prefillFlash: false) at a context length
        // above kAttentionMaxTk, the very first request will throw
        // deep in GpuOps::attentionPlainAsync. Catch that combination
        // at startup so the operator sees a clear message during boot,
        // not a stack trace during the first prod-facing request.
        {
            const auto effMaxCtx = e->maxContextTokens();
            // The plain-attention SLM-cap check is L0-only — the HIP
            // backend has no plain-attention path (flash is the only
            // impl there).
#ifdef MIMIRMIND_HAVE_L0
            if (effMaxCtx > ::mimirmind::compute::l0::GpuOps::kAttentionMaxTk
                && !cfg.features.flashPrefill) {
                const std::string msg =
                    "serve: model '" + m.id + "' has effective "
                    "runtime.maxContextTokens=" + std::to_string(effMaxCtx) +
                    " > kAttentionMaxTk=" +
                    std::to_string(
                        ::mimirmind::compute::l0::GpuOps::kAttentionMaxTk) +
                    " while features.prefillFlash=false — the "
                    "plain-attention fallback cannot hold "
                    "scores[ATTN_MAX_TK] in SLM at that context "
                    "length. Set features.prefillFlash=true (default) "
                    "OR reduce runtime.maxContextTokens below " +
                    std::to_string(
                        ::mimirmind::compute::l0::GpuOps::kAttentionMaxTk) + ".";
                MM_LOG_ERROR("main", "{}", msg);
                std::cerr << msg << "\n";
                return 2;
            }
#else
            (void)effMaxCtx;
#endif
            // Informational warn — long context + wide KV storage
            // pressures a 24 GiB DRAM host running Gemma 4 26B-A4B
            // weights (~22 GiB) alongside the KV cache. Rough per-token
            // KV size at F32 is ~430 KiB across all 30 layers; Q8_0 is
            // ~4× smaller. This is a warning, not an error — smaller
            // architectures (E4B / dense 4B) fit F32 KV comfortably.
            if (effMaxCtx > 24576
                && e->kvDtype() == ::mimirmind::runtime::KvDtype::F32) {
                MM_LOG_WARN("main",
                            "runtime.maxContextTokens={} for model '{}' "
                            "with kvDtype=f32: the KV cache will consume "
                            "several GiB on Gemma-4-class geometries. "
                            "Consider kvDtype=q8_0 or kvDtype=fp16 on "
                            "shared-24 GiB hosts.",
                            effMaxCtx, m.id);
            }
        }

        const auto d = e->kvDtype();
        const char* dName = (d == ::mimirmind::runtime::KvDtype::FP16 ? "fp16"
                           : d == ::mimirmind::runtime::KvDtype::Q8_0 ? "q8_0"
                           : d == ::mimirmind::runtime::KvDtype::FP8_E4M3 ? "fp8_e4m3"
                                                                      : "f32");
        MM_LOG_INFO("main",
                    "KV cache dtype for '{}': {} (block {} B × {} elem)",
                    m.id, dName,
                    ::mimirmind::runtime::kvBlockBytes(d),
                    ::mimirmind::runtime::kvBlockElements(d));

        ::mimirmind::server::LoadedEngine le{};
        le.id     = m.id;
        le.title  = m.title;
        le.engine = e.get();
        loadedEngines.push_back(std::move(le));
        ownedEngines.push_back(std::move(e));
    }

    if (ownedEngines.empty()) {
        std::cerr << "serve: no model with loadOnStart:true in config.json — "
                     "nothing to serve\n";
        return 2;
    }

    // The default engine drives all the per-process ancillaries below
    // (thermal guard, power monitor, governor, fan, perf-regression).
    // Additional engines share those same monitors transparently — the
    // hooks are stateless getters that any engine's generate() consults.
    // (`defaultId` itself is computed up-front, before the loop — see there.)
    ::mimirmind::runtime::InferenceEngine* defaultEnginePtr = nullptr;
    for (auto& le : loadedEngines) {
        if (le.id == defaultId) { defaultEnginePtr = le.engine; break; }
    }
    if (defaultEnginePtr == nullptr) {
        std::cerr << "serve: defaultModel='" << defaultId
                  << "' has no loadOnStart:true entry\n";
        return 2;
    }
    auto& engine = *defaultEnginePtr;
    // Re-use the effective runtime for the DEFAULT model when reporting
    // to the user later on (preserve_thinking flag).
    const auto effRuntime = cfg.effectiveRuntime(defaultId);

    // Optional speculative-decoding drafter (see ServerBootstrap). Held as a
    // local so `draftEngine`/`drafter` outlive ApiServer::run below; the
    // references keep the names the wiring further down already uses.
    auto  speculative = ServerBootstrap::buildSpeculative(cfg, engine);
    auto& draftEngine = speculative.draftEngine;
    auto& drafter     = speculative.drafter;

    ::mimirmind::server::ServerConfig scfg{};
    scfg.host    = "0.0.0.0";
    scfg.port    = args.port.value_or(static_cast<std::uint16_t>(cfg.server.port));
    // defaultModelId in the ServerConfig picks the fallback engine when a
    // request omits `model`. Must match one of the loaded engine ids —
    // computed above as `defaultId`.
    scfg.modelId = defaultId;
    scfg.preserveThinking = effRuntime.preserveThinking.value_or(false);
    scfg.speculative.enabled  = cfg.speculative.enabled;
    scfg.speculative.draftN   = static_cast<std::size_t>(cfg.speculative.n);
    scfg.speculativeTargetId  = cfg.speculative.target;

    // Zero-config security provisioning: in-process TLS + bearer auth, both
    // auto-provisioned (self-signed cert / auto-generated key) so a bare
    // `serve` still comes up secure. Files land next to the loaded config.
    const std::filesystem::path secDir = securityDir(args.configPath);
    if (!provisionTls(cfg.server.tls, scfg.host, secDir, scfg.tls)) {
        std::cerr << "serve: TLS is enabled but no usable certificate could "
                     "be provisioned — aborting.\n";
        return 1;
    }
    provisionAuth(cfg.server.auth, scfg.host, secDir, scfg.auth);

    // Per-tenant usage metrics (per-API-key token/request accounting for the
    // admin routes). Persisted next to the config so totals survive a restart;
    // an explicit path overrides. Empty path => in-memory only.
    if (cfg.server.metrics.enabled) {
        scfg.tenantMetricsPath =
            cfg.server.metrics.path.empty()
                ? (secDir / "mimir-tenant-metrics.json").string()
                : cfg.server.metrics.path;
    }

    // Thermal profile lives inline in config.json under governor.thermal.
    // Empty `name` means "no profile" and the guard runs unprotected.
    const bool hasThermalProfile = !cfg.governor.thermal.name.empty() ||
                                   cfg.governor.thermal.hasPackageLimits();

    std::unique_ptr<::mimirmind::runtime::SystemMonitor> monitor;
    std::unique_ptr<::mimirmind::runtime::ThermalGuard>  guard;
    // In attached mode Munin drives every sysfs-WRITE regulator —
    // GpuClockGovernor + FanController. Per M-Munin ADR "Governor —
    // Sonderregel" the worker MUST NOT install those. SystemMonitor
    // and ThermalGuard are read-only (sensor read + local pacing
    // decision), so the worker installs them in both modes. That
    // keeps /v1/system/status (package temp, RAM, throttle state) alive
    // for the pegenaut dashboard and lets each worker back off decode
    // based on its own thermal reading, which is belt-and-suspenders
    // to Munin's authoritative clock cap.
    if (hasThermalProfile) {
        const ::mimirmind::runtime::ThermalProfile& profile = cfg.governor.thermal;
        try {
            monitor = std::make_unique<::mimirmind::runtime::SystemMonitor>(
                /*requirePackageTemp=*/profile.hasPackageLimits(),
                /*requireRam=*/        false);
        } catch (const std::exception& e) {
            if (attachedMode) {
                // Non-fatal in attached mode: Munin still runs its own
                // regulators. Losing the local telemetry hurts the
                // dashboard but should not refuse the worker boot.
                std::cerr << "serve: attached mode — SystemMonitor sensor "
                             "probe failed (" << e.what() << "); "
                             "continuing without local thermal telemetry\n";
            } else {
                std::cerr << "serve: profile '" << profile.name
                          << "' requires sensors the host does not expose: "
                          << e.what() << "\n";
                return 1;
            }
        }
        if (monitor) {
            guard = std::make_unique<::mimirmind::runtime::ThermalGuard>(
                profile, *monitor);
            engine.setThermalGuard(guard.get());
        }
        if (attachedMode) {
            MM_LOG_INFO("main",
                        "serve: attached mode — SystemMonitor + ThermalGuard "
                        "installed (read-only); GpuClockGovernor / "
                        "FanController skipped (Munin owns the sysfs writes)");
        }
    }
    if (!attachedMode && hasThermalProfile) {
        const ::mimirmind::runtime::ThermalProfile& profile = cfg.governor.thermal;

        // GPU clock governor lives in the same profile (field
        // gpu_target_temp_c). If present AND the iGPU sysfs is
        // writable, we install it. Otherwise we move on without one —
        // the per-token thermal pace still runs as a safety net.
        //
        // `governor.gpuClockPin` pins the software cap for the whole
        // session and suppresses the P-controller tick. Meant for
        // perf-bench runs where the M9.6.5 asymmetric gains would
        // otherwise clock down aggressively and confound the
        // measurement. Package thermal safety still runs via the
        // ThermalGuard admission check + per-token pace. Do NOT ship
        // this to sustained workloads on a passively-cooled chassis.
        //
        // Accepted values (from config.json):
        //   "rp0"            → hardware max (RP0)
        //   "rpn"            → hardware min (RPn, ~800 MHz on Xe-LPG)
        //   "<MHz integer>"  → arbitrary cap, clamped to [RPn, RP0]
        //   null / "0" / "off" / "false" / "no" → no pin (governor ticks as normal)
        enum class ClockPinIntent { None, Rp0, Rpn, Numeric };
        struct ClockPinRequest {
            ClockPinIntent intent = ClockPinIntent::None;
            std::uint32_t  mhz    = 0;
            std::string    rawEnv;
            bool           malformed = false;
        };
        const auto parseClockPin = [](std::string_view sv) {
            ClockPinRequest req;
            if (sv.empty()) {
                return req;
            }
            if (sv == "0" || sv == "off" || sv == "false" || sv == "no") {
                return req; // treat as unset
            }
            req.rawEnv = std::string{sv};
            if (sv == "rp0" || sv == "RP0") {
                req.intent = ClockPinIntent::Rp0;
                return req;
            }
            if (sv == "rpn" || sv == "RPn" || sv == "RPN") {
                req.intent = ClockPinIntent::Rpn;
                return req;
            }
            char* end = nullptr;
            const std::string zSv{sv};
            const unsigned long v = std::strtoul(zSv.c_str(), &end, 10);
            if (end != zSv.c_str() && *end == '\0' && v > 0 && v < 100000) {
                req.intent = ClockPinIntent::Numeric;
                req.mhz    = static_cast<std::uint32_t>(v);
                return req;
            }
            req.malformed = true;
            return req;
        };

        static std::unique_ptr<::mimirmind::runtime::GpuClockGovernor> governor;
        const auto pinReq = parseClockPin(
            cfg.governor.gpuClockPin.value_or(""));
        const bool clockPinRequested = pinReq.intent != ClockPinIntent::None;

        if (profile.hasGpuClockTarget()) {
            governor = std::make_unique<::mimirmind::runtime::GpuClockGovernor>();
            governor->setTargetTempC(*profile.gpu_target_temp_c);
            if (!governor->available()) {
                MM_LOG_WARN("main",
                            "thermal profile asks for GPU clock governor "
                            "(gpu_target_temp_c={:.1f}) but it is not "
                            "available: {}",
                            *profile.gpu_target_temp_c,
                            governor->unavailableReason());
                governor.reset();
            } else if (clockPinRequested) {
                std::uint32_t   requestedMhz = 0;
                std::string_view intentName  = "";
                switch (pinReq.intent) {
                    case ClockPinIntent::Rp0:
                        requestedMhz = governor->rp0Mhz();
                        intentName   = "rp0";
                        break;
                    case ClockPinIntent::Rpn:
                        requestedMhz = governor->rpnMhz();
                        intentName   = "rpn";
                        break;
                    case ClockPinIntent::Numeric:
                        requestedMhz = pinReq.mhz;
                        intentName   = "numeric";
                        break;
                    case ClockPinIntent::None:
                        break;
                }
                const auto pinned = governor->pin(
                    requestedMhz, intentName, pinReq.rawEnv);
                MM_LOG_WARN("main",
                            "governor.gpuClockPin={} — cap pinned to "
                            "{} MHz (intent={}, envelope [{},{}]). "
                            "P-controller tick suppressed. Bench mode. "
                            "Thermal safety still via ThermalGuard.",
                            pinReq.rawEnv, pinned, intentName,
                            governor->rpnMhz(), governor->rp0Mhz());
                // Install the governor anyway so ApiServer can report
                // the pin state via /system/info + /system/status. The
                // engine's tick loop consults governor->pinned() and
                // skips its adjust call, so the pin survives the run.
                engine.setGpuClockGovernor(governor.get(), monitor.get());
            } else {
                if (pinReq.malformed) {
                    MM_LOG_WARN("main",
                                "governor.gpuClockPin={} not recognised — "
                                "expected rp0 / rpn / <MHz> / 0 / off. "
                                "Installing governor as if unset.",
                                pinReq.rawEnv);
                }
                engine.setGpuClockGovernor(governor.get(), monitor.get());
            }
        } else if (clockPinRequested) {
            MM_LOG_WARN("main",
                        "governor.gpuClockPin={} ignored — thermal "
                        "profile has no gpu_target_temp_c so no governor "
                        "was going to be installed anyway.",
                        pinReq.rawEnv);
        }

        // M9.6.6.0 tick sink. `governor.tickLog:true` gates the sink;
        // `governor.tickLogFile` names the NDJSON output. For one
        // release we still honour `diagnostics.traceDecodeFile` as the
        // path when tickLogFile is unset — that reuse conflates the
        // decode-trace and governor-tick streams and is being retired.
        if (governor != nullptr && cfg.governor.tickLog) {
            std::string tickPath = cfg.governor.tickLogFile;
            bool viaDeprecated  = false;
            if (tickPath.empty() && !cfg.diagnostics.traceDecodeFile.empty()) {
                tickPath      = cfg.diagnostics.traceDecodeFile;
                viaDeprecated = true;
            }
            if (tickPath.empty()) {
                MM_LOG_WARN("main",
                            "governor.tickLog:true but governor.tickLogFile "
                            "is empty — sink stays off. Set "
                            "governor.tickLogFile to a writable path.");
            } else if (governor->setTickLogPath(tickPath)) {
                if (viaDeprecated) {
                    MM_LOG_WARN("main",
                                "GovernorTickSink using deprecated "
                                "diagnostics.traceDecodeFile='{}' as its path "
                                "— move to governor.tickLogFile in config.json "
                                "before the next release.",
                                tickPath);
                }
                MM_LOG_INFO("main",
                            "GovernorTickSink open — writing NDJSON to '{}'",
                            tickPath);
            } else {
                MM_LOG_WARN("main",
                            "governor.tickLog set with path '{}' — "
                            "could not open for append. Sink stays off.",
                            tickPath);
            }
        }
    }

    // Power telemetry — always-on attempt, never fatal. If RAPL is
    // masked (Docker / unprivileged LXC without explicit mount) the
    // monitor reports unavailable and /v1/system/status shows the
    // reason; the engine still runs.
    auto powerMonitor = std::make_unique<::mimirmind::runtime::PowerMonitor>();
    engine.setPowerMonitor(powerMonitor.get());

    // M9.11.b chassis fan controller. Probes /sys/class/hwmon/* at
    // construction; if a writable pwm/pwm_enable pair is found, the
    // engine boosts the fan at the start of each generate() and
    // releases to auto at the end. Original BIOS values captured for
    // RAII restore on process exit.
    //
    // Config knobs:
    //   governor.fan.boost:    false → do not install (kill switch)
    //   governor.fan.pwmBoost: 0-255 override boost target
    //   governor.fan.pwmMin:   0-255 override safety floor
    // Kill switch is checked first so we can disable the whole feature
    // without touching sysfs at all — useful when the BIOS refuses
    // manual mode and hwmon writes are throwing kernel warnings.
    static std::unique_ptr<::mimirmind::runtime::FanController> fanController;
    {
        // Attached-mode workers never touch the fan (see M-Munin ADR
        // Governor-Sonderregel). Munin owns the fan install; the worker
        // just runs generate() and lets Munin cool the chassis.
        const bool disabled = !cfg.governor.fan.boost || attachedMode;
        if (!disabled) {
            fanController = std::make_unique<::mimirmind::runtime::FanController>();
            if (!fanController->available()) {
                MM_LOG_WARN("main",
                            "FanController unavailable — no proactive fan "
                            "boost. Reason: {}",
                            fanController->unavailableReason());
                fanController.reset();
            } else {
                if (const auto v = cfg.governor.fan.pwmBoost;
                    v.has_value() && *v >= 0 && *v <= 255) {
                    fanController->setBoostPwm(
                        static_cast<std::uint8_t>(*v));
                }
                if (const auto v = cfg.governor.fan.pwmMin;
                    v.has_value() && *v >= 0 && *v <= 255) {
                    fanController->setMinSafePwm(
                        static_cast<std::uint8_t>(*v));
                }
                MM_LOG_INFO("main",
                            "FanController ready — chip='{}' pwm='{}' "
                            "fan_input='{}' orig_pwm={} orig_enable={} "
                            "boost={} min_safe={}",
                            fanController->chipName(),
                            fanController->pwmPath(),
                            fanController->fanInputPath(),
                            fanController->originalPwm(),
                            fanController->originalEnableMode(),
                            fanController->boostPwm(),
                            fanController->minSafePwm());
                engine.setFanController(fanController.get());
            }
        }
    }

    // Thermal-safety cross-check: governor.gpuClockPin=rp0 disables
    // the P-controller entirely, so the FanController is the only
    // active thermal regulator during sustained decode. Warn loudly
    // when the operator asks for rp0 without a functioning fan-boost
    // path — this is the exact 2026-07-01 shutdown scenario.
    {
        const std::string_view pinSv =
            cfg.governor.gpuClockPin.has_value()
                ? std::string_view{*cfg.governor.gpuClockPin}
                : std::string_view{};
        const bool  wantsRp0 = (pinSv == "rp0" || pinSv == "RP0");
        const bool fanActive =
            fanController != nullptr && fanController->available();
        if (wantsRp0 && !fanActive) {
            MM_LOG_WARN("main",
                        "governor.gpuClockPin=rp0 is set but the "
                        "FanController is not active — the P-controller "
                        "is disabled AND no proactive cooling is "
                        "installed. Sustained decode on a passively "
                        "cooled chassis can trigger a hardware thermal "
                        "shutdown (see 2026-07-01 incident). Consider "
                        "governor.gpuClockPin=<numeric MHz> as a "
                        "safer bench mode.");
        }
    }

    // In-process perf-regression detector. Feeds off the same per-token
    // wall-time the NDJSON sink already computes, so it costs a couple
    // of doubles per token and one median at end-of-run. Kill-switch:
    // `diagnostics.regressionAlert: false` in config.json skips the
    // installer entirely for the case where the detector itself
    // misbehaves and needs to be silenced without a redeploy.
    std::unique_ptr<::mimirmind::runtime::PerfRegressionDetector> perfDetector;
    {
        const bool disabled = !cfg.diagnostics.regressionAlert;
        if (!disabled) {
            std::string baselinePath;
            if (const char* h = std::getenv("HOME"); h != nullptr && h[0] != '\0') {
                baselinePath = std::string{h} +
                               "/.cache/mimirmind/perf-baseline.json";
            } else {
                baselinePath = "/tmp/mimirmind-perf-baseline.json";
            }
            std::error_code ec;
            std::filesystem::create_directories(
                std::filesystem::path{baselinePath}.parent_path(), ec);
            // create_directories failure is not fatal — the detector
            // logs a warning on the first write and keeps running.
            perfDetector =
                std::make_unique<::mimirmind::runtime::PerfRegressionDetector>(
                    baselinePath);
            engine.setPerfRegressionDetector(perfDetector.get());
        } else {
            MM_LOG_WARN("main",
                        "diagnostics.regressionAlert=false — perf-regression "
                        "detector not installed for this session");
        }
    }

    // Propagate the process-wide ancillary monitors from the default
    // engine to any extras so their generate() paths also honour
    // thermal admission, RAPL joule accounting, fan-boost pre-warm and
    // perf-regression sampling. Governor propagation is intentionally
    // skipped — its per-tick control loop is process-scoped and driven
    // by the default engine; extras would fight for the same GPU cap.
    for (auto& e : ownedEngines) {
        if (e.get() == defaultEnginePtr) continue;
        if (auto* g = engine.thermalGuard())            e->setThermalGuard(g);
        if (auto* p = engine.powerMonitor())            e->setPowerMonitor(p);
        if (auto* d = engine.perfRegressionDetector()) e->setPerfRegressionDetector(d);
        if (auto* fc = engine.fanController())          e->setFanController(fc);
    }

    // M-Cuda.Batch D2e.2 — continuous-batching worker for the default
    // (serving-class) engine. Built only for qwen35moe once the startup
    // BatchCapacityProbe has recommended serving-class (sustainableBatch
    // >= min). Requests to the default engine are then serviced through the
    // batcher's worker thread (multi-tenant continuous batching) rather than
    // the serialised single-session generate() path. Kept alive for the
    // whole server.run() below; its dtor joins the worker on shutdown.
    // M9.1 — un-gated from qwen35moe-only to any backend that implements the
    // neutral synchronized batched decode: Gemma 4 MoE on L0/Xe-LPG serves
    // through the non-paged slab substrate (SlabDecodeStepper) instead of the
    // paged pool. qwen35moe keeps its CUDA paged path.
    // ---- 5.27.10 fix: main-thread warmup of process-global lazy CUDA state,
    // in particular the CUTLASS NVFP4-TC grouped GEMM ("nvfp4-tc-banks").
    // Serve requests run on the batcher _worker / httplib pool threads. If the
    // FIRST process-wide touch of the CUTLASS grouped kernel (its lazy
    // initialize()) happens on such a worker thread — which never called
    // cudaSetDevice and does not own the CudaContext — it returns kErrorInternal
    // (rc=2) and poisons the context, crashing the first request intermittently
    // (~60%, _Exit(70)). Forcing one tiny single-session generate() HERE — on
    // the main thread that constructed the CudaContext, before any batcher
    // worker exists — establishes that lazy state on the context-owning thread,
    // so the workers only ever re-use the already-initialized kernel. Guarded so
    // a warmup failure can never block serving; effectively a no-op for archs
    // that don't drive the TC-banks path.
    if (engine.servingClassEnabled()) {
        try {
            const auto& wtok = engine.tokenizer();
            auto warmIds = wtok.encode("Hi", /*addBos=*/true);
            if (warmIds.empty()) {
                warmIds.push_back(wtok.eosId());
            }
            ::mimirmind::runtime::GenerateParams wgp{};
            wgp.maxNewTokens         = 1;
            wgp.sampling.temperature = 0.0F;
            engine.resetCache();
            (void)engine.generate(warmIds, wgp, {}, nullptr, {}, {});
            engine.resetCache();
            MM_LOG_INFO("main",
                        "serve: main-thread kernel warmup done (default engine "
                        "'{}') — nvfp4-tc-banks initialized on the context "
                        "thread (5.27.10)",
                        defaultId);
        } catch (const std::exception& warmEx) {
            MM_LOG_WARN("main",
                        "serve: main-thread kernel warmup failed ({}); serving "
                        "continues (worker first-init may still race — 5.27.10)",
                        warmEx.what());
        }
    }

    std::unique_ptr<::mimirmind::runtime::serving::ContinuousBatcher> batcher;
    // 5.27.11.2: qwen4_exp re-enabled for the continuous batcher — the paged
    // stepServing (decode) and prefillSlot (single-slot prefill) forward now carry
    // per-slot PLE n-gram + the HC stream collapse, and the batcher forces
    // single-slot prefill for qwen4_exp (no ragged varlen). conc>1 batched forward
    // is slot-correct + coherent (5.27.11.2).
    if ((engine.config().architecture == "qwen35moe" ||
         engine.config().architecture == "qwen4_exp" ||
         engine.supportsBatchedDecode()) &&
        engine.servingClassEnabled()) {
        std::size_t maxBatch =
            std::max<std::size_t>(1, engine.batchCapacity().sustainableBatch);
        // Operator override of the batcher's slot count. The v1 BatchCapacity
        // probe is a coarse bandwidth-tier proxy (Xe-LPG ~70 GB/s pins it to
        // 1) — not a real device-memory calc — so an operator who knows the
        // model fits more concurrent slots can set MIMIRMIND_SERVING_MAXBATCH.
        if (const char* mb = std::getenv("MIMIRMIND_SERVING_MAXBATCH")) {
            const long v = std::atol(mb);
            if (v > 0) {
                maxBatch = static_cast<std::size_t>(v);
                MM_LOG_INFO("main",
                            "serve: MIMIRMIND_SERVING_MAXBATCH override — "
                            "batcher maxBatch={} (probe said {})",
                            maxBatch, engine.batchCapacity().sustainableBatch);
            }
        }
        const std::size_t maxContext = engine.maxContextTokens();
        // Total accepted-but-unfinished cap (running + queued). Beyond it the
        // batcher sheds load with a 503 instead of an unbounded queue.
        const std::size_t maxInflight = cfg.serving.maxActiveRequests;
        // Per-tenant fairness cap (0 = off). Keeps one API-key tenant from
        // eating the whole maxInflight budget and starving co-tenants.
        const std::size_t maxInflightPerTenant =
            cfg.serving.maxActiveRequestsPerTenant;
        try {
            batcher = std::make_unique<
                ::mimirmind::runtime::serving::ContinuousBatcher>(
                engine, maxBatch, maxContext, engine.tokenizer().eosId(),
                maxInflight, maxInflightPerTenant);
            scfg.batcher = batcher.get();
            MM_LOG_INFO("main",
                        "serve: continuous batcher ENABLED for default engine "
                        "'{}' (maxBatch={} maxContext={} maxInflight={} "
                        "maxInflightPerTenant={})",
                        defaultId, maxBatch, maxContext,
                        batcher->maxInflight(),
                        batcher->maxInflightPerTenant());
        } catch (const std::exception& e) {
            MM_LOG_WARN("main",
                        "serve: continuous batcher init failed ({}); falling "
                        "back to single-session generate()", e.what());
            batcher.reset();
            scfg.batcher = nullptr;
        }
    }

    // M-Munin.3 (full): worker-side materialize/evict pool for the
    // non-default chat models registered above (`poolChatModels`). Builds a
    // real AttachedModelProvider whose factory attaches to Munin,
    // materializes, and — mirroring the default engine's setup above —
    // wires this slot's OWN continuous batcher and, if it's the configured
    // speculative.target, its OWN spec-dec decoder sharing the default's
    // drafter. See decisions/2026-08-22-m-munin3-per-request-model-switch.md.
    std::unique_ptr<::mimirmind::server::AttachedModelProvider> modelProvider;
    if (!poolChatModels.empty()) {
        std::vector<::mimirmind::server::ProvidedModel> provided;
        provided.reserve(poolChatModels.size());
        for (const auto& m : poolChatModels) {
            provided.push_back({m.id, m.title});
        }
        // Captured by value/pointer: cfg and backendPool outlive the
        // server (stack frame of runServe, same as attachEngine itself,
        // which is captured by reference — both live for the whole
        // process). The ancillary monitor pointers and the shared drafter
        // are snapshotted now (constructed once, above, and never rebuilt).
        auto factory =
            [&cfg, &backendPool, &attachEngine, &scfg,
             thermalGuardPtr  = engine.thermalGuard(),
             powerMonitorPtr  = engine.powerMonitor(),
             fanControllerPtr = engine.fanController(),
             perfDetectorPtr  = engine.perfRegressionDetector(),
             drafterPtr       = drafter.get()]
            (const std::string& modelId)
                -> std::unique_ptr<::mimirmind::server::PooledEngine> {
            const ::mimirmind::core::config::ModelEntry* modelEntry = nullptr;
            for (const auto& mm : cfg.models) {
                if (mm.id == modelId) { modelEntry = &mm; break; }
            }
            if (modelEntry == nullptr) {
                throw std::runtime_error(
                    "M-Munin.3 pool: model '" + modelId +
                    "' not found in config");
            }
            const auto& m = *modelEntry;

            const std::string token =
                m.backend.empty() ? std::string{"auto"} : m.backend;
            auto& backendEntry = backendPool.selectByToken(token);

            auto payload = std::make_unique<::mimirmind::server::PooledEngine>();
            payload->engine = std::make_unique<::mimirmind::runtime::InferenceEngine>(
                cfg, backendEntry.kind);
            payload->title = m.title;

            auto ka = attachEngine(*payload->engine, m);
            if (!ka) {
                throw std::runtime_error(
                    "M-Munin.3 pool: attach failed for model '" + modelId + "'");
            }
            payload->keepAliveImporter = std::move(ka->first);
            payload->keepAliveClient   = std::move(ka->second);

            auto& e = *payload->engine;

            // Same per-model runtime overrides as the eager path.
            const auto rt = cfg.effectiveRuntime(m.id);
            if (rt.maxContextTokens.has_value() && *rt.maxContextTokens > 0) {
                e.setMaxContextTokens(*rt.maxContextTokens);
            }
            if (rt.kvDtype.has_value()) {
                const std::string_view v{*rt.kvDtype};
                if (v == "fp16")      e.setKvDtype(::mimirmind::runtime::KvDtype::FP16);
                else if (v == "q8_0") e.setKvDtype(::mimirmind::runtime::KvDtype::Q8_0);
                else if (v == "f32" || v.empty())
                                      e.setKvDtype(::mimirmind::runtime::KvDtype::F32);
            }
            if (cfg.serving.kvDtype.has_value() && !cfg.serving.kvDtype->empty()) {
                const std::string_view v{*cfg.serving.kvDtype};
                if (v == "fp8")       e.setServingKvDtype(::mimirmind::runtime::KvDtype::FP8_E4M3);
                else if (v == "fp16") e.setServingKvDtype(::mimirmind::runtime::KvDtype::FP16);
                else if (v == "f32")  e.setServingKvDtype(::mimirmind::runtime::KvDtype::F32);
            }

            const auto& arch = e.config().architecture;
            if (arch != "qwen2" && arch != "llama" && arch != "gemma4" &&
                arch != "qwen35moe" && arch != "qwen4_exp") {
                throw std::runtime_error(
                    "M-Munin.3 pool: architecture '" + arch + "' (model '" +
                    modelId + "') is not implemented");
            }

            // Propagate the process-wide ancillary monitors — same pattern
            // as the eager "extras" propagation, above.
            if (thermalGuardPtr  != nullptr) e.setThermalGuard(thermalGuardPtr);
            if (powerMonitorPtr  != nullptr) e.setPowerMonitor(powerMonitorPtr);
            if (perfDetectorPtr  != nullptr) e.setPerfRegressionDetector(perfDetectorPtr);
            if (fanControllerPtr != nullptr) e.setFanController(fanControllerPtr);

            // Per-slot continuous batcher — same eligibility + config knobs
            // as the default engine's, above.
            if ((arch == "qwen35moe" || arch == "qwen4_exp" ||
                 e.supportsBatchedDecode()) &&
                e.servingClassEnabled()) {
                std::size_t maxBatch =
                    std::max<std::size_t>(1, e.batchCapacity().sustainableBatch);
                if (const char* mb = std::getenv("MIMIRMIND_SERVING_MAXBATCH")) {
                    const long v = std::atol(mb);
                    if (v > 0) maxBatch = static_cast<std::size_t>(v);
                }
                try {
                    payload->batcher = std::make_unique<
                        ::mimirmind::runtime::serving::ContinuousBatcher>(
                        e, maxBatch, e.maxContextTokens(), e.tokenizer().eosId(),
                        cfg.serving.maxActiveRequests,
                        cfg.serving.maxActiveRequestsPerTenant);
                    MM_LOG_INFO("main",
                                "serve: pool slot '{}' continuous batcher "
                                "ENABLED (maxBatch={} maxContext={})",
                                modelId, maxBatch, e.maxContextTokens());
                } catch (const std::exception& x) {
                    MM_LOG_WARN("main",
                                "serve: pool slot '{}' continuous batcher init "
                                "failed ({}); single-session generate() only",
                                modelId, x.what());
                    payload->batcher.reset();
                }
            }

            // Per-slot spec-dec — ONLY when this model is the configured
            // speculative.target. `drafterPtr` is shared with the default
            // engine's SpeculativeDecoder, but RequestDispatcher's own
            // constructor already refuses to build ITS decoder unless
            // speculative.target names the default — so at most ONE of
            // {default engine, this pool slot} ever actually calls into the
            // shared draft model. Never both at once, so no cross-model
            // concurrent-draft-engine hazard.
            if (drafterPtr != nullptr && cfg.speculative.enabled &&
                cfg.speculative.target == modelId) {
                payload->spec = std::make_unique<::mimirmind::runtime::SpeculativeDecoder>(
                    e, *drafterPtr, scfg.speculative);
                MM_LOG_INFO("main",
                            "serve: pool slot '{}' is the speculative.target "
                            "— spec-dec decoder built", modelId);
            }

            return payload;
        };

        modelProvider = std::make_unique<::mimirmind::server::AttachedModelProvider>(
            cfg.serving.modelPoolCapacity, std::move(provided), defaultId,
            factory);
        scfg.modelProvider = modelProvider.get();
        MM_LOG_INFO("main",
                    "serve: M-Munin.3 pool ENABLED — capacity={} models={}",
                    cfg.serving.modelPoolCapacity, poolChatModels.size());
    }

    ::mimirmind::server::ApiServer server{std::move(loadedEngines), scfg,
                                          drafter.get(),
                                          std::move(loadedRerankers),
                                          std::move(loadedEmbedders),
                                          std::move(loadedTranscribers),
                                          std::move(loadedSpeakers),
                                          std::move(loadedDeciders)};

    g_runningServer.store(&server, std::memory_order_release);
    std::signal(SIGINT,  signalStop);
    std::signal(SIGTERM, signalStop);

    std::cout << "\n[M7d/M7e] OpenAI-compatible "
              << (scfg.tls.enabled ? "HTTPS" : "HTTP")
              << " API listening on "
              << (scfg.tls.enabled ? "https://" : "http://")
              << scfg.host << ":" << scfg.port
              << "\n  GET  /health\n"
                 "  GET  /v1/models\n"
                 "  GET  /v1/system/info\n"
                 "  GET  /v1/system/status\n"
                 "  POST /v1/chat/completions  (stream=true supported)\n"
                 "  model id:           " << scfg.modelId << "\n"
                 "  tls:                "
              << (scfg.tls.enabled
                      ? (scfg.tls.selfSigned ? "on (self-signed, use curl -k)"
                                             : "on (operator cert)")
                      : "off (cleartext)")
              << "\n  auth:               "
              << (scfg.auth.enabled
                      ? "on (Authorization: Bearer <key>)"
                      : "off (open)")
              << "\n  preserve-thinking:  "
              << (scfg.preserveThinking ? "on (raw deltas, KV-cache friendly)"
                                        : "off (cleaned text, channel-wrapper stripped)")
              << "\n  thermal profile:    ";
    if (guard) {
        std::cout << "'" << guard->profile().name
                  << "' (package=" << monitor->packageTempSource() << ")";
    } else {
        std::cout << "\033[1;33mNOT CONFIGURED — engine is unprotected\033[0m";
    }
    std::cout << "\n  power telemetry:    ";
    if (powerMonitor->available()) {
        std::cout << "on (" << powerMonitor->domainNames().size()
                  << " RAPL domain(s))";
    } else {
        std::cout << "off (" << powerMonitor->unavailableReason() << ")";
    }
    std::cout << "\n  gpu clock governor: ";
    if (auto* gov = engine.gpuClockGovernor()) {
        std::cout << "on (target=" << gov->targetTempC()
                  << "°C, " << gov->rpnMhz() << ".."
                  << gov->rp0Mhz() << " MHz on "
                  << gov->cardPath() << ")";
    } else {
        std::cout << "off";
    }
    std::cout << "\n  perf regression:    ";
    if (auto* det = engine.perfRegressionDetector()) {
        std::cout << "on (baseline=" << det->baselineSampleCount()
                  << " samples, threshold="
                  << ::mimirmind::runtime::PerfRegressionDetector::kAlertThreshold
                  << "x)";
    } else {
        std::cout << "off (diagnostics.regressionAlert=false)";
    }
    std::cout << "\n  spec decoding:      ";
    if (drafter != nullptr) {
        std::cout << "ready (drafter=" << drafter->kind();
        if (draftEngine != nullptr) {
            std::cout << ", draft arch=" << draftEngine->config().architecture
                      << ", d_model="   << draftEngine->config().embeddingLength;
        }
        std::cout << ")";
    } else if (cfg.speculative.enabled) {
        std::cout << "disabled (draft load or vocab check failed — see log)";
    } else {
        std::cout << "off (set speculative.enabled=true in config.json to enable)";
    }
    std::cout << "\n  max context tokens: " << engine.maxContextTokens()
              << "\n  Ctrl-C to stop.\n";
    std::cout.flush();

    if (!guard) {
        MM_LOG_WARN("main",
                    "serve: no thermal profile configured. The engine will "
                    "not throttle decode on temperature/RAM limits. Fill "
                    "the governor.thermal section of config.json to "
                    "protect the host.");
    }

    try {
        server.run();
    } catch (const std::exception& e) {
        g_runningServer.store(nullptr, std::memory_order_release);
        MM_LOG_ERROR("main", "serve: {}", e.what());
        std::cerr << "serve failed: " << e.what() << "\n";
        return 1;
    }
    g_runningServer.store(nullptr, std::memory_order_release);

    MM_LOG_INFO("main", "serve: stopped cleanly");
    return 0;
}

} // namespace mimirmind::cli