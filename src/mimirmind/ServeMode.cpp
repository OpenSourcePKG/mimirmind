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

/// Print the human-readable "listening on …" status block once the server is
/// wired but before `server.run()` blocks. Pure formatting over the already-
/// built config + ancillaries; factored out of runServe to keep the boot body
/// an orchestrator (8.30.4). `specConfigEnabled` mirrors cfg.speculative.enabled
/// so the block can distinguish "off" from "enabled but failed to load".
void printServeBanner(const ::mimirmind::server::ServerConfig&  scfg,
                      ::mimirmind::runtime::InferenceEngine&    engine,
                      ::mimirmind::runtime::SystemMonitor*      monitor,
                      ::mimirmind::runtime::ThermalGuard*       guard,
                      ::mimirmind::runtime::PowerMonitor*       powerMonitor,
                      ::mimirmind::runtime::Drafter*            drafter,
                      ::mimirmind::runtime::InferenceEngine*    draftEngine,
                      bool                                      specConfigEnabled) {
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
    if (guard != nullptr) {
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
    } else if (specConfigEnabled) {
        std::cout << "disabled (draft load or vocab check failed — see log)";
    } else {
        std::cout << "off (set speculative.enabled=true in config.json to enable)";
    }
    std::cout << "\n  max context tokens: " << engine.maxContextTokens()
              << "\n  Ctrl-C to stop.\n";
    std::cout.flush();

    if (guard == nullptr) {
        MM_LOG_WARN("main",
                    "serve: no thermal profile configured. The engine will "
                    "not throttle decode on temperature/RAM limits. Fill "
                    "the governor.thermal section of config.json to "
                    "protect the host.");
    }
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
    ServerBootstrap::AttachEngineFn attachEngine =
        [&args](::mimirmind::runtime::InferenceEngine&       e,
                const ::mimirmind::core::config::ModelEntry& m) {
            return ServerBootstrap::attachEngineToMunin(e, m, args.attachSocket);
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
    // Load every loadOnStart:true model into its own engine/stack (see
    // ServerBootstrap). backendPool + attachEngine stay owned here because
    // the M-Munin.3 pool factory below captures them by reference; `models`
    // is held for the whole serve lifetime (it owns the engines past
    // ApiServer::run).
    ServerBootstrap::LoadedModels models;
    ServerBootstrap::loadModels(models, cfg, args, backendPool, attachEngine,
                                defaultId, attachedMode);
    if (models.exitCode) {
        return *models.exitCode;
    }

    if (models.ownedEngines.empty()) {
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
    for (auto& le : models.loadedEngines) {
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

    // Process-wide thermal / power / fan ancillaries (see ServerBootstrap).
    // Held as a local so the owning monitors (monitor/guard/powerMonitor)
    // outlive ApiServer::run below; printServeBanner reads them via
    // `ancillaries`. The GpuClockGovernor/FanController stay function-local
    // statics inside the helper (program lifetime, as before).
    auto ancillaries =
        ServerBootstrap::wireThermalGovernorFan(engine, cfg, attachedMode);
    if (ancillaries.fatalExitCode) {
        return *ancillaries.fatalExitCode;
    }

    // Propagate the process-wide ancillary monitors from the default
    // engine to any extras so their generate() paths also honour
    // thermal admission, RAPL joule accounting, fan-boost pre-warm and
    // perf-regression sampling. Governor propagation is intentionally
    // skipped — its per-tick control loop is process-scoped and driven
    // by the default engine; extras would fight for the same GPU cap.
    for (auto& e : models.ownedEngines) {
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
    ServerBootstrap::warmupDefaultEngine(engine, defaultId);

    // Continuous batcher for the default engine (see ServerBootstrap). Held as
    // a runServe local so it outlives ApiServer::run; its dtor joins the worker
    // on shutdown. Null when the engine is not serving-class-eligible or init
    // failed — the handler then falls back to single-session generate().
    auto batcher = ServerBootstrap::buildDefaultBatcher(engine, cfg, defaultId);
    scfg.batcher = batcher.get();

    // M-Munin.3 (full): worker-side materialize/evict pool for the non-default
    // chat models (see ServerBootstrap::buildModelProvider). Null when there are
    // no pool models; the ApiServer then serves only the eager engines. Declared
    // before `server` so it outlives ApiServer::run.
    auto modelProvider = ServerBootstrap::buildModelProvider(
        engine, cfg, backendPool, attachEngine, scfg, drafter.get(),
        models.poolChatModels, defaultId);
    scfg.modelProvider = modelProvider.get();

    ::mimirmind::server::ApiServer server{std::move(models.loadedEngines), scfg,
                                          drafter.get(),
                                          std::move(models.loadedRerankers),
                                          std::move(models.loadedEmbedders),
                                          std::move(models.loadedTranscribers),
                                          std::move(models.loadedSpeakers),
                                          std::move(models.loadedDeciders)};

    g_runningServer.store(&server, std::memory_order_release);
    std::signal(SIGINT,  signalStop);
    std::signal(SIGTERM, signalStop);

    printServeBanner(scfg, engine, ancillaries.monitor.get(),
                     ancillaries.guard.get(), ancillaries.powerMonitor.get(),
                     drafter.get(), draftEngine.get(), cfg.speculative.enabled);

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
