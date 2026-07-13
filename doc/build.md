# Build & Run

Everything runs in Docker. The developer machine stays free of the
Intel compute SDK — only the source tree lives on the host.

## Host prerequisites

- A recent Docker (≥ 24) with `docker compose`.
- Intel iGPU exposed at `/dev/dri` (rendering passthrough).
- Host groups `render` (typically GID 104) and `video` (typically GID 44).
  If the host uses different GIDs, override via `MIMIRMIND_RENDER_GID` /
  `MIMIRMIND_VIDEO_GID` in `.env`. These two are the *only* env vars the
  container still consults; every runtime knob moved into `config.json`
  in commit `456bd2a`.

If you are running this from a Proxmox LXC container, see
[`setup-ct.md`](setup-ct.md) for the host-side configuration required
to forward `/dev/dri` properly. Without that the container will see
the device node but Level Zero will fail to initialise.

## Three compose files, three workflows

The repository ships three compose files. Pick the one that matches
what you are doing:

`docker-compose.yml` — **dev**: builds the image locally from your
checkout, runs the builder container against the host's source tree.
Use this when you are editing the engine.

`docker-compose.server.yml` — **deploy (standalone)**: pulls a tagged
image from the project registry, no build step. Single-process
mimirmind that owns everything — weights, KV cache, governor.

`docker-compose.munin.yml` — **deploy (attached)**: two services —
`munin` (long-lived, holds model tensors in USM) and `mimirmind` (short-
lived attached worker, restarted on redeploy). Use this when you want
the deploy-downtime win of M-Munin: worker restart is ~2 s instead of
~90 s because Munin keeps the models resident across worker
lifecycles.

## Iterative development (builder container)

```bash
# Configure + compile inside the toolchain container; build artefacts
# land in ./build on the host.
docker compose run --rm builder

# Interactive shell in the toolchain image (cmake, ninja, ocloc, gcc-14):
docker compose run --rm builder bash
```

The builder container bind-mounts the source tree read-write so that
incremental rebuilds work. The first build takes about a minute; the
SPV-kernel compile step alone takes a few seconds because each
`*.cl` file goes through `ocloc`.

To run the test binaries (no model file required):

```bash
docker compose run --rm builder bash -lc "/src/build/quant_tests"
docker compose run --rm builder bash -lc "/src/build/arch_tests"
docker compose run --rm builder bash -lc "/src/build/compute_tests"
docker compose run --rm builder bash -lc "/src/build/gpu_tests"
```

The first three are pure CPU. The last needs the iGPU and exercises
every SPIR-V kernel against its CPU reference; expect 16 tests to pass
in under a second.

## Running the runtime image

```bash
# Build the multi-stage runtime image (no toolchain in the final layer).
docker compose build mimirmind

# M1 device-enumeration smoke test:
docker compose run --rm mimirmind smoke

# Same, with a model file. The path is set in config.json under
# `models[<id>].path`; docker-compose maps `./models` to `/models`
# inside the container.
docker compose run --rm mimirmind smoke --chat \
    --prompt "Wer hat den Mond entdeckt?"
```

The runtime image is also what gets pushed to the project registry
for `docker-compose.server.yml`.

## Server deployment (registry image)

Configuration lives in two places:

- `.env` — only the container-level knobs: image reference, models
  directory bind-mount, host GIDs. Every runtime setting moved into
  `config.json`.
- `config.json` — every runtime knob (model path, port, KV dtype, thermal
  profile, feature flags, log level and file, USM probe cap, …). Copy
  `config.example.json` to `config.dev.json` (gitignored) and edit;
  point the compose file at it via `MIMIRMIND_CONFIG_HOST` in `.env`.

`.env` example:

```dotenv
# Defaults to mimirmind:latest — override to pull from your own registry.
MIMIRMIND_IMAGE=mimirmind:latest
MIMIRMIND_MODELS_DIR=/srv/llm-models
MIMIRMIND_RENDER_GID=104
MIMIRMIND_VIDEO_GID=44
MIMIRMIND_CONFIG_HOST=./config.dev.json
```

`config.dev.json` schema: see `config.example.json` for the full shape;
`README.md` § Config has a one-line-per-section overview. The migration
mapping (old env var → new JSON key) is documented in the header of
`docker-compose.server.yml`.

Then:

```bash
docker compose -f docker-compose.server.yml pull
docker compose -f docker-compose.server.yml up -d
docker compose -f docker-compose.server.yml logs -f mimirmind
```

The server takes about 90 seconds to come up from cold start with the
Gemma 4 model: ~3 s for the L0 + USM probes, then ~70 s for the
25 GiB USM copy from the mmap'd GGUF, then ~10 s for the per-tensor
inventory walk. Watch for the `listening on 0.0.0.0:8080` log line —
that means the server is ready.

Smoke / parity workflows override the default `serve` command by
passing argv after the service name:

```bash
# Per-stage tensor parity dump against a llama-cli reference
docker compose -f docker-compose.server.yml run --rm mimirmind parity

# 20-token autoregressive smoke with chat template
docker compose -f docker-compose.server.yml run --rm mimirmind \
    smoke --chat --temperature 0.0 --seed 42 \
    --prompt "What is the capital of France?"
```

## Attached deployment (M-Munin)

The M-Munin architecture splits the mimirmind runtime into two
processes: `munin` holds the model tensors in USM(host) across
worker restarts, and `mimirmind serve --attach` connects over a
Unix socket and imports the tensors zero-copy via L0 IPC
(`zeMemOpenIpcHandle` + `SCM_RIGHTS`). The full design lives in the
Synaipse note `Memory/mimirmind/decisions/2026-07-13-m-munin-scope.md`.

Two separate registry images:

- `munin:latest` — small (~340 MB), just the daemon. Long-lived,
  rarely redeployed. Runs `/usr/local/bin/munin`.
- `mimirmind:latest` — full inference engine (~460 MB). Short-lived,
  redeployed on every code change. Runs `mimirmind serve --attach ...`.

Build both from the same source tree via multi-stage targets:

```bash
docker build --target runtime       -t mimirmind:latest .
docker build --target munin_runtime -t munin:latest .
```

`.env` extends the standalone deployment's variables with one extra
image reference:

```dotenv
MUNIN_IMAGE=munin:latest
MIMIRMIND_IMAGE=mimirmind:latest
MIMIRMIND_MODELS_DIR=/srv/llm-models
MIMIRMIND_CONFIG_HOST=/etc/mimirmind/config.json
MIMIRMIND_RENDER_GID=104
MIMIRMIND_VIDEO_GID=44
MIMIRMIND_PORT=8080
```

Both containers share a tmpfs volume mounted at `/var/run/mimirmind/`
that carries the Unix socket (`munin.sock`) and the governor
ownership flock (`governor.lock`).

Initial bringup:

```bash
# Munin loads all loadOnStart:true models — takes as long as a
# standalone boot (~90 s for Gemma 4 4B), once, at daemon start.
docker compose -f docker-compose.munin.yml up -d munin

# Watch for "ModelStore: N model(s) resident in USM" before starting
# the worker.
docker compose -f docker-compose.munin.yml logs -f munin

# Worker attaches over IPC — takes ~2 s (healthz + manifest + N
# handle imports, no tensor copy).
docker compose -f docker-compose.munin.yml up -d mimirmind
```

Redeploy the worker (the entire business case):

```bash
docker compose -f docker-compose.munin.yml pull mimirmind
docker compose -f docker-compose.munin.yml up -d mimirmind
```

Munin stays up. The old worker's socket closes (Munin observes
implicit detach), the new worker connects, does one healthz probe,
verifies the manifest fingerprint against its own local GGUF header,
imports the handles, and starts serving. End-to-end downtime: ~2 s
on Gemma 4 4B, plus whatever depends_on's health-poll interval adds.

Munin redeploy is rare and unavoidable — every attached worker goes
down with Munin because their USM pointers are invalidated when
Munin's process exits. Plan a maintenance window:

```bash
docker compose -f docker-compose.munin.yml pull munin
docker compose -f docker-compose.munin.yml up -d --force-recreate munin
docker compose -f docker-compose.munin.yml restart mimirmind
```

Governor ownership is coordinated by an advisory `flock(2)` on
`/var/run/mimirmind/governor.lock`. Munin acquires it at startup and
holds it for its whole lifetime; the attached worker reads the
governor owner from Munin's healthz response and skips its own
`ThermalGuard` / `GpuClockGovernor` / `FanController` install. A
second standalone `mimirmind serve` on the same host fails fast with
a message pointing at the current holder's PID — this is the desired
behaviour.

## Troubleshooting

**`L0Context: zeInit failed`** — `/dev/dri` is not present in the
container, or the calling UID is not in the `render` and `video`
groups inside the container. Check that `group_add` in the compose
file matches your host's GIDs.

**`ze_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` during USM probe** — the
phase-2 ceiling sweep is asking for more memory than the host has
free. Set `runtime.usmProbeTotalGib: 4` in `config.json` (or `0` to
skip phase 2 entirely). The model loader does not need the phase-2
result; it's diagnostic only.

**`Q8_0 kernels loaded` not appearing in the startup log** — you are
running a pre-M8.G image. Pull a newer tag.

**Chat returns markup like `<|channel>thought\n<channel|>` in the
content field** — you are running a pre-`e2d1f2f` image. The
post-decode cleanup landed in that commit.

**Attached worker: `Munin healthz failed: connect(...) No such file
or directory`** — Munin's Unix socket is not visible in the worker
container. Check that both services mount the same named volume at
`/var/run/mimirmind/` (`docker volume inspect
mimirmind_mimirmind_run` lists the mountpoints).

**Attached worker: `Munin reports governor_owner='...', expected
'munin'`** — the healthz response says another process owns the
governor. Schritt 8-minimal hardcodes the owner to `"munin"` while
the daemon runs; if you see a different value, someone bumped the
protocol or Munin lost the flock. Check `/proc/locks` on the host.

**Attached worker: `fingerprint mismatch — Munin advertised '...'
but the local GGUF hashes to '...'`** — the model file mounted in
the worker container is a different revision than what Munin
loaded. Either Munin's `MIMIRMIND_MODELS_DIR` and the worker's do
not match, or someone re-downloaded a GGUF while Munin was up. Bring
Munin down and back up (which reloads from the current file) or fix
the mount.

**Standalone `mimirmind serve` fails with `GovernorLock: flock(...)
failed: Resource temporarily unavailable (held by pid X)`** — Munin
or another standalone mimirmind is already running on this host. If
it is Munin, use `--attach unix:/var/run/mimirmind/munin.sock`
instead. If it is stale, `kill X` and retry.

## Build dependencies (for reference)

The toolchain image is Ubuntu 24.04 + the Intel client graphics APT
repo:

- `build-essential`, `cmake ≥ 3.21`, `ninja-build`, `pkg-config`
- `level-zero`, `level-zero-dev`
- `intel-opencl-icd`, `intel-level-zero-gpu`
- `intel-ocloc` (for compiling `*.cl` to `.spv` at build time)
- GCC 14 (for C++20 with `<format>` and `<source_location>` support)

The runtime image is the same Ubuntu base with only the runtime
libraries — no toolchain, no source — plus `python3` and
`python3-numpy` for the parity-diff helper script.