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

## Two compose files, two workflows

The repository ships two compose files. Pick the one that matches what
you are doing:

`docker-compose.yml` — **dev**: builds the image locally from your
checkout, runs the builder container against the host's source tree.
Use this when you are editing the engine.

`docker-compose.server.yml` — **deploy**: pulls a tagged image from
the project registry, no build step. Use this when you are running
against a known-good image.

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

- `.env` — only the container-level knobs: image tag, models directory
  bind-mount, host GIDs. Every runtime setting moved into `config.json`.
- `config.json` — every runtime knob (model path, port, KV dtype, thermal
  profile, feature flags, log level and file, USM probe cap, …). Copy
  `config.example.json` to `config.dev.json` (gitignored) and edit;
  point the compose file at it via `MIMIRMIND_CONFIG_HOST` in `.env`.

`.env` example:

```dotenv
MIMIRMIND_IMAGE_TAG=latest
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