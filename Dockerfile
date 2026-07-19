# syntax=docker/dockerfile:1.7
#
# MimirMind — multi-stage build.
#
# Stages:
#   1. builder  — toolchain + Intel compute SDK only. No source baked in.
#                  Used by docker-compose's "builder" service: source is
#                  bind-mounted from the host, builds land in ./build/.
#   2. build    — extends builder, COPYs source, runs cmake. Only used
#                  internally to produce the runtime image.
#   3. runtime  — minimal Intel Level-Zero runtime + the produced binary.
#                  Runs on any host (or privileged LXC) with an Intel
#                  iGPU exposed through /dev/dri.
#
# Typical workflows:
#
#   # 1) Iterative dev (host source, container toolchain, host artefacts):
#   docker compose run --rm builder           # cmake configure + build
#   docker compose run --rm builder bash      # interactive shell in toolchain
#
#   # 2) Ship the runtime image (full reproducible multi-stage build):
#   docker compose build mimirmind
#   docker compose run --rm mimirmind         # M1 smoke test
#

# ============================================================================
# Stage 1 — Builder (toolchain only, source is NOT copied in)
# ============================================================================
FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        wget \
        gnupg \
        build-essential \
        cmake \
        ninja-build \
        pkg-config \
        git \
    && rm -rf /var/lib/apt/lists/*

# Intel GPU repo for current Level Zero (Ubuntu 24.04 = "noble").
# intel-ocloc compiles OpenCL C to SPIR-V at build time; required from M5
# onward where the GPU kernels live in kernels/*.cl.
RUN wget -qO - https://repositories.intel.com/gpu/intel-graphics.key \
        | gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] \
https://repositories.intel.com/gpu/ubuntu noble client" \
        > /etc/apt/sources.list.d/intel-gpu.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        level-zero \
        level-zero-dev \
        intel-opencl-icd \
        intel-level-zero-gpu \
        intel-ocloc \
    && rm -rf /var/lib/apt/lists/*

# Default working dir for the bind-mounted source path
WORKDIR /src

# Default command for the "builder" docker-compose service: configure + build.
# Override with anything else (e.g. "bash") when you need a shell.
CMD ["bash", "-lc", \
     "cmake -S /src -B /src/build -G Ninja -DCMAKE_BUILD_TYPE=${BUILD_TYPE:-Release} \
      && cmake --build /src/build --parallel"]


# ============================================================================
# Stage 2 — Build (toolchain + baked-in source, produces the binary)
# ============================================================================
FROM builder AS build

# CMake config first (only file that triggers a re-configure)
COPY CMakeLists.txt ./
COPY config.example.json ./
COPY src ./src
COPY kernels ./kernels
COPY tests ./tests
COPY tools ./tools
COPY third_party ./third_party

RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --parallel


# ============================================================================
# Stage 2b — llama.cpp CPU build (reference oracle for parity tests)
# ============================================================================
FROM builder AS llamacpp

WORKDIR /llamacpp
ARG LLAMACPP_REF=master
RUN git clone --depth 1 --branch ${LLAMACPP_REF} \
        https://github.com/ggml-org/llama.cpp.git src

# Inject our parity-dump tool as a new example under examples/parity-dump.
COPY tools/llama-parity-dump.cpp        /llamacpp/src/examples/parity-dump/main.cpp
COPY tools/parity-dump-CMakeLists.txt   /llamacpp/src/examples/parity-dump/CMakeLists.txt
RUN echo 'add_subdirectory(parity-dump)' >> /llamacpp/src/examples/CMakeLists.txt

# CPU-only build of just the targets we need. Skipping CUDA/Vulkan/Metal
# keeps the build under a minute and the runtime image lean.
RUN cmake -S src -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_BUILD_TESTS=OFF \
        -DGGML_NATIVE=OFF \
        -DGGML_CUDA=OFF \
        -DGGML_VULKAN=OFF \
        -DGGML_METAL=OFF \
        -DGGML_OPENMP=OFF \
    && cmake --build build --target llama-cli llama-parity-dump --parallel


# ============================================================================
# Stage 2c — Builder-HIP (ROCm/HIP toolchain, no source)
# ============================================================================
#
# Peer to the Intel L0 `builder` stage above but targets AMD gfx1101
# (RX 7800 XT and RDNA3 siblings). No source is baked in; docker-compose
# services can bind-mount from the host the same way. Produces
# mimirmind + hip_kernels via hipcc --genco.
#
# Verified locally:  HIP_TARGET_HOST (RX 7800 XT / gfx1101, ROCm 7.2.53)
# native builds already work — this stage lifts the setup into a
# reproducible Docker layer so the same image runs on other AMD hosts.
#
# NOT YET verified inside Docker: the stage below installs the ROCm apt
# repo and hipcc but has not been end-to-end built. Track "Task 1.10"
# in Synaipse (`todos/hip-tests-inline-migration.md`) for the follow-up
# once someone with a HIP-capable Docker build host validates it.

FROM ubuntu:24.04 AS builder-hip

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        wget \
        gnupg \
        build-essential \
        cmake \
        ninja-build \
        pkg-config \
        git \
        libnuma-dev \
        nlohmann-json3-dev \
    && rm -rf /var/lib/apt/lists/*

# ROCm apt repo. Version pin matches the tested HIP_TARGET_HOST setup
# (ROCm 7.2 series). Override at build time with
#   docker build --build-arg ROCM_VERSION=7.2 --target builder-hip
ARG ROCM_VERSION=7.2

RUN wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key \
        | gpg --yes --dearmor --output /usr/share/keyrings/rocm.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/rocm.gpg] \
https://repo.radeon.com/rocm/apt/${ROCM_VERSION} noble main" \
        > /etc/apt/sources.list.d/rocm.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        rocm-hip-runtime \
        rocm-hip-runtime-dev \
        hipcc \
        rocm-cmake \
    && rm -rf /var/lib/apt/lists/*

# hipcc lives at /opt/rocm/bin. Add to PATH so cmake's `find_program`
# picks it up without an explicit -DHIPCC_EXECUTABLE=... override.
ENV PATH=/opt/rocm/bin:${PATH}

WORKDIR /src

CMD ["bash", "-lc", \
     "cmake -S /src -B /src/build-hip -G Ninja -DCMAKE_BUILD_TYPE=${BUILD_TYPE:-Release} \
          -DMIMIRMIND_ENABLE_L0=OFF -DMIMIRMIND_ENABLE_HIP=ON \
          -DMIMIRMIND_HIP_ARCHS=${MIMIRMIND_HIP_ARCHS:-gfx1101} \
      && cmake --build /src/build-hip --parallel"]


# ============================================================================
# Stage 2d — Build-HIP (toolchain + baked source, HIP-only mimirmind exec)
# ============================================================================

FROM builder-hip AS build-hip

COPY CMakeLists.txt ./
COPY config.example.json ./
COPY src ./src
COPY kernels ./kernels
COPY kernels_hip ./kernels_hip
COPY tests ./tests
COPY tools ./tools
COPY third_party ./third_party

RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DMIMIRMIND_ENABLE_L0=OFF -DMIMIRMIND_ENABLE_HIP=ON \
        -DMIMIRMIND_HIP_ARCHS=${MIMIRMIND_HIP_ARCHS:-gfx1101} \
    && cmake --build build --parallel


# ============================================================================
# Stage 3 — Runtime (no toolchain, no source, just runtime libs + binary)
# ============================================================================
#
# Two runtime stages. Both live under the same base (ubuntu:24.04 + Intel
# Level Zero) but ship different binaries:
#
#   - `runtime`       — the full mimirmind inference engine + SPV kernels
#                       + llama.cpp parity oracle. Around 500 MB.
#   - `munin_runtime` — just the Munin persistent-memory daemon. No SPV
#                       (Munin never dispatches kernels), no httplib, no
#                       llama-cli, no python. Around 200 MB, updated
#                       independently from mimirmind so the model-load
#                       downtime savings survive worker deploys.
#
# The M-Munin ADR (decisions/2026-07-13-m-munin-scope.md) is the source
# of truth for what belongs where.

FROM ubuntu:24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        wget \
        gnupg \
    && wget -qO - https://repositories.intel.com/gpu/intel-graphics.key \
        | gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] \
https://repositories.intel.com/gpu/ubuntu noble client" \
        > /etc/apt/sources.list.d/intel-gpu.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        level-zero \
        intel-level-zero-gpu \
        intel-opencl-icd \
        python3 \
        python3-numpy \
    && apt-get purge -y wget gnupg \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# ubuntu:24.04 ships `video` (GID 44) but not `render` (GID 104). A
# caller that passes `--group-add 104` (see mimirmind's own compose
# and the pegenaut server-compose) fails at runc with
# `Unable to find group 104` when the group is absent from the
# container's /etc/group — even if the host has it. Create both at
# the well-known Ubuntu GIDs so DRM/render passthrough works.
RUN groupadd -g 44  video  2>/dev/null || true \
 && groupadd -g 104 render 2>/dev/null || true

# Defensive: lift the 4-GB single-allocation cap on Level Zero
ENV UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
ENV UR_L0_USE_RELAXED_ALLOCATION_LIMITS=1

# Models live on a host volume (mounted read-only in docker-compose.yml).
# The config that references them lives in a bind-mounted
# /etc/mimirmind/config.json — see docker-compose.server.yml.

COPY --from=build /src/build/mimirmind        /usr/local/bin/mimirmind
COPY --from=build /src/build/gpu_tests        /usr/local/bin/gpu_tests
COPY --from=build /src/build/prefill_bench    /usr/local/bin/prefill_bench
COPY --from=build /src/build/l0_ipc_testrig   /usr/local/bin/l0_ipc_testrig
COPY --from=build /src/tools/l0-ipc-testrig.sh /usr/local/bin/l0-ipc-testrig.sh
COPY --from=build /src/build/spv              /usr/local/share/mimirmind/spv
# Reference config that operators can copy + edit for their host. The
# actual runtime config comes in through a bind-mount at
# /etc/mimirmind/config.json (see docker-compose.server.yml).
COPY --from=build /src/config.example.json    /usr/local/share/mimirmind/config.example.json

# llama.cpp parity-test oracle: binaries + their shared libs.
COPY --from=llamacpp /llamacpp/build/bin/llama-cli           /usr/local/bin/llama-cli
COPY --from=llamacpp /llamacpp/build/bin/llama-parity-dump   /usr/local/bin/llama-parity-dump
# Shared libs (libllama.so, libggml*.so, libllama-common.so, ...) live next
# to the binaries in llama.cpp's build tree. Park them under /usr/local/lib
# and refresh the dynamic linker cache so the binaries pick them up.
COPY --from=llamacpp /llamacpp/build/bin/libllama.so.0           /usr/local/lib/
COPY --from=llamacpp /llamacpp/build/bin/libllama-common.so.0    /usr/local/lib/
COPY --from=llamacpp /llamacpp/build/bin/libggml.so.0            /usr/local/lib/
COPY --from=llamacpp /llamacpp/build/bin/libggml-base.so.0       /usr/local/lib/
COPY --from=llamacpp /llamacpp/build/bin/libggml-cpu.so.0        /usr/local/lib/
RUN ldconfig

# Python diff helper.
COPY tools/parity-diff.py /usr/local/bin/parity-diff
RUN chmod +x /usr/local/bin/parity-diff

# Note: `runtime.spvDir` in config.json can override this at runtime.
# When left unset in config.json, GpuModule falls back to
# /usr/local/share/mimirmind/spv (baked in above).

ENTRYPOINT ["/usr/local/bin/mimirmind"]


# ============================================================================
# Stage 3b — Runtime-HIP (AMD RDNA3 runtime, HIP-only mimirmind exec)
# ============================================================================
#
# Peer to the L0 `runtime` stage but ships the HIP build. Installs the
# ROCm runtime bits (no dev headers, no hipcc) plus the mimirmind binary
# and the pre-compiled hsaco kernel objects. Host needs an AMD GPU
# reachable via `/dev/kfd` and `/dev/dri` — `docker compose` runs must
# pass those through with `--device=/dev/kfd --device=/dev/dri` or the
# equivalent compose-file entries.
#
# Verified locally:  HIP_TARGET_HOST (RX 7800 XT / gfx1101) native runs.
# NOT YET verified inside Docker end-to-end — same follow-up as the
# builder-hip stage.

FROM ubuntu:24.04 AS runtime-hip

ENV DEBIAN_FRONTEND=noninteractive

ARG ROCM_VERSION=7.2

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        wget \
        gnupg \
    && wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key \
        | gpg --yes --dearmor --output /usr/share/keyrings/rocm.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/rocm.gpg] \
https://repo.radeon.com/rocm/apt/${ROCM_VERSION} noble main" \
        > /etc/apt/sources.list.d/rocm.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        rocm-hip-runtime \
        libnuma1 \
    && rm -rf /var/lib/apt/lists/*

# Copy just the mimirmind binary + compiled hsaco kernels. The runtime
# doesn't need the sources, doesn't need hipcc, doesn't need the L0
# SPV kernels (those are Intel-only).
COPY --from=build-hip /src/build/mimirmind /usr/local/bin/mimirmind
COPY --from=build-hip /src/build/hsaco     /usr/local/share/mimirmind/hsaco

ENTRYPOINT ["/usr/local/bin/mimirmind"]


# ============================================================================
# Stage 4 — Munin runtime (persistent model-memory daemon)
# ============================================================================
#
# Ships only the `munin` binary. Munin holds model tensors in USM(host)
# across mimirmind-worker restarts (see M-Munin ADR). No compute kernels,
# no HTTP surface, no reference-oracle tooling — Munin's whole job is
# owning memory and talking to attached workers over a Unix socket.
#
# The image is intentionally small so a Munin redeploy stays cheap; the
# whole point of splitting Munin out of mimirmind is that its lifecycle
# is decoupled from the compute code. Bumping mimirmind:latest does not
# touch munin:latest.

FROM ubuntu:24.04 AS munin_runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        wget \
        gnupg \
    && wget -qO - https://repositories.intel.com/gpu/intel-graphics.key \
        | gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] \
https://repositories.intel.com/gpu/ubuntu noble client" \
        > /etc/apt/sources.list.d/intel-gpu.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        level-zero \
        intel-level-zero-gpu \
    && apt-get purge -y wget gnupg \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# ubuntu:24.04 lacks the `render` GID that the compose file passes via
# --group-add 104. Create both at the well-known Ubuntu numbers so
# /dev/dri passthrough works — same rationale as the mimirmind runtime.
RUN groupadd -g 44  video  2>/dev/null || true \
 && groupadd -g 104 render 2>/dev/null || true

# Level Zero: lift the 4-GiB single-allocation cap. Munin allocates one
# USM buffer per tensor (720 for E4B, 800+ for 26B-A4B) and the largest
# individual weight can exceed 4 GiB on 26B-A4B.
ENV UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
ENV UR_L0_USE_RELAXED_ALLOCATION_LIMITS=1

COPY --from=build /src/build/munin              /usr/local/bin/munin
COPY --from=build /src/config.example.json      /usr/local/share/mimirmind/config.example.json

ENTRYPOINT ["/usr/local/bin/munin"]