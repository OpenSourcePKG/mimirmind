# syntax=docker/dockerfile:1.7
#
# Mimirmind — multi-stage build.
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
# Stage 3 — Runtime (no toolchain, no source, just runtime libs + binary)
# ============================================================================
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