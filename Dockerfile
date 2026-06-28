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

# Intel GPU repo for current Level Zero (Ubuntu 24.04 = "noble")
# NOTE: ocloc (SPIR-V offline compiler) is NOT installed here — its package
# name in the Intel repo changed and we will only need it at milestone M5.
# When adding it, probe the right name with:
#   apt-cache search ocloc; apt-cache search intel-ocl
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
COPY src ./src
# COPY kernels ./kernels   # enabled at M5

RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --parallel


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
    && apt-get purge -y wget gnupg \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Defensive: lift the 4-GB single-allocation cap on Level Zero
ENV UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
ENV UR_L0_USE_RELAXED_ALLOCATION_LIMITS=1

# Models live on a host volume (mounted read-only in docker-compose.yml)
ENV MIMIRMIND_MODELS_DIR=/models

COPY --from=build /src/build/mimirmind /usr/local/bin/mimirmind
# COPY --from=build /src/build/spv /usr/local/share/mimirmind/spv   # enabled at M5

ENTRYPOINT ["/usr/local/bin/mimirmind"]