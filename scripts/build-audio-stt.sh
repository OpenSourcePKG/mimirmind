#!/usr/bin/env bash
#
# Docker build + verify for the audio-STT (Whisper) path — roadmap 8.13.1.
#
# The dev box's system compiler (GCC 11.4) cannot build the server translation
# units (they pull C++23 <expected>/<format> via the InferenceEngine include
# chain), so the STT code is built inside the `mimirmind:builder` toolchain
# image (Ubuntu 24.04 / GCC 13) that docker-compose already provides. The
# Dockerfile COPYs `src/` and `tests/` wholesale, so no Dockerfile change is
# needed — every new file (compute/dsp, runtime/audio, server/Transcriptions*)
# is already in the build.
#
# What this does (all inside the container, artefacts land in ./build on host):
#   1. configure (Ninja, Release) — picks up the CMakeLists changes
#   2. build + run `compute_tests` — the pure-CPU suite, including the 12
#      wav_reader tests and the mel/whisper host tests (no GPU needed)
#   3. build the `mimirmind` serve binary — proves the full server wiring
#      (TranscriptionsHandler + ApiServer + ServeMode task==transcribe) links
#
# It does NOT run the serve binary (that needs a GPU + a Whisper checkpoint —
# see the curl smoke at the bottom of this header) and never pushes an image.
#
# Usage:
#   scripts/build-audio-stt.sh              # configure + tests + serve binary
#   scripts/build-audio-stt.sh tests        # configure + compute_tests only
#   scripts/build-audio-stt.sh serve        # configure + serve binary only
#
# On-box smoke test (once a Whisper checkpoint is staged and the serve
# container is up with a `task: transcribe` model entry in config.json):
#
#   # config.json models[] entry (Whisper checkpoint dir needs config.json +
#   # *.safetensors + tokenizer.json):
#   #   { "id": "whisper", "task": "transcribe",
#   #     "path": "/models/whisper-tiny", "loadOnStart": true }
#
#   curl -sS -X POST http://localhost:8080/v1/audio/transcriptions \
#        -F file=@clip.wav -F model=whisper -F language=en
#   # -> {"text":"<transcript>"}
#   # response_format=text returns the raw transcript as text/plain.

set -euo pipefail

MODE="${1:-all}"
COMPOSE_SVC="builder"
BUILD_DIR="/src/build"

run_in_builder() {
    docker compose run --rm "${COMPOSE_SVC}" bash -lc "$1"
}

echo "[build-audio-stt] configuring (Ninja, Release) in ${COMPOSE_SVC} container…"
run_in_builder "cmake -S /src -B ${BUILD_DIR} -G Ninja -DCMAKE_BUILD_TYPE=Release"

if [[ "${MODE}" == "all" || "${MODE}" == "tests" ]]; then
    echo "[build-audio-stt] building + running compute_tests (CPU-only)…"
    run_in_builder "cmake --build ${BUILD_DIR} --target compute_tests --parallel \
        && ${BUILD_DIR}/compute_tests"
fi

if [[ "${MODE}" == "all" || "${MODE}" == "serve" ]]; then
    echo "[build-audio-stt] building the mimirmind serve binary (L0 backend)…"
    run_in_builder "cmake --build ${BUILD_DIR} --target mimirmind --parallel"
    echo "[build-audio-stt] serve binary built: ./build/mimirmind"
fi

echo "[build-audio-stt] done."
