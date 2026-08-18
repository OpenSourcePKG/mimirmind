#!/usr/bin/env bash
#
# Reproducible deploy for a drifting on-box tree (Spark / NUC).
#
# Materialises an EXACT git ref into the deploy dir (no hand-scp, no tar
# mtime games that silently skip recompiles), records the deployed sha,
# rebuilds in the builder image, restarts the serving compose service,
# health-checks, and prints a one-line rollback command.
#
# Solves the fragility in roadmap 8.14: /opt/mimirmind is not a git repo
# and drifts from origin/main; syncs are hand-done; and a second GPU
# container next to prod crashes the box (CUDA illegal-access on the Spark,
# OOM-kill on the NUC). This script is idempotent and only ever touches the
# ONE serving service via compose (swap, never coexist).
#
# Source of truth is the git repo this script is run from. The deploy dir
# need NOT be a git repo: the script overwrites the tracked SOURCE files
# from the ref (on-box-only files — models, .env, config.*.json — are left
# untouched) and writes DEPLOYED_SHA so the on-box state is always
# identifiable and reversible.
#
# Usage:
#   scripts/deploy-repro.sh <git-ref> [flags]
#
# Flags (env override in parens = default):
#   --deploy-dir DIR   (DEPLOY_DIR=/opt/mimirmind)    on-box tree to sync into
#   --compose FILE     (COMPOSE_FILE=docker-compose.server.yml)
#   --service NAME     (SERVICE=mimirmind)            compose service to restart
#   --builder IMAGE    (BUILDER_IMAGE=mimirmind:builder-cuda)
#   --build-dir DIR    (BUILD_DIR=build-cuda)         build tree inside the image
#   --port N           (PORT=8080)                    HTTPS health-check port
#   --health-path P    (HEALTH_PATH=/health)
#   --gpu-containers "" (GPU_CONTAINERS="mimirmind mimirmind-cuda munin-worker")
#                        running containers from this list, other than the
#                        target service, are treated as a second GPU container
#   --no-build         config-only redeploy (skip the builder step)
#   --allow-dirty      permit deploying from a dirty source working tree
#   --allow-coexist    bypass the one-GPU-container guard (dangerous)
#   --yes              do not prompt before restarting the service
#   -h, --help
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DEPLOY_DIR="${DEPLOY_DIR:-/opt/mimirmind}"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.server.yml}"
SERVICE="${SERVICE:-mimirmind}"
BUILDER_IMAGE="${BUILDER_IMAGE:-mimirmind:builder-cuda}"
BUILD_DIR="${BUILD_DIR:-build-cuda}"
PORT="${PORT:-8080}"
HEALTH_PATH="${HEALTH_PATH:-/health}"
GPU_CONTAINERS="${GPU_CONTAINERS:-mimirmind mimirmind-cuda munin-worker}"
DO_BUILD=1
ALLOW_DIRTY=0
ALLOW_COEXIST=0
ASSUME_YES=0

REF=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --deploy-dir)     DEPLOY_DIR="$2"; shift 2;;
        --compose)        COMPOSE_FILE="$2"; shift 2;;
        --service)        SERVICE="$2"; shift 2;;
        --builder)        BUILDER_IMAGE="$2"; shift 2;;
        --build-dir)      BUILD_DIR="$2"; shift 2;;
        --port)           PORT="$2"; shift 2;;
        --health-path)    HEALTH_PATH="$2"; shift 2;;
        --gpu-containers) GPU_CONTAINERS="$2"; shift 2;;
        --no-build)       DO_BUILD=0; shift;;
        --allow-dirty)    ALLOW_DIRTY=1; shift;;
        --allow-coexist)  ALLOW_COEXIST=1; shift;;
        --yes)            ASSUME_YES=1; shift;;
        -h|--help)        sed -n '2,55p' "$0" | sed 's/^# \?//'; exit 0;;
        -*)               echo "unknown flag: $1" >&2; exit 2;;
        *)                REF="$1"; shift;;
    esac
done

[[ -n "$REF" ]] || { echo "usage: $0 <git-ref> [flags]  (see --help)" >&2; exit 2; }

log() { printf '\033[1;34m[deploy]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[deploy] ERROR:\033[0m %s\n' "$*" >&2; }

# --- 1. Resolve the ref against the source repo, refuse silent drift ------
SHA="$(git -C "$REPO_ROOT" rev-parse --verify "${REF}^{commit}" 2>/dev/null)" || {
    err "'$REF' is not a valid git ref in $REPO_ROOT"; exit 1; }
SHORT_SHA="${SHA:0:10}"

if [[ $ALLOW_DIRTY -eq 0 ]]; then
    if ! git -C "$REPO_ROOT" diff --quiet || ! git -C "$REPO_ROOT" diff --cached --quiet; then
        err "source working tree is dirty — commit or stash, or pass --allow-dirty"
        exit 1
    fi
fi
log "source ref $REF -> $SHORT_SHA"

[[ -d "$DEPLOY_DIR" ]] || { err "deploy dir $DEPLOY_DIR does not exist"; exit 1; }

# --- 2. One-GPU-container guard (encodes lesson_two_gpu_containers_*) ------
assert_single_gpu_container() {
    [[ $ALLOW_COEXIST -eq 1 ]] && { log "one-GPU-container guard bypassed (--allow-coexist)"; return; }
    command -v docker >/dev/null 2>&1 || return
    local running others=()
    running="$(docker ps --format '{{.Names}}' 2>/dev/null || true)"
    for name in $GPU_CONTAINERS; do
        [[ "$name" == "$SERVICE" ]] && continue          # the service we replace is a swap, not a coexist
        grep -qx "$name" <<<"$running" && others+=("$name")
    done
    if [[ ${#others[@]} -gt 0 ]]; then
        err "another GPU container is running: ${others[*]}"
        err "two GPU containers on one host crash the box (illegal-access / OOM-kill)."
        err "stop it first, or pass --allow-coexist if you are certain it is CPU-only."
        exit 1
    fi
}
assert_single_gpu_container

# --- 3. Record the current sha for rollback, then materialise the ref -----
PREV_SHA=""
[[ -f "$DEPLOY_DIR/DEPLOYED_SHA" ]] && PREV_SHA="$(cat "$DEPLOY_DIR/DEPLOYED_SHA" 2>/dev/null || true)"

log "materialising $SHORT_SHA into $DEPLOY_DIR (tracked source only; models/config/.env untouched)"
# git archive extracts a clean, deterministic snapshot of TRACKED files at the
# ref (fresh contents, fresh timestamps) over the deploy tree. It never deletes
# on-box-only paths (models/, .env, config.*.json), so the running config and
# weights survive. This is the mtime-safe replacement for tar/scp.
git -C "$REPO_ROOT" archive --format=tar "$SHA" | tar -x -C "$DEPLOY_DIR"

printf '%s\n' "$SHA" > "$DEPLOY_DIR/DEPLOYED_SHA"
prev_disp="${PREV_SHA:0:10}"; [[ -z "$prev_disp" ]] && prev_disp="none"
log "DEPLOYED_SHA -> $SHORT_SHA (was: $prev_disp)"

# --- 4. Build in the builder image ---------------------------------------
if [[ $DO_BUILD -eq 1 ]]; then
    log "building in $BUILDER_IMAGE ($BUILD_DIR)"
    docker run --rm --gpus all \
        -v "$DEPLOY_DIR":/src -w /src \
        "$BUILDER_IMAGE" \
        bash -c "cmake --build '$BUILD_DIR' -j \"\$(nproc)\""
    log "build green"
else
    log "skipping build (--no-build)"
fi

# --- 5. Restart the serving service (swap, never coexist) -----------------
if [[ $ASSUME_YES -eq 0 ]]; then
    read -r -p "[deploy] restart service '$SERVICE' from $COMPOSE_FILE? [y/N] " ans
    [[ "$ans" =~ ^[Yy]$ ]] || { err "aborted before restart (tree already synced to $SHORT_SHA)"; exit 1; }
fi
log "restarting $SERVICE"
docker compose -f "$DEPLOY_DIR/$COMPOSE_FILE" up -d --no-deps "$SERVICE"

# --- 6. Health check ------------------------------------------------------
log "waiting for https://localhost:$PORT$HEALTH_PATH"
ok=0
for _ in $(seq 1 60); do
    code="$(curl -ksS -o /dev/null -w '%{http_code}' "https://localhost:$PORT$HEALTH_PATH" 2>/dev/null || true)"
    # 200 = healthy; 401 = up but auth-gated (still alive) — both mean serving.
    if [[ "$code" == "200" || "$code" == "401" ]]; then ok=1; break; fi
    sleep 2
done
if [[ $ok -eq 1 ]]; then
    log "healthy (HTTP $code) — deployed $SHORT_SHA"
else
    err "health check did NOT come up (last code: ${code:-none})"
    err "rollback:  $0 ${PREV_SHA:-<previous-sha>}"
    exit 1
fi

# --- 7. Rollback hint -----------------------------------------------------
if [[ -n "$PREV_SHA" ]]; then
    log "rollback if needed:  $0 ${PREV_SHA:0:10}"
else
    log "no previous DEPLOYED_SHA recorded — capture this one as your baseline."
fi
log "done. verify parity with:  scripts/verify-parity.sh https://localhost:$PORT"