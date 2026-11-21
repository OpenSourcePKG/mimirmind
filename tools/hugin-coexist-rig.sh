#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Hugin coexist rig runner (roadmap 7.0): bandwidth per allocation kind, then two
# CUDA processes in THIS container on one memfd weight buffer, with and without
# the quiesce handshake.
#
#   ./tools/hugin-coexist-rig.sh <hugin_coexist_rig binary> [GiB] [seconds]
#
# Run inside ONE --gpus container on the GB10 host with prod serving stopped.
set -u
BIN="${1:?usage: hugin-coexist-rig.sh <binary> [GiB] [seconds]}"
GIB="${2:-8}"
SEC="${3:-60}"
DIR="$(mktemp -d)"
trap 'rm -rf "$DIR"' EXIT

echo "=== bandwidth ($GIB GiB) ==="
"$BIN" bw "$GIB" 20

for q in --quiesce ""; do
    tag=${q:-race}
    echo "=== soak same-container, $tag ==="
    "$BIN" owner "$DIR/s.sock" "$GIB" "$SEC" &
    opid=$!
    "$BIN" attacher "$DIR/s.sock" 40 $q --scratch-gib 4
    arc=$?
    wait "$opid"
    echo ">>> $tag attacher rc=$arc"
done
