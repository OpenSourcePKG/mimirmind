# Attached-Deploy Rollout — L0_TARGET_HOST

Step-by-step runbook for the first production rollout of M-Munin on
`L0_TARGET_HOST`. The design lives in the Synaipse ADR
`Memory/mimirmind/decisions/2026-07-13-m-munin-scope.md`; this file
covers the actual operator sequence.

The rollout deliberately keeps the standalone deployment reachable so
we can roll back at any point with one `docker compose down` +
`docker compose -f docker-compose.server.yml up -d`.

## Prerequisites

- L0_TARGET_HOST has the current standalone `mimirmind:latest`
  running via `docker-compose.server.yml`. Baseline works.
- Local build of both images has succeeded (see `doc/build.md`):
  - `mimirmind:latest` — target `runtime`
  - `munin:latest` — target `munin_runtime` (~340 MB)
- `docker-compose.munin.yml` is on the L0_TARGET_HOST host at the
  same path as the existing `docker-compose.server.yml`.
- `git log --oneline main..origin/main` on the pegenaut host shows
  the Schritt-7+8 commits (the ones that add `src/munin/`,
  `src/core/ipc/{MuninClient,WireOps}.hpp`,
  `src/core/os/GovernorLock.hpp`).

## Step 1 — Build and push images from the dev host

Runs on the dev host with source access. Push macht der User selbst
per Standardregel; keine autonome Push-Aktion.

```bash
# From the repo root, with docker configured for the project registry.
export MUNIN_IMAGE=docker-registry.example.com/munin:latest
export MIMIRMIND_IMAGE=docker-registry.example.com/mimirmind:latest

docker build --target runtime       -t "${MIMIRMIND_IMAGE}" .
docker build --target munin_runtime -t "${MUNIN_IMAGE}"     .

# Sanity: the small image should be well under the big one.
docker images "${MUNIN_IMAGE}"     --format 'munin:      {{.Size}}'
docker images "${MIMIRMIND_IMAGE}" --format 'mimirmind:  {{.Size}}'
# Expected: munin ~340 MB, mimirmind ~460 MB.

# Verify the munin binary comes up in its own image.
docker run --rm --entrypoint /usr/local/bin/munin "${MUNIN_IMAGE}" --help \
    | head -3

# ==> When both look right:
docker push "${MIMIRMIND_IMAGE}"
docker push "${MUNIN_IMAGE}"
```

## Step 2 — Stage the compose file on L0_TARGET_HOST

```bash
# SSH into L0_TARGET_HOST.
ssh L0_TARGET_HOST
cd /srv/mimirmind    # or wherever the existing docker-compose.server.yml lives

# Pull the new compose file into place. Either scp it from the dev
# host or fetch it from the git checkout.
git fetch && git checkout main -- docker-compose.munin.yml doc/build.md

# Extend the existing .env with the second image reference.
grep -q MUNIN_IMAGE .env || echo \
    'MUNIN_IMAGE=docker-registry.example.com/munin:latest' >> .env

# Same MIMIRMIND_MODELS_DIR, MIMIRMIND_CONFIG_HOST, GID vars as today.
cat .env
```

## Step 3 — Baseline measurement (before Munin)

We need a number for Ledger #32. Restart the standalone deployment
cold and time the boot to the "listening on 0.0.0.0:8080" log line.

```bash
docker compose -f docker-compose.server.yml down
sync

t0=$(date +%s)
docker compose -f docker-compose.server.yml up -d
docker compose -f docker-compose.server.yml logs -f mimirmind \
    | grep --line-buffered -m1 "listening on 0.0.0.0"
t1=$(date +%s)
echo "standalone boot: $((t1-t0)) s"
```

Expected: 80-100 s on L0_TARGET_HOST with Gemma 4 E4B Q4_K_M
(baseline in the perf ledger).

## Step 4 — Bring up Munin

```bash
# Take standalone down first — Munin acquires the same governor flock
# and would refuse to start otherwise.
docker compose -f docker-compose.server.yml down
sync

docker compose -f docker-compose.munin.yml pull munin
t0=$(date +%s)
docker compose -f docker-compose.munin.yml up -d munin
docker compose -f docker-compose.munin.yml logs -f munin \
    | grep --line-buffered -m1 "ModelStore: .* model(s) resident in USM"
t1=$(date +%s)
echo "munin boot: $((t1-t0)) s"

# Sanity: the socket is up.
docker exec mimirmind-munin ls -la /var/run/mimirmind/
# ==> munin.sock and governor.lock present.
```

If Munin does not come up: `docker compose logs munin` and check for
a `GovernorLock` error (something else holds the flock) or an
`L0Context` error (device passthrough missing).

## Step 5 — Attach the worker

```bash
docker compose -f docker-compose.munin.yml pull mimirmind
t0=$(date +%s)
docker compose -f docker-compose.munin.yml up -d mimirmind
docker compose -f docker-compose.munin.yml logs -f mimirmind \
    | grep --line-buffered -m1 "listening on 0.0.0.0"
t1=$(date +%s)
echo "attached worker boot: $((t1-t0)) s"
```

Expected: 3-5 s. That is the number for Ledger #32 — compare against
the standalone baseline from Step 3.

Watch also for these log lines during worker startup:

- `MuninClient: attached to model '<id>' fingerprint='...' tensors=N`
- `serve: Munin healthz ok — pid=... models=... owner=munin`
- `serve: acquired governor flock at ...` should NOT appear (the
  worker skipped it because Munin owns it).

## Step 6 — Functional check

```bash
curl -s http://L0_TARGET_HOST:8080/health
# ==> {"status":"ok"}

curl -s http://L0_TARGET_HOST:8080/v1/models | jq .
# ==> lists the same models as the standalone deployment.

curl -s http://L0_TARGET_HOST:8080/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Say hi in one word."}]}' \
    | jq .choices[0].message.content
# ==> a plausible single-word response, latency similar to standalone.
```

If the chat response is garbage (repeated tokens, wrong tokenizer),
that is the fingerprint-refuse having a hole: the manifest match
passed but the model semantics differ. Bring the worker down, dig
into what Munin loaded vs what the worker's config points at.

## Step 7 — Worker-redeploy exercise (the actual business case)

This is the whole point of M-Munin. Simulate a code-change deploy:

```bash
# Force-pull the same tag to avoid docker's "already have it" short-
# circuit; in prod this would be a new sha-tagged image.
docker compose -f docker-compose.munin.yml pull mimirmind

t0=$(date +%s)
docker compose -f docker-compose.munin.yml up -d mimirmind
docker compose -f docker-compose.munin.yml logs -f mimirmind \
    | grep --line-buffered -m1 "listening on 0.0.0.0"
t1=$(date +%s)
echo "worker redeploy: $((t1-t0)) s"

# Same chat request works immediately.
curl -s http://L0_TARGET_HOST:8080/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Say hi."}]}' \
    | jq .choices[0].message.content
```

Expected: ~2 s. Log-check on Munin should show:

- `session#N: peer closed (model='<id>') — implicit detach` when
  the old worker went down.
- `session#N+1 peer-pid=<new> connected` for the new worker.

## Step 8 — Ledger entry

Record what happened in the Synaipse Perf-Regression-Ledger. Format
matches existing entries.

```markdown
### #32 — 2026-07-14 — M-Munin attached-mode rollout on L0_TARGET_HOST

- Commits: <schritt-7 sha>, <schritt-8 sha>, <schritt-9 sha>
- Standalone cold boot:    <t> s (Step 3)
- Munin first boot:        <t> s (Step 4, matches standalone within ±5 s)
- Attached worker attach:  <t> s (Step 5)
- Worker redeploy:         <t> s (Step 7)
- Chat E2E latency vs standalone: ±<x> ms/tok (no regression expected)
- Notes: <anything unusual — flock conflicts, healthcheck flap, ...>
```

## Rollback

If anything looks wrong at any step:

```bash
docker compose -f docker-compose.munin.yml down
docker compose -f docker-compose.server.yml up -d
```

Standalone is back within the standard ~90 s boot. The two compose
files do not share any state — the `mimirmind_run` volume from
Munin is unused by standalone and gets pruned on the next
`docker volume prune`.

If the rollback itself hangs on a stuck flock, `sudo rm
/var/run/mimirmind/governor.lock` on the host is safe — the file
only carries an advisory lock and any live holder will get a fresh
one on next start.

## Post-rollout

- Update the Synaipse session-state note to `Schritt 9 live`.
- Update `MEMORY.md` with a short reference to the new compose file.
- If the numbers in Step 7 look good, this closes M-Munin Tier-1 and
  the next milestone is M-Munin.2 (KV persistence across worker
  restarts) — sequenced behind M10.2.