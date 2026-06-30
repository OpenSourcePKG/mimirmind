# Operations

Notes for running mimirmind on real hardware: monitoring, thermal limits,
and what to expect when the host is under sustained load.

## Thermal profile

Sustained Gemma 4 26B decode saturates the Intel iGPU shader array and
keeps several CPU cores busy for the duration of every request. On
small-form-factor hosts with marginal cooling that can drive the
package temperature into the thermal-trip range, which on Intel takes
the system down without a kernel log entry — the CPU shuts itself off
in microcode.

The thermal profile is mimirmind's mechanism for staying ahead of that.
It is **a per-host configuration** describing what the specific machine
can sustain thermally — not a generic preset. Two different hosts will
need two different profiles.

The profile is intentionally narrow: **only package temperature limits**
live here. Other host metrics (RAM, GPU frequency) are surfaced through
the status endpoint as observability, but they do not participate in
the throttle decision.

### Enabling the guard

Provide a JSON profile via either the CLI flag or the env var:

```bash
mimirmind serve \
  --model /models/gemma-4-26B-A4B-it-Q6_K.gguf \
  --thermal-profile /etc/mimirmind/thermal-nuc14.json

# or

MIMIRMIND_THERMAL_PROFILE=/etc/mimirmind/thermal-nuc14.json \
  mimirmind serve --model ...
```

If **neither is set**, the engine starts unprotected and logs a loud
WARN on startup. There is no built-in default profile — every host has
different cooling and the wrong defaults could be worse than nothing.

### Profile structure

```json
{
  "name":        "nuc14-attic",
  "description": "free-form notes about this host",

  "package_temp_soft_c":      75,
  "package_temp_hard_c":      82,
  "package_throttle_max_ms":  200
}
```

Both `package_temp_soft_c` and `package_temp_hard_c` must be present
together — supplying only one is a load-time error. Leaving both out
disables monitoring entirely and the guard becomes a no-op, which is
useful for testing the wiring without temperature-gating.

`name` is required (free-form identifier shown in logs and
`/v1/system/status`). `description` is optional. Fields unknown to the
loader are silently ignored, so adding future-version knobs to a
profile file does not break old binaries.

### How the limit is enforced

The guard computes a "fraction of the way from soft to hard":

```
reading at soft → 0
reading at hard → 1
linear in between
```

The decode-loop sleep between tokens is `fraction × package_throttle_max_ms`.
Throttling ramps in smoothly as the package heats up rather than
flipping on at one threshold.

Once the package hits its hard limit:

- The current generate() call still finishes (it would be worse to
  hang a client mid-stream).
- New `POST /v1/chat/completions` requests get **`HTTP 503` with
  `Retry-After: 10`** until the host cools off.

### Reading the host

The guard pulls package temperature from two sources, in this order:

1. `/sys/class/thermal/thermal_zone*/type` looking for `x86_pkg_temp`
   (kernel-managed package temperature, always present on Intel).
2. `/sys/class/hwmon/hwmon*/name` == `coretemp` with a `tempN_label`
   reading `Package id 0` (fallback when the thermal zone is named
   differently).

Both paths are visible inside an unprivileged Docker container — no
`--privileged` or capability tweaks needed. On startup the guard
verifies the package-temperature sensor is readable and refuses to
start if it is missing; better to fail at boot than discover a silent
gap during the next thermal incident.

### Status endpoint

`GET /v1/system/status` returns the current readings, the active
profile, and the current throttle decision:

```bash
curl -s https://your-mimirmind/v1/system/status | jq
```

```json
{
  "profile_active": true,
  "profile": {
    "name": "nuc14-attic",
    "description": "ASUS NUC14RVS-B in attic ...",
    "package_temp_soft_c":     75,
    "package_temp_hard_c":     82,
    "package_throttle_max_ms": 200
  },
  "readings": {
    "package_temp_c":    78.5,
    "ram_total_mib":     65484,
    "ram_available_mib": 32100
  },
  "throttle": {
    "state":                "throttling",
    "current_pause_ms":     100,
    "next_request_allowed": true,
    "reason":               "package_temp_c=78.5 above soft=75.0"
  }
}
```

`state` is one of `ok`, `throttling`, `critical`. `next_request_allowed`
is the same condition the chat endpoint uses to decide whether to
return `503`.

**RAM is reported in `readings` even though it is not part of the
profile.** Treat it as observability for external monitors — useful
for spotting trends, building dashboards, or correlating temperature
spikes with memory pressure. It does not affect admission or pacing.

If no profile is configured, the body is just
`{"profile_active": false, "warning": "..."}` and the chat endpoints
run without thermal protection.

## Tuning a profile for a new host

1. Run the host idle with `mimirmind serve --thermal-profile
   /tmp/dry-run.json` where the file sets *very* permissive limits
   (e.g. `soft=99, hard=110`). Poll `/v1/system/status` for a while
   to confirm the sensor source resolved to the expected zone.
2. Pick a workload that mirrors production traffic (Gemma 4 26B Q6_K
   with `max_tokens=400` is a reasonable stress test).
3. Watch package temperature climb under load. Note the **steady-state
   ceiling** — the temperature it converges to when generating for a
   minute or more. Call that `T_ceiling`.
4. Set `package_temp_hard_c` to roughly `T_ceiling + 5°C` if your
   cooling has headroom, or `T_ceiling - 5°C` if it doesn't (i.e. you
   want the throttle to actually trigger and slow things down).
5. Set `package_temp_soft_c` to `hard - 5 to 10°C`. Wider gap = gentler
   ramp; narrower gap = bigger pause once it starts. Try 5°C first and
   adjust if the throttle keeps yo-yoing.
6. `package_throttle_max_ms = 200` is a good starting point. Higher
   means more aggressive cooling, lower means less hit to throughput
   when throttle does activate.

Two example profiles live under `examples/`:

- `examples/thermal-profile-nuc14.json` — conservative profile for a
  small-form-factor host with marginal cooling.
- `examples/thermal-profile-workstation.json` — permissive profile for
  a tower with proper cooling, intended as a "trust the silicon"
  baseline.

## GPU clock governor

The per-token pacing in the thermal guard works, but the latency
profile is bursty — full-speed compute then idle pauses. For
sustained workloads (long generations, RAG with big context) the
nicer answer is to **lower the iGPU max frequency continuously**
so the chip generates less heat per second to begin with. That is
what the GPU clock governor does.

The governor reads the iGPU's hardware frequency envelope from
`/sys/class/drm/cardN/gt_RP0_freq_mhz` (max) and `gt_RPn_freq_mhz`
(min) at startup, then writes `gt_max_freq_mhz` dynamically based
on the package temperature. The control loop is an **asymmetric
P-controller** with a deadband:

```
error = current_temp_c - target_temp_c
if |error| < 0.5:                 # deadband — sit still
    new_cap_mhz = current_cap_mhz
elif error > 0:                   # too hot — drop fast
    new_cap_mhz = current_cap_mhz - error * 100
else:                             # too cool — creep up
    new_cap_mhz = current_cap_mhz - error * 10
new_cap_mhz = clamp(new_cap_mhz, rpn_mhz, rp0_mhz)
```

The 10× ratio between the down-gain and the up-gain reflects the
package thermals: heat-up reaches steady state in seconds, cool-
down takes 30-60 seconds once heat is in the heatsink. A symmetric
controller overshoots on the cool side and re-spikes on the next
workload burst. The deadband (±0.5 °C) is below the resolution of
`x86_pkg_temp` and prevents the cap from twitching by ±1 MHz at
steady state.

The governor ticks every 8 decode tokens (~1.2 s at typical
Gemma 4 decode rates) — fast enough to react to a thermal climb,
slow enough that sysfs write cost is negligible.

### Enabling

Add `gpu_target_temp_c` to the thermal profile JSON:

```json
{
  "name": "nuc14-attic",
  "package_temp_soft_c":     78,
  "package_temp_hard_c":     85,
  "package_throttle_max_ms": 200,
  "gpu_target_temp_c":       72
}
```

The loader rejects `gpu_target_temp_c >= package_temp_soft_c`
because the governor is meant to keep the chip below soft — the
per-token pause is only a safety net behind it.

Two prerequisites for the governor to actually run:

1. **Sysfs is writable**: the LXC + Docker config exposes
   `/sys/class/drm/cardN/gt_max_freq_mhz` rw to the engine.
   See [`setup-ct.md`](./setup-ct.md#lxc--docker-configuration-for-mimirmind-features).
2. **`gpu_target_temp_c`** is set in the active profile.

If either is missing, the engine logs a clear WARN and continues
without the governor; the per-token pace acts as the fallback
thermal management.

### Status

`/v1/system/status` reports a `gpu_clock` block when the governor
is installed:

```json
"gpu_clock": {
  "available": true,
  "card_path": "/sys/class/drm/card1",
  "rp0_mhz": 2350,
  "rpn_mhz": 800,
  "current_cap_mhz": 1850,
  "target_temp_c": 72.0
}
```

The current cap is the live state — polling this every few seconds
gives a fairly accurate picture of how hard the chip is being pushed.
At steady state under heavy load you should see the cap settle some
hundred MHz below RP0, with package temp near target.

### Tuning notes

- **Default target 72 °C.** Conservative for a chip rated to
  Tjmax ~110 °C; about 6-10 °C below typical thermal-paste
  inflection. If your chassis has actual airflow, 80-85 °C is
  fine and gives more throughput.
- **Profile-soft / hard above target.** The intent is that the
  governor keeps temp at or below `gpu_target_temp_c`; the
  per-token pace only kicks in when something prevents the
  governor from doing its job (workload pinning the iGPU at min
  freq while CPU cores also hammer, ambient temperature spike).
  Set soft = target + 5-10°C, hard = target + 12-15°C.
- **What happens on power outage / reboot.** `gt_max_freq_mhz` is
  kernel-runtime state — not persisted to NVRAM. After any
  hardware reset the kernel re-initializes the i915 driver and
  starts at RP0 (the hardware envelope max from the GPU PROM).
  The engine's destructor also restores RP0 on clean shutdown so
  a Docker stop without engine restart still leaves the chip
  unconstrained.

### Interaction with the thermal guard

Both layers can be active at once and they're designed to
cooperate:

| Layer | When it acts | What it does |
|---|---|---|
| GPU clock governor | proactively, every 8 tokens | nudges iGPU max-freq toward keeping package at target |
| Thermal guard pacing | reactively, every 4 tokens above soft | inserts a per-token sleep proportional to (current - soft) / (hard - soft) |
| Thermal guard admission | per request | returns 503 if current >= hard at request start |

Under normal operation only the governor is active — package stays
below soft, pacing computes 0 ms pause, admission stays open. If
the governor cannot keep up (cap already at RPn but temp still
climbs), pacing kicks in. If even pacing can't hold the line and
hard is breached, new requests are refused.

## Power telemetry (RAPL)

The engine snapshots Intel RAPL energy counters around every
`generate()` call to report how many Joules each request consumed,
and `/v1/system/status` exposes both rolling Watts and total Joules
since startup per RAPL domain. This is observability only — the
guard never throttles based on power.

### What gets reported

In `/v1/system/status`:

```json
"power": {
  "available": true,
  "uptime_s": 3812.4,
  "domains": [
    {"name": "package-0", "watts_now": 27.4, "total_joules": 184320.0},
    {"name": "core",      "watts_now": 18.5, "total_joules": 102003.0},
    {"name": "uncore",    "watts_now":  7.9, "total_joules":  47210.0},
    {"name": "psys",      "watts_now": 31.2, "total_joules": 230500.0}
  ]
}
```

- `watts_now` is the average power between this `/v1/system/status` call
  and the previous one. Poll every 5 s → 5-second smoothing window.
- `total_joules` is energy consumed since the baseline snapshot, which
  is taken once at server start, **immediately after the model finishes
  loading**. Subtract before-and-after to get a request-scoped figure;
  divide by `uptime_s` for average power.
- `uptime_s` runs from the same baseline so the math is consistent.

In `chat.completion` responses, the `usage` block gains:

```json
"usage": {
  "prompt_tokens":     147,
  "completion_tokens":  32,
  "total_tokens":      179,
  "package_joules":      4.7
}
```

That is the energy the CPU package burned for **this specific
generate() call** — prefill plus decode plus the bookkeeping around
it. Streaming responses do not currently surface this in the SSE
events (no `usage` field per OpenAI shape), but it is logged on the
server side.

The server log line for every completed request includes
`energy=4.7J` so you can grep for it.

### Container access

RAPL counters live under `/sys/devices/virtual/powercap/intel-rapl/`
on Linux. Both Docker and unprivileged LXC mask that subtree by
default as a [Platypus side-channel] mitigation — relevant for crypto
key recovery in SGX enclaves, not for LLM inference.

[Platypus side-channel]: https://platypusattack.com/

mimirmind is set up so the operator can opt in to RAPL visibility at
two layers:

- **Docker layer.** `docker-compose.yml` for the `mimirmind` service
  sets `security_opt: systempaths=unconfined`, which unmasks the
  whole `/sys` tree from Docker's side. Already in the file at the
  repo root.
- **Proxmox LXC layer.** If you run mimirmind inside an LXC
  container, add to `/etc/pve/lxc/<CTID>.conf` on the Proxmox host:
  ```
  lxc.mount.entry: /sys/devices/virtual/powercap sys/devices/virtual/powercap none bind,ro,create=dir 0 0
  ```
  then reboot the CT.

When either layer is still masking the path, the monitor reports
unavailable instead of failing:

```json
"power": {
  "available": false,
  "reason": "intel-rapl directories found but energy_uj is not readable — Docker likely masks /sys/devices/virtual/powercap; set security_opt: systempaths=unconfined on the runtime service"
}
```

The engine still serves chat completions; the per-request
`package_joules` field is just omitted.

### What you can do with the numbers

- **Idle baseline.** Snapshot `total_joules` right after startup with
  no traffic, divide by `uptime_s` after a minute — that is your
  idle power draw. Useful for sizing PSU / UPS.
- **Per-request cost.** `package_joules` in the response is the
  immediate signal. At greedy Gemma 4 26B Q6_K decode with ~145 ms/tok,
  expect roughly 4-6 J per 30-token reply, scaling with token count.
- **Throttle correlation.** When `throttle.state == "throttling"`
  you should see `watts_now` plateau — the pacing pauses give the
  package room to cool. If it does not plateau, the throttle
  `package_throttle_max_ms` is too low or the workload's idle power
  is too close to the throttling threshold.
- **Sanity check on the thermal profile.** If `watts_now` regularly
  sits at half the chip's TDP rating while `package_temp_c` is at
  your `package_temp_hard_c`, the cooling is the bottleneck (not
  the silicon) — chassis / fan investigation needed.

## Watching it from outside

The status endpoint is cheap (one sysfs read per call, capped at
~250 ms refresh). A Prometheus scrape or a simple every-5-seconds curl
both work. Useful alerts:

- `throttle.state == "critical"` for more than 60 s — host can't keep
  up with traffic, time to throttle the upstream queue.
- `throttle.state == "throttling"` for more than 5 min — profile may
  be too aggressive, or workload has shifted upward.
- `readings.ram_available_mib` trending toward zero — even though the
  guard does not enforce it, low free RAM under sustained load is
  worth investigating (KV cache leak, model bigger than the host).
- `next_request_allowed == false` returning true → false transitions
  per minute count > N — sustained pressure on the temperature limit.

## What this does NOT do

- **No GPU temperature reading separately.** The Intel iGPU shares its
  silicon with the CPU cores; `x86_pkg_temp` already measures the
  combined die. A separate GPU sensor would mostly track the same
  number a few hundred ms later.
- **No fan / power telemetry.** The profile is purely advisory based
  on thermal headroom. If the underlying cooling solution is broken
  (fan stopped, heatsink lifted), nothing here will detect it directly
  — temperature will rise faster than expected and the guard will
  throttle to the maximum. That is the intended failure mode.
- **No RAM enforcement.** RAM is shown in the status endpoint as a
  signal for external monitoring, but the guard never refuses or
  paces a request based on RAM. Memory pressure on this engine is
  typically slow-moving (KV cache + loaded weights), so it does not
  fit the per-token decision the thermal guard makes. If RAM ever
  needs to gate requests, that lives in a separate config — not
  here.
- **No automatic profile generation.** Profile tuning is a deliberate
  per-host activity — see "Tuning a profile" above.