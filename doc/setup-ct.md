# Setting up the runtime target for Mimirmind

> **TL;DR:** the recommended path is **Docker**. Pull / build the runtime
> image once, pass `/dev/dri` + groups `44`/`104`, done. The native install
> below is only needed if you want to run the binary directly on the host
> without Docker, or if you cannot use Docker at all on the target.

This document covers how to install the Intel compute runtime (Level Zero +
OpenCL ICD + Graphics Compiler) on the **runtime target** — a Linux host
or privileged LXC container that exposes the Intel iGPU through
`/dev/dri`.

The runtime target is expected to be set up with:

- Linux kernel ≥ 6.5 (for Meteor Lake `i915` / `xe` support).
- `/dev/dri/card0` and `/dev/dri/renderD128` visible to the container or
  host user.
- Owner groups `video` (GID **44**) and `render` (GID **104**) preserved
  through any LXC / Docker boundary.

Mimirmind itself is **not** built on the runtime target. We cross-build
SPIR-V on the developer machine and ship the binary + `.spv` files. The
runtime target only needs the loader, the GPU back-end and the ICD.

---

## Prerequisites

Quick sanity check on the target:

```bash
ls -la /dev/dri
intel_gpu_top -L      # part of intel-gpu-tools
```

`intel_gpu_top -L` should list the integrated GPU (`vendor=8086`,
`device=7d55` for Meteor Lake). If it does not, fix host-side
pass-through before doing anything else.

If you launch mimirmind from a Docker container, that container needs:

```yaml
devices:
    - /dev/dri:/dev/dri
group_add:
    - "44"     # video
    - "104"   # render (numeric — some Ubuntu base images do not name it)
```

The `docker-compose.yml` at the repo root already does this.

---

## Two installation paths

### Path A — Distribution-native packages (conservative)

Quickest, no third-party repos. Works for `ze_loader` + headers (so the
binary loads at runtime), but the **GPU back-end** (`intel-level-zero-gpu`)
in older distro repos may be too old to drive Meteor Lake correctly. Use
this only to get a loader-level smoke test going.

```bash
apt update
apt install -y \
    level-zero \
    level-zero-dev \
    clinfo
```

Expected outcome: `zeInit` succeeds, but enumeration may report zero GPU
devices, or report Meteor Lake as an unsupported / degraded device.
If that happens → switch to Path B.

### Path B — Intel client GPU repo (recommended)

This is what current Intel-iGPU stacks pull from. Adjust the suite name
(`jammy` for Ubuntu 22.04, `noble` for 24.04 — the repo is Ubuntu-flavoured
and installs cleanly on Debian 12 with the right pin). **No `--force-yes`,
no version-mismatch hacks** — if something refuses to install, stop and
diagnose.

```bash
# 1. Trust Intel's GPU signing key
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key \
    | gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg

# 2. Add the client repo
#    Replace "noble" with "jammy" on Ubuntu 22.04 / Debian 12.
echo "deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] \
https://repositories.intel.com/gpu/ubuntu noble client" \
    | tee /etc/apt/sources.list.d/intel-gpu.list

# 3. Pin so the Intel repo cannot flood other packages
cat > /etc/apt/preferences.d/intel-gpu <<'EOF'
Package: *
Pin: origin "repositories.intel.com"
Pin-Priority: 200

Package: intel-* libze* libigc* libigdgmm* level-zero* ocloc
Pin: origin "repositories.intel.com"
Pin-Priority: 1001

# Keep system libc / libstdc++ on the distro version
Package: libc6 libstdc++6 libgcc-s1
Pin: origin "repositories.intel.com"
Pin-Priority: -1
EOF

apt update

apt install -y \
    intel-opencl-icd \
    intel-level-zero-gpu \
    level-zero \
    level-zero-dev \
    libigc1 libigc2 libigdgmm12 libigdfcl1 \
    clinfo
```

If `apt` complains about `libstdc++6` or `libc6` upgrades from the Intel
repo, **decline**. Those are system-critical and the distro versions are
the source of truth.

---

## Verification

After either path:

```bash
# Loader visible?
ldconfig -p | grep -E 'libze_loader|libze_intel'

# Headers in place?
ls /usr/include/level_zero/ze_api.h

# OpenCL ICD detected?
clinfo -l

# What does Level Zero see?
cat > /tmp/zeprobe.cpp <<'EOF'
#include <level_zero/ze_api.h>
#include <cstdio>
int main() {
    if (zeInit(ZE_INIT_FLAG_GPU_ONLY) != ZE_RESULT_SUCCESS) {
        printf("zeInit failed\n"); return 1;
    }
    uint32_t n = 0;
    zeDriverGet(&n, nullptr);
    printf("drivers: %u\n", n);
    return 0;
}
EOF
g++ -std=c++20 /tmp/zeprobe.cpp -lze_loader -o /tmp/zeprobe && /tmp/zeprobe
```

Expected: `drivers: 1` (or more). `0` means the loader is installed but
the GPU back-end (`intel-level-zero-gpu`) is missing or too old.

---

## Smoke test with mimirmind

Once you have a built `mimirmind` binary from the developer machine, copy
it over and run:

```bash
./mimirmind
```

Expected first run (M1):

```
+------------------------------------------------------------+
|                          Mimirmind                         |
|       M1 - Level Zero device enumeration smoke test        |
+------------------------------------------------------------+
Found 1 Level-Zero device(s):

  * Intel(R) Arc(TM) Graphics  [GPU]
      vendor   : 0x8086
      deviceId : 0x7d55
      uuid     : ...
      compute  : 128 threads, 2250 MHz
      local mem: 59.42 GiB
      sub-devs : 0

Selected target device : Intel(R) Arc(TM) Graphics
Context created OK      : yes

M1 smoke test passed.
```

If `local mem` shows the full UMA size — that is the whole reason this
project exists. If it shows only a fraction (~4 GiB or so), the
single-allocation cap is biting enumeration; set both
`UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1` and
`UR_L0_USE_RELAXED_ALLOCATION_LIMITS=1` and retry.

---

## Running through Docker on the target (recommended)

The repo ships a `Dockerfile` (multi-stage build) and `docker-compose.yml`
that bundle the runtime image. **This is the primary path** — the host
needs only Docker + `/dev/dri` pass-through, no native Intel SDK.

```bash
docker compose build mimirmind
docker compose run --rm mimirmind
```

The compose file already passes `/dev/dri`, adds groups `44` (`video`) and
`104` (`render`), and exports the two `UR_L0_*` ENVs. The `models/`
directory is mounted read-only — drop your GGUF files there before
running.

If the host does have an Intel iGPU exposed but the container still reports
no devices, run the loader-level probe from the Verification section above
inside the container:

```bash
docker compose run --rm builder bash -lc '
  cat > /tmp/zeprobe.cpp <<EOF
  #include <level_zero/ze_api.h>
  #include <cstdio>
  int main() {
      zeInit(ZE_INIT_FLAG_GPU_ONLY);
      uint32_t n=0; zeDriverGet(&n,nullptr);
      printf("drivers: %u\n", n); return 0;
  }
  EOF
  g++ -std=c++20 /tmp/zeprobe.cpp -lze_loader -o /tmp/zeprobe && /tmp/zeprobe
'
```

---

## Known pitfalls

- **4 GiB single-allocation cap on Level Zero.** Mimirmind side-steps it
  by design (many small per-tensor allocations). Set both
  `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1` and
  `UR_L0_USE_RELAXED_ALLOCATION_LIMITS=1` as defensive ENVs anyway. The
  compose file does this.
- **The `render` group must map to GID 104** inside any container —
  Ubuntu / Debian base images do not always have that name, so use the
  numeric ID (`--group-add 104` for raw `docker run`, the compose file
  already does it).
- **OpenCL `OverrideMaxAllocSize`, `NEOReadDebugKeys`** etc. are
  OpenCL-NEO knobs only — they have **no effect** on the Level Zero
  path.
- **SYCL runtime bundles ignore `UR_L0_RELAXED`** — not relevant for
  mimirmind, since we link `libze_loader` directly and not through any
  bundled SYCL runtime.

---

## Cross-build / deploy

Two options once M5 lands and we start shipping SPIR-V kernels:

### Option 1 — push the runtime image (recommended)

Build the multi-stage image on the developer machine, push it to a
registry (or `docker save | ssh ... docker load`), pull on the target.

```bash
# Developer machine
docker compose build mimirmind
docker save mimirmind:dev | gzip > mimirmind-dev.tgz
scp mimirmind-dev.tgz <user>@<runtime-target>:/tmp/

# Runtime target
docker load < /tmp/mimirmind-dev.tgz
docker compose up mimirmind        # uses the loaded image, no rebuild
```

### Option 2 — bare binary + SPIR-V (no Docker on the target)

```bash
# Developer machine — build via docker, artefacts on host
docker compose run --rm builder

# Deploy
rsync -av --delete \
    build/mimirmind \
    build/spv/ \
    <user>@<runtime-target>:/opt/mimirmind/
```

`ocloc` lives only inside the builder image — the runtime target only
sees the produced `.spv` files and the `mimirmind` binary that loads them.