// Scatter a fused QKV matmul output into separate Q / K / V buffers.
//
// Fused output layout, row-major: [M, Nq + Nkv * (1 + hasV)]
// Per row m:
//   [m, 0        .. Nq-1                    ] → Y_q[m*Nq + i]
//   [m, Nq       .. Nq+Nkv-1                ] → Y_k[m*Nkv + (i - Nq)]
//   [m, Nq+Nkv   .. Nq+2*Nkv-1  (if hasV)   ] → Y_v[m*Nkv + (i - Nq - Nkv)]
//
// The kernel is dispatched with global_size = M * Nfused (1D). Y_v may
// be any valid pointer when hasV == 0 — the kernel never dereferences
// it because Nfused is set to Nq + Nkv in that case.

__kernel void qkv_split(
    __global const float* fused,
    __global       float* Yq,
    __global       float* Yk,
    __global       float* Yv,
    const int             M,
    const int             Nq,
    const int             Nkv,
    const int             hasV,
    const int             Nfused)
{
    const int idx = (int)get_global_id(0);
    const int total = M * Nfused;
    if (idx >= total) return;

    const int m = idx / Nfused;
    const int n = idx - m * Nfused;

    const float v = fused[idx];

    if (n < Nq) {
        Yq[m * Nq + n] = v;
    } else if (n < Nq + Nkv) {
        Yk[m * Nkv + (n - Nq)] = v;
    } else if (hasV != 0) {
        Yv[m * Nkv + (n - Nq - Nkv)] = v;
    }
}