// FP16-KV variant of qkv_split.cl. Fused matmul output is fp32; Q goes
// to an fp32 workspace (`Yq`) unchanged, K and V go to the fp16 KV
// cache via vstore_half.
//
// Layout / dispatch geometry identical to the f32 variant so the host
// wiring only differs in the pointer types of `Yk`/`Yv`.
//
// M10.2 Phase 0 Commit 4 — invoked only when
// `KvCache::dtype() == FP16`; the f32 dispatch stays on qkv_split.cl,
// preserving bit-parity against pre-M10.2 behaviour.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void qkv_split_fp16(
    __global const float* fused,
    __global       float* Yq,
    __global       half*  Yk,
    __global       half*  Yv,
    const int             M,
    const int             Nq,
    const int             Nkv,
    const int             hasV,
    const int             Nfused,
    __global const int*   curLenPtr)
{
    const int idx   = (int)get_global_id(0);
    const int total = M * Nfused;
    if (idx >= total) return;

    const int m = idx / Nfused;
    const int n = idx - m * Nfused;
    const int curLen = curLenPtr[0];

    const float v = fused[idx];

    if (n < Nq) {
        Yq[m * Nq + n] = v;
    } else if (n < Nq + Nkv) {
        vstore_half(v, (curLen + m) * Nkv + (n - Nq), Yk);
    } else if (hasV != 0) {
        vstore_half(v, (curLen + m) * Nkv + (n - Nq - Nkv), Yv);
    }
}