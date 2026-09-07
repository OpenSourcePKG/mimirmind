// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Qwen4-Exp (qwen4_exp) Hyper-Connections elementwise kernels (5.27 I-3). The
// GatedResidual forward keeps a 4-stream residual state H [T, hc*d] (token-major:
// per row the hc streams are the concatenated d-chunks) and, around each attn/MoE
// module, mixes the streams into a single d-wide module input and scatters the
// module output back into every stream. The three matmuls (input_mix down/up,
// block_inject) go through the normal BF16 matmul; these kernels cover the
// elementwise glue. Math + parity verified in isolation by
// tools/microbench/q4e_hc_{ref.py,parity.cu} (max abs err ~1e-7).
//
// -arch=compute_80 (PTX baseline, JIT-forward-compat to Blackwell) — plain f32,
// no arch-specific intrinsics.

extern "C" {

#define HC_RMS_BLOCK 256

// Grouped RMSNorm: normalize each `group_size` chunk of the hc*d row
// independently, then scale by the (1+w)-baked weight [hcd].
//   grid = (hc, T), block = HC_RMS_BLOCK, static-shared reduction over group_size.
//   normed[t, g*gs + j] = x[..] * rsqrt(mean_j(x^2) + eps) * wBaked[g*gs + j]
__global__ void __launch_bounds__(HC_RMS_BLOCK)
hc_grouped_rmsnorm(const float* __restrict__ x,
                   const float* __restrict__ wBaked,
                   float* __restrict__ normed,
                   float eps, int hcd, int group_size) {
    const int g    = blockIdx.x;   // group / stream index
    const int t    = blockIdx.y;   // row (token)
    const int base = g * group_size;
    const long row = (long)t * hcd;

    __shared__ float sm[HC_RMS_BLOCK];
    float acc = 0.0f;
    for (int j = threadIdx.x; j < group_size; j += blockDim.x) {
        const float v = x[row + base + j];
        acc += v * v;
    }
    sm[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sm[threadIdx.x] += sm[threadIdx.x + s];
        __syncthreads();
    }
    const float inv = rsqrtf(sm[0] / (float)group_size + eps);
    for (int j = threadIdx.x; j < group_size; j += blockDim.x) {
        const int c = base + j;
        normed[row + c] = x[row + c] * inv * wBaked[c];
    }
}

// In-place silu with a pre-scale: x = silu(x * scale) = (x*scale) * sigmoid(x*scale).
//   grid = ceil(n / 256), block = 256.
__global__ void hc_silu_scale(float* __restrict__ x, int n, float scale) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float z = x[i] * scale;
    x[i] = z / (1.0f + __expf(-z));
}

// Weighted mean over the hc streams:
//   mixed[t, j] = (1/hc) * sum_g w2[t, g*d + j] * normed[t, g*d + j]
//   grid = (ceil(d/256), T), block = 256.
__global__ void hc_weighted_mean_streams(const float* __restrict__ w2,
                                         const float* __restrict__ normed,
                                         float* __restrict__ mixed,
                                         int T, int hc, int d) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (j >= d || t >= T) return;
    const int  hcd = hc * d;
    const long row = (long)t * hcd;
    float acc = 0.0f;
    for (int g = 0; g < hc; ++g) acc += w2[row + g * d + j] * normed[row + g * d + j];
    mixed[(long)t * d + j] = acc / (float)hc;
}

// Injection scatter (the Hyper-Connections residual add):
//   x[t, g*d + j] += inj[t, g] * moduleOut[t, j]     for every stream g
//   grid = (ceil(hcd/256), T), block = 256; one thread per (t, c) over c in [0,hcd).
__global__ void hc_inject_scatter(float* __restrict__ x,
                                  const float* __restrict__ moduleOut,
                                  const float* __restrict__ inj,
                                  int T, int hc, int d) {
    const int hcd = hc * d;
    const int c   = blockIdx.x * blockDim.x + threadIdx.x;
    const int t   = blockIdx.y;
    if (c >= hcd || t >= T) return;
    const int g = c / d;
    const int j = c - g * d;
    x[(long)t * hcd + c] += inj[(long)t * hc + g] * moduleOut[(long)t * d + j];
}

// Stream broadcast (embed repeat x hc at forward start):
//   dst[t, g*d + j] = src[t, j]   for every stream g
//   grid = (ceil(hcd/256), T), block = 256.
__global__ void hc_stream_broadcast(const float* __restrict__ src,
                                    float* __restrict__ dst,
                                    int T, int hc, int d) {
    const int hcd = hc * d;
    const int c   = blockIdx.x * blockDim.x + threadIdx.x;
    const int t   = blockIdx.y;
    if (c >= hcd || t >= T) return;
    const int j = c - (c / d) * d;
    dst[(long)t * hcd + c] = src[(long)t * d + j];
}

} // extern "C"
