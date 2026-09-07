// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Qwen4-Exp PLE device-forward kernels (5.27 I-4c). The two ops the PLE layer
// needs beyond grouped-RMSNorm (hyper_connection.cu) and the BF16 projections:
//   ple_gate            — signed-sqrt stream gate + gated value broadcast
//   ple_conv_silu       — dilated depthwise causal conv1d + silu, with a decode
//                         state (prev short_conv_state_len rows)
// Math parity-verified in tools/microbench/q4e_ple_dev_{ref.py,parity.cu}.
// -arch=compute_80 (plain f32).

#define PLE_GATE_BLOCK 256

extern "C" {

// gate: block per (g,t); reduce dot over d; static-shared reduction.
//   g = (key·query)/sqrt(d); g = sign(g)*sqrt(max(|g|,1e-6)); s = sigmoid(g)
//   gated[t, g*d + j] = s * value[t, j]
__global__ void __launch_bounds__(PLE_GATE_BLOCK)
ple_gate(const float* __restrict__ key, const float* __restrict__ query,
         const float* __restrict__ value, float* __restrict__ gated,
         int T, int hc, int d) {
    const int g = blockIdx.x, t = blockIdx.y, hcd = hc * d;
    __shared__ float sm[PLE_GATE_BLOCK];
    const float* kr = key   + ((long)t * hcd + (long)g * d);
    const float* qr = query + ((long)t * hcd + (long)g * d);
    float acc = 0.0f;
    for (int j = threadIdx.x; j < d; j += blockDim.x) acc += kr[j] * qr[j];
    sm[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sm[threadIdx.x] += sm[threadIdx.x + s];
        __syncthreads();
    }
    float gate = sm[0] / sqrtf((float)d);
    gate = (gate < 0.0f ? -1.0f : 1.0f) * sqrtf(fmaxf(fabsf(gate), 1e-6f));
    const float sig = 1.0f / (1.0f + expf(-gate));
    const float* vr = value + (long)t * d;
    float* out = gated + ((long)t * hcd + (long)g * d);
    for (int j = threadIdx.x; j < d; j += blockDim.x) out[j] = sig * vr[j];
}

// dilated depthwise causal conv1d + silu with a decode state.
//   inpad = [state(stateLen rows) | x(T rows)] per channel c
//   out[t,c] = silu( sum_k w[c,k] * inpad[t + k*dilation, c] )
//   grid = (ceil(hcd/256), T), block 256; one thread per (t,c).
__global__ void ple_conv_silu(const float* __restrict__ x, const float* __restrict__ state,
                              const float* __restrict__ w, float* __restrict__ out,
                              int T, int hcd, int K, int dilation, int stateLen) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (c >= hcd || t >= T) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        const int idx = t + k * dilation - stateLen;   // < 0 -> into state, else x
        float v;
        if (idx >= 0) {
            v = x[(long)idx * hcd + c];
        } else {
            const int sidx = idx + stateLen;            // 0..stateLen-1
            v = (state != nullptr) ? state[(long)sidx * hcd + c] : 0.0f;
        }
        acc += w[(long)c * K + k] * v;
    }
    out[(long)t * hcd + c] = acc / (1.0f + expf(-acc));   // silu
}

} // extern "C"
