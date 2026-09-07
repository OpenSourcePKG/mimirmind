// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Qwen4-Exp PLE device-forward parity harness (5.27 I-4c). The two NEW PLE
// kernels — the signed-sqrt gate (+gated value) and the dilated depthwise causal
// conv1d+silu — run on the GPU and are checked against the numpy reference
// (q4e_ple_dev_ref.py). These are the production kernels, lifted here to prove
// the math before wiring into Qwen4ExpBackend::pleForward. All f32.
//
//   nvcc -O2 -arch=native tools/microbench/q4e_ple_dev_parity.cu -o /tmp/q4e_ple_dev_parity
//   python3 tools/microbench/q4e_ple_dev_ref.py /tmp/q4e_ple_dev_dump
//   /tmp/q4e_ple_dev_parity /tmp/q4e_ple_dev_dump

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

namespace {
#define CK(c) do{cudaError_t e=(c); if(e){std::fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);std::exit(2);} }while(0)

std::vector<float> readBin(const std::string& p, std::size_t n) {
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "open %s\n", p.c_str()); std::exit(2); }
    std::vector<float> v(n);
    if (std::fread(v.data(), 4, n, f) != n) { std::fprintf(stderr, "read %s\n", p.c_str()); std::exit(2); }
    std::fclose(f); return v;
}

#define PLE_GATE_BLOCK 256

// gate: block per (g,t); reduce dot over d; gated[t,g,:] = sigmoid(signed_sqrt(dot/sqrt(d)))*value[t,:]
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
    sm[threadIdx.x] = acc; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (threadIdx.x < s) sm[threadIdx.x] += sm[threadIdx.x + s]; __syncthreads(); }
    float gate = sm[0] / sqrtf((float)d);
    gate = (gate < 0 ? -1.0f : 1.0f) * sqrtf(fmaxf(fabsf(gate), 1e-6f));
    const float s = 1.0f / (1.0f + expf(-gate));
    const float* vr = value + (long)t * d;
    float* out = gated + ((long)t * hcd + (long)g * d);
    for (int j = threadIdx.x; j < d; j += blockDim.x) out[j] = s * vr[j];
}

// dilated depthwise causal conv1d + silu. inpad = [zero state_len | x]; per channel c:
// out[t,c] = silu( sum_k w[c,k] * inpad[t + k*dilation, c] ), inpad[state_len+i]=x[i].
__global__ void ple_dilated_conv_silu(const float* __restrict__ x, const float* __restrict__ w,
                                      float* __restrict__ out, int T, int hcd, int K,
                                      int dilation, int stateLen) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (c >= hcd || t >= T) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        const int pos = t + k * dilation - stateLen;   // index into x (>=0), else zero state
        const float v = (pos >= 0) ? x[(long)pos * hcd + c] : 0.0f;
        acc += w[(long)c * K + k] * v;
    }
    out[(long)t * hcd + c] = acc / (1.0f + expf(-acc));   // silu
}

void cmp(const char* tag, const std::vector<float>& a, const std::vector<float>& b) {
    double mx = 0, ss = 0;
    for (std::size_t i = 0; i < b.size(); ++i) { double e = std::fabs((double)a[i] - b[i]); mx = e > mx ? e : mx; ss += e * e; }
    std::printf("  %-8s maxAbs=%.3e rmse=%.3e (n=%zu)\n", tag, mx, std::sqrt(ss / b.size()), b.size());
}
} // namespace

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "/tmp/q4e_ple_dev_dump";
    std::map<std::string, int> m;
    { FILE* f = std::fopen((dir + "/meta.txt").c_str(), "r"); if (!f) { std::fprintf(stderr, "no meta\n"); return 2; }
      char k[64]; int v; while (std::fscanf(f, "%63s %d", k, &v) == 2) m[k] = v; std::fclose(f); }
    const int T = m["T"], d = m["d"], hc = m["hc"], K = m["K"], dil = m["dilation"], sl = m["state_len"];
    const int hcd = hc * d;
    std::printf("q4e_ple_dev_parity: T=%d d=%d hc=%d K=%d dil=%d\n", T, d, hc, K, dil);

    auto key = readBin(dir + "/key_normed.bin", (size_t)T * hcd);
    auto qry = readBin(dir + "/query_normed.bin", (size_t)T * hcd);
    auto val = readBin(dir + "/value.bin", (size_t)T * d);
    auto gatedRef = readBin(dir + "/gated.bin", (size_t)T * hcd);
    auto convw = readBin(dir + "/convw.bin", (size_t)hcd * K);
    auto xconv = readBin(dir + "/xconv.bin", (size_t)T * hcd);
    auto convRef = readBin(dir + "/convout.bin", (size_t)T * hcd);

    float *dK, *dQ, *dV, *dG, *dW, *dX, *dO;
    CK(cudaMalloc(&dK, 4ull * key.size())); CK(cudaMalloc(&dQ, 4ull * qry.size()));
    CK(cudaMalloc(&dV, 4ull * val.size())); CK(cudaMalloc(&dG, 4ull * gatedRef.size()));
    CK(cudaMalloc(&dW, 4ull * convw.size())); CK(cudaMalloc(&dX, 4ull * xconv.size()));
    CK(cudaMalloc(&dO, 4ull * convRef.size()));
    CK(cudaMemcpy(dK, key.data(), 4ull * key.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dQ, qry.data(), 4ull * qry.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dV, val.data(), 4ull * val.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dW, convw.data(), 4ull * convw.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dX, xconv.data(), 4ull * xconv.size(), cudaMemcpyHostToDevice));

    { dim3 grid(hc, T); ple_gate<<<grid, PLE_GATE_BLOCK>>>(dK, dQ, dV, dG, T, hc, d); }
    { dim3 grid((hcd + 255) / 256, T); ple_dilated_conv_silu<<<grid, 256>>>(dX, dW, dO, T, hcd, K, dil, sl); }
    CK(cudaDeviceSynchronize());

    std::vector<float> gated(gatedRef.size()), convo(convRef.size());
    CK(cudaMemcpy(gated.data(), dG, 4ull * gated.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(convo.data(), dO, 4ull * convo.size(), cudaMemcpyDeviceToHost));

    std::printf("parity vs numpy:\n");
    cmp("gated", gated, gatedRef);
    cmp("conv", convo, convRef);
    double mx = 0;
    for (std::size_t i = 0; i < gatedRef.size(); ++i) mx = std::max(mx, std::fabs((double)gated[i] - gatedRef[i]));
    for (std::size_t i = 0; i < convRef.size(); ++i) mx = std::max(mx, std::fabs((double)convo[i] - convRef[i]));
    const bool pass = mx < 1e-3;
    std::printf("%s (max abs %.3e, tol 1e-3)\n", pass ? "PLE-DEV PARITY PASS" : "PLE-DEV PARITY FAIL", mx);
    return pass ? 0 : 1;
}
